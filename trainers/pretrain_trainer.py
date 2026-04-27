from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.config import ExperimentConfig
from dataloader.data_reader import DataReader
from dataloader.dataset_graph import GraphDatasetParser, GraphPretrainDataset, collate_graph
from dataloader.sample_negative import NegativeSampler
from model.lightgcn import LightGCN
from utils.losses import bpr_loss
from utils.metrics import evaluate_all_ranking, get_user_item_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PretrainResult:
    best_epoch: int
    best_ndcg20: float
    checkpoint_path: str


class PretrainTrainer:
    """
    Object-oriented collaborative LightGCN pretraining pipeline.

    Stages:
      1) Build and persist parser
      2) Train LightGCN with BPR loss
      3) Evaluate best checkpoint on test split
    """

    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path
        self.cfg = ExperimentConfig.from_yaml(cfg_path)
        set_seed(int(self.cfg.seed))
        self.device = torch.device(
            self.cfg.pretrain.device if torch.cuda.is_available() else "cpu"
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.parser_path = f"{self.cfg.data.dataset}_parser.pkl"

    def build_parser(self) -> GraphDatasetParser:
        reader = DataReader(
            self.cfg.data.data_dir,
            min_user_interactions=self.cfg.data.min_user_interactions,
            min_item_interactions=self.cfg.data.min_item_interactions,
        )
        train, val, test = reader.load_all()

        parser = GraphDatasetParser(train, val, test)
        parser.remap_ids()
        parser.build_user_pos_items()
        parser.build_adj_mat()

        with open(self.parser_path, "wb") as f:
            pickle.dump(parser, f)
        return parser

    def load_parser(self) -> GraphDatasetParser:
        if not os.path.exists(self.parser_path):
            return self.build_parser()
        with open(self.parser_path, "rb") as f:
            return pickle.load(f)

    def _build_model(self, parser: GraphDatasetParser) -> LightGCN:
        return LightGCN(
            num_users=parser.num_users,
            num_items=parser.num_items,
            embedding_dim=self.cfg.lightgcn.embedding_dim,
            n_layers=self.cfg.lightgcn.n_layers,
            adj_mat=parser.adj_mat,
        ).to(self.device)

    def train(self) -> PretrainResult:
        parser = self.load_parser()
        dataset = GraphPretrainDataset(
            train_pairs=parser.train,
            user_pos_items=parser.user_pos_items,
        )
        neg_sampler = NegativeSampler(parser.num_items, parser.user_pos_items)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.pretrain.batch_size,
            shuffle=True,
            num_workers=self.cfg.pretrain.num_workers,
            collate_fn=lambda batch: collate_graph(batch, neg_sampler),
        )

        model = self._build_model(parser)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.pretrain.lr,
            weight_decay=float(self.cfg.pretrain.weight_decay),
        )

        save_path = self.cfg.pretrain.save_path
        if save_path is None:
            save_path = str(Path("checkpoints") / f"lightgcn_best_{self.cfg.data.dataset}_pretrain.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        best_epoch = -1
        best_ndcg20 = -1.0
        eval_interval = 10
        train_user_items = get_user_item_dict(parser.train)
        val_user_items = get_user_item_dict(parser.val)

        epoch_bar = tqdm(range(self.cfg.pretrain.epochs), desc="Pretrain", dynamic_ncols=True)
        for epoch in epoch_bar:
            model.train()
            running_loss = 0.0

            batch_bar = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{self.cfg.pretrain.epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            for batch in batch_bar:
                users = batch["user"].to(self.device)
                pos_items = batch["pos"].to(self.device)
                neg_items = batch["neg"].to(self.device)

                user_g, pos_g, neg_g = model(users, pos_items, neg_items)
                user_id_emb = model.user_embedding(users)
                pos_id_emb = model.item_embedding(pos_items)
                neg_id_emb = model.item_embedding(neg_items)

                loss = bpr_loss(
                    z_user=user_g,
                    z_pos=pos_g,
                    z_neg=neg_g,
                    reg=float(self.cfg.pretrain.reg),
                    user_id_emb=user_id_emb,
                    pos_id_emb=pos_id_emb,
                    neg_id_emb=neg_id_emb,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                batch_bar.set_postfix(loss=f"{running_loss / max(1, batch_bar.n):.4f}")

            avg_loss = running_loss / max(1, len(loader))
            epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")

            if (epoch + 1) % eval_interval != 0:
                continue

            model.eval()
            with torch.no_grad():
                recall_res, ndcg_res = evaluate_all_ranking(
                    model=model,
                    users=torch.LongTensor(list(val_user_items.keys())).to(self.device),
                    train_user_items=train_user_items,
                    eval_user_items=val_user_items,
                    K=[10, 20],
                    device=self.device,
                )
            cur_ndcg20 = float(ndcg_res[20])
            if cur_ndcg20 > best_ndcg20:
                best_ndcg20 = cur_ndcg20
                best_epoch = epoch + 1
                torch.save(model.state_dict(), save_path)

        if best_epoch < 0:
            # Ensure at least one checkpoint is saved for short runs.
            torch.save(model.state_dict(), save_path)
            best_epoch = self.cfg.pretrain.epochs
            best_ndcg20 = 0.0

        return PretrainResult(
            best_epoch=best_epoch,
            best_ndcg20=best_ndcg20,
            checkpoint_path=save_path,
        )

    def test(self, checkpoint_path: str | None = None) -> dict[str, float]:
        parser = self.load_parser()
        model = self._build_model(parser)
        ckpt = checkpoint_path or self.cfg.pretrain.save_path
        if ckpt is None:
            raise ValueError("No checkpoint path provided for test.")
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.eval()

        recall_res, ndcg_res = evaluate_all_ranking(
            model=model,
            users=torch.LongTensor(list(get_user_item_dict(parser.test).keys())).to(self.device),
            train_user_items=get_user_item_dict(parser.train),
            eval_user_items=get_user_item_dict(parser.test),
            K=[10, 20],
            device=self.device,
        )
        return {
            "test_recall@10": float(recall_res[10]),
            "test_ndcg@10": float(ndcg_res[10]),
            "test_recall@20": float(recall_res[20]),
            "test_ndcg@20": float(ndcg_res[20]),
        }
