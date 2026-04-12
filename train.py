from __future__ import annotations

import argparse
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import ExperimentConfig
from dataloader.data_reader import load_pickle_compat
from dataloader.dataset_graph import GraphPretrainDataset, collate_graph
from dataloader.manager import GeneralItemProfileManager
from dataloader.sample_negative import NegativeSampler
from model.lightgcn import LightGCN, LightGCN_retrain
from utils.losses import bpr_loss
from utils.metrics import evaluate_all_ranking, get_user_item_dict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_profile_embedding_dict(raw_emb, num_items: int) -> dict[int, torch.Tensor]:
    if isinstance(raw_emb, dict):
        result = {}
        for k, v in raw_emb.items():
            iid = int(k)
            if 0 <= iid < num_items:
                result[iid] = v if torch.is_tensor(v) else torch.as_tensor(v)
        return result

    tensor_emb = raw_emb if torch.is_tensor(raw_emb) else torch.as_tensor(raw_emb)
    if tensor_emb.ndim != 2:
        raise ValueError(f"item embeddings must be 2D, got shape={tuple(tensor_emb.shape)}")
    cap = min(num_items, tensor_emb.size(0))
    return {iid: tensor_emb[iid] for iid in range(cap)}


def _load_item_profile_embeddings(cfg: ExperimentConfig, parser) -> dict[int, torch.Tensor]:
    emb_path = os.path.join(cfg.data.data_dir, "itm_emb_np.pkl")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing item embedding file: {emb_path}")
    raw = load_pickle_compat(emb_path)
    return _to_profile_embedding_dict(raw, parser.num_items)


def _load_user_semantic_features(cfg: ExperimentConfig, parser, device: torch.device):
    user_emb_path = os.path.join(cfg.data.data_dir, "usr_emb_np.pkl")
    if not os.path.exists(user_emb_path):
        cluster_emb = torch.zeros((1, 1024), dtype=torch.float32, device=device)
        user_cluster = torch.zeros(parser.num_users, dtype=torch.long, device=device)
        return cluster_emb, user_cluster

    raw_user_emb = load_pickle_compat(user_emb_path)
    user_emb = torch.as_tensor(raw_user_emb, dtype=torch.float32, device=device)
    if user_emb.ndim != 2:
        raise ValueError(f"user embeddings must be 2D, got shape={tuple(user_emb.shape)}")

    if user_emb.size(0) < parser.num_users:
        pad = torch.zeros(
            (parser.num_users - user_emb.size(0), user_emb.size(1)),
            dtype=user_emb.dtype,
            device=device,
        )
        user_emb = torch.cat([user_emb, pad], dim=0)
    else:
        user_emb = user_emb[:parser.num_users]

    user_cluster = torch.arange(parser.num_users, dtype=torch.long, device=device)
    return user_emb, user_cluster


def train(cfg_path: str):
    cfg = ExperimentConfig.from_yaml(cfg_path)
    set_seed(cfg.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    parser_path = f"{cfg.data.dataset}_parser.pkl"
    if not os.path.exists(parser_path):
        raise FileNotFoundError(
            f"Missing parser: {parser_path}. "
            "Run `python pretrain.py --config <config>` first."
        )
    with open(parser_path, "rb") as f:
        parser = pickle.load(f)

    manager = GeneralItemProfileManager(
        dataset_name=cfg.data.dataset,
        parser=parser,
        profile_path=cfg.data.profile_path,
    )
    item_profiles = manager.load(format=cfg.data.dataset)
    parser.item_profiles = item_profiles
    print(f"Loaded {len(item_profiles)} item profiles.")

    pretrain_model = LightGCN(
        num_users=parser.num_users,
        num_items=parser.num_items,
        embedding_dim=cfg.lightgcn.embedding_dim,
        n_layers=cfg.lightgcn.n_layers,
        adj_mat=parser.adj_mat,
    ).to("cpu")
    pretrain_model.load_state_dict(torch.load(cfg.pretrain.save_path, map_location=device))
    pretrain_model.eval()

    item_profile_embeddings = _load_item_profile_embeddings(cfg, parser)
    cluster_emb, user_cluster = _load_user_semantic_features(cfg, parser, device)

    model = LightGCN_retrain(
        num_users=parser.num_users,
        num_items=parser.num_items,
        embedding_dim=cfg.lightgcn.embedding_dim,
        n_layers=cfg.lightgcn.n_layers,
        adj_mat=parser.adj_mat,
        cluster_emb=cluster_emb,
        user_cluster=user_cluster,
        item_profile_embeddings=item_profile_embeddings,
        device=device,
    ).to(device)

    missing, unexpected = model.load_state_dict(pretrain_model.state_dict(), strict=False)
    print("Model initialized from pretrain checkpoint.")
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=float(cfg.train.weight_decay),
    )

    dataset = GraphPretrainDataset(
        train_pairs=parser.train,
        user_pos_items=parser.user_pos_items,
    )
    neg_sample = NegativeSampler(parser.num_items, parser.user_pos_items)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=lambda batch: collate_graph(batch, neg_sample),
    )

    best_ndcg20 = -1.0
    best_epoch = -1
    save_path = cfg.train.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Starting retrain with fusion...")
    eval_interval = max(1, getattr(cfg.train, "eval_interval", 1))
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            users = batch["user"].to(device)
            pos_items = batch["pos"].to(device)
            neg_items = batch["neg"].to(device)

            user_g, pos_g, neg_g = model(users, pos_items, neg_items)
            user_id_emb = model.user_embedding(users)
            pos_id_emb = model.item_embedding(pos_items)
            neg_id_emb = model.item_embedding(neg_items)

            loss = bpr_loss(
                z_user=user_g,
                z_pos=pos_g,
                z_neg=neg_g,
                reg=float(cfg.train.reg),
                user_id_emb=user_id_emb,
                pos_id_emb=pos_id_emb,
                neg_id_emb=neg_id_emb,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))
        print(f"[Epoch {epoch+1}/{cfg.train.epochs}] Train Loss: {avg_loss:.4f}")

        if (epoch + 1) % eval_interval != 0:
            continue

        model.eval()
        with torch.no_grad():
            recall_res, ndcg_res = evaluate_all_ranking(
                model,
                users=torch.LongTensor(list(get_user_item_dict(parser.val).keys())).to(device),
                train_user_items=get_user_item_dict(parser.train),
                eval_user_items=get_user_item_dict(parser.val),
                K=[10, 20],
                device=device,
            )

        print(
            f"[Val @ Epoch {epoch+1}] "
            f"Recall@10={recall_res[10]:.4f}, NDCG@10={ndcg_res[10]:.4f}, "
            f"Recall@20={recall_res[20]:.4f}, NDCG@20={ndcg_res[20]:.4f}"
        )

        if ndcg_res[20] > best_ndcg20:
            best_ndcg20 = ndcg_res[20]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {best_epoch} with NDCG@20={best_ndcg20:.4f}")

    print(f"Retrain completed. Best NDCG@20={best_ndcg20:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yelp.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    train(args.config)
