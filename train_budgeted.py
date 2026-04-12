from __future__ import annotations

import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import ExperimentConfig
from dataloader.dataset_graph import GraphPretrainDataset, collate_graph
from dataloader.manager import GeneralItemProfileManager
from dataloader.sample_negative import NegativeSampler
from model.lightgcn import LightGCN, LightGCNBudgetedSemantic
from utils.losses import (
    bpr_loss,
    semantic_alignment_loss,
    embedding_consistency_loss,
)
from utils.metrics import evaluate_all_ranking, get_user_item_dict
from utils.semantic_acquisition import (
    ItemBudgetSelector,
    LocalLlamaSemanticAcquirer,
    SemanticTextEncoder,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_parser(dataset: str):
    parser_path = f"{dataset}_parser.pkl"
    if not os.path.exists(parser_path):
        raise FileNotFoundError(
            f"Missing parser file: {parser_path}. "
            "Please run `python pretrain.py --config <your_config>` first."
        )
    with open(parser_path, "rb") as f:
        parser = pickle.load(f)
    return parser


def _save_semantic_artifacts(
    save_dir: str,
    selected_items: torch.Tensor,
    semantic_dict: dict[int, dict],
    semantic_emb: dict[int, torch.Tensor],
    score: torch.Tensor,
):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "selected_item_ids.json"), "w", encoding="utf-8") as f:
        json.dump([int(i) for i in selected_items.tolist()], f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "selected_item_semantics.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in semantic_dict.items()}, f, ensure_ascii=False, indent=2)
    torch.save(
        {int(k): v.detach().cpu() for k, v in semantic_emb.items()},
        os.path.join(save_dir, "selected_item_semantic_embeddings.pt"),
    )
    torch.save(score.detach().cpu(), os.path.join(save_dir, "item_budget_score.pt"))


def train(cfg_path: str):
    cfg = ExperimentConfig.from_yaml(cfg_path)
    set_seed(cfg.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = _load_parser(cfg.data.dataset)
    print(f"Loaded parser: users={parser.num_users}, items={parser.num_items}")

    manager = GeneralItemProfileManager(
        dataset_name=cfg.data.dataset,
        parser=parser,
        profile_path=cfg.data.profile_path,
    )
    item_profiles = manager.load(format=cfg.data.dataset)
    print(f"Loaded {len(item_profiles)} item profiles.")

    pretrain_model = LightGCN(
        num_users=parser.num_users,
        num_items=parser.num_items,
        embedding_dim=cfg.lightgcn.embedding_dim,
        n_layers=cfg.lightgcn.n_layers,
        adj_mat=parser.adj_mat,
    ).to(device)
    pretrain_model.load_state_dict(torch.load(cfg.pretrain.save_path, map_location=device))
    pretrain_model.eval()
    print(f"Loaded pretrained checkpoint from {cfg.pretrain.save_path}")

    with torch.no_grad():
        item_emb_layers = pretrain_model.propagate_with_layers()
        item_id_emb = pretrain_model.item_embedding.weight.detach()

    selector = ItemBudgetSelector(parser, popularity_penalty=cfg.semantic.popularity_penalty)
    selected_items, budget_score = selector.select(
        item_emb_layers=item_emb_layers,
        item_id_emb=item_id_emb,
        budget_ratio=cfg.semantic.budget_ratio,
        min_selected_items=cfg.semantic.min_selected_items,
    )
    print(
        f"Budgeted semantic acquisition selected {selected_items.numel()} "
        f"/ {parser.num_items} items (ratio={cfg.semantic.budget_ratio})."
    )

    semantic_dir = os.path.join(cfg.semantic.cache_dir, cfg.data.dataset)
    acquirer = LocalLlamaSemanticAcquirer(
        llama_url=cfg.semantic.llama_url,
        llama_model=cfg.semantic.llama_model,
        max_tokens=cfg.semantic.llama_max_tokens,
        temperature=cfg.semantic.llama_temperature,
        timeout=cfg.semantic.request_timeout,
        cache_dir=os.path.join(semantic_dir, "llama_cache"),
        disable_proxy_for_local=cfg.semantic.disable_proxy_for_local,
    )
    selected_semantic = acquirer.acquire_batch(
        item_ids=[int(i) for i in selected_items.tolist()],
        item_profiles=item_profiles,
    )
    print(f"Acquired semantic annotations for {len(selected_semantic)} items.")

    encoder = SemanticTextEncoder(
        model_path=cfg.semantic.embedding_model_path,
        device=str(device),
        batch_size=cfg.semantic.embedding_batch_size,
    )
    semantic_emb = encoder.encode(selected_semantic)
    if len(semantic_emb) == 0:
        raise RuntimeError("No semantic embeddings were produced.")

    semantic_dim = int(next(iter(semantic_emb.values())).numel())
    selected_mask = torch.zeros(parser.num_items, dtype=torch.float32)
    selected_mask[selected_items] = 1.0
    print(f"Semantic embedding dim: {semantic_dim}")

    if cfg.semantic.save_artifacts:
        _save_semantic_artifacts(
            save_dir=semantic_dir,
            selected_items=selected_items,
            semantic_dict=selected_semantic,
            semantic_emb=semantic_emb,
            score=budget_score,
        )
        print(f"Saved semantic artifacts under {semantic_dir}")

    model = LightGCNBudgetedSemantic(
        num_users=parser.num_users,
        num_items=parser.num_items,
        embedding_dim=cfg.lightgcn.embedding_dim,
        n_layers=cfg.lightgcn.n_layers,
        adj_mat=parser.adj_mat,
        item_semantic_embeddings=semantic_emb,
        selected_item_mask=selected_mask,
        semantic_dim=semantic_dim,
        device=device,
    ).to(device)
    missing, unexpected = model.load_state_dict(pretrain_model.state_dict(), strict=False)
    print("Initialized budgeted model from pretrained LightGCN.")
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=float(cfg.train.weight_decay),
    )

    def set_backbone_trainable(trainable: bool):
        model.user_embedding.weight.requires_grad = trainable
        model.item_embedding.weight.requires_grad = trainable

    best_ndcg20 = -1.0
    best_epoch = -1
    os.makedirs(os.path.dirname(cfg.train.save_path), exist_ok=True)

    print("Start budgeted semantic training...")
    for epoch in range(cfg.train.epochs):
        freeze = epoch < int(cfg.semantic.freeze_backbone_epochs)
        set_backbone_trainable(not freeze)

        model.train()
        total_loss = 0.0
        total_rank = 0.0
        total_align = 0.0
        total_cons = 0.0

        for batch in loader:
            users = batch["user"].to(device)
            pos_items = batch["pos"].to(device)
            neg_items = batch["neg"].to(device)

            out = model(users, pos_items, neg_items)
            user_id_emb = model.user_embedding(users)
            pos_id_emb = model.item_embedding(pos_items)
            neg_id_emb = model.item_embedding(neg_items)

            loss_rank = bpr_loss(
                z_user=out["user"],
                z_pos=out["pos_fused"],
                z_neg=out["neg_fused"],
                reg=float(cfg.train.reg),
                user_id_emb=user_id_emb,
                pos_id_emb=pos_id_emb,
                neg_id_emb=neg_id_emb,
            )

            loss_align = (
                semantic_alignment_loss(out["pos_base"], out["pos_sem_proj"], out["pos_mask"])
                + semantic_alignment_loss(out["neg_base"], out["neg_sem_proj"], out["neg_mask"])
            )
            loss_cons = (
                embedding_consistency_loss(out["pos_fused"], out["pos_base"], out["pos_mask"])
                + embedding_consistency_loss(out["neg_fused"], out["neg_base"], out["neg_mask"])
            )

            loss = (
                loss_rank
                + float(cfg.semantic.align_weight) * loss_align
                + float(cfg.semantic.consistency_weight) * loss_cons
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_rank += float(loss_rank.item())
            total_align += float(loss_align.item())
            total_cons += float(loss_cons.item())

        num_batches = max(1, len(loader))
        print(
            f"[Epoch {epoch+1}/{cfg.train.epochs}] "
            f"loss={total_loss/num_batches:.4f} "
            f"rank={total_rank/num_batches:.4f} "
            f"align={total_align/num_batches:.4f} "
            f"cons={total_cons/num_batches:.4f} "
            f"freeze_backbone={freeze}"
        )

        if (epoch + 1) % max(1, cfg.train.eval_interval) != 0:
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
            torch.save(model.state_dict(), cfg.train.save_path)
            print(f"Best model saved to {cfg.train.save_path}")

    print(f"Training completed. best_ndcg20={best_ndcg20:.4f} at epoch={best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yelp_.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    train(args.config)
