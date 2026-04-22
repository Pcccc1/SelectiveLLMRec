from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import ExperimentConfig
from dataloader.data_reader import load_pickle_compat
from dataloader.dataset_graph import GraphPretrainDataset, collate_graph
from dataloader.manager import GeneralItemProfileManager
from dataloader.sample_negative import NegativeSampler
from model.lightgcn import LightGCN, LightGCNBudgetedSemantic
from utils.losses import (
    bpr_loss,
    embedding_consistency_loss,
    fusion_gate_l2_loss,
    semantic_alignment_loss,
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


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _save_semantic_artifacts(
    save_dir: str,
    selected_items: torch.Tensor,
    semantic_dict: dict[int, dict[str, Any]],
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


def _load_profile_embedding_matrix(path: str, num_items: int) -> torch.Tensor:
    raw = load_pickle_compat(path)
    if isinstance(raw, dict):
        sample = None
        for v in raw.values():
            sample = torch.as_tensor(v).view(-1)
            break
        if sample is None:
            raise ValueError(f"Empty embedding dict: {path}")
        matrix = torch.zeros((num_items, int(sample.numel())), dtype=torch.float32)
        for k, v in raw.items():
            iid = int(k)
            if 0 <= iid < num_items:
                vv = torch.as_tensor(v).view(-1).float()
                dim = min(vv.numel(), matrix.size(1))
                matrix[iid, :dim] = vv[:dim]
        return matrix

    emb = torch.as_tensor(raw).float()
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D item embedding matrix, got shape={tuple(emb.shape)} from {path}")

    if emb.size(0) < num_items:
        pad = torch.zeros((num_items - emb.size(0), emb.size(1)), dtype=emb.dtype)
        emb = torch.cat([emb, pad], dim=0)
    return emb[:num_items]


def _acquire_selected_semantics_llm(
    cfg: ExperimentConfig,
    selected_items: torch.Tensor,
    item_profiles: dict[int, Any],
    device: torch.device,
    semantic_dir: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, torch.Tensor], int]:
    acquirer = LocalLlamaSemanticAcquirer(
        llama_url=cfg.semantic.llama_url,
        llama_model=cfg.semantic.llama_model,
        max_tokens=cfg.semantic.llama_max_tokens,
        temperature=cfg.semantic.llama_temperature,
        timeout=cfg.semantic.request_timeout,
        cache_dir=os.path.join(semantic_dir, "llama_cache"),
        llm_fail_fast=cfg.semantic.llm_fail_fast,
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
    return selected_semantic, semantic_emb, semantic_dim


def _acquire_selected_semantics_profile_embedding(
    cfg: ExperimentConfig,
    parser,
    selected_items: torch.Tensor,
) -> tuple[dict[int, dict[str, Any]], dict[int, torch.Tensor], int]:
    emb_path = cfg.semantic.profile_embedding_path or os.path.join(cfg.data.data_dir, "itm_emb_np.pkl")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing profile embedding file: {emb_path}")

    matrix = _load_profile_embedding_matrix(path=emb_path, num_items=parser.num_items)
    semantic_emb = {int(iid): matrix[int(iid)].clone() for iid in selected_items.tolist()}
    semantic_dim = int(matrix.size(1))
    print(f"Loaded profile embeddings from {emb_path} with dim={semantic_dim}.")
    return {}, semantic_emb, semantic_dim


def _evaluate_split(
    model,
    parser,
    train_user_items: dict[int, set[int]],
    split_name: str,
    split_data: list[tuple[int, int]],
    device: torch.device,
) -> dict[str, float]:
    eval_users = list(get_user_item_dict(split_data).keys())
    recall, ndcg = evaluate_all_ranking(
        model=model,
        users=torch.LongTensor(eval_users).to(device),
        train_user_items=train_user_items,
        eval_user_items=get_user_item_dict(split_data),
        K=[10, 20],
        device=device,
    )
    return {
        f"{split_name}_recall@10": float(recall[10]),
        f"{split_name}_ndcg@10": float(ndcg[10]),
        f"{split_name}_recall@20": float(recall[20]),
        f"{split_name}_ndcg@20": float(ndcg[20]),
    }


def _apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    seed = getattr(args, "seed", None)
    if seed is not None:
        cfg.seed = int(seed)

    epochs = getattr(args, "epochs", None)
    if epochs is not None:
        cfg.train.epochs = int(epochs)

    eval_interval = getattr(args, "eval_interval", None)
    if eval_interval is not None:
        cfg.train.eval_interval = int(eval_interval)

    budget_ratio = getattr(args, "budget_ratio", None)
    if budget_ratio is not None:
        cfg.semantic.budget_ratio = float(budget_ratio)

    align_weight = getattr(args, "align_weight", None)
    if align_weight is not None:
        cfg.semantic.align_weight = float(align_weight)

    consistency_weight = getattr(args, "consistency_weight", None)
    if consistency_weight is not None:
        cfg.semantic.consistency_weight = float(consistency_weight)

    gate_l2_weight = getattr(args, "gate_l2_weight", None)
    if gate_l2_weight is not None:
        cfg.semantic.gate_l2_weight = float(gate_l2_weight)

    semantic_source = getattr(args, "semantic_source", None)
    if semantic_source is not None:
        cfg.semantic.source = str(semantic_source)

    save_path = getattr(args, "save_path", None)
    if save_path is not None:
        cfg.train.save_path = str(save_path)

    run_name = getattr(args, "run_name", None)
    if run_name is not None:
        cfg.train.run_name = str(run_name)

    selection_metric = getattr(args, "selection_metric", None)
    if selection_metric is not None:
        cfg.train.selection_metric = str(selection_metric)

    lr = getattr(args, "lr", None)
    if lr is not None:
        cfg.train.lr = float(lr)

    freeze_backbone_epochs = getattr(args, "freeze_backbone_epochs", None)
    if freeze_backbone_epochs is not None:
        cfg.semantic.freeze_backbone_epochs = int(freeze_backbone_epochs)

    gate_temperature = getattr(args, "gate_temperature", None)
    if gate_temperature is not None:
        cfg.semantic.gate_temperature = float(gate_temperature)

    max_residual_scale = getattr(args, "max_residual_scale", None)
    if max_residual_scale is not None:
        cfg.semantic.max_residual_scale = float(max_residual_scale)

    gate_bias = getattr(args, "gate_bias", None)
    if gate_bias is not None:
        cfg.semantic.gate_bias = float(gate_bias)

    fusion_hidden_dim = getattr(args, "fusion_hidden_dim", None)
    if fusion_hidden_dim is not None:
        cfg.semantic.fusion_hidden_dim = int(fusion_hidden_dim)

    popularity_penalty = getattr(args, "popularity_penalty", None)
    if popularity_penalty is not None:
        cfg.semantic.popularity_penalty = float(popularity_penalty)

    long_tail_boost = getattr(args, "long_tail_boost", None)
    if long_tail_boost is not None:
        cfg.semantic.long_tail_boost = float(long_tail_boost)

    value_alpha = getattr(args, "value_alpha", None)
    if value_alpha is not None:
        cfg.semantic.value_alpha = float(value_alpha)

    value_beta = getattr(args, "value_beta", None)
    if value_beta is not None:
        cfg.semantic.value_beta = float(value_beta)

    value_gamma = getattr(args, "value_gamma", None)
    if value_gamma is not None:
        cfg.semantic.value_gamma = float(value_gamma)


def train(cfg_path: str, args: argparse.Namespace | None = None):
    cfg = ExperimentConfig.from_yaml(cfg_path)
    if args is not None:
        _apply_overrides(cfg, args)
    set_seed(cfg.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = _load_parser(cfg.data.dataset)
    print(f"Loaded parser: users={parser.num_users}, items={parser.num_items}")

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

    selector = ItemBudgetSelector(
        parser,
        popularity_penalty=cfg.semantic.popularity_penalty,
        value_alpha=cfg.semantic.value_alpha,
        value_beta=cfg.semantic.value_beta,
        value_gamma=cfg.semantic.value_gamma,
        long_tail_boost=cfg.semantic.long_tail_boost,
    )
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
    semantic_source = str(cfg.semantic.source).strip().lower()
    if semantic_source == "profile_embedding":
        selected_semantic, semantic_emb, semantic_dim = _acquire_selected_semantics_profile_embedding(
            cfg=cfg,
            parser=parser,
            selected_items=selected_items,
        )
    elif semantic_source == "llm_summary":
        manager = GeneralItemProfileManager(
            dataset_name=cfg.data.dataset,
            parser=parser,
            profile_path=cfg.data.profile_path,
        )
        item_profiles = manager.load(format=cfg.data.dataset)
        print(f"Loaded {len(item_profiles)} item profiles.")
        selected_semantic, semantic_emb, semantic_dim = _acquire_selected_semantics_llm(
            cfg=cfg,
            selected_items=selected_items,
            item_profiles=item_profiles,
            device=device,
            semantic_dir=semantic_dir,
        )
    else:
        raise ValueError(f"Unsupported semantic source: {cfg.semantic.source}")

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
        fusion_hidden_dim=cfg.semantic.fusion_hidden_dim,
        gate_temperature=cfg.semantic.gate_temperature,
        max_residual_scale=cfg.semantic.max_residual_scale,
        gate_bias=cfg.semantic.gate_bias,
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

    train_user_items = get_user_item_dict(parser.train)
    val_user_items = get_user_item_dict(parser.val)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.train.run_name or (
        f"{timestamp}_seed{cfg.seed}_b{cfg.semantic.budget_ratio:.3f}_{semantic_source}"
    )
    run_dir = Path(cfg.train.log_dir) / cfg.data.dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"

    if cfg.train.save_path:
        checkpoint_path = Path(cfg.train.save_path)
    else:
        checkpoint_path = run_dir / "best_model.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_epoch = -1

    print(f"Start budgeted semantic training... run_dir={run_dir}")
    for epoch in range(cfg.train.epochs):
        freeze = epoch < int(cfg.semantic.freeze_backbone_epochs)
        set_backbone_trainable(not freeze)

        model.train()
        total_loss = 0.0
        total_rank = 0.0
        total_align = 0.0
        total_cons = 0.0
        total_gate = 0.0

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
            loss_gate = (
                fusion_gate_l2_loss(out["pos_alpha"], out["pos_mask"])
                + fusion_gate_l2_loss(out["neg_alpha"], out["neg_mask"])
            )

            loss = (
                loss_rank
                + float(cfg.semantic.align_weight) * loss_align
                + float(cfg.semantic.consistency_weight) * loss_cons
                + float(cfg.semantic.gate_l2_weight) * loss_gate
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_rank += float(loss_rank.item())
            total_align += float(loss_align.item())
            total_cons += float(loss_cons.item())
            total_gate += float(loss_gate.item())

        num_batches = max(1, len(loader))
        train_payload = {
            "epoch": epoch + 1,
            "split": "train",
            "loss": total_loss / num_batches,
            "loss_rank": total_rank / num_batches,
            "loss_align": total_align / num_batches,
            "loss_consistency": total_cons / num_batches,
            "loss_gate": total_gate / num_batches,
            "freeze_backbone": bool(freeze),
        }
        _append_jsonl(metrics_path, train_payload)
        print(
            f"[Epoch {epoch+1}/{cfg.train.epochs}] "
            f"loss={train_payload['loss']:.4f} "
            f"rank={train_payload['loss_rank']:.4f} "
            f"align={train_payload['loss_align']:.4f} "
            f"cons={train_payload['loss_consistency']:.4f} "
            f"gate={train_payload['loss_gate']:.4f} "
            f"freeze_backbone={freeze}"
        )

        if (epoch + 1) % max(1, cfg.train.eval_interval) != 0:
            continue

        model.eval()
        with torch.no_grad():
            recall_res, ndcg_res = evaluate_all_ranking(
                model=model,
                users=torch.LongTensor(list(val_user_items.keys())).to(device),
                train_user_items=train_user_items,
                eval_user_items=val_user_items,
                K=[10, 20],
                device=device,
            )

        val_payload = {
            "epoch": epoch + 1,
            "split": "val",
            "recall@10": float(recall_res[10]),
            "ndcg@10": float(ndcg_res[10]),
            "recall@20": float(recall_res[20]),
            "ndcg@20": float(ndcg_res[20]),
        }
        _append_jsonl(metrics_path, val_payload)
        print(
            f"[Val @ Epoch {epoch+1}] "
            f"Recall@10={recall_res[10]:.4f}, NDCG@10={ndcg_res[10]:.4f}, "
            f"Recall@20={recall_res[20]:.4f}, NDCG@20={ndcg_res[20]:.4f}"
        )

        metric_name = str(cfg.train.selection_metric).strip().lower()
        metric_pool = {
            "ndcg10": float(ndcg_res[10]),
            "ndcg20": float(ndcg_res[20]),
            "recall10": float(recall_res[10]),
            "recall20": float(recall_res[20]),
        }
        if metric_name not in metric_pool:
            raise ValueError(
                f"Unsupported selection_metric={cfg.train.selection_metric}. "
                f"Use one of {sorted(metric_pool.keys())}."
            )
        cur_score = metric_pool[metric_name]
        if cur_score > best_score:
            best_score = cur_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")

    if best_epoch < 0:
        raise RuntimeError("No validation checkpoint was saved. Check eval_interval and training epochs.")

    best_state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_state, strict=False)
    model.eval()
    val_metrics = _evaluate_split(
        model=model,
        parser=parser,
        train_user_items=train_user_items,
        split_name="val",
        split_data=parser.val,
        device=device,
    )
    test_metrics = _evaluate_split(
        model=model,
        parser=parser,
        train_user_items=train_user_items,
        split_name="test",
        split_data=parser.test,
        device=device,
    )

    summary = {
        "dataset": cfg.data.dataset,
        "seed": int(cfg.seed),
        "semantic_source": semantic_source,
        "budget_ratio": float(cfg.semantic.budget_ratio),
        "selected_items": int(selected_items.numel()),
        "best_epoch": int(best_epoch),
        "selection_metric": str(cfg.train.selection_metric),
        "best_selection_score": float(best_score),
        "best_model_path": str(checkpoint_path),
        "metrics_file": str(metrics_path),
        **val_metrics,
        **test_metrics,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "Training completed. "
        f"best_{cfg.train.selection_metric}={best_score:.4f} at epoch={best_epoch}. "
        f"Summary saved to {summary_path}"
    )
    return summary


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--config", default="configs/yelp_.yaml", help="Path to YAML config file.")
    argp.add_argument("--seed", type=int, default=None)
    argp.add_argument("--epochs", type=int, default=None)
    argp.add_argument("--eval_interval", type=int, default=None)
    argp.add_argument("--budget_ratio", type=float, default=None)
    argp.add_argument("--align_weight", type=float, default=None)
    argp.add_argument("--consistency_weight", type=float, default=None)
    argp.add_argument("--gate_l2_weight", type=float, default=None)
    argp.add_argument("--semantic_source", type=str, default=None)
    argp.add_argument("--save_path", type=str, default=None)
    argp.add_argument("--run_name", type=str, default=None)
    argp.add_argument("--selection_metric", type=str, default=None)
    argp.add_argument("--lr", type=float, default=None)
    argp.add_argument("--freeze_backbone_epochs", type=int, default=None)
    argp.add_argument("--gate_temperature", type=float, default=None)
    argp.add_argument("--max_residual_scale", type=float, default=None)
    argp.add_argument("--gate_bias", type=float, default=None)
    argp.add_argument("--fusion_hidden_dim", type=int, default=None)
    argp.add_argument("--popularity_penalty", type=float, default=None)
    argp.add_argument("--long_tail_boost", type=float, default=None)
    argp.add_argument("--value_alpha", type=float, default=None)
    argp.add_argument("--value_beta", type=float, default=None)
    argp.add_argument("--value_gamma", type=float, default=None)
    parsed_args = argp.parse_args()
    train(parsed_args.config, parsed_args)
