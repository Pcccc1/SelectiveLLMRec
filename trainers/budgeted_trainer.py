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
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.config import ExperimentConfig
from dataloader.data_reader import load_pickle_compat
from dataloader.dataset_graph import GraphPretrainDataset, collate_graph
from dataloader.manager import GeneralItemProfileManager
from dataloader.sample_negative import NegativeSampler
from model.lightgcn import LightGCN, LightGCNBudgetedDualSemantic, LightGCNBudgetedSemantic
from utils.losses import (
    bpr_loss,
    confidence_aware_distillation_loss,
    embedding_consistency_loss,
    fusion_gate_l2_loss,
    semantic_alignment_loss,
)
from utils.metrics import evaluate_all_ranking, get_user_item_dict
from utils.semantic_acquisition import (
    ItemBudgetSelector,
    SemanticTextEncoder,
    TransformersSemanticAcquirer,
    UserBudgetSelector,
    UserClusterSemanticAcquirer,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _str2bool(raw: str | bool | None) -> bool | None:
    if raw is None or isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool value from {raw!r}")


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


def _save_item_semantic_artifacts(
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


def _save_user_semantic_artifacts(
    save_dir: str,
    selected_users: torch.Tensor,
    user_need_score: torch.Tensor,
    user_cluster_ids: np.ndarray,
    cluster_semantic: dict[int, dict[str, Any]],
    cluster_semantic_emb: dict[int, torch.Tensor],
):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "selected_user_ids.json"), "w", encoding="utf-8") as f:
        json.dump([int(u) for u in selected_users.tolist()], f, ensure_ascii=False, indent=2)
    torch.save(user_need_score.detach().cpu(), os.path.join(save_dir, "user_need_score.pt"))
    np.save(os.path.join(save_dir, "user_cluster_ids.npy"), user_cluster_ids.astype(np.int64))
    with open(os.path.join(save_dir, "cluster_semantics.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in cluster_semantic.items()}, f, ensure_ascii=False, indent=2)
    torch.save(
        {int(k): v.detach().cpu() for k, v in cluster_semantic_emb.items()},
        os.path.join(save_dir, "cluster_semantic_embeddings.pt"),
    )


def _load_profile_embedding_matrix(path: str, num_rows: int) -> torch.Tensor:
    raw = load_pickle_compat(path)
    if isinstance(raw, dict):
        sample = None
        for v in raw.values():
            sample = torch.as_tensor(v).view(-1)
            break
        if sample is None:
            raise ValueError(f"Empty embedding dict: {path}")
        matrix = torch.zeros((num_rows, int(sample.numel())), dtype=torch.float32)
        for k, v in raw.items():
            rid = int(k)
            if 0 <= rid < num_rows:
                vv = torch.as_tensor(v).view(-1).float()
                dim = min(vv.numel(), matrix.size(1))
                matrix[rid, :dim] = vv[:dim]
        return matrix

    emb = torch.as_tensor(raw).float()
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape={tuple(emb.shape)} from {path}")

    if emb.size(0) < num_rows:
        pad = torch.zeros((num_rows - emb.size(0), emb.size(1)), dtype=emb.dtype)
        emb = torch.cat([emb, pad], dim=0)
    return emb[:num_rows]


def _assert_not_data_new_embedding_path(path: str, field_name: str) -> None:
    norm = os.path.normpath(str(path)).replace("\\", "/")
    if "/data_new/" in f"/{norm}/" or norm.startswith("data_new/"):
        raise ValueError(
            f"{field_name} points to data_new embeddings ({path}), which are disabled by policy. "
            "Please use LLM semantic acquisition with local Qwen models, "
            "or provide a custom non-data_new embedding file."
        )


def _normalize_profile_text(profile: Any) -> str:
    if profile is None:
        return "N/A"
    if isinstance(profile, str):
        return profile.strip()[:256]
    if not isinstance(profile, dict):
        return str(profile)[:256]

    if "profile" in profile and isinstance(profile["profile"], str):
        return profile["profile"].strip()[:256]

    parts = []
    for key in ["title", "name", "categories", "genres", "description"]:
        val = profile.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            val = ", ".join(str(x) for x in val)
        text = str(val).strip()
        if text:
            parts.append(f"{key}={text}")
    if not parts:
        return json.dumps(profile, ensure_ascii=False)[:256]
    return "; ".join(parts)[:256]


def _load_item_profiles_if_needed(
    cfg: ExperimentConfig,
    parser,
    required: bool,
) -> dict[int, Any]:
    if not required:
        return {}
    manager = GeneralItemProfileManager(
        dataset_name=cfg.data.dataset,
        parser=parser,
        profile_path=cfg.data.profile_path,
    )
    item_profiles = manager.load(format=cfg.data.dataset)
    print(f"Loaded {len(item_profiles)} item profiles.")
    return item_profiles


def _acquire_selected_semantics_llm(
    cfg: ExperimentConfig,
    selected_items: torch.Tensor,
    item_profiles: dict[int, Any],
    device: torch.device,
    semantic_dir: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, torch.Tensor], int]:
    acquirer = TransformersSemanticAcquirer(
        model_path=cfg.semantic.llm_model_path,
        max_tokens=cfg.semantic.llm_max_tokens,
        temperature=cfg.semantic.llm_temperature,
        timeout=cfg.semantic.request_timeout,
        cache_dir=os.path.join(semantic_dir, cfg.semantic.llm_cache_subdir),
        llm_fail_fast=cfg.semantic.llm_fail_fast,
        load_in_8bit=cfg.semantic.llm_load_in_8bit,
        device_map=cfg.semantic.llm_device_map,
        trust_remote_code=cfg.semantic.llm_trust_remote_code,
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
    emb_path = cfg.semantic.profile_embedding_path
    if not emb_path:
        raise ValueError(
            "semantic.source=profile_embedding requires semantic.profile_embedding_path to be explicitly set. "
            "Fallback to data_new/itm_emb_np.pkl is disabled."
        )
    _assert_not_data_new_embedding_path(emb_path, "semantic.profile_embedding_path")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing profile embedding file: {emb_path}")

    matrix = _load_profile_embedding_matrix(path=emb_path, num_rows=parser.num_items)
    semantic_emb = {int(iid): matrix[int(iid)].clone() for iid in selected_items.tolist()}
    semantic_dim = int(matrix.size(1))
    print(f"Loaded item profile embeddings from {emb_path} with dim={semantic_dim}.")
    return {}, semantic_emb, semantic_dim


def _build_cluster_payloads(
    parser,
    cluster_ids: np.ndarray,
    need_score: torch.Tensor,
    item_profiles: dict[int, Any],
    num_clusters: int,
    representatives: int,
    prompt_items_per_user: int,
    seed: int,
) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    need_np = need_score.detach().cpu().numpy().reshape(-1)
    payloads: dict[int, str] = {}

    for cid in range(num_clusters):
        members = np.where(cluster_ids == cid)[0]
        if len(members) == 0:
            payloads[cid] = "Empty cluster."
            continue

        member_need = need_np[members]
        order = np.argsort(-member_need)
        reps = members[order[: min(representatives, len(members))]]

        lines = [
            f"cluster_id={cid}",
            f"member_count={len(members)}",
            "Representative user interactions:",
        ]

        for uid in reps.tolist():
            history = list(parser.user_pos_items.get(int(uid), set()))
            if not history:
                continue
            if len(history) > prompt_items_per_user:
                history = rng.choice(history, size=prompt_items_per_user, replace=False).tolist()

            item_lines = []
            for iid in history:
                item_text = _normalize_profile_text(item_profiles.get(int(iid), {"profile": f"item_id={iid}"}))
                item_lines.append(f"item_{iid}: {item_text}")
            if item_lines:
                lines.append(f"user_{uid}: " + " || ".join(item_lines))

        lines.append("Summarize this cluster's stable preferences in concise JSON fields.")
        payloads[cid] = "\n".join(lines)

    return payloads


def _acquire_user_semantics_profile_embedding(
    cfg: ExperimentConfig,
    parser,
) -> tuple[dict[int, dict[str, Any]], dict[int, torch.Tensor], np.ndarray, int]:
    emb_path = cfg.semantic.user_profile_embedding_path
    if emb_path is None:
        raise ValueError(
            "semantic.user_source=profile_embedding requires semantic.user_profile_embedding_path to be explicitly set. "
            "Fallback to data_new/usr_emb_np.pkl is disabled."
        )
    _assert_not_data_new_embedding_path(emb_path, "semantic.user_profile_embedding_path")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing user profile embedding file: {emb_path}")
    matrix = _load_profile_embedding_matrix(path=emb_path, num_rows=parser.num_users)
    user_semantic = {int(uid): matrix[int(uid)].clone() for uid in range(parser.num_users)}
    semantic_dim = int(matrix.size(1))
    # Trivial one-user-one-cluster bookkeeping for compatibility.
    user_cluster = np.arange(parser.num_users, dtype=np.int64)
    print(f"Loaded user profile embeddings from {emb_path} with dim={semantic_dim}.")
    return {}, user_semantic, user_cluster, semantic_dim


def _acquire_user_semantics_cluster_llm(
    cfg: ExperimentConfig,
    parser,
    user_repr: torch.Tensor,
    user_need_score: torch.Tensor,
    item_profiles: dict[int, Any],
    device: torch.device,
    semantic_dir: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, torch.Tensor], dict[int, torch.Tensor], np.ndarray, int]:
    num_clusters = max(1, int(cfg.semantic.user_num_clusters))
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=int(cfg.seed),
        batch_size=min(4096, max(256, parser.num_users)),
        n_init="auto",
    )
    user_cluster_ids = kmeans.fit_predict(user_repr.detach().cpu().numpy()).astype(np.int64)
    cluster_payload = _build_cluster_payloads(
        parser=parser,
        cluster_ids=user_cluster_ids,
        need_score=user_need_score,
        item_profiles=item_profiles,
        num_clusters=num_clusters,
        representatives=int(cfg.semantic.user_cluster_representatives),
        prompt_items_per_user=int(cfg.semantic.user_prompt_items_per_user),
        seed=int(cfg.seed),
    )

    acquirer = UserClusterSemanticAcquirer(
        model_path=cfg.semantic.llm_model_path,
        max_tokens=cfg.semantic.llm_max_tokens,
        temperature=cfg.semantic.llm_temperature,
        timeout=cfg.semantic.request_timeout,
        cache_dir=os.path.join(semantic_dir, cfg.semantic.llm_cache_subdir, "user_clusters"),
        llm_fail_fast=cfg.semantic.llm_fail_fast,
        load_in_8bit=cfg.semantic.llm_load_in_8bit,
        device_map=cfg.semantic.llm_device_map,
        trust_remote_code=cfg.semantic.llm_trust_remote_code,
    )
    cluster_semantic = acquirer.acquire_batch(
        item_ids=list(range(num_clusters)),
        item_profiles=cluster_payload,
    )
    print(f"Acquired semantic annotations for {len(cluster_semantic)} user clusters.")

    encoder = SemanticTextEncoder(
        model_path=cfg.semantic.embedding_model_path,
        device=str(device),
        batch_size=cfg.semantic.embedding_batch_size,
    )
    cluster_semantic_emb = encoder.encode(cluster_semantic)
    if len(cluster_semantic_emb) == 0:
        raise RuntimeError("No cluster semantic embeddings were produced.")
    semantic_dim = int(next(iter(cluster_semantic_emb.values())).numel())
    user_semantic_emb = {
        int(uid): cluster_semantic_emb[int(user_cluster_ids[uid])].clone()
        for uid in range(parser.num_users)
    }
    return cluster_semantic, cluster_semantic_emb, user_semantic_emb, user_cluster_ids, semantic_dim


def _evaluate_split(
    model,
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
    pairs = [
        ("seed", ("seed", int)),
        ("epochs", ("train.epochs", int)),
        ("eval_interval", ("train.eval_interval", int)),
        ("budget_ratio", ("semantic.budget_ratio", float)),
        ("user_budget_ratio", ("semantic.user_budget_ratio", float)),
        ("align_weight", ("semantic.align_weight", float)),
        ("consistency_weight", ("semantic.consistency_weight", float)),
        ("gate_l2_weight", ("semantic.gate_l2_weight", float)),
        ("semantic_source", ("semantic.source", str)),
        ("user_source", ("semantic.user_source", str)),
        ("save_path", ("train.save_path", str)),
        ("run_name", ("train.run_name", str)),
        ("selection_metric", ("train.selection_metric", str)),
        ("lr", ("train.lr", float)),
        ("freeze_backbone_epochs", ("semantic.freeze_backbone_epochs", int)),
        ("ranking_preserve_weight", ("semantic.ranking_preserve_weight", float)),
        ("gate_temperature", ("semantic.gate_temperature", float)),
        ("max_residual_scale", ("semantic.max_residual_scale", float)),
        ("gate_bias", ("semantic.gate_bias", float)),
        ("fusion_hidden_dim", ("semantic.fusion_hidden_dim", int)),
        ("popularity_penalty", ("semantic.popularity_penalty", float)),
        ("long_tail_boost", ("semantic.long_tail_boost", float)),
        ("value_alpha", ("semantic.value_alpha", float)),
        ("value_beta", ("semantic.value_beta", float)),
        ("value_gamma", ("semantic.value_gamma", float)),
        ("user_num_clusters", ("semantic.user_num_clusters", int)),
        ("user_align_weight", ("semantic.user_align_weight", float)),
        ("user_consistency_weight", ("semantic.user_consistency_weight", float)),
        ("user_distill_weight", ("semantic.user_distill_weight", float)),
        ("user_gate_l2_weight", ("semantic.user_gate_l2_weight", float)),
        ("user_gate_temperature", ("semantic.user_gate_temperature", float)),
        ("user_max_residual_scale", ("semantic.user_max_residual_scale", float)),
        ("user_gate_bias", ("semantic.user_gate_bias", float)),
    ]

    for arg_name, (path, cast_fn) in pairs:
        val = getattr(args, arg_name, None)
        if val is None:
            continue
        keys = path.split(".")
        target = cfg
        for k in keys[:-1]:
            target = getattr(target, k)
        setattr(target, keys[-1], cast_fn(val))

    user_enable = _str2bool(getattr(args, "user_enable", None))
    if user_enable is not None:
        cfg.semantic.user_enable = bool(user_enable)

    frozen = _str2bool(getattr(args, "frozen", None))
    freeze_backbone = _str2bool(getattr(args, "freeze_backbone", None))
    if frozen is not None and freeze_backbone is not None and bool(frozen) != bool(freeze_backbone):
        print("[Config] Both `--frozen` and `--freeze_backbone` were set; using `--frozen`.")
    backbone_frozen = frozen if frozen is not None else freeze_backbone
    if backbone_frozen is not None:
        cfg.semantic.frozen = bool(backbone_frozen)
        cfg.semantic.freeze_backbone = bool(backbone_frozen)
    else:
        cfg.semantic.frozen = bool(cfg.semantic.freeze_backbone)

    detach_alignment_target = _str2bool(getattr(args, "detach_alignment_target", None))
    if detach_alignment_target is not None:
        cfg.semantic.detach_alignment_target = bool(detach_alignment_target)

    use_amp = _str2bool(getattr(args, "use_amp", None))
    if use_amp is not None:
        cfg.train.use_amp = bool(use_amp)

    grad_clip_norm = getattr(args, "grad_clip_norm", None)
    if grad_clip_norm is not None:
        cfg.train.grad_clip_norm = float(grad_clip_norm)

    early_stop_patience = getattr(args, "early_stop_patience", None)
    if early_stop_patience is not None:
        cfg.train.early_stop_patience = int(early_stop_patience)


def train(cfg_path: str, args: argparse.Namespace | None = None):
    cfg = ExperimentConfig.from_yaml(cfg_path)
    if args is not None:
        _apply_overrides(cfg, args)
    set_seed(cfg.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_enabled = bool(cfg.train.use_amp and device.type == "cuda")
    print(f"Using device: {device}, amp={amp_enabled}")

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
        user_emb_layers, item_emb_layers = pretrain_model.propagate_with_user_item_layers()
        user_id_emb = pretrain_model.user_embedding.weight.detach()
        item_id_emb = pretrain_model.item_embedding.weight.detach()
        user_repr = user_emb_layers[-1].detach()

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
        f"Item budget selection: {selected_items.numel()} / {parser.num_items} "
        f"(ratio={cfg.semantic.budget_ratio})."
    )

    semantic_dir = os.path.join(cfg.semantic.cache_dir, cfg.data.dataset)
    semantic_source = str(cfg.semantic.source).strip().lower()
    need_item_profiles = semantic_source == "llm_summary" or (
        bool(cfg.semantic.user_enable) and str(cfg.semantic.user_source).strip().lower() == "cluster_llm"
    )
    item_profiles = _load_item_profiles_if_needed(cfg, parser, required=need_item_profiles)

    if semantic_source == "profile_embedding":
        selected_semantic, item_semantic_emb, item_semantic_dim = _acquire_selected_semantics_profile_embedding(
            cfg=cfg,
            parser=parser,
            selected_items=selected_items,
        )
    elif semantic_source == "llm_summary":
        selected_semantic, item_semantic_emb, item_semantic_dim = _acquire_selected_semantics_llm(
            cfg=cfg,
            selected_items=selected_items,
            item_profiles=item_profiles,
            device=device,
            semantic_dir=semantic_dir,
        )
    else:
        raise ValueError(f"Unsupported semantic source: {cfg.semantic.source}")

    selected_item_mask = torch.zeros(parser.num_items, dtype=torch.float32)
    selected_item_mask[selected_items] = 1.0
    print(f"Item semantic dim: {item_semantic_dim}")

    if cfg.semantic.save_artifacts:
        _save_item_semantic_artifacts(
            save_dir=semantic_dir,
            selected_items=selected_items,
            semantic_dict=selected_semantic,
            semantic_emb=item_semantic_emb,
            score=budget_score,
        )
        print(f"Saved item semantic artifacts under {semantic_dir}")

    user_enable = bool(cfg.semantic.user_enable)
    selected_users = torch.arange(parser.num_users, dtype=torch.long)
    user_need_score = torch.zeros(parser.num_users, dtype=torch.float32)
    user_semantic_emb: dict[int, torch.Tensor] = {}
    user_cluster_ids = np.arange(parser.num_users, dtype=np.int64)
    cluster_semantic: dict[int, dict[str, Any]] = {}
    cluster_semantic_emb: dict[int, torch.Tensor] = {}
    user_semantic_dim = int(item_semantic_dim)

    if user_enable:
        user_selector = UserBudgetSelector(
            parser=parser,
            activity_penalty=cfg.semantic.user_activity_penalty,
            value_alpha=cfg.semantic.user_value_alpha,
            value_beta=cfg.semantic.user_value_beta,
            value_gamma=cfg.semantic.user_value_gamma,
            cold_start_boost=cfg.semantic.user_cold_start_boost,
        )
        selected_users, user_need_score = user_selector.select(
            user_emb_layers=user_emb_layers,
            user_id_emb=user_id_emb,
            budget_ratio=cfg.semantic.user_budget_ratio,
            min_selected_users=cfg.semantic.min_selected_users,
        )
        print(
            f"User need selection: {selected_users.numel()} / {parser.num_users} "
            f"(ratio={cfg.semantic.user_budget_ratio})."
        )

        user_source = str(cfg.semantic.user_source).strip().lower()
        if user_source == "profile_embedding":
            cluster_semantic, user_semantic_emb, user_cluster_ids, user_semantic_dim = (
                _acquire_user_semantics_profile_embedding(cfg=cfg, parser=parser)
            )
        elif user_source == "cluster_llm":
            (
                cluster_semantic,
                cluster_semantic_emb,
                user_semantic_emb,
                user_cluster_ids,
                user_semantic_dim,
            ) = _acquire_user_semantics_cluster_llm(
                cfg=cfg,
                parser=parser,
                user_repr=user_repr,
                user_need_score=user_need_score,
                item_profiles=item_profiles,
                device=device,
                semantic_dir=semantic_dir,
            )
        else:
            raise ValueError(f"Unsupported semantic.user_source: {cfg.semantic.user_source}")

        if cfg.semantic.save_artifacts:
            _save_user_semantic_artifacts(
                save_dir=semantic_dir,
                selected_users=selected_users,
                user_need_score=user_need_score,
                user_cluster_ids=user_cluster_ids,
                cluster_semantic=cluster_semantic,
                cluster_semantic_emb=cluster_semantic_emb,
            )
            print(f"Saved user semantic artifacts under {semantic_dir}")

    selected_user_mask = torch.zeros(parser.num_users, dtype=torch.float32)
    selected_user_mask[selected_users] = 1.0

    if user_enable:
        model = LightGCNBudgetedDualSemantic(
            num_users=parser.num_users,
            num_items=parser.num_items,
            embedding_dim=cfg.lightgcn.embedding_dim,
            n_layers=cfg.lightgcn.n_layers,
            adj_mat=parser.adj_mat,
            item_semantic_embeddings=item_semantic_emb,
            selected_item_mask=selected_item_mask,
            item_semantic_dim=item_semantic_dim,
            user_semantic_embeddings=user_semantic_emb,
            selected_user_mask=selected_user_mask,
            user_need_score=user_need_score,
            user_semantic_dim=user_semantic_dim,
            fusion_hidden_dim=cfg.semantic.fusion_hidden_dim,
            gate_temperature=cfg.semantic.gate_temperature,
            max_residual_scale=cfg.semantic.max_residual_scale,
            gate_bias=cfg.semantic.gate_bias,
            user_gate_temperature=cfg.semantic.user_gate_temperature,
            user_max_residual_scale=cfg.semantic.user_max_residual_scale,
            user_gate_bias=cfg.semantic.user_gate_bias,
            device=device,
        ).to(device)
    else:
        model = LightGCNBudgetedSemantic(
            num_users=parser.num_users,
            num_items=parser.num_items,
            embedding_dim=cfg.lightgcn.embedding_dim,
            n_layers=cfg.lightgcn.n_layers,
            adj_mat=parser.adj_mat,
            item_semantic_embeddings=item_semantic_emb,
            selected_item_mask=selected_item_mask,
            semantic_dim=item_semantic_dim,
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

    def set_backbone_trainable(trainable: bool):
        model.user_embedding.weight.requires_grad = trainable
        model.item_embedding.weight.requires_grad = trainable

    backbone_frozen = bool(cfg.semantic.frozen)
    if backbone_frozen:
        set_backbone_trainable(False)
    print(
        "Backbone freeze config: "
        f"frozen={backbone_frozen}, freeze_backbone_epochs={cfg.semantic.freeze_backbone_epochs}"
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=cfg.train.lr,
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg.train.scheduler_tmax)),
        eta_min=float(cfg.train.scheduler_eta_min),
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        autocast_cm = lambda: torch.amp.autocast("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        autocast_cm = lambda: torch.cuda.amp.autocast(enabled=amp_enabled)

    train_user_items = get_user_item_dict(parser.train)
    val_user_items = get_user_item_dict(parser.val)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.train.run_name or (
        f"{timestamp}_seed{cfg.seed}_bi{cfg.semantic.budget_ratio:.3f}_"
        f"bu{cfg.semantic.user_budget_ratio:.3f}_{semantic_source}"
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

    metric_name = str(cfg.train.selection_metric).strip().lower()
    valid_metric_names = {"ndcg10", "ndcg20", "recall10", "recall20"}
    if metric_name not in valid_metric_names:
        raise ValueError(
            f"Unsupported selection_metric={cfg.train.selection_metric}. "
            f"Use one of {sorted(valid_metric_names)}."
        )

    def _selection_score(recall_res: dict[int, float], ndcg_res: dict[int, float]) -> float:
        metric_pool = {
            "ndcg10": float(ndcg_res[10]),
            "ndcg20": float(ndcg_res[20]),
            "recall10": float(recall_res[10]),
            "recall20": float(recall_res[20]),
        }
        return metric_pool[metric_name]

    model.eval()
    with torch.no_grad():
        base_recall, base_ndcg = evaluate_all_ranking(
            model=model,
            users=torch.LongTensor(list(val_user_items.keys())).to(device),
            train_user_items=train_user_items,
            eval_user_items=val_user_items,
            K=[10, 20],
            device=device,
        )
    best_score = _selection_score(base_recall, base_ndcg)
    best_epoch = 0
    torch.save(model.state_dict(), checkpoint_path)
    _append_jsonl(
        metrics_path,
        {
            "epoch": 0,
            "split": "val",
            "recall@10": float(base_recall[10]),
            "ndcg@10": float(base_ndcg[10]),
            "recall@20": float(base_recall[20]),
            "ndcg@20": float(base_ndcg[20]),
            "guardrail_baseline": True,
        },
    )
    print(
        f"Baseline guardrail: best_{cfg.train.selection_metric}={best_score:.4f} "
        "at epoch=0 before semantic training."
    )
    bad_epochs = 0
    patience = max(1, int(cfg.train.early_stop_patience))

    print(f"Start dual-side budgeted semantic training... run_dir={run_dir}")
    epoch_bar = tqdm(range(cfg.train.epochs), desc="Training", dynamic_ncols=True)
    for epoch in epoch_bar:
        freeze = backbone_frozen or epoch < int(cfg.semantic.freeze_backbone_epochs)
        if not backbone_frozen:
            set_backbone_trainable(not freeze)

        model.train()
        total_loss = 0.0
        total_rank = 0.0
        total_item_align = 0.0
        total_item_cons = 0.0
        total_item_gate = 0.0
        total_user_align = 0.0
        total_user_cons = 0.0
        total_user_gate = 0.0
        total_user_distill = 0.0
        total_rank_preserve = 0.0

        batch_bar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{cfg.train.epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in batch_bar:
            users = batch["user"].to(device)
            pos_items = batch["pos"].to(device)
            neg_items = batch["neg"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_cm():
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
                pos_align_base = (
                    out["pos_base"].detach()
                    if bool(cfg.semantic.detach_alignment_target)
                    else out["pos_base"]
                )
                neg_align_base = (
                    out["neg_base"].detach()
                    if bool(cfg.semantic.detach_alignment_target)
                    else out["neg_base"]
                )
                loss_item_align = (
                    semantic_alignment_loss(pos_align_base, out["pos_sem_proj"], out["pos_mask"])
                    + semantic_alignment_loss(neg_align_base, out["neg_sem_proj"], out["neg_mask"])
                )
                loss_item_cons = (
                    embedding_consistency_loss(out["pos_fused"], pos_align_base, out["pos_mask"])
                    + embedding_consistency_loss(out["neg_fused"], neg_align_base, out["neg_mask"])
                )
                loss_item_gate = (
                    fusion_gate_l2_loss(out["pos_alpha"], out["pos_mask"])
                    + fusion_gate_l2_loss(out["neg_alpha"], out["neg_mask"])
                )

                loss_user_align = torch.zeros((), device=device)
                loss_user_cons = torch.zeros((), device=device)
                loss_user_gate = torch.zeros((), device=device)
                loss_user_distill = torch.zeros((), device=device)
                user_base_for_preserve = out.get("user_base", out["user"])
                if user_enable:
                    user_align_base = (
                        out["user_base"].detach()
                        if bool(cfg.semantic.detach_alignment_target)
                        else out["user_base"]
                    )
                    loss_user_align = semantic_alignment_loss(
                        user_align_base,
                        out["user_sem_proj"],
                        out["user_mask"],
                    )
                    loss_user_cons = embedding_consistency_loss(
                        out["user"],
                        user_align_base,
                        out["user_mask"],
                    )
                    loss_user_gate = fusion_gate_l2_loss(out["user_alpha"], out["user_mask"])
                    if float(cfg.semantic.user_distill_weight) > 0:
                        loss_user_distill = confidence_aware_distillation_loss(
                            student_emb=out["user"],
                            teacher_emb=out["user_sem_proj"],
                            confidence=out["user_need"],
                            selected_mask=out["user_mask"],
                        )
                base_margin = (
                    (user_base_for_preserve.detach() * out["pos_base"].detach()).sum(dim=-1)
                    - (user_base_for_preserve.detach() * out["neg_base"].detach()).sum(dim=-1)
                )
                fused_margin = (
                    (out["user"] * out["pos_fused"]).sum(dim=-1)
                    - (out["user"] * out["neg_fused"]).sum(dim=-1)
                )
                loss_rank_preserve = F.smooth_l1_loss(fused_margin, base_margin)

                loss = (
                    loss_rank
                    + float(cfg.semantic.align_weight) * loss_item_align
                    + float(cfg.semantic.consistency_weight) * loss_item_cons
                    + float(cfg.semantic.gate_l2_weight) * loss_item_gate
                    + float(cfg.semantic.user_align_weight) * loss_user_align
                    + float(cfg.semantic.user_consistency_weight) * loss_user_cons
                    + float(cfg.semantic.user_gate_l2_weight) * loss_user_gate
                    + float(cfg.semantic.user_distill_weight) * loss_user_distill
                    + float(cfg.semantic.ranking_preserve_weight) * loss_rank_preserve
                )

            scaler.scale(loss).backward()
            if float(cfg.train.grad_clip_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            total_rank += float(loss_rank.item())
            total_item_align += float(loss_item_align.item())
            total_item_cons += float(loss_item_cons.item())
            total_item_gate += float(loss_item_gate.item())
            total_user_align += float(loss_user_align.item())
            total_user_cons += float(loss_user_cons.item())
            total_user_gate += float(loss_user_gate.item())
            total_user_distill += float(loss_user_distill.item())
            total_rank_preserve += float(loss_rank_preserve.item())

            seen = max(1, batch_bar.n)
            postfix = {
                "loss": f"{total_loss / seen:.4f}",
                "rank": f"{total_rank / seen:.4f}",
                "i_align": f"{total_item_align / seen:.4f}",
                "kd": f"{total_rank_preserve / seen:.4f}",
            }
            if float(cfg.semantic.user_distill_weight) > 0:
                postfix["u_dist"] = f"{total_user_distill / seen:.4f}"
            batch_bar.set_postfix(**postfix)

        scheduler.step()

        num_batches = max(1, len(loader))
        train_payload = {
            "epoch": epoch + 1,
            "split": "train",
            "loss": total_loss / num_batches,
            "loss_rank": total_rank / num_batches,
            "loss_item_align": total_item_align / num_batches,
            "loss_item_consistency": total_item_cons / num_batches,
            "loss_item_gate": total_item_gate / num_batches,
            "loss_user_align": total_user_align / num_batches,
            "loss_user_consistency": total_user_cons / num_batches,
            "loss_user_gate": total_user_gate / num_batches,
            "loss_user_distill": total_user_distill / num_batches,
            "loss_rank_preserve": total_rank_preserve / num_batches,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "frozen": bool(backbone_frozen),
            "freeze_backbone": bool(freeze),
        }
        _append_jsonl(metrics_path, train_payload)

        epoch_bar.set_postfix(
            loss=f"{train_payload['loss']:.4f}",
            rank=f"{train_payload['loss_rank']:.4f}",
            lr=f"{train_payload['lr']:.2e}",
            freeze=freeze,
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

        cur_score = _selection_score(recall_res, ndcg_res)
        if cur_score > best_score:
            best_score = cur_score
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}, patience={patience}.")
            break

    if best_epoch < 0:
        raise RuntimeError("No validation checkpoint was saved. Check eval_interval and training epochs.")

    best_state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_state, strict=False)
    model.eval()
    val_metrics = _evaluate_split(
        model=model,
        train_user_items=train_user_items,
        split_name="val",
        split_data=parser.val,
        device=device,
    )
    test_metrics = _evaluate_split(
        model=model,
        train_user_items=train_user_items,
        split_name="test",
        split_data=parser.test,
        device=device,
    )

    summary = {
        "dataset": cfg.data.dataset,
        "seed": int(cfg.seed),
        "semantic_source": semantic_source,
        "user_semantic_source": str(cfg.semantic.user_source),
        "user_enable": bool(user_enable),
        "item_budget_ratio": float(cfg.semantic.budget_ratio),
        "user_budget_ratio": float(cfg.semantic.user_budget_ratio),
        "selected_items": int(selected_items.numel()),
        "selected_users": int(selected_users.numel()),
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


class BudgetedSemanticTrainer:
    """
    OOP wrapper for budgeted dual-side semantic training.

    This keeps a stable class interface for future extension while reusing
    the validated training implementation above.
    """

    def __init__(self, config_path: str, cli_args: argparse.Namespace | None = None):
        self.config_path = config_path
        self.cli_args = cli_args

    def run(self) -> dict[str, Any]:
        return train(self.config_path, self.cli_args)


def build_budgeted_arg_parser() -> argparse.ArgumentParser:
    argp = argparse.ArgumentParser(
        description="Dual-side selective semantic enhancement training entrypoint."
    )
    argp.add_argument("--config", default="configs/yelp_budgeted_sota.yaml", help="Path to YAML config file.")
    argp.add_argument("--seed", type=int, default=None)
    argp.add_argument("--epochs", type=int, default=None)
    argp.add_argument("--eval_interval", type=int, default=None)
    argp.add_argument("--budget_ratio", type=float, default=None)
    argp.add_argument("--user_budget_ratio", type=float, default=None)
    argp.add_argument("--align_weight", type=float, default=None)
    argp.add_argument("--consistency_weight", type=float, default=None)
    argp.add_argument("--gate_l2_weight", type=float, default=None)
    argp.add_argument("--semantic_source", type=str, default=None)
    argp.add_argument("--user_source", type=str, default=None)
    argp.add_argument("--user_enable", type=str, default=None)
    argp.add_argument("--save_path", type=str, default=None)
    argp.add_argument("--run_name", type=str, default=None)
    argp.add_argument("--selection_metric", type=str, default=None)
    argp.add_argument("--lr", type=float, default=None)
    argp.add_argument("--freeze_backbone_epochs", type=int, default=None)
    argp.add_argument("--frozen", type=str, default=None)
    argp.add_argument("--freeze_backbone", type=str, default=None)
    argp.add_argument("--detach_alignment_target", type=str, default=None)
    argp.add_argument("--ranking_preserve_weight", type=float, default=None)
    argp.add_argument("--gate_temperature", type=float, default=None)
    argp.add_argument("--max_residual_scale", type=float, default=None)
    argp.add_argument("--gate_bias", type=float, default=None)
    argp.add_argument("--fusion_hidden_dim", type=int, default=None)
    argp.add_argument("--popularity_penalty", type=float, default=None)
    argp.add_argument("--long_tail_boost", type=float, default=None)
    argp.add_argument("--value_alpha", type=float, default=None)
    argp.add_argument("--value_beta", type=float, default=None)
    argp.add_argument("--value_gamma", type=float, default=None)
    argp.add_argument("--user_num_clusters", type=int, default=None)
    argp.add_argument("--user_align_weight", type=float, default=None)
    argp.add_argument("--user_consistency_weight", type=float, default=None)
    argp.add_argument("--user_distill_weight", type=float, default=None)
    argp.add_argument("--user_gate_l2_weight", type=float, default=None)
    argp.add_argument("--user_gate_temperature", type=float, default=None)
    argp.add_argument("--user_max_residual_scale", type=float, default=None)
    argp.add_argument("--user_gate_bias", type=float, default=None)
    argp.add_argument("--use_amp", type=str, default=None)
    argp.add_argument("--grad_clip_norm", type=float, default=None)
    argp.add_argument("--early_stop_patience", type=int, default=None)
    return argp
