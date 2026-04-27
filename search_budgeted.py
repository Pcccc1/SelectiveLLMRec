from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from train_budgeted import train


def _parse_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _sample_candidates(
    mode: str,
    max_trials: int,
    rnd: random.Random,
    space: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    keys = list(space.keys())
    if mode == "grid":
        all_candidates = [dict(zip(keys, vals)) for vals in product(*(space[k] for k in keys))]
        rnd.shuffle(all_candidates)
        return all_candidates[:max_trials]

    candidates = []
    seen = set()
    for _ in range(max_trials * 20):
        cand = {k: rnd.choice(space[k]) for k in keys}
        sig = tuple((k, cand[k]) for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(cand)
        if len(candidates) >= max_trials:
            break
    return candidates


def _build_train_args(
    config: str,
    trial_id: int,
    run_name: str,
    checkpoint_dir: Path,
    params: dict[str, Any],
    epochs: int,
    eval_interval: int,
    semantic_source: str,
    selection_metric: str,
):
    save_path = checkpoint_dir / f"trial_{trial_id:03d}.pth"
    return SimpleNamespace(
        seed=params["seed"],
        epochs=epochs,
        eval_interval=eval_interval,
        budget_ratio=params["budget_ratio"],
        user_budget_ratio=params["user_budget_ratio"],
        align_weight=params["align_weight"],
        consistency_weight=params["consistency_weight"],
        gate_l2_weight=params["gate_l2_weight"],
        semantic_source=semantic_source,
        user_source=params["user_source"],
        user_enable=True,
        save_path=str(save_path),
        run_name=run_name,
        selection_metric=selection_metric,
        lr=params["lr"],
        frozen=params["frozen"],
        freeze_backbone=params["frozen"],
        freeze_backbone_epochs=params["freeze_backbone_epochs"],
        gate_temperature=params["gate_temperature"],
        max_residual_scale=params["max_residual_scale"],
        gate_bias=params["gate_bias"],
        fusion_hidden_dim=params["fusion_hidden_dim"],
        popularity_penalty=params["popularity_penalty"],
        long_tail_boost=params["long_tail_boost"],
        value_alpha=params["value_alpha"],
        value_beta=params["value_beta"],
        value_gamma=params["value_gamma"],
        user_num_clusters=params["user_num_clusters"],
        user_distill_weight=params["user_distill_weight"],
    )


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--config", default="configs/yelp_budgeted_aggressive.yaml")
    argp.add_argument("--search_name", default=None)
    argp.add_argument("--mode", choices=["random", "grid"], default="random")
    argp.add_argument("--max_trials", type=int, default=8)
    argp.add_argument("--epochs", type=int, default=8)
    argp.add_argument("--eval_interval", type=int, default=1)
    argp.add_argument("--semantic_source", default="llm_summary")
    argp.add_argument("--selection_metric", default="ndcg10")
    argp.add_argument("--objective_metric", default="val_ndcg@10")
    argp.add_argument("--random_seed", type=int, default=2026)

    argp.add_argument("--seeds", default="42,7,2024")
    argp.add_argument("--budget_ratios", default="0.5,0.8,1.0")
    argp.add_argument("--user_budget_ratios", default="0.1,0.2,0.3")
    argp.add_argument("--user_sources", default="cluster_llm")
    argp.add_argument("--user_num_clusters", default="64,128,256")
    argp.add_argument("--user_distill_weights", default="0.03,0.05,0.08")
    argp.add_argument("--lrs", default="0.0003,0.0005,0.0008")
    argp.add_argument("--frozen", default="false")
    argp.add_argument("--align_weights", default="0.01,0.02,0.03")
    argp.add_argument("--consistency_weights", default="0.0,0.005,0.01")
    argp.add_argument("--gate_l2_weights", default="0.0,0.0001")
    argp.add_argument("--freeze_backbone_epochs", default="0,1")
    argp.add_argument("--gate_temperatures", default="0.4,0.5,0.7")
    argp.add_argument("--max_residual_scales", default="0.8,1.0,1.2")
    argp.add_argument("--gate_biases", default="-0.8,-0.5,-0.2")
    argp.add_argument("--fusion_hidden_dims", default="128,256")
    argp.add_argument("--popularity_penalties", default="0.0,0.05")
    argp.add_argument("--long_tail_boosts", default="0.0,0.02")
    argp.add_argument("--value_alpha", default="0.0")
    argp.add_argument("--value_beta", default="0.0")
    argp.add_argument("--value_gamma", default="1.0")
    args = argp.parse_args()

    rnd = random.Random(args.random_seed)
    search_name = args.search_name or datetime.now().strftime("auto_%Y%m%d_%H%M%S")
    root = Path("runs/hparam_search") / search_name
    ckpt_dir = Path("checkpoints/hparam_search") / search_name
    root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    space = {
        "seed": _parse_list(args.seeds, int),
        "budget_ratio": _parse_list(args.budget_ratios, float),
        "user_budget_ratio": _parse_list(args.user_budget_ratios, float),
        "user_source": _parse_list(args.user_sources, str),
        "lr": _parse_list(args.lrs, float),
        "frozen": _parse_list(args.frozen, lambda x: x.lower() in {"1", "true", "yes", "y", "on"}),
        "align_weight": _parse_list(args.align_weights, float),
        "consistency_weight": _parse_list(args.consistency_weights, float),
        "gate_l2_weight": _parse_list(args.gate_l2_weights, float),
        "freeze_backbone_epochs": _parse_list(args.freeze_backbone_epochs, int),
        "gate_temperature": _parse_list(args.gate_temperatures, float),
        "max_residual_scale": _parse_list(args.max_residual_scales, float),
        "gate_bias": _parse_list(args.gate_biases, float),
        "fusion_hidden_dim": _parse_list(args.fusion_hidden_dims, int),
        "popularity_penalty": _parse_list(args.popularity_penalties, float),
        "long_tail_boost": _parse_list(args.long_tail_boosts, float),
        "value_alpha": _parse_list(args.value_alpha, float),
        "value_beta": _parse_list(args.value_beta, float),
        "value_gamma": _parse_list(args.value_gamma, float),
        "user_num_clusters": _parse_list(args.user_num_clusters, int),
        "user_distill_weight": _parse_list(args.user_distill_weights, float),
    }

    candidates = _sample_candidates(args.mode, args.max_trials, rnd, space)
    if not candidates:
        raise RuntimeError("No candidates generated for hyperparameter search.")

    trials_path = root / "trials.jsonl"
    best = None
    best_score = None
    print(f"Hyperparam search start: name={search_name}, trials={len(candidates)}")

    for idx, params in enumerate(candidates, start=1):
        run_name = f"{search_name}_trial_{idx:03d}"
        train_args = _build_train_args(
            config=args.config,
            trial_id=idx,
            run_name=run_name,
            checkpoint_dir=ckpt_dir,
            params=params,
            epochs=args.epochs,
            eval_interval=args.eval_interval,
            semantic_source=args.semantic_source,
            selection_metric=args.selection_metric,
        )

        print(
            f"[Trial {idx}/{len(candidates)}] "
            f"seed={params['seed']} budget={params['budget_ratio']} lr={params['lr']} "
            f"gate_bias={params['gate_bias']} gate_temp={params['gate_temperature']}"
        )
        summary = train(args.config, train_args)
        score = float(summary[args.objective_metric])

        payload = {
            "trial_id": idx,
            "run_name": run_name,
            "objective_metric": args.objective_metric,
            "objective_score": score,
            "params": params,
            "summary": summary,
        }
        with trials_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if best is None or score > best_score:
            best = payload
            best_score = score

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_path = root / "best_result.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(
        "Search completed. "
        f"best {args.objective_metric}={best_score:.6f}, run={best['run_name']}. "
        f"saved: {best_path}"
    )


if __name__ == "__main__":
    main()
