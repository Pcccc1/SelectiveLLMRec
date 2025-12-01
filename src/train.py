from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import optim
from tqdm import trange

from .config import ExperimentConfig
from .data_loading import BPRSampler, build_sparse_graph, load_interactions
from .model import FusionRecModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_profiles(artifacts_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    user_p = torch.load(artifacts_dir / "user_profiles.pt")
    item_p = torch.load(artifacts_dir / "item_profiles.pt")
    return user_p, item_p


def train(cfg_path: str):
    cfg = ExperimentConfig.from_yaml(cfg_path)
    set_seed(cfg.seed)

    data = load_interactions(cfg.data)
    adj = build_sparse_graph(data)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    model = FusionRecModel(
        data.num_users,
        data.num_items,
        cfg.lightgcn.embedding_dim,
        cfg.lightgcn.n_layers,
        adj,
        cfg.profile.profile_dim,
        hidden_dim=cfg.lightgcn.embedding_dim,
    ).to(device)

    artifacts_dir = Path(cfg.artifacts_dir) / cfg.data.dataset
    user_p, item_p = load_profiles(artifacts_dir)
    model.load_profiles(user_p.to(device), item_p.to(device))

    sampler = BPRSampler(data.user_pos, data.num_items)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    steps_per_epoch = max(
        1, len(data.df) // (cfg.train.batch_size * cfg.train.neg_samples)
    )

    for epoch in trange(cfg.train.epochs, desc="epochs"):
        model.train()
        for _ in range(steps_per_epoch):
            batch = sampler.sample(cfg.train.batch_size)
            batch = tuple(torch.tensor(x, device=device) for x in batch)
            loss = model.training_step(
                batch,
                lambda_user=cfg.train.lambda_user_cl,
                lambda_item=cfg.train.lambda_item_cl,
                temperature=cfg.train.temperature,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            ckpt = artifacts_dir / f"checkpoint_epoch{epoch+1}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/example.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    train(args.config)
