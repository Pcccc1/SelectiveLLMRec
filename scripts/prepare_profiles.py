from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim
from tqdm import trange

from src.config import ExperimentConfig
from src.data_loading import BPRSampler, build_sparse_graph, load_interactions
from src.lightgcn import LightGCN
from src.losses import bpr_loss
from src.profiles import prepare_profiles


def pretrain_lightgcn(data, cfg, device, epochs: int = 10, batch_size: int = 4096):
    adj = build_sparse_graph(data)
    model = LightGCN(
        data.num_users,
        data.num_items,
        cfg.lightgcn.embedding_dim,
        cfg.lightgcn.n_layers,
        adj,
    ).to(device)
    sampler = BPRSampler(data.user_pos, data.num_items)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps_per_epoch = max(1, len(data.df) // batch_size)

    for _ in trange(epochs, desc="pretrain-gcn"):
        for _ in range(steps_per_epoch):
            u, p, n = sampler.sample(batch_size)
            u = torch.tensor(u, device=device)
            p = torch.tensor(p, device=device)
            n = torch.tensor(n, device=device)
            user_e, item_e, g_user, g_item = model.full_embeddings()
            z_u = user_e[u] + g_user[u]
            z_p = item_e[p] + g_item[p]
            z_n = item_e[n] + g_item[n]
            loss = bpr_loss(z_u, z_p, z_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def main(args):
    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    data = load_interactions(cfg.data)
    pretrain_epochs = args.pretrain_epochs or max(10, cfg.train.epochs // 5)
    gcn = pretrain_lightgcn(data, cfg, device, pretrain_epochs, cfg.train.batch_size)
    _, _, g_user, _ = gcn.full_embeddings()
    artifacts_dir = Path(cfg.artifacts_dir) / cfg.data.dataset
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(g_user.cpu(), artifacts_dir / "pretrain_user_g.pt")
    profiles = prepare_profiles(
        data,
        g_user.cpu(),
        cfg.profile,
        artifacts_dir,
        recency_half_life=cfg.data.recency_half_life,
    )
    print(f"Saved profiles to {artifacts_dir}")
    print(
        f"user_profiles: {profiles.user_profiles.shape}, item_profiles: {profiles.item_profiles.shape}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/example.yaml", help="Path to YAML config file."
    )
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    args = parser.parse_args()
    main(args)
