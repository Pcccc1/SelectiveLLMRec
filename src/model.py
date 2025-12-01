from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .lightgcn import LightGCN
from .losses import bpr_loss, info_nce


class FusionRecModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        n_layers: int,
        adj_mat,
        profile_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.lightgcn = LightGCN(num_users, num_items, embedding_dim, n_layers, adj_mat)
        hidden_dim = hidden_dim or embedding_dim
        fusion_dim = embedding_dim * 2 + profile_dim
        self.user_proj = nn.Linear(fusion_dim, hidden_dim)
        self.item_proj = nn.Linear(fusion_dim, hidden_dim)

        self.register_buffer(
            "user_profiles", torch.zeros((num_users, profile_dim), dtype=torch.float32)
        )
        self.register_buffer(
            "item_profiles", torch.zeros((num_items, profile_dim), dtype=torch.float32)
        )

    def load_profiles(self, user_profiles: torch.Tensor, item_profiles: torch.Tensor):
        self.user_profiles = user_profiles.to(self.user_profiles.device)
        self.item_profiles = item_profiles.to(self.item_profiles.device)

    def representations(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        id_user, id_item, g_user, g_item = self.lightgcn.full_embeddings()
        z_u = self._fuse_user(id_user[users], g_user[users], self.user_profiles[users])
        z_i = self._fuse_item(id_item[items], g_item[items], self.item_profiles[items])
        return z_u, z_i

    def _fuse_user(self, e: torch.Tensor, g: torch.Tensor, p: torch.Tensor):
        x = torch.cat([e, g, p], dim=-1)
        return self.user_proj(x)

    def _fuse_item(self, e: torch.Tensor, g: torch.Tensor, p: torch.Tensor):
        x = torch.cat([e, g, p], dim=-1)
        return self.item_proj(x)

    def score(self, users: torch.Tensor, items: torch.Tensor):
        z_u, z_i = self.representations(users, items)
        return (z_u * z_i).sum(dim=-1)

    def training_step(
        self,
        batch,
        lambda_user: float,
        lambda_item: float,
        temperature: float,
    ):
        users, pos_items, neg_items = batch
        users = users.to(self.user_profiles.device)
        pos_items = pos_items.to(self.user_profiles.device)
        neg_items = neg_items.to(self.user_profiles.device)

        id_user, id_item, g_user, g_item = self.lightgcn.full_embeddings()
        z_u = self._fuse_user(id_user[users], g_user[users], self.user_profiles[users])
        z_pos = self._fuse_item(
            id_item[pos_items], g_item[pos_items], self.item_profiles[pos_items]
        )
        z_neg = self._fuse_item(
            id_item[neg_items], g_item[neg_items], self.item_profiles[neg_items]
        )

        loss = bpr_loss(z_u, z_pos, z_neg)

        if lambda_user > 0:
            loss = loss + lambda_user * self.user_contrastive(users, temperature, g_user)
        if lambda_item > 0:
            loss = loss + lambda_item * self.item_contrastive(
                pos_items, temperature, g_item
            )
        return loss

    def user_contrastive(
        self, users: torch.Tensor, temperature: float, g_user_full: torch.Tensor
    ):
        anchor = g_user_full[users]
        positive = self.user_profiles[users]
        return info_nce(anchor, positive, temperature)

    def item_contrastive(
        self, items: torch.Tensor, temperature: float, g_item_full: torch.Tensor
    ):
        profiles = self.item_profiles[items]
        mask = profiles.norm(dim=-1) > 0
        if mask.sum() <= 1:
            return torch.tensor(0.0, device=profiles.device)
        anchor = g_item_full[items][mask]
        positive = profiles[mask]
        return info_nce(anchor, positive, temperature)
