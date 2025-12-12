from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .lightgcn import LightGCN
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import torch.nn.functional as F


class ClusterSemanticFusion(nn.Module):
    """
    Fusion module for:
        user graph embedding (g_u)
        + distance-aware cluster semantic embedding (LLM)
    
    final_u = g_u + alpha_u * proj(cluster_emb)
    
    Components:
    - Projection 768 → embed_dim
    - LayerNorm
    - Alpha computation using Gaussian kernel
    """

    def __init__(
        self,
        embed_dim: int,                           # LightGCN dim
        cluster_emb: torch.Tensor,                # [K, 768]
        user_feature: torch.Tensor,               # [num_users, D]  (pretrain g_u)
        cluster_centers: torch.Tensor,            # [K, D]
        user_cluster: torch.Tensor,               # [num_users]
        device="cuda",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device

        # --------------------------
        # 1. Register cluster embedding (LLM)
        # --------------------------
        cluster_emb = cluster_emb.to(device)
        self.cluster_emb = nn.Embedding(cluster_emb.size(0), cluster_emb.size(1))
        self.cluster_emb.weight = nn.Parameter(cluster_emb, requires_grad=False)

        # --------------------------
        # 2. Projection 768 → embed_dim
        # --------------------------
        self.proj = nn.Linear(cluster_emb.size(1), embed_dim)

        # --------------------------
        # 3. LayerNorm for stability
        # --------------------------
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------
        # 4. Precompute alpha (Gaussian decay)
        # --------------------------
        print("[Fusion] Computing alpha using Gaussian distance decay...")

        user_feature = user_feature.to(device)
        cluster_centers = cluster_centers.to(device=device, dtype=user_feature.dtype)
        user_cluster = user_cluster.to(device).long()

        # use the same normalized space as KMeans clustering for distance
        user_feature = F.normalize(user_feature, p=2, dim=1)

        # distance to cluster center
        dist = torch.norm(user_feature - cluster_centers[user_cluster], dim=1)  # [num_users]

        # compute sigma for each cluster
        num_clusters = cluster_centers.size(0)
        sigma = torch.zeros(num_clusters, device=device, dtype=user_feature.dtype)

        for c in range(num_clusters):
            mask = (user_cluster == c)
            if mask.any():
                sigma[c] = dist[mask].mean()
            else:
                # rare case: empty cluster from KMeans, back off to global mean
                sigma[c] = dist.mean()

        sigma_u = sigma[user_cluster]  # per-user sigma

        alpha = torch.exp(-(dist ** 2) / (2 * sigma_u ** 2 + 1e-8))  # Gaussian kernel
        alpha = alpha.clamp(min=0.0, max=1.0)

        self.register_buffer("alpha", alpha)               # [num_users]
        self.register_buffer("user_cluster", user_cluster) # [num_users]

        print("[Fusion] Alpha computed. Example:", alpha[:10])

    # --------------------------
    #  Fusion forward
    # --------------------------
    def forward(self, users: torch.Tensor, user_g: torch.Tensor):
        """
        users: [B]
        user_g: [B, embed_dim]
        """

        cid = self.user_cluster[users]           # [B]
        cluster_vec = self.cluster_emb(cid)      # [B, 768]

        # project into LightGCN space
        cluster_proj = self.proj(cluster_vec)    # [B, embed_dim]
        cluster_proj = self.norm(cluster_proj)

        # get alpha for each user
        alpha_u = self.alpha[users].unsqueeze(1)  # [B, 1]

        # fusion
        fused = user_g + alpha_u * cluster_proj   # [B, embed_dim]

        return fused



class UserClusterer:
    def __init__(self, num_clusters=100):   
        self.num_clusters = num_clusters
    
    def cluster(self, user_g):
        emb = user_g.detach().cpu().numpy()
        emb = normalize(emb, norm='l2')
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init='auto')
        cluster_ids = kmeans.fit_predict(emb)

        return cluster_ids, kmeans.cluster_centers_
