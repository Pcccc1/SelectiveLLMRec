from __future__ import annotations

import torch
from torch import nn

from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


class UserClusterer:
    def __init__(self, num_clusters=100):   
        self.num_clusters = num_clusters
    
    def cluster(self, user_g):
        emb = user_g.detach().cpu().numpy()
        emb = normalize(emb, norm='l2')
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init='auto')
        cluster_ids = kmeans.fit_predict(emb)

        return cluster_ids, kmeans.cluster_centers_



class FusionHead(nn.Module):
    def __init__(self, dim: int, user_cluster: torch.Tensor, cluster_emb: torch.Tensor):
        super().__init__()
        
        self.register_buffer('cluster_emb', cluster_emb)
        self.register_buffer('user_cluster', user_cluster) 

        self.proj_u = nn.Sequential(
            nn.Linear(cluster_emb.size(1), 256),
            nn.ReLU(),
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
        )

        self.proj_i = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
        )

        self.mlp_i = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.gate_i = nn.Linear(2 * dim, 1)

        # 关键初始化：让模型一开始 ≈ 原 GNN
        nn.init.zeros_(self.mlp_i[-1].weight)
        nn.init.zeros_(self.mlp_i[-1].bias)

        self.mlp_u = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.gate_u = nn.Linear(2 * dim, 1)

        # 关键初始化：让模型一开始 ≈ 原 GNN
        nn.init.zeros_(self.mlp_u[-1].weight)
        nn.init.zeros_(self.mlp_u[-1].bias)



    def fusion_item(
        self,
        gnn_emb: torch.Tensor,      # [B, d]
        llm_emb: torch.Tensor,      # [B, d] or zeros
        mask: torch.Tensor          # [B, 1]  (0 or 1)
    ):
        llm_emb = self.proj_i(llm_emb)        # [B, d]
        x = torch.cat([gnn_emb, llm_emb], dim=-1)

        delta = self.mlp_i(x)                       # [B, d]
        alpha = torch.sigmoid(self.gate_i(x))       # [B, 1]

        out = gnn_emb + mask * alpha * delta
        return out


    def fusion_user(self, users: torch.Tensor, user_emb: torch.Tensor):
        """
        users: [B]
        user_g: [B, embed_dim]
        """

        cid = self.user_cluster[users]           # [B]
        cluster_vec = self.cluster_emb[cid]      # [B, 768]

        # project into LightGCN space
        cluster_proj = self.proj_u(cluster_vec)  # [B, embed_dim]

        fused = torch.cat([user_emb, cluster_proj], dim=-1) # [B, embed_dim * 2]

        delta = self.mlp_u(fused)                       # [B, d]
        alpha = torch.sigmoid(self.gate_u(fused))       # [B,

        out = user_emb + alpha * delta
        
        return out