from __future__ import annotations

import torch
from torch import nn



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
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
        )

        self.mlp_i = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.mlp_u = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.scale_i = nn.Parameter(torch.tensor(0.5))
        self.scale_u = nn.Parameter(torch.tensor(0.5))



    def fusion_item(
        self,
        gnn_emb: torch.Tensor,      # [B, d]
        llm_emb: torch.Tensor,      # [B, d] or zeros
        mask: torch.Tensor          # [B, 1]  (0 or 1)
    ):
        llm_emb = self.proj_i(llm_emb)        # [B, d]
        x = gnn_emb + self.scale_i * self.mlp_i(torch.cat([gnn_emb, llm_emb], dim=-1))
        return x


    def fusion_user(self, users: torch.Tensor, user_emb: torch.Tensor):
        """
        users: [B]
        user_g: [B, embed_dim]
        """

        cid = self.user_cluster[users]           # [B]
        cluster_vec = self.cluster_emb[cid]      # [B, 768]

        cluster_proj = self.proj_u(cluster_vec)  # [B, embed_dim]
        fused = user_emb + self.scale_u * self.mlp_u(torch.cat([user_emb, cluster_proj], dim=-1))
        return fused


class ItemSemanticFusionHead(nn.Module):
    """
    Conservative item-only semantic fusion:
        z = g + mask * alpha * delta
    """

    def __init__(self, gnn_dim: int, semantic_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, gnn_dim),
            nn.LayerNorm(gnn_dim),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(gnn_dim * 2, gnn_dim),
            nn.ReLU(),
            nn.Linear(gnn_dim, gnn_dim),
        )
        self.gate = nn.Linear(gnn_dim * 2, 1)

        # Start close to identity mapping.
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(
        self,
        gnn_emb: torch.Tensor,      # [B, d]
        semantic_emb: torch.Tensor, # [B, ds]
        selected_mask: torch.Tensor # [B, 1], 0/1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sem_proj = self.projector(semantic_emb)
        x = torch.cat([gnn_emb, sem_proj], dim=-1)
        delta = self.delta_mlp(x)
        alpha = torch.sigmoid(self.gate(x))
        fused = gnn_emb + selected_mask * alpha * delta
        return fused, sem_proj, alpha
