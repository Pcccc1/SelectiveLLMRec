from __future__ import annotations

import torch
from torch import nn

class ItemSemanticFusionHead(nn.Module):
    """
    Conservative item-only semantic fusion:
        z = g + mask * alpha * delta
    """

    def __init__(
        self,
        gnn_dim: int,
        semantic_dim: int,
        hidden_dim: int = 256,
        gate_temperature: float = 1.0,
        max_residual_scale: float = 0.35,
        gate_bias: float = -2.5,
    ):
        super().__init__()
        self.gnn_dim = int(gnn_dim)
        self.gate_temperature = float(max(1e-3, gate_temperature))
        self.max_residual_scale = float(max_residual_scale)
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
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Start close to identity mapping.
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(
        self,
        gnn_emb: torch.Tensor,      # [B, d]
        semantic_emb: torch.Tensor, # [B, ds]
        selected_mask: torch.Tensor # [B, 1], 0/1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sem_proj = self.projector(semantic_emb)
        x = torch.cat([gnn_emb, sem_proj], dim=-1)
        # Bound residual and gate strength for stable conservative fusion.
        delta = torch.tanh(self.delta_mlp(x))
        alpha = torch.sigmoid(self.gate(x) / self.gate_temperature)
        scale = torch.clamp(self.residual_scale, min=0.0, max=self.max_residual_scale)
        fused = gnn_emb + selected_mask * scale * alpha * delta
        return fused, sem_proj, alpha, delta


class UserSemanticFusionHead(nn.Module):
    """
    Conservative user-side semantic transfer:
        z = g + mask * alpha(need) * delta
    """

    def __init__(
        self,
        gnn_dim: int,
        semantic_dim: int,
        hidden_dim: int = 256,
        gate_temperature: float = 1.0,
        max_residual_scale: float = 0.35,
        gate_bias: float = -2.5,
    ):
        super().__init__()
        self.gnn_dim = int(gnn_dim)
        self.gate_temperature = float(max(1e-3, gate_temperature))
        self.max_residual_scale = float(max_residual_scale)
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
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(
        self,
        gnn_emb: torch.Tensor,      # [B, d]
        semantic_emb: torch.Tensor, # [B, ds]
        selected_mask: torch.Tensor, # [B, 1], 0/1
        need_score: torch.Tensor,   # [B, 1], [0, 1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sem_proj = self.projector(semantic_emb)
        x = torch.cat([gnn_emb, sem_proj], dim=-1)
        delta = torch.tanh(self.delta_mlp(x))
        alpha = torch.sigmoid(self.gate(x) / self.gate_temperature)
        alpha = alpha * torch.clamp(need_score, min=0.0, max=1.0)
        scale = torch.clamp(self.residual_scale, min=0.0, max=self.max_residual_scale)
        fused = gnn_emb + selected_mask * scale * alpha * delta
        return fused, sem_proj, alpha, delta
