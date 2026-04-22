from __future__ import annotations

import torch
from torch import nn
from .fusion import FusionHead, ItemSemanticFusionHead
import numpy as np
from scipy.sparse import diags
import torch.nn.functional as F


def _convert_sp_mat_to_sp_tensor(sp_mat) -> torch.sparse.FloatTensor:
    """scipy.spmatrix -> torch.sparse.FloatTensor (coalesced)."""
    coo = sp_mat.tocoo()
    indices = torch.stack(
        (torch.from_numpy(coo.row), torch.from_numpy(coo.col)), dim=0
    )
    values = torch.from_numpy(coo.data.astype("float32"))
    shape = coo.shape
    sp_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
    return sp_tensor.coalesce()


class LightGCN(nn.Module):
    """
    Standard LightGCN for user–item图上的推荐任务。

    节点顺序固定为: [所有 user; 所有 item]，邻接矩阵 adj_mat 的行列顺序必须一致。
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        n_layers: int,
        adj_mat,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        # ID embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        # 归一化后的拉普拉斯/邻接矩阵，稀疏格式
        # 作为 buffer 注册，不参与梯度更新

        degree = np.array(adj_mat.sum(axis=1)).flatten()
        degree[degree == 0] = 1.0  # avoid divide-by-zero for isolated nodes
        d_inv_sqrt = np.power(degree, -0.5, dtype=np.float32)
        d_mat_inv_sqrt = diags(d_inv_sqrt)

        # Symmetric normalized adjacency used by LightGCN: D^{-1/2} A D^{-1/2}
        norm_adj = d_mat_inv_sqrt @ adj_mat @ d_mat_inv_sqrt


        self.register_buffer("adj_torch", _convert_sp_mat_to_sp_tensor(norm_adj))

    # ------------------------------------------------------------------
    # 核心：图传播（不区分 train / eval，单纯计算 GNN embedding）
    # ------------------------------------------------------------------
    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:

        # 初始 embedding: [U+I, d]
        all_emb = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )                                   # [N, d], N = U + I

        # 累加每一层的 embedding（包括 0 层）
        out = all_emb                       # 累加器，先放第 0 层
        g = all_emb
        for _ in range(self.n_layers):
            g = torch.sparse.mm(self.adj_torch, g)   # [N, d]
            out = out + g 

        # 按论文，等价于对 (K+1) 层取平均，这里直接 / (K+1)
        out = out / (self.n_layers + 1)

        # 切回 user / item
        user_g, item_g = torch.split(
            out, [self.num_users, self.num_items], dim=0
        )
        return user_g, item_g
    

    # Get embeddings at each layer for uncertainty estimation
    def propagate_with_layers(self) -> list[torch.Tensor]:
        """
        Return item embeddings at each GNN layer (including layer 0).

        Returns:
            item_emb_layers: List[Tensor]
                length = n_layers + 1
                each shape = [num_items, dim]
        """
        # 初始 embedding（layer 0）
        g = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0
        )  # [N, d]

        item_layers = []

        # layer 0
        _, item_emb = torch.split(
            g, [self.num_users, self.num_items], dim=0
        )
        item_layers.append(item_emb)

        # layer 1 ~ L
        for _ in range(self.n_layers):
            g = torch.sparse.mm(self.adj_torch, g)

            _, item_emb = torch.split(
                g, [self.num_users, self.num_items], dim=0
            )
            item_layers.append(item_emb)

        return item_layers


    # ------------------------------------------------------------------
    # Embedding 接口：方便训练/评估统一调用
    # ------------------------------------------------------------------
    def get_all_embeddings(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            user_e: ID-level user embedding (nn.Embedding)
            item_e: ID-level item embedding
            user_g: GNN 聚合后的 user embedding
            item_g: GNN 聚合后的 item embedding
        """
        user_e = self.user_embedding.weight  # [U, d]
        item_e = self.item_embedding.weight  # [I, d]
        user_g, item_g = self.propagate()
        return user_e, item_e, user_g, item_g

    # ------------------------------------------------------------------
    # 训练阶段：BPR 用的 forward
    # ------------------------------------------------------------------
    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        BPR 训练使用的前向:

        输入:
            users:     [B]
            pos_items: [B]
            neg_items: [B]

        返回:
            u_g:   [B, d]   对应 users 的 GNN embedding
            pos_g: [B, d]   正样本 item 的 GNN embedding
            neg_g: [B, d]   负样本 item 的 GNN embedding
        """
        _, _, user_g, item_g = self.get_all_embeddings()

        u_g = user_g[users]           # [B, d]
        pos_g = item_g[pos_items]     # [B, d]
        neg_g = item_g[neg_items]     # [B, d]

        return u_g, pos_g, neg_g

    # ------------------------------------------------------------------
    # 推理接口：预测任意 (user, item) 对的得分
    # ------------------------------------------------------------------
    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor | None = None,
        use_graph_embedding: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        统一推理接口：
        1) items is None: 返回 (batch_user_emb, all_item_emb)，用于 full-ranking 评估
        2) items is not None: 返回 (user, item) pair 的点积得分
        """
        user_e, item_e, user_g, item_g = self.get_all_embeddings()

        if use_graph_embedding:
            all_user_emb, all_item_emb = user_g, item_g
        else:
            all_user_emb, all_item_emb = user_e, item_e

        users = users.to(all_user_emb.device)

        # full ranking path
        if items is None:
            return all_user_emb[users], all_item_emb

        # pair scoring path
        items = items.to(all_item_emb.device)
        u = all_user_emb[users]
        v = all_item_emb[items]
        return (u * v).sum(dim=-1)


    

class LightGCN_retrain(LightGCN):
    """
    Retrain/Fine-tune version with cluster semantic fusion:
        final_u = user_g + alpha_u * LN(Proj(cluster_emb[cid]))

    Requirements:
    - cluster_embeddings.pt: dict{cid: Tensor(768,)} OR Tensor[K,768]
    - user_cluster.pt: Tensor[num_users] (long)
    - alpha.pt: Tensor[num_users] (float), computed from pretrain user_feature + cluster_centers by Gaussian decay
      (recommended: precompute offline)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        n_layers: int,
        adj_mat,
        cluster_emb: torch.Tensor | dict[int, torch.Tensor],
        user_cluster: torch.Tensor,
        item_profile_embeddings: dict[int, torch.Tensor] | torch.Tensor,
        device: str | torch.device = "cuda",
    ):
        super().__init__(num_users, num_items, embedding_dim, n_layers, adj_mat)
        self.device = torch.device(device)
        self.fusion_head = FusionHead(embedding_dim, user_cluster, cluster_emb)
        self.item_profile_embeddings = item_profile_embeddings

        self.register_buffer('cluster_emb', cluster_emb)

        item_llm_emb = torch.zeros((num_items, 1536), device=self.device)
        item_llm_mask = torch.zeros((num_items, 1), device=self.device)
        if isinstance(self.item_profile_embeddings, dict):
            embedding_iter = self.item_profile_embeddings.items()
        else:
            embedding_iter = enumerate(self.item_profile_embeddings)

        for iid, emb in embedding_iter:
            iid = int(iid)
            if iid < 0 or iid >= num_items:
                continue

            if not torch.is_tensor(emb):
                emb = torch.as_tensor(emb)
            emb = emb.to(self.device).view(-1)

            copy_dim = min(item_llm_emb.size(1), emb.numel())
            item_llm_emb[iid, :copy_dim] = emb[:copy_dim]
            item_llm_mask[iid] = 1.0

        self.register_buffer('item_llm_emb', item_llm_emb)
        self.register_buffer('item_llm_mask', item_llm_mask)
        

    def build_llm_items_embedings(self, item_ids, device):
        llm_emb = self.item_llm_emb[item_ids].to(device)      # [B,d]
        mask = self.item_llm_mask[item_ids].to(device)        # [B,1]
        return llm_emb, mask


    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ):
        
        user_e, item_e, user_g, item_g = self.get_all_embeddings()
        u_g = user_g[users]           # [B,d]
        pos_g = item_g[pos_items]     # [B,d]
        neg_g = item_g[neg_items]     # [B,d]
        # fusion
        u_g = self.fusion_head.fusion_user(users, u_g)

        pos_llm, pos_mask = self.build_llm_items_embedings(pos_items, pos_g.device)
        pos_final = self.fusion_head.fusion_item(pos_g, pos_llm, pos_mask)

        neg_llm, neg_mask = self.build_llm_items_embedings(neg_items, pos_g.device)
        neg_final = self.fusion_head.fusion_item(neg_g, neg_llm, neg_mask)


        return u_g, pos_final, neg_final
    

    @torch.no_grad()
    def predict(self, users: torch.Tensor):

        self.eval()

        user_e, item_e, user_g, item_g = self.get_all_embeddings()
        device = self.device

        # ---------- user ----------
        u_g = user_g[users]                       # [B, d]
        u_final = self.fusion_head.fusion_user(
            users,
            u_g
        )                                         # [B, d]

        # ---------- item (ALL items) ----------
        num_items = self.num_items
        all_item_ids = torch.arange(
            num_items, device=device, dtype=torch.long
        )

        item_g_all = item_g                       # [num_items, d]

        # build llm emb + mask for all items
        item_llm, item_mask = self.build_llm_items_embedings(
            all_item_ids,
            device
        )                                         # [num_items, 768], [num_items, 1]

        item_final = self.fusion_head.fusion_item(
            item_g_all,
            item_llm,
            item_mask
        )                                         # [num_items, d]
        
        return u_final, item_final

    

class LightGCNBudgetedSemantic(LightGCN):
    """
    Item-only budgeted semantic acquisition model.

    - Users keep collaborative embeddings from LightGCN.
    - Items receive semantic residual corrections only when selected mask == 1.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        n_layers: int,
        adj_mat,
        item_semantic_embeddings: dict[int, torch.Tensor] | torch.Tensor,
        selected_item_mask: torch.Tensor,
        semantic_dim: int,
        fusion_hidden_dim: int = 256,
        gate_temperature: float = 1.0,
        max_residual_scale: float = 0.35,
        gate_bias: float = -2.5,
        device: str | torch.device = "cuda",
    ):
        super().__init__(num_users, num_items, embedding_dim, n_layers, adj_mat)
        self.device = torch.device(device)
        self.semantic_dim = int(semantic_dim)
        self.item_fusion = ItemSemanticFusionHead(
            gnn_dim=embedding_dim,
            semantic_dim=self.semantic_dim,
            hidden_dim=int(fusion_hidden_dim),
            gate_temperature=float(gate_temperature),
            max_residual_scale=float(max_residual_scale),
            gate_bias=float(gate_bias),
        )

        semantic_matrix = torch.zeros((num_items, self.semantic_dim), dtype=torch.float32)
        if isinstance(item_semantic_embeddings, dict):
            iterator = item_semantic_embeddings.items()
        else:
            iterator = enumerate(item_semantic_embeddings)

        for iid, emb in iterator:
            iid = int(iid)
            if iid < 0 or iid >= num_items:
                continue
            if not torch.is_tensor(emb):
                emb = torch.as_tensor(emb)
            emb = emb.view(-1).detach().cpu().float()
            copy_dim = min(self.semantic_dim, emb.numel())
            semantic_matrix[iid, :copy_dim] = emb[:copy_dim]

        selected_item_mask = torch.as_tensor(selected_item_mask, dtype=torch.float32).view(-1, 1)
        if selected_item_mask.size(0) != num_items:
            raise ValueError(
                f"selected_item_mask length mismatch: {selected_item_mask.size(0)} vs {num_items}"
            )

        self.register_buffer("item_semantic_emb", semantic_matrix)
        self.register_buffer("item_selected_mask", selected_item_mask)

    def _fuse_items(
        self,
        item_ids: torch.Tensor,
        item_gnn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_emb = self.item_semantic_emb[item_ids].to(item_gnn.device)
        selected_mask = self.item_selected_mask[item_ids].to(item_gnn.device)
        fused, sem_proj, alpha, _ = self.item_fusion(item_gnn, semantic_emb, selected_mask)
        return fused, sem_proj, selected_mask, alpha

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _, _, user_g, item_g = self.get_all_embeddings()

        user_batch = user_g[users]

        pos_base = item_g[pos_items]
        neg_base = item_g[neg_items]

        pos_fused, pos_sem_proj, pos_mask, pos_alpha = self._fuse_items(pos_items, pos_base)
        neg_fused, neg_sem_proj, neg_mask, neg_alpha = self._fuse_items(neg_items, neg_base)

        return {
            "user": user_batch,
            "pos_fused": pos_fused,
            "neg_fused": neg_fused,
            "pos_base": pos_base,
            "neg_base": neg_base,
            "pos_sem_proj": pos_sem_proj,
            "neg_sem_proj": neg_sem_proj,
            "pos_mask": pos_mask,
            "neg_mask": neg_mask,
            "pos_alpha": pos_alpha,
            "neg_alpha": neg_alpha,
        }

    @torch.no_grad()
    def predict(self, users: torch.Tensor):
        _, _, user_g, item_g = self.get_all_embeddings()
        users = users.to(user_g.device)
        all_item_ids = torch.arange(self.num_items, device=item_g.device, dtype=torch.long)
        item_fused, _, _, _ = self._fuse_items(all_item_ids, item_g)
        return user_g[users], item_fused
