from __future__ import annotations

import torch
from torch import nn
from .model import ClusterSemanticFusion
import numpy as np
from scipy.sparse import diags


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
        items: torch.Tensor,
        use_graph_embedding: bool = True,
    ) -> torch.Tensor:
        """
        预测一批 (user, item) pair 的偏好得分。

        输入:
            users: [B]
            items: [B] 或 [B, K] (后者一般自己写广播或展开)
            use_graph_embedding: True 时用 GNN embedding，否则用 ID embedding

        返回:
            scores: [B] 对应 user-item 的打分
        """
        user_e, item_e, user_g, item_g = self.get_all_embeddings()

        if use_graph_embedding:
            u = user_g[users]
            v = item_g[items]
        else:
            u = user_e[users]
            v = item_e[items]

        # 点积
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
        user_feature: torch.Tensor,
        cluster_centers: torch.Tensor,
        user_cluster: torch.Tensor,
        device: str | torch.device = "cuda",
    ):
        super().__init__(num_users, num_items, embedding_dim, n_layers, adj_mat)
        self.device = torch.device(device)
        self.fusion = ClusterSemanticFusion(
            embed_dim=embedding_dim,
            cluster_emb=cluster_emb,
            user_feature=user_feature,
            cluster_centers=cluster_centers,
            user_cluster=user_cluster,
            device=device,
        )


    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ):
        _, _, user_g, item_g = self.get_all_embeddings()
        u_g = user_g[users]           # [B,d]
        u_g = self.fusion(users, u_g)

        pos_g = item_g[pos_items]     # [B,d]
        neg_g = item_g[neg_items]     # [B,d]
        return u_g, pos_g, neg_g