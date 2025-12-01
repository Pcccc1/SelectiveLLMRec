from __future__ import annotations

import torch
from torch import nn


def _convert_sp_mat_to_sp_tensor(sp_mat) -> torch.sparse.FloatTensor:
    coo = sp_mat.tocoo()
    indices = torch.stack((torch.from_numpy(coo.row), torch.from_numpy(coo.col)), dim=0)
    values = torch.from_numpy(coo.data.astype("float32"))
    shape = coo.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape))


class LightGCN(nn.Module):
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

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.register_buffer("adj_torch", _convert_sp_mat_to_sp_tensor(adj_mat))

    def propagate(self):
        all_embeddings = []
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings.append(embeddings)
        g = embeddings
        for _ in range(self.n_layers):
            g = torch.sparse.mm(self.adj_torch, g)
            all_embeddings.append(g)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_g, item_g = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return user_g, item_g

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        user_g, item_g = self.propagate()
        user_e = self.user_embedding(users)
        item_e = self.item_embedding(items)
        return user_e, item_e, user_g[users], item_g[items]

    def full_embeddings(self):
        user_g, item_g = self.propagate()
        return self.user_embedding.weight, self.item_embedding.weight, user_g, item_g

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        user_e, item_e, user_g, item_g = self.forward(users, items)
        return (user_e + user_g) * (item_e + item_g)
