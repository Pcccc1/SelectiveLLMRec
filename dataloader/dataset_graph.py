import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix, diags

class GraphPretrainDataset(Dataset):
    def __init__(self, user_pos_items, num_users, num_items):
        self.user_pos_items = user_pos_items  
        self.users = [u for u in range(num_users) if len(user_pos_items[u]) > 0]
        self.num_users = num_users
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos_item = random.choice(self.user_pos_items[u])
        return u, pos_item




class GraphDatasetParser:
    #TODO: 增强的metedata还没有做映射
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

    def remap_ids(self):
        # 收集所有 ID
        users = set()
        items = set()
        for u, i in self.train + self.val + self.test:
            users.add(u)
            items.add(i)

        # 连续映射
        self.user2id = {u: idx for idx, u in enumerate(sorted(users))}
        self.item2id = {i: idx for idx, i in enumerate(sorted(items))}

        # 映射后的数据
        def map_one(data):
            mapped = []
            for u, i in data:
                if u in self.user2id and i in self.item2id:
                    mapped.append((self.user2id[u], self.item2id[i]))
            return mapped


        self.train = map_one(self.train)
        self.val = map_one(self.val)
        self.test = map_one(self.test)

        self.num_users = len(users)
        self.num_items = len(items)

    # 构建用户-正样本项的字典
    def build_user_pos_items(self):
        self.user_pos_items = {u: [] for u in range(self.num_users)}
        for u, i in self.train:
            self.user_pos_items[u].append(i)

    # 构建用户-正样本项的稀疏矩阵
    def build_adj_mat(self):
        row, col, data = [], [], []
        for u, i in self.train:
            j = i + self.num_users
            row.append(u); col.append(j); data.append(1)
            row.append(j); col.append(u); data.append(1)

        N = self.num_users + self.num_items
        adj = coo_matrix((data, (row, col)), shape=(N, N), dtype=np.float32)

        degree = np.array(adj.sum(axis=1)).flatten()
        degree[degree == 0] = 1.0  # avoid divide-by-zero for isolated nodes
        d_inv_sqrt = np.power(degree, -0.5, dtype=np.float32)
        d_mat_inv_sqrt = diags(d_inv_sqrt)

        # Symmetric normalized adjacency used by LightGCN: D^{-1/2} A D^{-1/2}
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        self.adj_mat = norm_adj.tocsr()


def collate_graph(batch, neg_sample):
    users, pos, neg = [], [], []

    for u, p in batch:
        users.append(u)
        pos.append(p)
        neg.append(neg_sample.sample(u))

    return {
        "user": torch.LongTensor(users),
        "pos" : torch.LongTensor(pos),
        "neg" : torch.LongTensor(neg),
    }
