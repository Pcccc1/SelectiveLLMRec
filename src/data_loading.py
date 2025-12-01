from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import DataConfig


@dataclass
class InteractionData:
    df: pd.DataFrame
    user2id: Dict[str, int]
    item2id: Dict[str, int]
    user_pos: Dict[int, set]
    num_users: int
    num_items: int
    max_ts: float


def load_interactions(cfg: DataConfig) -> InteractionData:
    """
    Expected schema: user_id, item_id, rating, timestamp (unix or pandas-friendly).
    """
    path = Path(cfg.data_dir) / cfg.dataset / "interactions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing interactions file at {path}")

    df = pd.read_csv(path)
    for col in ["user_id", "item_id"]:
        if col not in df.columns:
            raise ValueError(f"{path} must contain column {col}")

    if "rating" in df.columns:
        df = df[df["rating"] > 0]

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
    else:
        ts = pd.Series(pd.Timestamp.now().repeat(len(df)))
    df["timestamp"] = ts.astype("int64") // 1_000_000_000

    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()
    df = df[
        (df["user_id"].isin(user_counts[user_counts >= cfg.min_user_interactions].index))
        & (df["item_id"].isin(item_counts[item_counts >= cfg.min_item_interactions].index))
    ]

    user2id = {u: idx for idx, u in enumerate(df["user_id"].unique())}
    item2id = {i: idx for idx, i in enumerate(df["item_id"].unique())}
    df["uidx"] = df["user_id"].map(user2id)
    df["iidx"] = df["item_id"].map(item2id)

    user_pos: Dict[int, set] = {}
    for row in df[["uidx", "iidx"]].itertuples(index=False):
        user_pos.setdefault(row.uidx, set()).add(row.iidx)

    return InteractionData(
        df=df,
        user2id=user2id,
        item2id=item2id,
        user_pos=user_pos,
        num_users=len(user2id),
        num_items=len(item2id),
        max_ts=float(df["timestamp"].max()),
    )


def build_sparse_graph(data: InteractionData) -> sp.csr_matrix:
    """
    Build a normalized bi-adjacency matrix for LightGCN.
    """
    users = data.df["uidx"].to_numpy()
    items = data.df["iidx"].to_numpy() + data.num_users
    n_nodes = data.num_users + data.num_items

    values = np.ones_like(users, dtype=np.float32)
    mat = sp.coo_matrix((values, (users, items)), shape=(n_nodes, n_nodes))
    mat = mat + mat.transpose()

    rowsum = np.array(mat.sum(axis=1)).squeeze()
    d_inv = np.power(rowsum + 1e-8, -0.5)
    d_mat_inv = sp.diags(d_inv)
    norm_mat = d_mat_inv @ mat @ d_mat_inv
    return norm_mat.tocsr()


class BPRSampler:
    def __init__(self, user_pos: Dict[int, set], num_items: int):
        self.user_pos = user_pos
        self.num_items = num_items
        self.users = list(user_pos.keys())

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        users, pos_items, neg_items = [], [], []
        for _ in range(batch_size):
            u = random.choice(self.users)
            pos = random.choice(tuple(self.user_pos[u]))
            neg = random.randrange(self.num_items)
            while neg in self.user_pos[u]:
                neg = random.randrange(self.num_items)
            users.append(u)
            pos_items.append(pos)
            neg_items.append(neg)
        return (
            np.asarray(users, dtype=np.int64),
            np.asarray(pos_items, dtype=np.int64),
            np.asarray(neg_items, dtype=np.int64),
        )


def compute_item_recency(
    data: InteractionData, half_life_days: float = 30.0
) -> np.ndarray:
    timestamps = data.df.groupby("iidx")["timestamp"].max()
    recency_seconds = np.maximum(data.max_ts - timestamps, 0)
    half_life = half_life_days * 24 * 3600
    decay = np.exp(-recency_seconds / half_life)
    recency = np.zeros(data.num_items, dtype=np.float32)
    recency[timestamps.index] = decay
    return recency
