from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import random

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer

from .config import ProfileConfig
from .data_loading import InteractionData, compute_item_recency


@dataclass
class ProfileArtifacts:
    user_profiles: torch.Tensor
    item_profiles: torch.Tensor
    user_cluster_ids: np.ndarray
    top_item_indices: np.ndarray


class HashingEncoder:
    def __init__(self, dim: int):
        self.vectorizer = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm=None
        )

    def encode(self, texts: List[str]) -> torch.Tensor:
        matrix = self.vectorizer.transform(texts)
        return torch.tensor(matrix.toarray(), dtype=torch.float32)


class LLMClient:
    def __init__(self, placeholder: bool = True):
        self.placeholder = placeholder

    def generate_cluster_profile(self, prompt: str) -> str:
        if self.placeholder:
            return (
                "Summarized cluster interests: "
                + " ".join(prompt.splitlines()[:3])[:200]
            )
        raise NotImplementedError("Hook up real LLM client here.")

    def generate_item_profile(self, prompt: str) -> str:
        if self.placeholder:
            return "Item focus summary: " + prompt[:200]
        raise NotImplementedError("Hook up real LLM client here.")


def cluster_users(
    user_embeddings: torch.Tensor, cfg: ProfileConfig
) -> Tuple[np.ndarray, MiniBatchKMeans]:
    model = MiniBatchKMeans(
        n_clusters=cfg.num_clusters, random_state=0, batch_size=10_000, n_init="auto"
    )
    cluster_ids = model.fit_predict(user_embeddings.cpu().numpy())
    return cluster_ids, model


def _sample_user_history(
    data: InteractionData, user_idx: int, max_examples: int
) -> List[str]:
    user_rows = data.df[data.df["uidx"] == user_idx]
    sampled = user_rows.sample(
        n=min(max_examples, len(user_rows)), random_state=random.randint(0, 10_000)
    )
    text_cols = [c for c in ["title", "description", "category"] if c in data.df.columns]
    lines = []
    for _, row in sampled.iterrows():
        parts = [f"item_id={row['item_id']}"]
        for col in text_cols:
            parts.append(f"{col}={row[col]}")
        lines.append(", ".join(parts))
    return lines


def build_cluster_prompts(
    data: InteractionData,
    cluster_ids: np.ndarray,
    cfg: ProfileConfig,
) -> List[str]:
    prompts = []
    for c in range(cfg.num_clusters):
        members = np.where(cluster_ids == c)[0]
        if len(members) == 0:
            prompts.append("Empty cluster.")
            continue
        reps = np.random.choice(
            members,
            size=min(cfg.cluster_representatives, len(members)),
            replace=False,
        )
        lines = ["Below are interactions from similar users:"]
        for u in reps:
            history = _sample_user_history(data, int(u), cfg.prompt_user_examples)
            if history:
                lines.append(f"User {u}: " + " | ".join(history))
        lines.append("Summarize the core interests in 3-5 short bullets.")
        prompts.append("\n".join(lines))
    return prompts


def item_value_scores(data: InteractionData, half_life: float) -> np.ndarray:
    deg = data.df.groupby("iidx").size()
    deg_scores = np.zeros(data.num_items, dtype=np.float32)
    deg_scores[deg.index] = np.log1p(deg.values)

    recency = compute_item_recency(data, half_life)
    return deg_scores + recency


def build_item_prompts(data: InteractionData, item_indices: Iterable[int], cfg: ProfileConfig) -> List[str]:
    prompts = []
    columns = [c for c in ["title", "description", "category"] if c in data.df.columns]
    for i in item_indices:
        rows = data.df[data.df["iidx"] == i].head(3)
        parts = [f"Item {i}"]
        for _, row in rows.iterrows():
            for col in columns:
                parts.append(f"{col}={row[col]}")
        prompt = (
            "This item needs a recommendation profile. "
            + " ".join(parts)
            + f" Write a concise summary (<={cfg.prompt_item_max_words} words) covering scenario, audience, attributes."
        )
        prompts.append(prompt)
    return prompts


def prepare_profiles(
    data: InteractionData,
    user_embeddings: torch.Tensor,
    cfg: ProfileConfig,
    artifacts_dir: Path,
    recency_half_life: float = 30.0,
) -> ProfileArtifacts:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    encoder = HashingEncoder(cfg.profile_dim)
    llm = LLMClient(cfg.llm_placeholder)

    cluster_ids, _ = cluster_users(user_embeddings, cfg)
    cluster_prompts = build_cluster_prompts(data, cluster_ids, cfg)
    cluster_texts = [llm.generate_cluster_profile(p) for p in cluster_prompts]
    cluster_embeds = encoder.encode(cluster_texts)

    user_profile = cluster_embeds[cluster_ids]

    scores = item_value_scores(data, recency_half_life)
    top_m = max(1, int(cfg.item_top_ratio * data.num_items))
    top_idx = np.argsort(-scores)[:top_m]
    item_prompts = build_item_prompts(data, top_idx, cfg)
    item_texts = [llm.generate_item_profile(p) for p in item_prompts]
    item_profile = torch.zeros((data.num_items, cfg.profile_dim), dtype=torch.float32)
    if len(item_texts) > 0:
        item_profile[top_idx] = encoder.encode(item_texts)

    torch.save(user_profile, artifacts_dir / "user_profiles.pt")
    torch.save(item_profile, artifacts_dir / "item_profiles.pt")
    np.save(artifacts_dir / "user_cluster_ids.npy", cluster_ids)
    np.save(artifacts_dir / "top_items.npy", top_idx)

    return ProfileArtifacts(
        user_profiles=user_profile,
        item_profiles=item_profile,
        user_cluster_ids=cluster_ids,
        top_item_indices=top_idx,
    )
