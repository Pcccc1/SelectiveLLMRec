import os
import json
import requests
from typing import Dict, Any, Optional, List

import numpy as np


class ClusterProfileSummarizer:
    """
    Full pipeline:
    cluster_profile (dict)
        → formatted item text
        → llama.cpp server summary (HTTP)
        → embedding (BGE or custom sentence model)
        → saved for later use
    """

    def __init__(
        self,
        llama_url: str = "http://127.0.0.1:8080/v1/chat/completions",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        cache_dir: str = "./cache/cluster_profiles",
        max_tokens: int = 128,
        temperature: float = 0.2,
    ):
        self.llama_url = llama_url
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Load embedding model
        #self.encoder = SentenceTransformer(embedding_model_name)

        # Cache directory
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    # -------------------------------------------------------------
    # Cache utility
    # -------------------------------------------------------------
    def _cache_path(self, cluster_id: int) -> str:
        return os.path.join(self.cache_dir, f"cluster_{cluster_id}.json")

    def _load_cache(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        path = self._cache_path(cluster_id)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def _save_cache(self, cluster_id: int, data: Dict[str, Any]):
        path = self._cache_path(cluster_id)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # -------------------------------------------------------------
    # Build item text
    # -------------------------------------------------------------
    def build_item_text(self, cluster: Dict[str, Any]) -> str:
        lines = []
        for it in cluster["top_item_profiles"]:
            name = it["profile"]["name"]
            cats = ", ".join(it["profile"]["categories"])
            freq = it["freq"]
            lines.append(f"{name} — {cats} (freq {freq})")
        return "\n".join(lines)

    # -------------------------------------------------------------
    # Llama request
    # -------------------------------------------------------------
    def llama_query(self, prompt: str) -> str:
        payload = {
            "model": "local",  # llama.cpp requires but ignores the value
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        resp = requests.post(self.llama_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # -------------------------------------------------------------
    # Build prompt for cluster summary
    # -------------------------------------------------------------
    def build_prompt(self, item_text: str) -> str:
        return f"""
You are an expert analyst for recommendation systems.
Please summarize the preference profile of this user cluster based on their most frequently interacted items.

Requirements:
- Identify the cluster’s main interest themes.
- Extract key categories and cuisine styles.
- Capture stable behavioral or taste tendencies.
- Maximum 50 words.
- One concise paragraph only.

Items:
{item_text}

Final summary:
"""

    # -------------------------------------------------------------
    # Encode summary → embedding
    # -------------------------------------------------------------
    # def encode_summary(self, summary: str) -> List[float]:
    #     emb = self.encoder.encode(summary, normalize_embeddings=True)
    #     return emb.tolist()

    # -------------------------------------------------------------
    # Full pipeline: cluster → summary → embedding
    # -------------------------------------------------------------
    def summarize_cluster(self, cluster_id: int, cluster_data: Dict[str, Any]) -> Dict[str, Any]:

        cluster_id = int(cluster_id)
        # Check cache
        cached = self._load_cache(cluster_id)
        if cached is not None:
            return cached

        # 1. Build item text
        item_text = self.build_item_text(cluster_data)

        # 2. Build prompt
        prompt = self.build_prompt(item_text)

        # 3. Query llama.cpp server
        summary = self.llama_query(prompt)

        # 4. Generate embedding
        # embedding = self.encode_summary(summary)

        result = {
            "cluster_id": cluster_id,
            "summary": summary,
            # "embedding": embedding,
        }

        # Save cache
        self._save_cache(cluster_id, result)

        return result
# Example usage: