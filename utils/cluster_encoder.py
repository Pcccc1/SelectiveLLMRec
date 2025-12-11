import os
import json
from typing import Dict, Any

import torch
from sentence_transformers import SentenceTransformer

model_path = "/home/stu256475/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"

class ClusterEmbeddingEncoder:
    """
    Pipeline:
        cluster_summaries.json (one big JSON)
            → load summaries
            → Qwen3-Embedding-0.6B encode
            → {cluster_id: tensor}
            → save to cluster_embeddings.pt
    """

    def __init__(
        self,
        summary_json_path: str = "./cluster_summaries.json",
        save_path: str = "./cluster_embeddings.pt",
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = None,
    ):
        self.summary_json_path = summary_json_path
        self.save_path = save_path
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load embedding model
        print(f"[ClusterEmbeddingEncoder] Loading model: {model_name} on {device}")
        self.model = SentenceTransformer(model_path, tokenizer_kwargs={"padding_side": "left"}, device=device)

    # ---------------------------------------------------------
    # Load all summaries from single JSON file
    # ---------------------------------------------------------
    def load_summaries(self) -> Dict[int, str]:
        if not os.path.exists(self.summary_json_path):
            raise FileNotFoundError(f"Cluster summary file not found: {self.summary_json_path}")

        with open(self.summary_json_path, "r") as f:
            data = json.load(f)

        summaries = {}
        for cid_str, info in data.items():
            # Convert JSON key to int
            cid = int(cid_str)
            summaries[cid] = info["summary"]

        print(f"[ClusterEmbeddingEncoder] Loaded {len(summaries)} cluster summaries.")
        return summaries

    # ---------------------------------------------------------
    # Encode all cluster summaries with Qwen embedding model
    # ---------------------------------------------------------
    def encode_all(self, summaries: Dict[int, str]) -> Dict[int, torch.Tensor]:
        cluster_ids = sorted(summaries.keys())
        texts = [summaries[cid] for cid in cluster_ids]

        print(f"[ClusterEmbeddingEncoder] Encoding {len(texts)} summaries...")

        # Batch encode for speed
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=self.device,
        )  # shape = (num_clusters, dim)

        cluster_embeddings = {
            cid: embeddings[i].cpu()
            for i, cid in enumerate(cluster_ids)
        }

        print(f"[ClusterEmbeddingEncoder] Done. Embedding dim = {embeddings[0].shape[0]}")
        return cluster_embeddings

    # ---------------------------------------------------------
    # Save embeddings to .pt
    # ---------------------------------------------------------
    def save_embeddings(self, cluster_embeddings: Dict[int, torch.Tensor]):
        torch.save(cluster_embeddings, self.save_path)
        print(f"[ClusterEmbeddingEncoder] Saved embeddings → {self.save_path}")

    # ---------------------------------------------------------
    # Full pipeline
    # ---------------------------------------------------------
    def run(self):
        summaries = self.load_summaries()
        embeddings = self.encode_all(summaries)
        self.save_embeddings(embeddings)
        return embeddings
