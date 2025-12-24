import os
import json
from typing import Dict

import torch
from sentence_transformers import SentenceTransformer


class ItemEmbeddingEncoder:
    """
    Pipeline:
        item_profiles.json
            → build item text
            → Qwen3-Embedding encode
            → {item_id: tensor}
            → save to item_embeddings.pt
    """

    def __init__(
        self,
        profile_json_path: str,
        save_path: str = ".static/item_embeddings.pt",
        device: str = None,
    ):
        self.profile_json_path = profile_json_path
        self.save_path = save_path

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model_path = "/home/stu256475/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"

        print(f"[ItemEmbeddingEncoder] Loading Qwen embedding on {device}")
        self.model = SentenceTransformer(
            model_path,
            tokenizer_kwargs={"padding_side": "left"},
            device=device,
        )

    # --------------------------------------------------
    # Load item profiles
    # --------------------------------------------------
    def load_profiles(self) -> Dict[int, Dict]:
        with open(self.profile_json_path, "r") as f:
            data = json.load(f)

        profiles = {int(k): v for k, v in data.items()}
        print(f"[ItemEmbeddingEncoder] Loaded {len(profiles)} item profiles.")
        return profiles

    # --------------------------------------------------
    # Build text for embedding
    # --------------------------------------------------
    def build_text(self, profile: Dict) -> str:
        parts = []

        if "character" in profile:
            parts.append(f"Item name: {profile['character']}.")

        if "description" in profile:
            parts.append(f"Description: {profile['description']}")

        if "reasoning" in profile:
            parts.append(f"User preference reasoning: {profile['reasoning']}")

        return "\n".join(parts)

    # --------------------------------------------------
    # Encode
    # --------------------------------------------------
    def encode_all(self, profiles: Dict[int, Dict]) -> Dict[int, torch.Tensor]:
        item_ids = sorted(profiles.keys())
        texts = [self.build_text(profiles[iid]) for iid in item_ids]

        print(f"[ItemEmbeddingEncoder] Encoding {len(texts)} items...")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_tensor=True,
            batch_size=32,   # 明确写出来，避免默认坑
            device=self.device,
        )

        item_embeddings = {
            iid: embeddings[i].cpu()
            for i, iid in enumerate(item_ids)
        }

        print(f"[ItemEmbeddingEncoder] Done. Dim = {embeddings.shape[1]}")
        return item_embeddings

    # --------------------------------------------------
    def save(self, item_embeddings: Dict[int, torch.Tensor]):
        torch.save(item_embeddings, self.save_path)
        print(f"[ItemEmbeddingEncoder] Saved → {self.save_path}")

    def run(self):
        profiles = self.load_profiles()
        emb = self.encode_all(profiles)
        self.save(emb)
        return emb
