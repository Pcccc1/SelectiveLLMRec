from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, Any

import requests
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer

from utils.item_node_value_evaluation import Node_value_Evaluator


SYSTEM_PROMPT = """You are an expert in recommendation semantics.
Given one item profile text, output ONLY valid JSON in this schema:
{
  "semantic_summary": "short summary <= 60 words",
  "category_tags": ["tag1", "tag2", "tag3"],
  "target_preferences": ["pref1", "pref2", "pref3"]
}
Keep content concise and recommendation-oriented.
"""


class ItemBudgetSelector:
    def __init__(
        self,
        parser,
        popularity_penalty: float = 0.1,
        value_alpha: float = 0.33,
        value_beta: float = 0.33,
        value_gamma: float = 0.34,
        long_tail_boost: float = 0.0,
    ):
        self.parser = parser
        self.popularity_penalty = float(popularity_penalty)
        self.value_alpha = float(value_alpha)
        self.value_beta = float(value_beta)
        self.value_gamma = float(value_gamma)
        self.long_tail_boost = float(long_tail_boost)

    def _item_popularity(self) -> torch.Tensor:
        pop = torch.zeros(self.parser.num_items, dtype=torch.float32)
        for _, i in self.parser.train:
            pop[i] += 1.0
        if pop.max() > pop.min():
            pop = (pop - pop.min()) / (pop.max() - pop.min())
        else:
            pop.zero_()
        return pop

    def select(
        self,
        item_emb_layers: list[torch.Tensor],
        item_id_emb: torch.Tensor,
        budget_ratio: float,
        min_selected_items: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cpu_item_layers = [x.detach().cpu() for x in item_emb_layers]
        cpu_item_id_emb = item_id_emb.detach().cpu()
        evaluator = Node_value_Evaluator(
            parser=self.parser,
            item_emb_layers=cpu_item_layers,
            item_id_emb=cpu_item_id_emb,
            alpha=self.value_alpha,
            beta=self.value_beta,
            gamma=self.value_gamma,
        )
        value_score = evaluator.calculate().detach().cpu()
        pop_penalty = self._item_popularity()
        long_tail_signal = 1.0 - pop_penalty
        final_score = (
            value_score
            - self.popularity_penalty * pop_penalty
            + self.long_tail_boost * long_tail_signal
        )

        num_items = self.parser.num_items
        k = int(num_items * float(budget_ratio))
        k = max(int(min_selected_items), k)
        k = max(1, min(k, num_items))

        selected = torch.topk(final_score, k=k).indices
        selected, _ = torch.sort(selected)
        return selected, final_score


class LocalLlamaSemanticAcquirer:
    def __init__(
        self,
        llama_url: str,
        llama_model: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
        cache_dir: str,
        llm_fail_fast: bool = True,
        disable_proxy_for_local: bool = True,
    ):
        self.llama_url = llama_url
        self.llama_model = llama_model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self.cache_dir = cache_dir
        self.llm_fail_fast = bool(llm_fail_fast)
        self._llm_available = True
        os.makedirs(self.cache_dir, exist_ok=True)

        self.session = requests.Session()
        # Local LLM endpoints are often blocked by shell proxy env vars.
        if disable_proxy_for_local:
            self.session.trust_env = False

    def _cache_path(self, item_id: int) -> str:
        return os.path.join(self.cache_dir, f"item_{item_id}.json")

    def _load_cache(self, item_id: int) -> dict[str, Any] | None:
        path = self._cache_path(item_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, item_id: int, data: dict[str, Any]) -> None:
        path = self._cache_path(item_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _profile_to_text(self, profile: Any) -> str:
        if profile is None:
            return ""
        if isinstance(profile, str):
            return profile
        if not isinstance(profile, dict):
            return str(profile)

        if "profile" in profile and isinstance(profile["profile"], str):
            return profile["profile"]

        fields = []
        for key in ["title", "name", "categories", "genres", "description"]:
            val = profile.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                val = ", ".join(str(x) for x in val)
            fields.append(f"{key}: {val}")

        if not fields:
            fields.append(json.dumps(profile, ensure_ascii=False))
        return "\n".join(fields)

    def _build_prompt(self, item_id: int, profile_text: str) -> str:
        return (
            f"item_id: {item_id}\n"
            "item_profile:\n"
            f"{profile_text}\n"
            "\nOutput JSON only."
        )

    def _extract_json_obj(self, raw_text: str) -> dict[str, Any]:
        raw_text = raw_text.strip()
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        for matched in re.findall(r"\{.*?\}", raw_text, flags=re.DOTALL):
            try:
                obj = json.loads(matched)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

        return {
            "semantic_summary": raw_text[:400],
            "category_tags": [],
            "target_preferences": [],
        }

    def _normalize_semantic_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        summary = str(data.get("semantic_summary", "")).strip()
        category_tags = data.get("category_tags", [])
        target_preferences = data.get("target_preferences", [])

        if not isinstance(category_tags, list):
            category_tags = [str(category_tags)]
        if not isinstance(target_preferences, list):
            target_preferences = [str(target_preferences)]

        category_tags = [str(x).strip() for x in category_tags if str(x).strip()]
        target_preferences = [str(x).strip() for x in target_preferences if str(x).strip()]

        return {
            "semantic_summary": summary,
            "category_tags": category_tags[:8],
            "target_preferences": target_preferences[:8],
        }

    def acquire_one(self, item_id: int, profile: Any) -> dict[str, Any]:
        cached = self._load_cache(item_id)
        if cached is not None:
            return cached

        profile_text = self._profile_to_text(profile)
        if not self._llm_available:
            data = self._normalize_semantic_dict(
                {
                    "semantic_summary": profile_text[:400],
                    "category_tags": [],
                    "target_preferences": [],
                }
            )
            self._save_cache(item_id, data)
            return data

        prompt = self._build_prompt(item_id, profile_text)
        payload = {
            "model": self.llama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = self.session.post(self.llama_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"]
            data = self._normalize_semantic_dict(self._extract_json_obj(raw_text))
        except Exception:
            if self.llm_fail_fast:
                self._llm_available = False
            # Fallback keeps training pipeline runnable even when local LLM is unstable.
            data = self._normalize_semantic_dict(
                {
                    "semantic_summary": profile_text[:400],
                    "category_tags": [],
                    "target_preferences": [],
                }
            )

        self._save_cache(item_id, data)
        return data

    def acquire_batch(self, item_ids: Iterable[int], item_profiles: dict[int, Any]) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for iid in item_ids:
            iid = int(iid)
            profile = item_profiles.get(iid, {})
            result[iid] = self.acquire_one(iid, profile)
        return result


class SemanticTextEncoder:
    def __init__(self, model_path: str, device: str, batch_size: int = 16):
        self.model = None
        self.hashing = None
        try:
            self.model = SentenceTransformer(
                model_path,
                tokenizer_kwargs={"padding_side": "left"},
                device=device,
            )
        except Exception as exc:
            print(
                "[SemanticTextEncoder] Failed to load SentenceTransformer model, "
                f"fallback to HashingVectorizer. reason={exc}"
            )
            self.hashing = HashingVectorizer(
                n_features=1024,
                alternate_sign=False,
                norm="l2",
            )
        self.device = device
        self.batch_size = int(batch_size)

    @staticmethod
    def _to_text(data: dict[str, Any]) -> str:
        summary = str(data.get("semantic_summary", "")).strip()
        category_tags = data.get("category_tags", [])
        target_preferences = data.get("target_preferences", [])
        if not isinstance(category_tags, list):
            category_tags = [str(category_tags)]
        if not isinstance(target_preferences, list):
            target_preferences = [str(target_preferences)]

        lines = [
            f"summary: {summary}",
            f"category_tags: {', '.join(str(x) for x in category_tags)}",
            f"target_preferences: {', '.join(str(x) for x in target_preferences)}",
        ]
        return "\n".join(lines)

    def encode(self, semantics: dict[int, dict[str, Any]]) -> dict[int, torch.Tensor]:
        if not semantics:
            return {}
        item_ids = sorted(semantics.keys())
        texts = [self._to_text(semantics[iid]) for iid in item_ids]
        if self.model is not None:
            emb = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_tensor=True,
                batch_size=self.batch_size,
                device=self.device,
            )
            return {iid: emb[i].detach().cpu() for i, iid in enumerate(item_ids)}

        sparse = self.hashing.transform(texts)
        dense = sparse.toarray().astype(np.float32)
        emb = torch.from_numpy(dense)
        return {iid: emb[i].detach().cpu() for i, iid in enumerate(item_ids)}
