from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, Any

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.item_node_value_evaluation import NodeValueEvaluator, UserNodeValueEvaluator

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


SYSTEM_PROMPT = """You are an expert in recommendation semantics.
Given one item profile text, output ONLY valid JSON in this schema:
{
  "semantic_summary": "short summary <= 60 words",
  "category_tags": ["tag1", "tag2", "tag3"],
  "target_preferences": ["pref1", "pref2", "pref3"]
}
Keep content concise and recommendation-oriented.
"""

USER_CLUSTER_SYSTEM_PROMPT = """You are an expert in recommendation semantics.
Given one user-cluster profile text, output ONLY valid JSON in this schema:
{
  "semantic_summary": "short summary <= 60 words",
  "category_tags": ["tag1", "tag2", "tag3"],
  "target_preferences": ["pref1", "pref2", "pref3"]
}
Summarize stable interests and preference styles for this cluster.
Keep content concise and recommendation-oriented.
"""


def _minmax_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + eps)


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
        evaluator = NodeValueEvaluator(
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


class UserBudgetSelector:
    """
    Select high-need users for semantic transfer.

    The score is a normalized semantic-need proxy:
      value_score - activity_penalty * activity + cold_start_boost * (1 - activity)
    """

    def __init__(
        self,
        parser,
        activity_penalty: float = 0.0,
        value_alpha: float = 0.33,
        value_beta: float = 0.33,
        value_gamma: float = 0.34,
        cold_start_boost: float = 0.2,
    ):
        self.parser = parser
        self.activity_penalty = float(activity_penalty)
        self.value_alpha = float(value_alpha)
        self.value_beta = float(value_beta)
        self.value_gamma = float(value_gamma)
        self.cold_start_boost = float(cold_start_boost)

    def _user_activity(self) -> torch.Tensor:
        act = torch.zeros(self.parser.num_users, dtype=torch.float32)
        for u, _ in self.parser.train:
            act[u] += 1.0
        return _minmax_norm(act)

    def select(
        self,
        user_emb_layers: list[torch.Tensor],
        user_id_emb: torch.Tensor,
        budget_ratio: float,
        min_selected_users: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cpu_user_layers = [x.detach().cpu() for x in user_emb_layers]
        cpu_user_id_emb = user_id_emb.detach().cpu()
        evaluator = UserNodeValueEvaluator(
            parser=self.parser,
            user_emb_layers=cpu_user_layers,
            user_id_emb=cpu_user_id_emb,
            alpha=self.value_alpha,
            beta=self.value_beta,
            gamma=self.value_gamma,
            cold_start_boost=0.0,
        )
        value_score = evaluator.calculate().detach().cpu()
        activity = self._user_activity()
        low_activity = 1.0 - activity
        need_score = value_score - self.activity_penalty * activity + self.cold_start_boost * low_activity
        need_score = _minmax_norm(need_score)

        num_users = self.parser.num_users
        k = int(num_users * float(budget_ratio))
        k = max(int(min_selected_users), k)
        k = max(1, min(k, num_users))
        selected = torch.topk(need_score, k=k).indices
        selected, _ = torch.sort(selected)
        return selected, need_score


class TransformersSemanticAcquirer:
    def __init__(
        self,
        model_path: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
        cache_dir: str,
        llm_fail_fast: bool = True,
        load_in_8bit: bool = True,
        device_map: str = "auto",
        trust_remote_code: bool = False,
        system_prompt: str = SYSTEM_PROMPT,
        cache_prefix: str = "item",
        entity_name: str = "item",
        entity_field_name: str = "item_profile",
    ):
        self.model_path = model_path
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self.cache_dir = cache_dir
        self.llm_fail_fast = bool(llm_fail_fast)
        self.load_in_8bit = bool(load_in_8bit)
        self.device_map = str(device_map)
        self.trust_remote_code = bool(trust_remote_code)
        self.system_prompt = str(system_prompt)
        self.cache_prefix = str(cache_prefix)
        self.entity_name = str(entity_name)
        self.entity_field_name = str(entity_field_name)
        self._llm_available = True
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = None
        self.model = None

    def _init_model(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or BitsAndBytesConfig is None:
            raise ImportError(
                "transformers/bitsandbytes dependencies are missing. "
                "Please install: transformers, accelerate, bitsandbytes."
            )

        if self.load_in_8bit and not torch.cuda.is_available():
            raise RuntimeError("8-bit quantization requires CUDA, but CUDA is not available.")

        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if self.load_in_8bit else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self.model.eval()

    def _cache_path(self, item_id: int) -> str:
        return os.path.join(self.cache_dir, f"{self.cache_prefix}_{item_id}.json")

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
            f"{self.entity_name}_id: {item_id}\n"
            f"{self.entity_field_name}:\n"
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

        # Best-effort fallback for truncated JSON (e.g., max token cutoff).
        # It recovers fields even when closing braces are missing.
        summary_match = re.search(
            r'"semantic_summary"\s*:\s*"((?:\\.|[^"\\])*)"',
            raw_text,
            flags=re.DOTALL,
        )
        category_match = re.search(
            r'"category_tags"\s*:\s*\[(.*?)\]',
            raw_text,
            flags=re.DOTALL,
        )
        pref_match = re.search(
            r'"target_preferences"\s*:\s*\[(.*?)\]',
            raw_text,
            flags=re.DOTALL,
        )
        if summary_match or category_match or pref_match:
            summary = ""
            if summary_match:
                summary = bytes(summary_match.group(1), "utf-8").decode("unicode_escape")

            def _extract_list_items(block_match) -> list[str]:
                if not block_match:
                    return []
                block = block_match.group(1)
                vals = re.findall(r'"((?:\\.|[^"\\])*)"', block, flags=re.DOTALL)
                out: list[str] = []
                for v in vals:
                    try:
                        out.append(bytes(v, "utf-8").decode("unicode_escape"))
                    except Exception:
                        out.append(v)
                return out

            return {
                "semantic_summary": summary,
                "category_tags": _extract_list_items(category_match),
                "target_preferences": _extract_list_items(pref_match),
            }

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

    def _generate(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            self._init_model()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            rendered_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        model_inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        model_device = next(self.model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        do_sample = self.temperature > 0
        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_time": float(max(1, self.timeout)),
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        generated_ids = outputs[0, input_ids.shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def acquire_one(self, item_id: int, profile: Any) -> dict[str, Any]:
        cached = self._load_cache(item_id)
        if cached is not None:
            return cached

        profile_text = self._profile_to_text(profile)
        if not self._llm_available:
            raise RuntimeError(
                f"LLM became unavailable before item_id={item_id}. "
                "Fallback is disabled; please fix LLM generation and rerun."
            )

        prompt = self._build_prompt(item_id, profile_text)

        try:
            raw_text = self._generate(prompt)
            data = self._normalize_semantic_dict(self._extract_json_obj(raw_text))
        except Exception as exc:
            print(f"[TransformersSemanticAcquirer] generation failed for item_id={item_id}: {exc}")
            if self.llm_fail_fast:
                self._llm_available = False
            raise RuntimeError(
                f"Failed to acquire semantic JSON for item_id={item_id}. "
                "Fallback is disabled."
            ) from exc

        self._save_cache(item_id, data)
        return data

    def acquire_batch(self, item_ids: Iterable[int], item_profiles: dict[int, Any]) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for iid in item_ids:
            iid = int(iid)
            profile = item_profiles.get(iid, {})
            result[iid] = self.acquire_one(iid, profile)
        return result


class UserClusterSemanticAcquirer(TransformersSemanticAcquirer):
    def __init__(
        self,
        model_path: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
        cache_dir: str,
        llm_fail_fast: bool = True,
        load_in_8bit: bool = True,
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ):
        super().__init__(
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            cache_dir=cache_dir,
            llm_fail_fast=llm_fail_fast,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            system_prompt=USER_CLUSTER_SYSTEM_PROMPT,
            cache_prefix="cluster",
            entity_name="cluster",
            entity_field_name="cluster_profile",
        )

class SemanticTextEncoder:
    def __init__(self, model_path: str, device: str, batch_size: int = 16):
        self.model = None
        try:
            self.model = SentenceTransformer(
                model_path,
                tokenizer_kwargs={"padding_side": "left"},
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the configured semantic embedding model. "
                "HashingVectorizer fallback is disabled because experiments must use local Qwen embeddings. "
                f"embedding_model_path={model_path!r}, reason={exc}"
            ) from exc
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

        raise RuntimeError("SemanticTextEncoder model is not initialized.")
