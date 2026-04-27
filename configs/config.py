from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import yaml


_T = TypeVar("_T")


@dataclass
class DataConfig:
    dataset: str = "yelp"
    data_dir: str = "data_new/yelp"
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    profile_path: Optional[str] = None


@dataclass
class LightGCNConfig:
    embedding_dim: int = 64
    n_layers: int = 3


@dataclass
class PretrainingConfig:
    epochs: int = 50
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    num_workers: int = 4
    reg: float = 1e-4
    save_path: Optional[str] = None


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    num_workers: int = 4
    reg: float = 1e-4
    save_path: Optional[str] = None
    eval_interval: int = 1
    selection_metric: str = "ndcg20"
    log_dir: str = "runs/budgeted"
    run_name: Optional[str] = None
    use_amp: bool = True
    grad_clip_norm: float = 5.0
    scheduler_tmax: int = 20
    scheduler_eta_min: float = 1e-6
    early_stop_patience: int = 8


@dataclass
class SemanticConfig:
    source: str = "llm_summary"  # one of: llm_summary, profile_embedding
    budget_ratio: float = 0.01
    min_selected_items: int = 10
    popularity_penalty: float = 0.1
    long_tail_boost: float = 0.0
    value_alpha: float = 0.33
    value_beta: float = 0.33
    value_gamma: float = 0.34
    cache_dir: str = "cache/semantic_items"
    llm_model_path: str = (
        "/home/stu256475/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
        "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    llm_max_tokens: int = 256
    llm_temperature: float = 0.2
    llm_load_in_8bit: bool = True
    llm_device_map: str = "auto"
    llm_trust_remote_code: bool = False
    llm_cache_subdir: str = "transformers_cache"
    request_timeout: int = 60
    llm_fail_fast: bool = True
    disable_proxy_for_local: bool = True
    profile_embedding_path: Optional[str] = None
    embedding_model_path: str = (
        "/home/stu256475/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/"
        "snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    )
    embedding_batch_size: int = 16
    align_weight: float = 0.05
    consistency_weight: float = 0.05
    gate_l2_weight: float = 0.001
    fusion_hidden_dim: int = 256
    gate_temperature: float = 1.0
    max_residual_scale: float = 0.35
    gate_bias: float = -2.5
    frozen: bool = True
    freeze_backbone_epochs: int = 1
    freeze_backbone: bool = True
    detach_alignment_target: bool = True
    ranking_preserve_weight: float = 0.1
    save_artifacts: bool = True
    # User-side selective semantic transfer
    user_enable: bool = True
    user_source: str = "cluster_llm"  # one of: cluster_llm, profile_embedding
    user_budget_ratio: float = 0.2
    min_selected_users: int = 32
    user_num_clusters: int = 256
    user_cluster_representatives: int = 24
    user_prompt_items_per_user: int = 5
    user_value_alpha: float = 0.3
    user_value_beta: float = 0.4
    user_value_gamma: float = 0.3
    user_activity_penalty: float = 0.05
    user_cold_start_boost: float = 0.2
    user_align_weight: float = 0.03
    user_consistency_weight: float = 0.03
    user_distill_weight: float = 0.05
    user_gate_l2_weight: float = 0.001
    user_gate_temperature: float = 0.8
    user_max_residual_scale: float = 0.35
    user_gate_bias: float = -2.2
    user_profile_embedding_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    lightgcn: LightGCNConfig = field(default_factory=LightGCNConfig)
    pretrain: PretrainingConfig = field(default_factory=PretrainingConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    seed: int = 42

    @staticmethod
    def _from_section(cls: Type[_T], section: dict, section_name: str) -> _T:
        valid = {f.name for f in fields(cls)}
        unknown = sorted(k for k in section.keys() if k not in valid)
        if unknown:
            print(f"[Config] Ignored unknown keys in `{section_name}`: {unknown}")
        filtered = {k: v for k, v in section.items() if k in valid}
        return cls(**filtered)

    @staticmethod
    def from_yaml(path: Union[str, Path]) -> "ExperimentConfig":
        cfg_dict = yaml.safe_load(Path(path).read_text())
        semantic_cfg = dict(cfg_dict.get("semantic", {}))
        if "frozen" in semantic_cfg:
            if (
                "freeze_backbone" in semantic_cfg
                and bool(semantic_cfg["freeze_backbone"]) != bool(semantic_cfg["frozen"])
            ):
                print(
                    "[Config] Both `semantic.frozen` and `semantic.freeze_backbone` "
                    "were set; using `semantic.frozen`."
                )
            semantic_cfg["freeze_backbone"] = bool(semantic_cfg["frozen"])
        elif "freeze_backbone" in semantic_cfg:
            semantic_cfg["frozen"] = bool(semantic_cfg["freeze_backbone"])
        return ExperimentConfig(
            data=ExperimentConfig._from_section(DataConfig, cfg_dict.get("data", {}), "data"),
            lightgcn=ExperimentConfig._from_section(
                LightGCNConfig, cfg_dict.get("lightgcn", {}), "lightgcn"
            ),
            pretrain=ExperimentConfig._from_section(
                PretrainingConfig, cfg_dict.get("pretrain", {}), "pretrain"
            ),
            train=ExperimentConfig._from_section(TrainingConfig, cfg_dict.get("train", {}), "train"),
            semantic=ExperimentConfig._from_section(SemanticConfig, semantic_cfg, "semantic"),
            seed=cfg_dict.get("seed", 42),
        )
