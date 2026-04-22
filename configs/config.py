from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "yelp"
    data_dir: str = "data_new/yelp"
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    recency_half_life: float = 30.0  # days
    profile_path: Optional[str] = None


@dataclass
class LightGCNConfig:
    embedding_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.0


@dataclass
class PretrainingConfig:
    epochs: int = 50
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    num_workers: int = 4
    neg_samples: int = 1
    lambda_user_cl: float = 0.1
    lambda_item_cl: float = 0.1
    temperature: float = 0.2
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
    neg_samples: int = 1
    lambda_user_cl: float = 0.1
    lambda_item_cl: float = 0.1
    temperature: float = 0.2
    reg: float = 1e-4
    save_path: Optional[str] = None
    item_top_ratio: float = 0.05
    eval_interval: int = 1
    selection_metric: str = "ndcg20"
    log_dir: str = "runs/budgeted"
    run_name: Optional[str] = None


@dataclass
class ProfileConfig:
    num_clusters: int = 200
    top_k: int = 20
    cluster_representatives: int = 50
    item_top_ratio: float = 0.05
    profile_dim: int = 128
    prompt_user_examples: int = 5
    prompt_item_max_words: int = 80
    llm_placeholder: bool = True


@dataclass
class SemanticConfig:
    enable: bool = True
    source: str = "llm_summary"  # one of: llm_summary, profile_embedding
    budget_ratio: float = 0.01
    min_selected_items: int = 10
    popularity_penalty: float = 0.1
    long_tail_boost: float = 0.0
    value_alpha: float = 0.33
    value_beta: float = 0.33
    value_gamma: float = 0.34
    cache_dir: str = "cache/semantic_items"
    llama_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    llama_model: str = "local"
    llama_max_tokens: int = 128
    llama_temperature: float = 0.2
    request_timeout: int = 60
    llm_fail_fast: bool = True
    disable_proxy_for_local: bool = True
    profile_embedding_path: Optional[str] = None
    embedding_model_path: str = (
        "/home/stu256475/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/"
        "snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    )
    embedding_batch_size: int = 16
    align_weight: float = 0.05
    consistency_weight: float = 0.05
    gate_l2_weight: float = 0.001
    fusion_hidden_dim: int = 256
    gate_temperature: float = 1.0
    max_residual_scale: float = 0.35
    gate_bias: float = -2.5
    freeze_backbone_epochs: int = 1
    save_artifacts: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    lightgcn: LightGCNConfig = field(default_factory=LightGCNConfig)
    pretrain: PretrainingConfig = field(default_factory=PretrainingConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    seed: int = 42

    from typing import Union

    @staticmethod
    def from_yaml(path: Union[str, Path]) -> "ExperimentConfig":
        cfg_dict = yaml.safe_load(Path(path).read_text())
        return ExperimentConfig(
            data=DataConfig(**cfg_dict.get("data", {})),
            lightgcn=LightGCNConfig(**cfg_dict.get("lightgcn", {})),
            pretrain=PretrainingConfig(**cfg_dict.get("pretrain", {})),
            train=TrainingConfig(**cfg_dict.get("train", {})),
            profile=ProfileConfig(**cfg_dict.get("profile", {})),
            semantic=SemanticConfig(**cfg_dict.get("semantic", {})),
            seed=cfg_dict.get("seed", 42),
        )
