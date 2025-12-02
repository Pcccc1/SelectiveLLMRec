from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "yelp"
    data_dir: str = "data"
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    recency_half_life: float = 30.0  # days


@dataclass
class LightGCNConfig:
    embedding_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.0


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


@dataclass
class ProfileConfig:
    num_clusters: int = 200
    cluster_representatives: int = 50
    item_top_ratio: float = 0.05
    profile_dim: int = 128
    prompt_user_examples: int = 5
    prompt_item_max_words: int = 80
    llm_placeholder: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = DataConfig()
    lightgcn: LightGCNConfig = LightGCNConfig()
    train: TrainingConfig = TrainingConfig()
    profile: ProfileConfig = ProfileConfig()
    seed: int = 42
    artifacts_dir: str = "artifacts"

    from typing import Union

    @staticmethod
    def from_yaml(path: Union[str, Path]) -> "ExperimentConfig":
        cfg_dict = yaml.safe_load(Path(path).read_text())
        return ExperimentConfig(
            data=DataConfig(**cfg_dict.get("data", {})),
            lightgcn=LightGCNConfig(**cfg_dict.get("lightgcn", {})),
            train=TrainingConfig(**cfg_dict.get("train", {})),
            profile=ProfileConfig(**cfg_dict.get("profile", {})),
            seed=cfg_dict.get("seed", 42),
            artifacts_dir=cfg_dict.get("artifacts_dir", "artifacts"),
        )
