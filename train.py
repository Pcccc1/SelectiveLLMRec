from __future__ import annotations

import wandb
import os
import argparse
import random
from configs.config import ExperimentConfig
import numpy as np
import torch
import pickle
import json
from collections import defaultdict

from model.lightgcn import LightGCN
from dataloader.manager import GeneralItemProfileManager
from utils.cluster_statistic import ClusterProfile
from utils.cluster_encoder import ClusterEmbeddingEncoder

from prompt.cluster_summer import ClusterProfileSummarizer

from model.model import UserClusterer, ClusterSemanticFusion

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg_path: str):
    cfg = ExperimentConfig.from_yaml(cfg_path)
    #wandb.init(project="SelectiveLLMRec", config=cfg)
    set_seed(cfg.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    with open(f"{cfg.data.dataset}_parser.pkl", "rb") as f:
        parser = pickle.load(f)

    manager = GeneralItemProfileManager(
        dataset_name=cfg.data.dataset,
        parser=parser,
        profile_path=cfg.data.profile_path,
    )
    item_profiles = manager.load(format="yelp")
    parser.item_profiles = item_profiles
    print(f"Loaded {len(item_profiles)} item profiles.")

    pretrain_model = LightGCN(
        num_users=parser.num_users,
        num_items=parser.num_items,
        embedding_dim=cfg.lightgcn.embedding_dim,
        n_layers=cfg.lightgcn.n_layers,
        adj_mat=parser.adj_mat,
    ).to(device)

    pretrain_model.load_state_dict(torch.load(cfg.train.save_path, map_location=device))
    pretrain_model.eval()

    with torch.no_grad():
        user_g = pretrain_model.get_all_embeddings()[2]

    cluster_id, cluster_centers = UserClusterer(num_clusters=cfg.profile.num_clusters).cluster(user_g)

    cluster_users = defaultdict(list)
    
    for c, uid in enumerate(cluster_id):
        cluster_users[uid].append(c)

    cp = ClusterProfile(
        parser=parser,
        cluster_users=cluster_users,
        item_profiles=item_profiles,
    )
    cluster_profile = cp.get_cluster_profiles(top_k=cfg.profile.top_k)
    
    # summarizer = ClusterProfileSummarizer()

    # all_summaries = {}

    # for cid, cluster in cluster_profile.items():
    #     summary = summarizer.summarize_cluster(cluster_id=cid, cluster_data=cluster)
    #     all_summaries[int(cid)] = summary
    
    # with open("cluster_summaries.json", "w") as f:
    #     json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    
    # encoder = ClusterEmbeddingEncoder(
    #     summary_json_path="./cluster_summaries.json",
    #     save_path="./cluster_embeddings.pt"
    # )

    # embeddings = encoder.run()

    cluster_embeddings = torch.load("cluster_embeddings.pt")
    cluster_emb = torch.stack([cluster_embeddings[c] for c in sorted(cluster_embeddings.keys())])
    fusion = ClusterSemanticFusion(
        embed_dim=cfg.lightgcn.embedding_dim,
        cluster_emb=cluster_emb,
        user_feature=user_g,
        cluster_centers=torch.Tensor(cluster_centers),
        user_cluster=torch.LongTensor(cluster_id)
    )

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/yelp.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    train(args.config)

