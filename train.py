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


from torch.utils.data import DataLoader
from dataloader.data_reader import DataReader
from dataloader.dataset_graph import GraphDatasetParser, GraphPretrainDataset, collate_graph
from dataloader.sample_negative import NegativeSampler

from model.lightgcn import LightGCN, LightGCN_retrain
from dataloader.manager import GeneralItemProfileManager
from utils.cluster_statistic import ClusterProfile
from utils.cluster_encoder import ClusterEmbeddingEncoder
from utils.item_node_value_evaluation import Node_value_Evaluator
from utils.losses import bpr_loss
from utils.metrics import evaluate_all_ranking, get_user_item_dict

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

    node_evaluator = Node_value_Evaluator(
        parser=parser,
    )

    # load item profiles
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
        adj_mat=parser.adj_mat
    ).to("cpu")

    pretrain_model.load_state_dict(torch.load(cfg.pretrain.save_path, map_location=device))
    pretrain_model.eval()

    with torch.no_grad():
        g_u_pretrain = pretrain_model.get_all_embeddings()[2].detach().cpu()

    cluster_id, cluster_centers = UserClusterer(num_clusters=cfg.profile.num_clusters).cluster(g_u_pretrain)


    """
    data prepare for cluster profile summarization
    """
    # cluster_users = defaultdict(list)
    
    # for user_idx, cid in enumerate(cluster_id):
    #     cluster_users[cid].append(user_idx)

    # cp = ClusterProfile(
    #     parser=parser,
    #     cluster_users=cluster_users,
    #     item_profiles=item_profiles,
    # )
    # cluster_profile = cp.get_cluster_profiles(top_k=cfg.profile.top_k)
    
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

    cluster_embeddings = torch.load("static/cluster_embeddings.pt", map_location=device)
    cluster_emb = torch.stack([cluster_embeddings[c] for c in sorted(cluster_embeddings.keys())]).to(device=device, dtype=g_u_pretrain.dtype)
    cluster_centers = torch.tensor(cluster_centers, device=device, dtype=g_u_pretrain.dtype)
    user_cluster = torch.tensor(cluster_id, device=device, dtype=torch.long)
    model = LightGCN_retrain(
        num_users=parser.num_users,
        num_items=parser.num_items,
        embedding_dim=cfg.lightgcn.embedding_dim,
        n_layers=cfg.lightgcn.n_layers,
        adj_mat=parser.adj_mat,
        cluster_emb=cluster_emb,
        cluster_centers=cluster_centers,
        user_cluster=user_cluster,
        user_feature=g_u_pretrain,
        device=cfg.train.device
    ).to(device)
    missing, unexpected = model.load_state_dict(pretrain_model.state_dict(), strict=False)
    print("Model loaded.")
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=float(cfg.train.weight_decay))

    criterion = bpr_loss

    dataset = GraphPretrainDataset(
        train_pairs=parser.train,
        user_pos_items=parser.user_pos_items,
    )

    neg_sample = NegativeSampler(parser.num_items, parser.user_pos_items)

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=lambda batch: collate_graph(batch, neg_sample),
    )

    best_ndcg20 = -1.0
    best_epoch = -1
    save_path = cfg.train.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Starting retrain with fusion...")

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            users = batch["user"].to(device)
            pos_items = batch["pos"].to(device)
            neg_items = batch["neg"].to(device)

            # ------------------------------------------------
            # Forward (already fused user embedding)
            # ------------------------------------------------
            user_g, pos_g, neg_g = model(users, pos_items, neg_items)

            # ------------------------------------------------
            # ID embeddings (for L2 regularization, same as pretrain)
            # ------------------------------------------------
            user_id_emb = model.user_embedding(users)
            pos_id_emb = model.item_embedding(pos_items)
            neg_id_emb = model.item_embedding(neg_items)

            # ------------------------------------------------
            # BPR loss (unchanged)
            # ------------------------------------------------
            loss = criterion(
                z_user=user_g,
                z_pos=pos_g,
                z_neg=neg_g,
                reg=float(cfg.train.reg),
                user_id_emb=user_id_emb,
                pos_id_emb=pos_id_emb,
                neg_id_emb=neg_id_emb,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        wandb.log({"Train/Loss": avg_loss})
        print(f"[Epoch {epoch+1}/{cfg.train.epochs}] Train Loss: {avg_loss:.4f}")

        # ------------------------------------------------
        # Validation (same protocol as pretrain)
        # ------------------------------------------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                recall_res, ndcg_res = evaluate_all_ranking(
                    model,
                    users=torch.LongTensor(
                        list(get_user_item_dict(parser.val).keys())
                    ).to(device),
                    train_user_items=get_user_item_dict(parser.train),
                    eval_user_items=get_user_item_dict(parser.val),
                    K=[10, 20],
                    device=device,
                )

            print(
                f"[Val @ Epoch {epoch+1}] "
                f"Recall@10={recall_res[10]:.4f}, NDCG@10={ndcg_res[10]:.4f}, "
                f"Recall@20={recall_res[20]:.4f}, NDCG@20={ndcg_res[20]:.4f}"
            )

            wandb.log({
                "Val/Recall@10": recall_res[10],
                "Val/NDCG@10": ndcg_res[10],
                "Val/Recall@20": recall_res[20],
                "Val/NDCG@20": ndcg_res[20],
            })

            # ------------------------------------------------
            # Save best model (by NDCG@20)
            # ------------------------------------------------
            if ndcg_res[20] > best_ndcg20:
                best_ndcg20 = ndcg_res[20]
                best_epoch = epoch + 1

                assert save_path is not None
                torch.save(model.state_dict(), save_path)

                print(
                    f"Best model saved at epoch {best_epoch} "
                    f"with NDCG@20 = {best_ndcg20:.4f}"
                )

    print(
        f"Retrain completed. "
        f"Best NDCG@20 = {best_ndcg20:.4f} @ epoch {best_epoch}"
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
