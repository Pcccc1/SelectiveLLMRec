from __future__ import annotations

import wandb

import argparse
import random
import os
from configs.config import ExperimentConfig
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader

from dataloader.data_reader import DataReader
from dataloader.dataset_graph import GraphDatasetParser, GraphPretrainDataset, collate_graph
from dataloader.sample_negative import NegativeSampler

from model.lightgcn import LightGCN

from utils.losses import bpr_loss
from utils.metrics import evaluate_all_ranking, get_user_item_dict

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(cfg_path: str):
    cfg = ExperimentConfig.from_yaml(cfg_path)
    set_seed(cfg.seed)
    device = torch.device(cfg.pretrain.device if torch.cuda.is_available() else "cpu")


    reader = DataReader(
        cfg.data.data_dir,
        min_user_interactions=cfg.data.min_user_interactions,
        min_item_interactions=cfg.data.min_item_interactions,
    )
    train, val, test = reader.load_all()

    # step 2 : parse dataset and remap ids
    parser = GraphDatasetParser(train, val, test)
    parser.remap_ids()
    parser.build_user_pos_items()
    parser.build_adj_mat()

    print("Data loaded and parsing.")

    with open(cfg.data.dataset + "_parser.pkl", "wb") as f:
        pickle.dump(parser, f)

    print("Parser saved to parser.pkl")

def train(cfg_path: str):
    cfg = ExperimentConfig.from_yaml(cfg_path)
    wandb.init(project="SelectiveLLMRec_pretrain", config=cfg)
    set_seed(cfg.seed)

    device = torch.device(cfg.pretrain.device if torch.cuda.is_available() else "cpu")

    print(f"Loading parser from {cfg.data.dataset}_parser.pkl")
    with open(cfg.data.dataset + "_parser.pkl", "rb") as f:
        parser: GraphDatasetParser = pickle.load(f)

    print("Parser loaded.")

    # step 3 : build dataset
    dataset = GraphPretrainDataset(
        train_pairs=parser.train,
        user_pos_items=parser.user_pos_items,
    )

    # step 4 : dataloader
    neg_sample = NegativeSampler(parser.num_items, parser.user_pos_items)

    loader = DataLoader(
        dataset,
        batch_size=cfg.pretrain.batch_size,
        shuffle=True,
        num_workers=cfg.pretrain.num_workers,
        collate_fn=lambda batch: collate_graph(batch, neg_sample),
    )

    # step 5 : build model
    model = LightGCN(
        num_users=parser.num_users,
        num_items=parser.num_items,
        embedding_dim=cfg.lightgcn.embedding_dim,
        n_layers=cfg.lightgcn.n_layers,
        adj_mat=parser.adj_mat,
    ).to(device)

    # step 6 : training setup
    criterion = bpr_loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.pretrain.lr,
        weight_decay=float(cfg.pretrain.weight_decay),
    )
    

    best_ncdg20 = -1.0
    best_epoch = -1
    save_path = cfg.pretrain.save_path
    os.makedirs("checkpoints", exist_ok=True)

    print("Starting training...")
    # step 7 : training loop
    for epoch in range(cfg.pretrain.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            users = batch["user"].to(device)
            pos_items = batch["pos"].to(device)
            neg_items = batch["neg"].to(device)

            user_g, pos_g, neg_g = model(users, pos_items, neg_items)

            user_id_emb = model.user_embedding(users)
            pos_id_emb = model.item_embedding(pos_items)
            neg_id_emb = model.item_embedding(neg_items)

            loss = criterion(
                z_user=user_g,
                z_pos=pos_g,
                z_neg=neg_g,
                reg=float(cfg.pretrain.reg),
                user_id_emb=user_id_emb,
                pos_id_emb=pos_id_emb,
                neg_id_emb=neg_id_emb,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        wandb.log({"Loss": avg_loss})
        print(f"Epoch {epoch+1}/{cfg.pretrain.epochs}, Loss: {avg_loss:.4f}")
    
        if (epoch+1) % 10 == 0:

        # -----------------------------
        # Validation
        # -----------------------------
            recall_res, ndcg_res = evaluate_all_ranking(
                model,
                users=torch.LongTensor(list(get_user_item_dict(parser.val).keys())),
                train_user_items=get_user_item_dict(parser.train),
                eval_user_items=get_user_item_dict(parser.val),
                K=[10, 20],
                device=device,
            )  
            print(
                f"Validation - Recall@10: {recall_res[10]:.4f}, NDCG@10: {ndcg_res[10]:.4f}, "
                f"Recall@20: {recall_res[20]:.4f}, NDCG@20: {ndcg_res[20]:.4f}"
            ) 
            wandb.log({
                "Val_Recall@10": recall_res[10],
                "Val_NDCG@10": ndcg_res[10],
                "Val_Recall@20": recall_res[20],
                "Val_NDCG@20": ndcg_res[20],
            })
            
            if ndcg_res[20] > best_ncdg20:
                best_ncdg20 = ndcg_res[20]
                best_epoch = epoch + 1
                assert save_path is not None
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved at epoch {best_epoch} with NDCG@20: {best_ncdg20:.4f}")
            
    print(f"Training completed. Best NDCG@20: {best_ncdg20:.4f} at epoch {best_epoch}.")


def test(cfg_path: str):
    cfg = ExperimentConfig.from_yaml(cfg_path)
    set_seed(cfg.seed)
    device = torch.device(cfg.pretrain.device if torch.cuda.is_available() else "cpu")

    reader = DataReader(
        cfg.data.data_dir,
        min_user_interactions=cfg.data.min_user_interactions,
        min_item_interactions=cfg.data.min_item_interactions,
    )
    train, val, test = reader.load_all()


    parser = GraphDatasetParser(train, val, test)
    parser.remap_ids()
    parser.build_user_pos_items()
    parser.build_adj_mat()
    model = LightGCN(
        num_users=parser.num_users,
        num_items=parser.num_items,
        embedding_dim=cfg.lightgcn.embedding_dim,
        n_layers=cfg.lightgcn.n_layers,
        adj_mat=parser.adj_mat,
    ).to(device)
    model.load_state_dict(torch.load(cfg.pretrain.save_path, map_location=device))
    model.eval()

    
    test_users = torch.LongTensor(list(get_user_item_dict(parser.test).keys()))

    # 把 train/val 放成元组传入，并把返回的两个结果先保存为一个元组 `results`
    results = evaluate_all_ranking(
        model,
        users=torch.LongTensor(list(get_user_item_dict(parser.test).keys())),
        train_user_items=get_user_item_dict(parser.train),
        eval_user_items=get_user_item_dict(parser.test),
        K=[10, 20],
        device=device,
    )
    recall_res, ndcg_res = results

    print(
        f"Test Results - Recall@10: {recall_res[10]:.4f}, NDCG@10: {ndcg_res[10]:.4f}, "
        f"Recall@20: {recall_res[20]:.4f}, NDCG@20: {ndcg_res[20]:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/movie.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    load_data(args.config)
    train(args.config)
    test(args.config)
