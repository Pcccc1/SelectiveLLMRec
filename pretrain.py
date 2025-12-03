from __future__ import annotations

import argparse
import random
from configs.config import ExperimentConfig
import numpy as np
import torch
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


def train(cfg_path: str):
    cfg = ExperimentConfig.from_yaml(cfg_path)
    set_seed(cfg.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # step 1 : load data
    reader = DataReader(cfg.data.data_dir)
    train, val, test = reader.load_all()

    # step 2 : parse dataset and remap ids
    parser = GraphDatasetParser(train, val, test)
    parser.remap_ids()
    parser.build_user_pos_items()
    parser.build_adj_mat()

    # step 3 : build dataset
    dataset = GraphPretrainDataset(
        parser.user_pos_items, parser.num_users, parser.num_items
    )

    # step 4 : dataloader
    neg_sample = NegativeSampler(parser.num_items, parser.user_pos_items)

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
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
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    
    # step 7 : training loop
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            users = batch["user"].to(device)
            pos_items = batch["pos"].to(device)
            neg_items = batch["neg"].to(device)

            user_g, pos_g, neg_g = model(users, pos_items, neg_items)

            loss = criterion(z_user=user_g, z_pos=pos_g, z_neg=neg_g, reg=float(cfg.train.reg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{cfg.train.epochs}, Loss: {avg_loss:.4f}")
    


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/yelp.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    train(args.config)
