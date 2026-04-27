from __future__ import annotations

import argparse

from trainers.pretrain_trainer import PretrainTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Collaborative LightGCN pretraining entrypoint.")
    parser.add_argument("--config", default="configs/yelp.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "train", "test"],
        default="all",
        help="Pipeline stage to run.",
    )
    args = parser.parse_args()

    trainer = PretrainTrainer(args.config)

    if args.stage in {"all", "prepare"}:
        p = trainer.build_parser()
        print(f"Parser ready: users={p.num_users}, items={p.num_items}")

    train_result = None
    if args.stage in {"all", "train"}:
        train_result = trainer.train()
        print(
            "Pretrain completed. "
            f"best_ndcg20={train_result.best_ndcg20:.4f} at epoch={train_result.best_epoch}, "
            f"checkpoint={train_result.checkpoint_path}"
        )

    if args.stage in {"all", "test"}:
        checkpoint = train_result.checkpoint_path if train_result is not None else None
        metrics = trainer.test(checkpoint)
        print(
            "Test metrics: "
            f"Recall@10={metrics['test_recall@10']:.4f}, NDCG@10={metrics['test_ndcg@10']:.4f}, "
            f"Recall@20={metrics['test_recall@20']:.4f}, NDCG@20={metrics['test_ndcg@20']:.4f}"
        )


if __name__ == "__main__":
    main()
