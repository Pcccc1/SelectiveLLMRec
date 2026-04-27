from __future__ import annotations

from trainers.budgeted_trainer import (
    BudgetedSemanticTrainer,
    build_budgeted_arg_parser,
    train,
)


def main() -> None:
    parser = build_budgeted_arg_parser()
    args = parser.parse_args()
    trainer = BudgetedSemanticTrainer(args.config, args)
    trainer.run()


if __name__ == "__main__":
    main()
