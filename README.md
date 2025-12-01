# SelectiveLLMRec
Cost-aware LLM-enhanced LightGCN recommender skeleton for Yelp / Amazon-Book / MovieLens1M.

## Layout
- `requirements.txt` dependencies (PyTorch, sklearn, pandas, tqdm).
- `configs/example.yaml` default hyper-parameters.
- `scripts/prepare_profiles.py` offline step: pretrain LightGCN, cluster users, score items, generate profile embeddings with placeholder LLM.
- `src/train.py` main training (LightGCN + ID + profile fusion + multi-view contrastive).
- `src/` modules: data loading, LightGCN, profile builder, losses, fused model.

## Data format
Place raw data under `data/<dataset>/interactions.csv` where `<dataset>` is `yelp`, `amazon-book`, or `movielens1m`.

Required columns:
- `user_id`, `item_id`
- optional: `rating` (keeps rows with rating>0), `timestamp` (unix or datetime), and text fields `title`, `description`, `category` for better prompts.

## Workflow
1) Install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2) Offline prep (pretrain LightGCN, clustering, selective LLM enrichment with placeholder summaries):  
   `PYTHONPATH=. python scripts/prepare_profiles.py --config configs/example.yaml`
3) Final training with fusion + contrastive:  
   `PYTHONPATH=. python -m src.train --config configs/example.yaml`
4) Artifacts land in `artifacts/<dataset>/` (user/item profiles, pretrain embeddings, checkpoints).

## Notes
- LLM calls are stubbed with a placeholder (`LLMClient`) to keep the repo offline-friendly; plug in your own client in `src/profiles.py`.
- User profiles are cluster-level (K-Means), item profiles only for top-K items ranked by degree/recency; others default to zero vectors.
- Multi-view contrastive aligns GNN structural embeddings with semantic profiles on both user and high-value item sides.
