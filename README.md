# SelectiveLLMRec
Cost-aware LLM-enhanced LightGCN recommender for RLMRec datasets (`yelp` / `amazon` / `steam`).

## Layout
- `requirements.txt` dependencies (PyTorch, sklearn, pandas, tqdm).
- `pretrain.py` collaborative LightGCN pretraining.
- `train.py` previous fusion training path.
- `train_budgeted.py` budgeted semantic acquisition path (item-only, local llama + conservative fusion).
- `utils/semantic_acquisition.py` routing, local llama semantic acquisition, semantic text encoding.
- `configs/yelp_budgeted_debug.yaml` minimal runnable debug config for budgeted training.
- `BUDGETED_SEMANTIC_ACQUISITION.md` method definition and implementation notes.

## Data format
Use RLMRec split files under `data_new/<dataset>/`, where `<dataset>` is `yelp`, `amazon`, or `steam`:

- `trn_mat.pkl`: training sparse matrix
- `val_mat.pkl`: validation sparse matrix
- `tst_mat.pkl`: test sparse matrix
- `usr_prf.pkl`: user text profiles
- `itm_prf.pkl`: item text profiles
- `usr_emb_np.pkl`: user text embeddings
- `itm_emb_np.pkl`: item text embeddings

Legacy `datasets/*` text/json format has been deprecated in the runtime loader.

## Workflow
1) Install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2) Pretrain (uses validation set for early model selection):  
   `python pretrain.py --config configs/yelp.yaml`
3) Budgeted semantic acquisition training (new path):  
   `python train_budgeted.py --config configs/yelp_budgeted_debug.yaml`

## Notes
- Budgeted path only enhances selected items (`top-B` by routing score) under a fixed budget ratio.
- Local llama endpoint defaults to `http://127.0.0.1:8080/v1/chat/completions`.
- If shell/global proxy is enabled, set `NO_PROXY=127.0.0.1,localhost` when launching training.
- Semantic encoder first tries SentenceTransformer; if model loading fails, it falls back to a local hashing encoder so the pipeline remains runnable.
