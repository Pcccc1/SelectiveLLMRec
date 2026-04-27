# SelectiveLLMRec

Selective dual-side semantic enhancement for LightGCN under a fixed LLM budget.

## Project Structure

- `pretrain.py`: entrypoint for collaborative LightGCN pretraining.
- `trainers/pretrain_trainer.py`: object-oriented pretraining pipeline (`PretrainTrainer`).
- `train_budgeted.py`: dual-side budgeted semantic training entrypoint.
- `trainers/budgeted_trainer.py`: object-oriented dual-side semantic pipeline (`BudgetedSemanticTrainer`).
- `model/lightgcn.py`: LightGCN backbone and dual semantic fusion models.
- `model/fusion.py`: conservative item/user semantic fusion heads.
- `utils/semantic_acquisition.py`: budget selectors + LLM semantic acquisition + semantic encoding.
- `utils/item_node_value_evaluation.py`: node utility and user semantic need scoring.
- `configs/`: dataset and training configs.

## Data

Use RLMRec splits under `data_new/<dataset>/`:

- `trn_mat.pkl`
- `val_mat.pkl`
- `tst_mat.pkl`
- `itm_prf.pkl`

## Quick Start

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Build parser + pretrain collaborative model:

```bash
.venv/bin/python pretrain.py --config configs/yelp.yaml --stage all
```

3. Run dual-side budgeted training:

```bash
.venv/bin/python train_budgeted.py --config configs/yelp_budgeted_sota.yaml
```

To quickly toggle the pretrained LightGCN backbone during budgeted training:

```yaml
semantic:
  frozen: true   # true: keep user/item embeddings frozen; false: finetune them
  freeze_backbone_epochs: 0  # warmup freeze epochs when frozen is false
```

The CLI override is also available:

```bash
.venv/bin/python train_budgeted.py --config configs/yelp_budgeted_sota.yaml --frozen false --freeze_backbone_epochs 1
```

## Notes

- The current default semantic pipeline uses local Qwen models:
  - LLM: local `Qwen3.5-9B`
  - Encoder: local `Qwen3-Embedding-0.6B`
- Fallback to `data_new/*_emb_np.pkl` is disabled in budgeted training.
