# Running experiments

## Layout
- `data/restaurant_v1/` — cleaned data (`scripts/prepare_data.py`); never edit by hand.
- `src/asqp_common.py` — shared target format, parser and scorer (all models and augmentation methods use it).
- `src/train_paraphrase.py` — Paraphrase on ViT5-base, any number of seeds, optional `--aug_path`.
- `src/summarize_runs.py` — mean ± std over seeds, recomputed from saved predictions.
- `scripts/kaggle_paraphrase.sh` — 5 seeds on Kaggle 2×T4 (seeds 42, 2024, 99 on GPU 0; 123, 7 on GPU 1).
- `outputs/<run_name>/seed_<s>/` — one folder per seed (predictions, metrics, history, run config, log).

## Protocol (identical for every augmentation method)
- Seeds 42, 123, 2024, 7, 99. Model ViT5-base (`VietAI/vit5-base`), AdamW lr 3e-4, batch 16, ≤ 20 epochs,
  linear decay, no warmup, grad clip 1.0, fp32, greedy decoding, max source 256 / target 384 tokens.
- Checkpoint = best dev quad F1 (evaluated every epoch, early stop after 5 epochs without gain).
- Test is decoded once, with that checkpoint. Metrics are always recomputed from `*_predictions.jsonl`.
- Augmented data: a JSONL in the same format as `train.jsonl`, passed with `--aug_path`; nothing else changes.
- Step-matched control: `--max_steps N` with N = the steps used by the augmented run.

## Kaggle (2×T4, internet on)
1. Upload this folder (at least `src/`, `scripts/`, `data/`) as a Kaggle dataset, e.g. `asqp-da`.
2. In the notebook:
   ```bash
   cp -r /kaggle/input/asqp-da /kaggle/working/DataAugmentation
   cd /kaggle/working/DataAugmentation
   bash scripts/kaggle_paraphrase.sh para_none                                   # no augmentation
   bash scripts/kaggle_paraphrase.sh para_ada --aug_path data/aug/ada.jsonl       # an augmentation method
   ```
3. Download `/kaggle/working/DataAugmentation/results_<run_name>.zip`, unzip into `outputs/` here, then
   `python3 src/summarize_runs.py outputs/<run_name>`.

## Local smoke test (CPU, ~1 min)
```bash
HF_HUB_OFFLINE=1 python3 src/train_paraphrase.py --output_dir /tmp/smoke --seeds 42 --limit 8 --epochs 2 --cpu
```

## QUADRA (our method)
Steps 1, 3 and LLM fusion run on CPU (locally, OpenAI key in `.env`); steps 2 and 4b need a GPU (Kaggle).

1. **Kaggle, once** — difficulty + verifier (~3 h on 2×T4):
   ```bash
   bash scripts/kaggle_quadra_prep.sh
   ```
   Download `results_quadra_prep.zip` (contains `data/aug/difficulty.json`) and the verifier checkpoint
   `outputs/verifier/seed_42/checkpoint` (save it as a Kaggle dataset/output for step 3).
2. **Local** — generate candidates (frequency × difficulty budget, recomposition, GPT fusion):
   ```bash
   python3 src/aug/quadra.py --difficulty data/aug/difficulty.json --out data/aug/quadra_rule.jsonl
   python3 src/aug/quadra.py --difficulty data/aug/difficulty.json --out data/aug/quadra_gpt.jsonl \
       --llm_model gpt-4.1-mini-2025-04-14
   ```
   Commit and push the two `.jsonl` files (+ `.report.json`; the `.llm_cache.jsonl` keeps every prompt/answer).
3. **Kaggle** — model-based check, then train with the same protocol as every method:
   ```bash
   python src/aug/verify_model.py --ckpt <verifier checkpoint> --in data/aug/quadra_gpt.jsonl --out data/aug/quadra_gpt_v.jsonl
   bash scripts/kaggle_paraphrase.sh para_quadra_gpt --aug_path data/aug/quadra_gpt_v.jsonl
   ```
   Same for the rule-only variant (`quadra_rule.jsonl` → `para_quadra_rule`).
