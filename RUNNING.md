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

## Rounds and sessions
The plan lives in `experiments/plan.json`: **round** (one experimental question) → **session** (one Kaggle run,
≤ 12 h) → **jobs**. Runs keep fixed names (`outputs/para_<run>/seed_<s>/`); each session also writes
`outputs/_sessions/<round>_<session>/` (session.log, jobs.txt, commit.txt).

On Kaggle (GPU T4 ×2, Save & Run All):
```bash
git clone https://github.com/KiRzEa/ASQP-Augmentation.git && cd ASQP-Augmentation && pip install -q -r requirements.txt
bash scripts/kaggle_session.sh r3 s1          # → results_r3_s1.zip
```
Locally:
```bash
unzip -o results_r3_s1.zip && mv results_r3_s1.zip outputs/_zips_archive/
python3 src/report_rounds.py                  # → experiments/README.md (local only; results are unpublished)
```
To add work: add a round or session to `experiments/plan.json`, commit, push, run it on Kaggle.
Ad hoc runs still work: `bash scripts/kaggle_session.sh <name> <job> [<job> ...]`.

## QUADRA (our method)
1. Kaggle `prep` (in s1): 5-fold out-of-fold F1 per cell → `data/aug/difficulty.json`; verifier = Paraphrase
   seed 42 on the full train set with checkpoint → `outputs/verifier/seed_42/checkpoint`.
2. Local (CPU, OpenAI key in `.env`): generate 1.5× candidates, then keep exactly |train| reviews that the
   verifier re-extracts correctly; commit and push the verified files.
   ```bash
   python3 src/aug/quadra.py --difficulty data/aug/difficulty.json --ratio 1.5 --out data/aug/quadra_rule.jsonl
   python3 src/aug/quadra.py --difficulty data/aug/difficulty.json --ratio 1.5 --out data/aug/quadra_gpt.jsonl \
       --llm_model gpt-4.1-mini-2025-04-14
   for v in rule gpt; do python3 src/aug/verify_model.py --ckpt outputs/verifier/seed_42/checkpoint \
       --in data/aug/quadra_$v.jsonl --out data/aug/quadra_${v}_v.jsonl --target_n 2135; done
   ```
3. Kaggle s2: train both verified sets with the baseline protocol (5 seeds).
