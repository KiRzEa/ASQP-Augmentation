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

## Rounds, jobs, sessions (same workflow as ACSA / fox-repro)
- **Round** = one experimental question, its targets in `experiments/rounds.json`.
- **Job** = one run × one seed (`outputs/<run>/seed_<s>/`), the unit that is planned, timed and resumed.
- **Session** = one Kaggle run (≤ 10.5 h on 2 × T4), packed automatically from the round's unfinished jobs.

```bash
# local
python3 scripts/kaggle_queue.py list                         # done / to run / estimated minutes per round
python3 scripts/kaggle_queue.py plan --round r4              # -> experiments/run_plan.json (calibrated from outputs/_job_times/)
git add experiments && git commit -m "Plan r4" && git push

# Kaggle (GPU T4 x2, Internet on, Save & Run All), session N of the plan
git clone https://github.com/KiRzEa/ASQP-Augmentation.git && cd ASQP-Augmentation
bash scripts/run_kaggle.sh 1                                 # -> results_r4_s1.zip, re-packed after every job

# local, after downloading the zip(s) into runs_inbox/
python3 scripts/kaggle_queue.py ingest                       # unzip into outputs/, archive the zip (idempotent)
python3 src/report_rounds.py                                 # -> experiments/README.md (local only)
```
On Kaggle each GPU runs one job at a time (longest first); a job that no longer fits the budget is skipped and
stays in the plan for the next session (re-plan after ingesting). Per job: log in `outputs/_logs/`, measured
minutes in `outputs/_job_times/<round>_s<N>.tsv` (used by `plan` for the next estimates).
Legacy: `scripts/kaggle_session.sh` (r1, r2) — kept for the `prep` and `quadra_data` jobs.

OpenAI key: each round in `rounds.json` states `openai_key` (also shown by `list`; `run` stops early if a round
needs a key that is not set). Only rounds that generate data with GPT need it; training rounds do not.
On Kaggle: Add-ons → Secrets → `OPENAI_API_KEY`, then in a cell before `run_kaggle.sh`:
```python
import os; from kaggle_secrets import UserSecretsClient
os.environ["OPENAI_API_KEY"] = UserSecretsClient().get_secret("OPENAI_API_KEY")
```
Never type the key into a notebook cell.

## Augmentation baselines (data committed in data/aug/, each with .report.json)
Where each step runs: rule/CPU and GPT steps locally (OpenAI key in `.env`); model steps on Kaggle.
```bash
python3 src/aug/ada.py --strategy {pattern|category|joint} --out data/aug/ada_<s>.jsonl   # local, CPU
python3 src/aug/baselines.py {dup|eda|spansub} --out data/aug/<m>.jsonl                     # local, CPU
python3 src/aug/baselines.py concat --n 3632 --out data/aug/concat.jsonl                     # local, CPU
python3 src/aug/llm_aug.py {paraphrase|quad2text} --out data/aug/llm_{para|q2t}.jsonl        # local, GPT
python -u src/aug/genda.py --out data/aug/genda.jsonl                                       # Kaggle (job genda_gen, r5)
```

## Multi-view targets and consensus relabelling (r7 pilot)
```bash
# Kaggle: every review trained in k element orders (asqp_common.VIEWS), decoded per view, majority vote
python -u src/train_paraphrase.py --output_dir outputs/para_views3 --views 3
#   -> test_predictions.jsonl = voted; test_view0_predictions.jsonl = view 0 alone (summarize_runs.py --split test_view0)
# Kaggle (GPU inference): teachers trained on train only label augmented reviews, majority of teachers kept
python -u src/aug/relabel.py --ckpts <ckpt1>,<ckpt2>,<ckpt3> --in data/aug/llm_para.jsonl --out data/aug/llm_para_cons.jsonl
```
--views 1 (default) is the single AOCS target, byte-identical to earlier runs.

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
