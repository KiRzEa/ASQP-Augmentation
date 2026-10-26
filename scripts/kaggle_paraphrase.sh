#!/usr/bin/env bash
# Run Paraphrase (ViT5-base) for 5 seeds on Kaggle 2×T4: one seed per GPU, two in parallel.
#
#   bash scripts/kaggle_paraphrase.sh <run_name> [extra args for src/train_paraphrase.py ...]
#   bash scripts/kaggle_paraphrase.sh para_none
#   bash scripts/kaggle_paraphrase.sh para_ada --aug_path data/aug/ada.jsonl
#
# Every augmentation method must be run through this script with no change except --aug_path,
# so all methods share model, hyperparameters, seeds and evaluation.
set -uo pipefail
RUN=${1:?run name}; shift || true
cd "$(dirname "$0")/.."
OUT=outputs/$RUN
mkdir -p "$OUT"
GPU0_SEEDS=${GPU0_SEEDS:-42,2024,99}
GPU1_SEEDS=${GPU1_SEEDS:-123,7}

echo "[$RUN] seeds $GPU0_SEEDS on GPU 0, $GPU1_SEEDS on GPU 1; logs also in $OUT/gpu0.log, $OUT/gpu1.log"
( CUDA_VISIBLE_DEVICES=0 python -u src/train_paraphrase.py --output_dir "$OUT" --seeds "$GPU0_SEEDS" "$@" 2>&1 \
  | tee "$OUT/gpu0.log" | sed -u "s/^/[$RUN gpu0] /" ) &
P0=$!
( CUDA_VISIBLE_DEVICES=1 python -u src/train_paraphrase.py --output_dir "$OUT" --seeds "$GPU1_SEEDS" "$@" 2>&1 \
  | tee "$OUT/gpu1.log" | sed -u "s/^/[$RUN gpu1] /" ) &
P1=$!
wait $P0; RC0=$?
wait $P1; RC1=$?
echo "exit codes: gpu0=$RC0 gpu1=$RC1"

python src/summarize_runs.py "$OUT" --json "$OUT/summary_test.json"
python src/summarize_runs.py "$OUT" --split dev --json "$OUT/summary_dev.json"
# results to download: predictions, metrics, configs, logs (no checkpoints); paths relative to the repo root
if [[ -z "${NOZIP:-}" ]]; then
  zip -qr "results_${RUN}.zip" "$OUT" -x "*/checkpoint/*"
  echo "packed results_${RUN}.zip (unzip in the repo root)"
fi
