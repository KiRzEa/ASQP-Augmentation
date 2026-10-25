#!/usr/bin/env bash
# QUADRA GPU preparation on Kaggle 2×T4 (run once, before generating the augmented data):
#   1) 5-fold out-of-fold predictions on train  -> data/aug/difficulty.json (per-cell difficulty for the budget)
#   2) verifier: Paraphrase seed 42 on the full train set, checkpoint kept -> outputs/verifier/seed_42/checkpoint
# Folds 0,2,4 run on GPU 0 and folds 1,3 + the verifier on GPU 1. Seed 42 everywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/aug/kfold.py split --k 5 --out_dir data/folds

run_fold() {  # $1 = gpu, $2.. = folds
  local gpu=$1; shift
  for k in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu python src/train_paraphrase.py --output_dir outputs/kfold/fold$k --seeds 42 \
      --train_path data/folds/fold${k}_train.jsonl --test_path data/folds/fold${k}_heldout.jsonl
  done
}
( run_fold 0 0 2 4 ) > outputs_kfold_gpu0.log 2>&1 &
P0=$!
( run_fold 1 1 3 && CUDA_VISIBLE_DEVICES=1 python src/train_paraphrase.py --output_dir outputs/verifier --seeds 42 --keep_ckpt ) \
  > outputs_kfold_gpu1.log 2>&1 &
P1=$!
wait $P0; wait $P1

python src/aug/kfold.py difficulty --runs outputs/kfold --out data/aug/difficulty.json
echo "difficulty: data/aug/difficulty.json; verifier checkpoint: outputs/verifier/seed_42/checkpoint"
