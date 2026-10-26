#!/usr/bin/env bash
# Kaggle (GPU T4 x2, Internet on, Save & Run All): run one session of experiments/run_plan.json.
#
#   bash scripts/run_kaggle.sh <N>          N = session number from `python3 scripts/kaggle_queue.py plan`
#   BUDGET=9 bash scripts/run_kaggle.sh <N>  override the hour budget
#
# Result: results_<round>_s<N>.zip in the repo root, re-packed after every finished job.
# Locally: put it in runs_inbox/ and run `python3 scripts/kaggle_queue.py ingest`.
set -uo pipefail
cd "$(dirname "$0")/.."
N=${1:?"usage: bash scripts/run_kaggle.sh <session number>"}
pip install -q -r requirements.txt
python -u scripts/kaggle_queue.py run --session "$N" ${BUDGET:+--budget-hours "$BUDGET"}
