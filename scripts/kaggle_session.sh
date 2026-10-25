#!/usr/bin/env bash
# One Kaggle session = one command = one zip.
#
#   bash scripts/kaggle_session.sh <session_name> <job> [<job> ...]
#
# Jobs (run in the order given, each on both GPUs):
#   none                       Paraphrase baseline, no augmentation      -> outputs/para_none
#   <name>=<aug.jsonl>         Paraphrase + augmented data, 5 seeds      -> outputs/para_<name>
#   prep                       QUADRA prep: 5-fold difficulty + verifier -> data/aug/difficulty.json, outputs/kfold,
#                              outputs/verifier (checkpoint included in the zip, ~0.9 GB)
#   quadra_data                QUADRA data on the GPU machine: train the verifier if no checkpoint is present (GPU 0)
#                              while generating 1.5x candidates (rules + GPT fusion, needs OPENAI_API_KEY), then keep
#                              exactly |train| verified reviews -> data/aug/quadra_{rule,gpt}_v.jsonl (+ reports, cache)
# Examples:
#   bash scripts/kaggle_session.sh s1 none prep
#   bash scripts/kaggle_session.sh s2 quadra_data quadra_gpt=data/aug/quadra_gpt_v.jsonl quadra_rule=data/aug/quadra_rule_v.jsonl
#
# Result: session_<session_name>.zip in the repo root. Paths inside are relative to the repo root, so locally:
#   cd <repo> && unzip -o session_<session_name>.zip
set -uo pipefail
SESSION=${1:?session name}; shift
cd "$(dirname "$0")/.."
FILES=()
LOG=session_${SESSION}.log
echo "session $SESSION started $(date -u)" | tee "$LOG"

for job in "$@"; do
  echo "=== job: $job  $(date -u)" | tee -a "$LOG"
  if [[ "$job" == "none" ]]; then
    NOZIP=1 bash scripts/kaggle_paraphrase.sh para_none 2>&1 | tee -a "$LOG"
    FILES+=(outputs/para_none)
  elif [[ "$job" == "prep" ]]; then
    bash scripts/kaggle_quadra_prep.sh 2>&1 | tee -a "$LOG"
    FILES+=(data/aug/difficulty.json outputs/kfold outputs/verifier outputs_kfold_gpu0.log outputs_kfold_gpu1.log)
  elif [[ "$job" == "quadra_data" ]]; then
    NTRAIN=$(wc -l < data/restaurant_v1/train.jsonl)
    CKPT=outputs/verifier/seed_42/checkpoint
    VPID=""
    if [[ ! -d "$CKPT" ]]; then
      CUDA_VISIBLE_DEVICES=0 python src/train_paraphrase.py --output_dir outputs/verifier --seeds 42 --keep_ckpt \
        > outputs_verifier.log 2>&1 &
      VPID=$!
    fi
    python src/aug/quadra.py --difficulty data/aug/difficulty.json --ratio 1.5 --out data/aug/quadra_rule.jsonl 2>&1 | tee -a "$LOG"
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
      python src/aug/quadra.py --difficulty data/aug/difficulty.json --ratio 1.5 --out data/aug/quadra_gpt.jsonl \
        --llm_model gpt-4.1-mini-2025-04-14 2>&1 | tee -a "$LOG"
    else
      echo "OPENAI_API_KEY not set: skipping the GPT variant" | tee -a "$LOG"
    fi
    [[ -n "$VPID" ]] && wait "$VPID"
    for v in rule gpt; do
      [[ -f data/aug/quadra_$v.jsonl ]] || continue
      CUDA_VISIBLE_DEVICES=0 python src/aug/verify_model.py --ckpt "$CKPT" --in data/aug/quadra_$v.jsonl \
        --out data/aug/quadra_${v}_v.jsonl --target_n "$NTRAIN" 2>&1 | tee -a "$LOG"
    done
    FILES+=(data/aug outputs/verifier/seed_42/metrics.json outputs/verifier/seed_42/run_config.json)
    [[ -f outputs_verifier.log ]] && FILES+=(outputs_verifier.log)
  elif [[ "$job" == *=* ]]; then
    name=${job%%=*}; aug=${job#*=}
    [[ -f "$aug" ]] || { echo "missing $aug" | tee -a "$LOG"; continue; }
    NOZIP=1 bash scripts/kaggle_paraphrase.sh "para_$name" --aug_path "$aug" 2>&1 | tee -a "$LOG"
    FILES+=("outputs/para_$name")
  else
    echo "unknown job $job" | tee -a "$LOG"
  fi
done

echo "session $SESSION finished $(date -u)" | tee -a "$LOG"
rm -f "session_${SESSION}.zip"
zip -qr "session_${SESSION}.zip" "${FILES[@]}" "$LOG" -x "outputs/kfold/*/checkpoint/*"
echo "packed session_${SESSION}.zip ($(du -h "session_${SESSION}.zip" | cut -f1)) with: ${FILES[*]}"
