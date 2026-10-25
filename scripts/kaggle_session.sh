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
    # Verifier: VERIFIER_CKPT (e.g. the s1 checkpoint attached as Kaggle input) if given, else train one in
    # outputs/verifier_<session> (own folder, so unzipping never overwrites another session's verifier).
    NTRAIN=$(wc -l < data/restaurant_v1/train.jsonl)
    VDIR=outputs/verifier_${SESSION}
    CKPT=${VERIFIER_CKPT:-$VDIR/seed_42/checkpoint}
    VPID=""
    if [[ ! -d "$CKPT" ]]; then
      CUDA_VISIBLE_DEVICES=0 python src/train_paraphrase.py --output_dir "$VDIR" --seeds 42 --keep_ckpt \
        > "$VDIR.log" 2>&1 &
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
    python src/aug/verify_model.py --ckpt "$CKPT" --in /dev/null --out /dev/null --card_only > "data/aug/verifier_card_${SESSION}.json"
    for v in rule gpt; do
      [[ -f data/aug/quadra_$v.jsonl ]] || continue
      CUDA_VISIBLE_DEVICES=0 python src/aug/verify_model.py --ckpt "$CKPT" --in data/aug/quadra_$v.jsonl \
        --out data/aug/quadra_${v}_v.jsonl --target_n "$NTRAIN" 2>&1 | tee -a "$LOG"
    done
    FILES+=(data/aug)
    [[ -d "$VDIR" ]] && FILES+=("$VDIR" "$VDIR.log")   # checkpoint weights excluded below; hash is in the card
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
zip -qr "session_${SESSION}.zip" "${FILES[@]}" "$LOG" -x "outputs/kfold/*/checkpoint/*" -x "outputs/verifier_*/seed_*/checkpoint/*"
echo "packed session_${SESSION}.zip ($(du -h "session_${SESSION}.zip" | cut -f1)) with: ${FILES[*]}"
