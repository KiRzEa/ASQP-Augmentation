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
mkdir -p outputs data/aug   # outputs/ is git-ignored, so it does not exist in a fresh clone
LOG=session_${SESSION}.log
echo "session $SESSION started $(date -u)" | tee "$LOG"

# GPU check: stop instead of silently training on CPU (set ALLOW_CPU=1 to override)
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | sed "s/^/[gpu] /" | tee -a "$LOG"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "[gpu] torch sees $NGPU CUDA device(s); torch $(python -c 'import torch;print(torch.__version__)'), transformers $(python -c 'import transformers;print(transformers.__version__)')" | tee -a "$LOG"
if [[ "$NGPU" -lt 1 && -z "${ALLOW_CPU:-}" ]]; then
  echo "ERROR: no GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2 (also for 'Save & Run All'). Stopping." | tee -a "$LOG"
  exit 1
fi

pack() {  # update the zip after every job, so a timeout keeps finished work
  zip -qr "session_${SESSION}.zip" ${FILES[@]+"${FILES[@]}"} "$LOG" -x "outputs/kfold/*/checkpoint/*" -x "outputs/verifier_*/seed_*/checkpoint/*"
  echo "[pack] session_${SESSION}.zip updated ($(du -h "session_${SESSION}.zip" | cut -f1))" | tee -a "$LOG"
}
rm -f "session_${SESSION}.zip"

for job in "$@"; do
  JSTART=$(date +%s)
  echo "=== job: $job  started $(date -u)" | tee -a "$LOG"
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
      echo "[verifier] training Paraphrase seed 42 on GPU 0 (~45 min on a T4) -> $CKPT" | tee -a "$LOG"
      ( CUDA_VISIBLE_DEVICES=0 python -u src/train_paraphrase.py --output_dir "$VDIR" --seeds 42 --keep_ckpt 2>&1 \
        | tee "$VDIR.log" | sed -u "s/^/[verifier] /" ) &
      VPID=$!
    fi
    # Candidates: reuse the committed files (fixed data for the paper); REGEN=1 generates new ones.
    if [[ -n "${REGEN:-}" || ! -f data/aug/quadra_gpt.jsonl ]]; then
      if [[ -n "${OPENAI_API_KEY:-}" ]]; then
        python src/aug/quadra.py --difficulty data/aug/difficulty.json --ratio 1.5 --out data/aug/quadra_gpt.jsonl \
          --llm_model gpt-4.1-mini-2025-04-14 2>&1 | tee -a "$LOG"
        python src/aug/pre_fusion.py data/aug/quadra_gpt.jsonl data/aug/quadra_rule.jsonl 2>&1 | tee -a "$LOG"
      else
        python src/aug/quadra.py --difficulty data/aug/difficulty.json --ratio 1.5 --out data/aug/quadra_rule.jsonl 2>&1 | tee -a "$LOG"
        echo "OPENAI_API_KEY not set: rules-only variant generated, GPT variant skipped" | tee -a "$LOG"
      fi
    else
      echo "using committed candidates data/aug/quadra_gpt.jsonl and quadra_rule.jsonl" | tee -a "$LOG"
    fi
    [[ -n "$VPID" ]] && wait "$VPID"
    if [[ ! -f "$CKPT/config.json" ]]; then
      echo "ERROR: no verifier checkpoint at $CKPT; last lines of $VDIR.log:" | tee -a "$LOG"
      tail -n 30 "$VDIR.log" 2>/dev/null | tee -a "$LOG"
      FILES+=(data/aug); [[ -f "$VDIR.log" ]] && FILES+=("$VDIR.log")
      pack; continue
    fi
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
    [[ -f "$aug" ]] || { echo "missing $aug" | tee -a "$LOG"; pack; continue; }
    NOZIP=1 bash scripts/kaggle_paraphrase.sh "para_$name" --aug_path "$aug" 2>&1 | tee -a "$LOG"
    FILES+=("outputs/para_$name")
  else
    echo "unknown job $job" | tee -a "$LOG"
  fi
  echo "=== job: $job  finished $(date -u) after $(( ($(date +%s) - JSTART) / 60 )) min" | tee -a "$LOG"
  pack
done

echo "session $SESSION finished $(date -u); download session_${SESSION}.zip" | tee -a "$LOG"
pack
