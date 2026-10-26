#!/usr/bin/env bash
# LEGACY (rounds r1, r2). New rounds use scripts/kaggle_queue.py + scripts/run_kaggle.sh (one job = run x seed).
# One Kaggle session = one command = one zip. Sessions are defined in experiments/plan.json (round -> session -> jobs).
#
#   bash scripts/kaggle_session.sh <round> <session>          e.g. bash scripts/kaggle_session.sh r3 s1
#   bash scripts/kaggle_session.sh <name> <job> [<job> ...]   ad hoc (jobs on the command line)
#
# Jobs (run in the order given, each on both GPUs):
#   none                       Paraphrase baseline, no augmentation      -> outputs/para_none
#   <name>=<aug.jsonl>         Paraphrase + augmented data, 5 seeds      -> outputs/para_<name>
#   prep                       QUADRA prep: 5-fold difficulty + verifier -> data/aug/difficulty.json, outputs/kfold,
#                              outputs/verifier (checkpoint included in the zip, ~0.9 GB)
#   quadra_data                QUADRA candidates + verification (verifier trained in-session if needed)
#
# Result: results_<round>_<session>.zip (paths relative to the repo root) containing the runs and
# outputs/_sessions/<round>_<session>/ (session.log, git commit, jobs). Locally: cd <repo> && unzip -o <zip>
set -uo pipefail
cd "$(dirname "$0")/.."
A1=${1:?"usage: kaggle_session.sh <round> <session> | <name> <job>..."}; shift
if [[ $# -eq 1 ]] && python - "$A1" "$1" <<'PY'
import json, sys
r, s = sys.argv[1], sys.argv[2]
plan = json.load(open("experiments/plan.json"))["rounds"]
sys.exit(0 if r in plan and s in plan[r]["sessions"] else 1)
PY
then
  SESSION="${A1}_$1"
  JOBS=()
  while IFS= read -r line; do [[ -n "$line" ]] && JOBS+=("$line"); done < <(python -c \
    "import json,sys;print('\n'.join(json.load(open('experiments/plan.json'))['rounds'][sys.argv[1]]['sessions'][sys.argv[2]]))" "$A1" "$1")
else
  SESSION="$A1"; JOBS=("$@")
fi
[[ ${#JOBS[@]} -gt 0 ]] || { echo "no jobs for $SESSION"; exit 1; }  # bash 3.2-safe below
FILES=()
mkdir -p outputs data/aug "outputs/_sessions/$SESSION"   # outputs/ is git-ignored, so it does not exist in a fresh clone
LOG=outputs/_sessions/$SESSION/session.log
ZIP=results_${SESSION}.zip
echo "session $SESSION started $(date -u); commit $(git rev-parse --short HEAD 2>/dev/null); jobs: ${JOBS[*]}" | tee "$LOG"
printf '%s\n' "${JOBS[@]}" > "outputs/_sessions/$SESSION/jobs.txt"
git rev-parse HEAD > "outputs/_sessions/$SESSION/commit.txt" 2>/dev/null
FILES+=("outputs/_sessions/$SESSION")

# GPU check: stop instead of silently training on CPU (set ALLOW_CPU=1 to override)
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | sed "s/^/[gpu] /" | tee -a "$LOG"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "[gpu] torch sees $NGPU CUDA device(s); torch $(python -c 'import torch;print(torch.__version__)'), transformers $(python -c 'import transformers;print(transformers.__version__)')" | tee -a "$LOG"
if [[ "$NGPU" -lt 1 && -z "${ALLOW_CPU:-}" ]]; then
  echo "ERROR: no GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2 (also for 'Save & Run All'). Stopping." | tee -a "$LOG"
  exit 1
fi

pack() {  # update the zip after every job, so a timeout keeps finished work
  zip -qr "$ZIP" ${FILES[@]+"${FILES[@]}"} -x "outputs/kfold/*/checkpoint/*" -x "outputs/verifier_*/seed_*/checkpoint/*"
  echo "[pack] $ZIP updated ($(du -h "$ZIP" | cut -f1))" | tee -a "$LOG"
}
rm -f "$ZIP"

for job in "${JOBS[@]}"; do
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

echo "session $SESSION finished $(date -u); download $ZIP" | tee -a "$LOG"
pack
