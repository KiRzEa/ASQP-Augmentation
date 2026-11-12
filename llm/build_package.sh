#!/usr/bin/env bash
# Build the offline package for a server without Internet (run on a machine WITH Internet, from the repo root):
#   bash llm/build_package.sh            -> dist/asqp_llm_package.zip
# Contents: code (llm/, src/asqp_common.py, src/aug/{mvd,st_scorer}.py), data (3 datasets + 3 unlabelled pools),
# Linux x86_64 wheels for Python 3.10-3.12 (llm/wheels), llm/MANIFEST.json (sha256 of every data file).
set -euo pipefail
cd "$(dirname "$0")/.."
WH=llm/wheels
rm -rf "$WH" && mkdir -p "$WH"
for pv in 3.10 3.11 3.12; do
  python3 -m pip download -q --no-deps --only-binary=:all: --dest "$WH" --implementation cp --python-version "$pv" \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
    --platform manylinux_2_5_x86_64 --platform manylinux1_x86_64 \
    -r <(grep -v '^#' llm/requirements-lock.txt)
done
DATA="data/restaurant_v1/train.jsonl data/restaurant_v1/dev.jsonl data/restaurant_v1/test.jsonl
data/acos/rest/train.jsonl data/acos/rest/dev.jsonl data/acos/rest/test.jsonl
data/acos/laptop/train.jsonl data/acos/laptop/dev.jsonl data/acos/laptop/test.jsonl
data/aug/st_pool_train.jsonl data/aug/acos_rest_pool_train.jsonl data/aug/acos_laptop_pool_train.jsonl"
python3 - $DATA <<'PY'
import hashlib, json, sys
m = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sys.argv[1:]}
json.dump(m, open("llm/MANIFEST.json", "w"), indent=1)
PY
mkdir -p dist && rm -f dist/asqp_llm_package.zip
zip -q -r dist/asqp_llm_package.zip llm/inlg.py llm/pipeline.py llm/setup.sh llm/README.md llm/requirements.txt \
  llm/requirements-lock.txt llm/MANIFEST.json llm/wheels src/asqp_common.py src/aug/mvd.py src/aug/st_scorer.py $DATA \
  -x '*/__pycache__/*'
echo "dist/asqp_llm_package.zip: $(du -h dist/asqp_llm_package.zip | cut -f1), $(ls $WH | wc -l | tr -d ' ') wheels"
