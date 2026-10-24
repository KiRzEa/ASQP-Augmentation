"""QUADRA step 2 support — k-fold splits of train and per-cell out-of-fold difficulty.

    python src/aug/kfold.py split --k 5 --out_dir data/folds          # write fold{i}_train.jsonl / fold{i}_heldout.jsonl
    # train src/train_paraphrase.py on each fold_train, predict on fold_heldout (scripts/kaggle_quadra_prep.sh)
    python src/aug/kfold.py difficulty --runs outputs/kfold --out data/aug/difficulty.json

Difficulty = out-of-fold quad F1 per category|polarity cell, computed only on the training set, so the dev set
stays reserved for model selection.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import load_jsonl, prf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def split(a):
    items = load_jsonl(a.train_path)
    idx = list(range(len(items)))
    random.Random(a.seed).shuffle(idx)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(a.k):
        held = set(idx[k::a.k])
        for name, sel in (("train", [i for i in range(len(items)) if i not in held]), ("heldout", sorted(held))):
            with open(out / f"fold{k}_{name}.jsonl", "w", encoding="utf-8") as f:
                for i in sel:
                    f.write(json.dumps(items[i], ensure_ascii=False) + "\n")
        print(f"fold {k}: train {len(items) - len(held)}, heldout {len(held)}")


def difficulty(a):
    counts = collections.defaultdict(collections.Counter)
    files = sorted(Path(a.runs).glob("fold*/seed_*/test_predictions.jsonl"))
    if not files:
        sys.exit(f"no fold predictions under {a.runs}")
    for p in files:
        for r in load_jsonl(p):
            g, pr = {tuple(q) for q in r["gold"]}, {tuple(q) for q in r["pred"]}
            for q in g:
                counts[f"{q[0]}|{q[3]}"]["tp" if q in pr else "fn"] += 1
            for q in pr - g:
                counts[f"{q[0]}|{q[3]}"]["fp"] += 1
    out = {c: prf(v["tp"], v["fp"], v["fn"])["f1"] for c, v in sorted(counts.items())}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(files)} folds; per-cell out-of-fold F1 written to {a.out}")
    for c, f in sorted(out.items(), key=lambda kv: kv[1]):
        print(f"  {c:36s} {f:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out_dir", default=str(ROOT / "data/folds"))
    d = sub.add_parser("difficulty")
    d.add_argument("--runs", default=str(ROOT / "outputs/kfold"))
    d.add_argument("--out", default=str(ROOT / "data/aug/difficulty.json"))
    a = ap.parse_args()
    split(a) if a.cmd == "split" else difficulty(a)


if __name__ == "__main__":
    main()
