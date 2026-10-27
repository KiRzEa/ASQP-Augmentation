"""Nested random subsets of an augmented set (amount sweep): the first n of one fixed shuffle, so smaller
subsets are contained in larger ones and differ only in amount.

    python src/aug/subsample.py --in data/aug/quadra_gpt_v.jsonl --n 534 1068 --out_prefix data/aug/quadra_gpt_v
    -> data/aug/quadra_gpt_v534.jsonl, data/aug/quadra_gpt_v1068.jsonl (+ .subsample_report.json)
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    items = load_jsonl(a.inp)
    order = list(range(len(items)))
    random.Random(a.seed).shuffle(order)
    for n in sorted(a.n):
        if n > len(items):
            sys.exit(f"--n {n} > {len(items)} items in {a.inp}")
        chosen = [items[i] for i in order[:n]]
        out = Path(f"{a.out_prefix}{n}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for it in chosen:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        cells = collections.Counter(it.get("meta", {}).get("target_cell", "?") for it in chosen)
        report = {"input": a.inp, "seed": a.seed, "n": n, "n_input": len(items),
                  "quads": sum(len(it["quads"]) for it in chosen), "per_target_cell": dict(sorted(cells.items()))}
        json.dump(report, open(out.with_suffix(".subsample_report.json"), "w"), indent=1)
        print(f"{n}/{len(items)} reviews, {report['quads']} quads -> {out}")


if __name__ == "__main__":
    main()
