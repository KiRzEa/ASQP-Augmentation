"""Review combination (Algorithm 1 of "Using Review Combination and Pseudo-Tokens for ASQP", Findings NAACL 2025).

    python src/aug/review_combination.py --train_path data/restaurant_v1/train.jsonl --out data/aug/rc_train.jsonl

Reviews are split into D_I (every aspect and opinion term implicit), D_E (every term explicit) and D_M (mixed).
D_I and D_E reviews are paired one to one (random order inside the pair); the rest of the larger of the two is
paired with D_M reviews; whatever is left stays alone. The output is the combined training set D' that REPLACES
the training set (--train_path of train_paraphrase.py), as in the paper. Only the combination component is
reproduced: pseudo-token prompts, multi-order templates and voting are other components of that framework.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import NULL, load_jsonl  # noqa: E402
from ada import concat  # noqa: E402
from quadra import grounded  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def kind(x):
    terms = [t for q in x["quads"] for t in (q["aspect"], q["opinion"])]
    return "I" if all(t == NULL for t in terms) else "E" if all(t != NULL for t in terms) else "M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/restaurant_v1/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    sets = collections.defaultdict(list)
    for x in load_jsonl(a.train_path):
        sets[kind(x)].append(x)
    for v in sets.values():
        rng.shuffle(v)
    sizes = {k: len(sets[k]) for k in "IEM"}
    pairs = []

    def pair(p, q):
        while sets[p] and sets[q]:
            x, y = sets[p].pop(), sets[q].pop()
            pairs.append((x, y) if rng.random() < 0.5 else (y, x))
    pair("I", "E")
    n_ie = len(pairs)
    pair("I" if sets["I"] else "E", "M")
    out = []
    for x, y in pairs:
        c = concat(x, y, len(out))
        assert grounded(c), (x["id"], y["id"])
        out.append({"id": f"rc-{len(out):05d}", "text": c["text"], "quads": c["quads"],
                    "meta": {"method": "review_combination", "sources": [x["id"], y["id"]]}})
    for k in "IEM":
        for x in sets[k]:
            out.append({**x, "meta": {**x.get("meta", {}), "method": "review_combination", "alone": k}})
    with open(a.out, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    report = {"train_path": a.train_path, "seed": a.seed, "sets": sizes, "pairs_I_E": n_ie,
              "pairs_with_M": len(pairs) - n_ie, "alone": {k: len(sets[k]) for k in "IEM"}, "n_out": len(out),
              "quads_in": sum(len(x["quads"]) for x in load_jsonl(a.train_path)),
              "quads_out": sum(len(x["quads"]) for x in out)}
    json.dump(report, open(Path(a.out).with_suffix(".report.json"), "w"), indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
