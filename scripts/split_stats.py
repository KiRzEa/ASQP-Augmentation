#!/usr/bin/env python3
"""Descriptive comparison of dev and test with train (dataset section of the paper; not used to design methods).

    python3 scripts/split_stats.py [--json out.json]

For every label and text distribution, dev and test are compared with train, and the same comparison is repeated
for random train subsets of the same size (the rest of train as reference, --boot draws): a dev/test value is a
real shift only if it falls outside the 95 % range of those train-vs-train values. The dev/test value itself is the
mean over the same number of random train references of the same size as in the null (|train| - |split|), so both
sides compare against equally large references. Above the range = shift; below = closer to train than random
sampling would be (e.g. a split stratified by category).
Label distributions: quads per review, polarity, category, implicit type, category x polarity, number of
polarities per review. Terms: share of aspect / opinion terms (per quad) unseen in train or seen 1-2 times.
Cells (category x polarity x implicit type) that are rare (<= 2 quads) or absent in train but present in dev/test.
Text: word length, comma / emoji / slang shares.
"""
import argparse
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from asqp_common import NULL, implicit_type, load_jsonl, quad_tuple  # noqa: E402

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
SLANG = {"ko", "k", "dc", "đc", "j", "mn", "hok", "hông", "kh", "bt", "mik", "mk", "vs", "nc", "ng"}


def tv(p, q):
    """Total variation distance between two count distributions (0 = identical, 1 = disjoint)."""
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    return 0.5 * sum(abs(p.get(k, 0) / sp - q.get(k, 0) / sq) for k in set(p) | set(q))


def quads(items):
    return [quad_tuple(q) for x in items for q in x["quads"]]


DISTS = {
    "quads_per_review": lambda xs: collections.Counter(min(len(x["quads"]), 6) for x in xs),
    "polarities_per_review": lambda xs: collections.Counter(len({q["polarity"] for q in x["quads"]}) for x in xs),
    "polarity": lambda xs: collections.Counter(q[3] for q in quads(xs)),
    "category": lambda xs: collections.Counter(q[0] for q in quads(xs)),
    "implicit_type": lambda xs: collections.Counter(implicit_type(q) for q in quads(xs)),
    "category_x_polarity": lambda xs: collections.Counter((q[0], q[3]) for q in quads(xs)),
}


def term_shares(items, ref):
    ca = collections.Counter(q[1] for q in quads(ref)); co = collections.Counter(q[2] for q in quads(ref))
    qs = quads(items)
    out = {}
    for name, idx, c in (("aspect", 1, ca), ("opinion", 2, co)):
        ex = [q[idx] for q in qs if q[idx] != NULL]
        out[f"{name}_unseen"] = sum(c[t] == 0 for t in ex) / max(len(ex), 1)
        out[f"{name}_rare_1_2"] = sum(1 <= c[t] <= 2 for t in ex) / max(len(ex), 1)
    out["quad_with_unseen_or_rare_term"] = sum((q[1] != NULL and ca[q[1]] <= 2) or (q[2] != NULL and co[q[2]] <= 2)
                                               for q in qs) / max(len(qs), 1)
    return out


def text_stats(items):
    t = [x["text"] for x in items]
    return {"words_mean": statistics.mean(len(s.split()) for s in t),
            "comma_share": sum("," in s for s in t) / len(t),
            "emoji_share": sum(bool(EMOJI.search(s)) for s in t) / len(t),
            "slang_share": sum(any(w in s.split() for w in SLANG) for s in t) / len(t)}


def profile(items, ref):
    p = {f"TV_{k}": tv(f(items), f(ref)) for k, f in DISTS.items()}
    p.update(term_shares(items, ref))
    p.update(text_stats(items))
    qs = quads(items)
    p["quads_per_review_mean"] = len(qs) / len(items)
    for k, v in collections.Counter(implicit_type(q) for q in qs).items():
        p[f"share_{k}"] = v / len(qs)
    for k, v in collections.Counter(q[3] for q in qs).items():
        p[f"share_{k}"] = v / len(qs)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--json")
    a = ap.parse_args()
    train = load_jsonl(ROOT / "data/restaurant_v1/train.jsonl")
    splits = {"dev": load_jsonl(ROOT / "data/restaurant_v1/dev.jsonl"),
              "test": load_jsonl(ROOT / "data/restaurant_v1/test.jsonl")}
    rng = random.Random(a.seed)
    res = {}
    for name, items in splits.items():
        real, null = collections.defaultdict(list), collections.defaultdict(list)
        for _ in range(a.boot):
            idx = set(rng.sample(range(len(train)), len(items)))
            sub = [x for i, x in enumerate(train) if i in idx]
            rest = [x for i, x in enumerate(train) if i not in idx]
            for k, v in profile(sub, rest).items():  # null: train subset of the split's size vs the rest of train
                null[k].append(v)
            for k, v in profile(items, rest).items():  # the split vs a reference of the same size
                real[k].append(v)
        res[name] = {}
        for k in real:
            v, xs = statistics.mean(real[k]), sorted(null[k])
            lo, hi = xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]
            res[name][k] = {"value": v, "null_lo": lo, "null_hi": hi,
                            "flag": "HIGHER" if v > hi else "lower" if v < lo else ""}
    print(f"{'measure':34s} " + " ".join(f"{n + ' (train-subset 95% range)':>38s}" for n in splits))
    for k in res["dev"]:
        cells = []
        for n in splits:
            r = res[n].get(k)
            if r is None:
                cells.append(" " * 38)
                continue
            cells.append(f"{r['value']:8.3f} [{r['null_lo']:.3f}, {r['null_hi']:.3f}] {r['flag']:6s}".rjust(38))
        print(f"{k:34s} " + " ".join(cells))
    # cells rare/absent in train but present in dev/test
    cell = lambda q: (q[0], q[3], implicit_type(q))
    ct = collections.Counter(cell(q) for q in quads(train))
    print("\ncells (category, polarity, implicit type) with <= 2 training quads that occur in dev/test:")
    for n, items in splits.items():
        c = collections.Counter(cell(q) for q in quads(items))
        rare = sorted(((k, v, ct[k]) for k, v in c.items() if ct[k] <= 2), key=lambda z: -z[1])
        tot = sum(c.values())
        print(f"  {n}: {len(rare)} cells, {sum(v for _, v, _ in rare)} quads ({100 * sum(v for _, v, _ in rare) / tot:.1f}% of {n})")
        for k, v, t in rare[:10]:
            print(f"     {k}: {n} {v}, train {t}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
