"""ADA (Zhang et al., ICASSP 2024, arXiv 2401.06394): adaptive concatenation augmentation, reimplemented.

Two training reviews are concatenated (text and quads) when the condition function allows both. CPU only.

    python src/aug/ada.py --strategy joint    --out data/aug/ada_joint.jsonl
    python src/aug/ada.py --strategy pattern  --out data/aug/ada_pattern.jsonl
    python src/aug/ada.py --strategy category --out data/aug/ada_category.jsonl

Condition function (Eq. 3-4), per class with index pos in the descending count array of the raw train set:
    F = max(e^(-gamma*pos) + eta - n'/(kappa*n1), 0)
n1 = count of the largest class, n' = current count (raw count, increased by every concatenation). A review is
augmentable under
  pattern   F(its quad-pattern class) > 0
  category  F(c) > 0 for every category c in it (the min in Eq. 4)
  joint     both.
Classes: categories are counted per quad; a pattern class is the review's aspect-opinion DAG (Eq. 1-2).
Procedure (as in the released code): in each pass, every augmentable review in shuffled order is paired with
one random augmentable partner, both are re-checked against the updated counts, and the pair is concatenated;
passes repeat until no pair is formed or --max_passes is reached.
Defaults gamma 0.05, eta 0.4, kappa 2 = the paper's setting for the ACOS Restaurant dataset (implicit
aspects and opinions, like ours). Differences from the paper are listed in notes/DEVIATIONS.md.
"""
import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def pattern(item):
    """Canonical aspect-opinion DAG of a review (Eq. 1-2): nodes = distinct aspect / opinion spans (implicit
    "null" is a node), edges aspect -> opinion. Canonical form (sorted degree structure) so that the class does
    not depend on quad order."""
    edges = {(q["aspect"], q["opinion"]) for q in item["quads"]}
    a_out = collections.Counter(a for a, _ in edges)
    o_in = collections.Counter(o for _, o in edges)
    a_sig = sorted((a_out[a], tuple(sorted(o_in[o] for x, o in edges if x == a))) for a in a_out)
    return json.dumps([a_sig, sorted(o_in.values())])


def categories(item):
    return sorted({q["category"] for q in item["quads"]})


class Condition:
    def __init__(self, counts, gamma, eta, kappa):
        order = [c for c, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
        self.pos = {c: i + 1 for i, c in enumerate(order)}  # 1-based index in the descending array
        self.n1 = counts[order[0]]
        self.n = collections.Counter(counts)  # n', starts at the raw counts
        self.gamma, self.eta, self.kappa = gamma, eta, kappa

    def f(self, c):
        return max(math.exp(-self.gamma * self.pos[c]) + self.eta - self.n[c] / (self.kappa * self.n1), 0.0)


def concat(x, y, idx):
    shift = len(x["text"]) + 1
    quads = [dict(q) for q in x["quads"]]
    for q in y["quads"]:
        q = dict(q)
        for k in ("aspect_offsets", "opinion_offsets"):
            if q.get(k):
                q[k] = [q[k][0] + shift, q[k][1] + shift]
        quads.append(q)
    return {"id": f"ada-{idx:05d}", "text": f"{x['text']} {y['text']}", "quads": quads,
            "meta": {"method": "ada", "sources": [x["id"], y["id"]]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    ap.add_argument("--strategy", choices=["pattern", "category", "joint"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=0.4)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--max_passes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    train = load_jsonl(a.train_path)
    pat = {it["id"]: pattern(it) for it in train}
    cat = {it["id"]: categories(it) for it in train}
    cond_p = Condition(collections.Counter(pat.values()), a.gamma, a.eta, a.kappa)
    cond_c = Condition(collections.Counter(q["category"] for it in train for q in it["quads"]), a.gamma, a.eta, a.kappa)
    use_p, use_c = a.strategy in ("pattern", "joint"), a.strategy in ("category", "joint")

    def ok(it):
        return ((not use_p or cond_p.f(pat[it["id"]]) > 0)
                and (not use_c or min(cond_c.f(c) for c in cat[it["id"]]) > 0))

    def add(it):
        cond_p.n[pat[it["id"]]] += 1
        for q in it["quads"]:
            cond_c.n[q["category"]] += 1

    out, passes = [], []
    for p in range(1, a.max_passes + 1):
        pool = [it for it in train if ok(it)]
        rng.shuffle(pool)
        made = 0
        for x in pool:
            if not ok(x):
                continue
            y = rng.choice(pool)
            if y["id"] == x["id"] or not ok(y):
                continue
            out.append(concat(x, y, len(out)))
            add(x); add(y)
            made += 1
        passes.append({"pass": p, "augmentable": len(pool), "made": made})
        print(f"pass {p}: {len(pool)} augmentable, {made} concatenated, total {len(out)}")
        if made == 0:
            break
    with open(a.out, "w", encoding="utf-8") as f:
        for it in out:
            it["meta"]["strategy"] = a.strategy
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    raw_c = collections.Counter(q["category"] for it in train for q in it["quads"])
    raw_p = collections.Counter(pat.values())
    report = {"strategy": a.strategy, "gamma": a.gamma, "eta": a.eta, "kappa": a.kappa, "seed": a.seed,
              "max_passes": a.max_passes, "n_train": len(train), "n_aug": len(out),
              "quads_aug": sum(len(it["quads"]) for it in out), "passes": passes,
              "n_pattern_classes": len(raw_p),
              "category_counts": {c: {"raw": raw_c[c], "final": cond_c.n[c], "pos": cond_c.pos[c]} for c in cond_c.pos},
              "pattern_counts_top": {c: {"raw": raw_p[c], "final": cond_p.n[c]} for c in list(cond_p.pos)[:15]}}
    json.dump(report, open(Path(a.out).with_suffix(".report.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{a.strategy}: {len(out)} concatenated reviews ({report['quads_aug']} quads), "
          f"{len(raw_p)} pattern classes -> {a.out}")


if __name__ == "__main__":
    main()
