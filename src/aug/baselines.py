"""Rule-based augmentation baselines (CPU, no model). Every method keeps the quads valid: labelled spans are
never changed except by span substitution, and offsets are recomputed and checked (grounded()).

    python src/aug/baselines.py dup      --out data/aug/dup.jsonl            # copy of train (more steps, no new data)
    python src/aug/baselines.py eda      --out data/aug/eda.jsonl            # span-safe EDA (RI, RS, RD)
    python src/aug/baselines.py spansub  --out data/aug/spansub.jsonl        # aspect / opinion span substitution
    python src/aug/baselines.py concat   --out data/aug/concat.jsonl --n 3632  # random concatenation (ADA without F)

Default size --n = |train| (one augmented review per training review, sources taken in a shuffled cycle).
EDA (Wei & Zou, 2019) with alpha 0.1 and one operation per review: random insertion of a word of the same
review, random swap, random deletion, applied to words outside labelled spans only. Synonym replacement is
omitted (no Vietnamese WordNet). Span substitution: each explicit aspect (opinion) is replaced with p --p_sub by
an aspect (opinion) of the same category (category and polarity) from train; at least one replacement.
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import NULL, load_jsonl  # noqa: E402
from ada import concat  # noqa: E402
from quadra import grounded  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def regions(item):
    """Merged character intervals covered by labelled spans."""
    iv = sorted({tuple(q[f + "_offsets"]) for q in item["quads"] for f in ("aspect", "opinion") if q.get(f + "_offsets")})
    out = []
    for s, e in iv:
        if out and s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def units(item):
    """Review as units: (text, region index or None, glued to the previous unit)."""
    text, us, pos = item["text"], [], 0
    for ri, (s, e) in enumerate(regions(item)):
        for m in re.finditer(r"\S+", text[pos:s]):
            us.append([m.group(), None, m.start() == 0 and pos > 0 and not text[pos - 1].isspace() if pos else False])
        us.append([text[s:e], ri, s > 0 and not text[s - 1].isspace()])
        pos = e
    for m in re.finditer(r"\S+", text[pos:]):
        us.append([m.group(), None, m.start() == 0 and pos > 0])
    return us


def rebuild(item, us):
    """Join units, move every quad offset with its region; None if the result is not grounded."""
    old = regions(item)
    text, start = "", {}
    for i, (u, ri, glued) in enumerate(us):
        if text and not glued:
            text += " "
        if ri is not None:
            start[ri] = len(text)
        text += u
    quads = []
    for q in item["quads"]:
        q = dict(q)
        for f in ("aspect", "opinion"):
            o = q.get(f + "_offsets")
            if o:
                ri = next(i for i, (s, e) in enumerate(old) if s <= o[0] and o[1] <= e)
                d = start[ri] - old[ri][0]
                q[f + "_offsets"] = [o[0] + d, o[1] + d]
        quads.append(q)
    new = {"text": text, "quads": quads}
    return new if grounded(new) else None


def eda(item, rng, alpha):
    us = units(item)
    free = [i for i, u in enumerate(us) if u[1] is None]
    if len(free) < 2:
        return None
    n = max(1, round(alpha * len(us)))
    op = rng.choice(["ri", "rs", "rd"])
    if op == "ri":
        for _ in range(n):
            w = us[rng.choice(free)][0]
            us.insert(rng.randint(0, len(us)), [w, None, False])
    elif op == "rs":
        for _ in range(n):
            free = [i for i, u in enumerate(us) if u[1] is None]
            i, j = rng.sample(free, 2)
            us[i][0], us[j][0] = us[j][0], us[i][0]
    else:
        keep = [u for u in us if u[1] is not None or rng.random() > alpha]
        if len(keep) == len(us):
            drop = rng.choice(free)
            keep = [u for i, u in enumerate(us) if i != drop]
        us = keep
    for u in us[:1]:
        u[2] = False
    new = rebuild(item, us)
    return new and new["text"] != item["text"] and {**new, "meta": {"op": op}}


def spansub(item, rng, pools, p):
    text = item["text"]
    reg = {}  # (start, end) -> (field, key)
    for q in item["quads"]:
        for f in ("aspect", "opinion"):
            o = q.get(f + "_offsets")
            if o:
                key = q["category"] if f == "aspect" else (q["category"], q["polarity"])
                if tuple(o) in reg and reg[tuple(o)][0] != f:
                    return None
                reg.setdefault(tuple(o), (f, key))
    spans = sorted(reg)
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):  # overlapping aspect / opinion regions
        return None
    new_span = {}
    for o in spans:
        f, key = reg[o]
        if rng.random() < p:
            cand = [s for s in pools[f][key] if s != text[o[0]:o[1]]]
            if cand:
                new_span[o] = rng.choice(cand)
    if not new_span:
        return None
    out, pos, moved = "", 0, {}
    for o in spans:
        out += text[pos:o[0]]
        s = new_span.get(o, text[o[0]:o[1]])
        moved[o] = [len(out), len(out) + len(s)]
        out += s
        pos = o[1]
    out += text[pos:]
    quads = []
    for q in item["quads"]:
        q = dict(q)
        for f in ("aspect", "opinion"):
            o = q.get(f + "_offsets")
            if o:
                q[f + "_offsets"] = moved[tuple(o)]
                q[f] = out[moved[tuple(o)][0]:moved[tuple(o)][1]]
        quads.append(q)
    new = {"text": out, "quads": quads}
    return new if grounded(new) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dup", "eda", "spansub", "concat"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="augmented reviews; default |train|")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--p_sub", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    train = load_jsonl(a.train_path)
    n = a.n or len(train)
    pools = {"aspect": collections.defaultdict(set), "opinion": collections.defaultdict(set)}
    for it in train:
        for q in it["quads"]:
            if q["aspect"] != NULL:
                pools["aspect"][q["category"]].add(q["aspect"])
            if q["opinion"] != NULL:
                pools["opinion"][(q["category"], q["polarity"])].add(q["opinion"])
    pools = {f: {k: sorted(v) for k, v in d.items()} for f, d in pools.items()}

    out, stats, tries = [], collections.Counter(), 0
    order = []
    while len(out) < n:
        if not order:
            order = list(train)
            rng.shuffle(order)
        x = order.pop()
        tries += 1
        if tries > 50 * n:
            sys.exit(f"only {len(out)} of {n} produced")
        if a.method == "dup":
            new = {"text": x["text"], "quads": x["quads"], "meta": {}}
        elif a.method == "eda":
            new = eda(x, rng, a.alpha)
        elif a.method == "spansub":
            new = spansub(x, rng, pools, a.p_sub)
        else:
            y = rng.choice(train)
            new = None if y["id"] == x["id"] else concat(x, y, len(out))
        if not new:
            stats["failed"] += 1
            continue
        meta = {**new.get("meta", {}), "method": a.method, "source": x["id"]}
        if a.method == "concat":
            meta["sources"] = new["meta"]["sources"]
        stats[meta.get("op", a.method)] += 1
        out.append({"id": f"{a.method}-{len(out):05d}", "text": new["text"], "quads": new["quads"], "meta": meta})
    with open(a.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    report = {"method": a.method, "seed": a.seed, "n": len(out), "quads": sum(len(it["quads"]) for it in out),
              "alpha": a.alpha if a.method == "eda" else None, "p_sub": a.p_sub if a.method == "spansub" else None,
              "attempts": tries, "stats": dict(stats)}
    json.dump(report, open(Path(a.out).with_suffix(".report.json"), "w"), indent=1)
    print(f"{a.method}: {len(out)} reviews, {report['quads']} quads, {dict(stats)} -> {a.out}")


if __name__ == "__main__":
    main()
