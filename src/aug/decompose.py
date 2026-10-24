"""QUADRA step 1 — decompose reviews into quad-bearing clause units (rule-based).

A review is cut at clause boundaries (punctuation, Vietnamese connectives). A cut is allowed only if it does
not fall inside any quad's region, i.e. the stretch from the first to the last character of its explicit
aspect/opinion spans. Reviews containing a quad with both spans implicit are not split (the quad cannot be
located). Pieces without quads are dropped; leading connectives and edge punctuation are stripped and the
offsets of every span are recomputed.

    python src/aug/decompose.py data/restaurant_v1/train.jsonl --out data/aug/units_train.jsonl --report
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import NULL, load_jsonl  # noqa: E402

# Longest first so that "nhưng mà" wins over "nhưng".
CONNECTIVES = sorted(["nhưng mà", "nhưng", "tuy nhiên", "tuy", "còn", "cơ mà", "có điều", "được cái", "song",
                      "với lại", "ngoài ra", "và", "mà", "nhưng được cái", "bù lại", "thêm nữa", "riêng"],
                     key=len, reverse=True)
CONN_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(c) for c in CONNECTIVES) + r")(?!\w)")
PUNCT_RE = re.compile(r"[,.;!?…]+|\s[-+]\s")
STRIP_CHARS = " ,.;!?…-+:•*~>"


def cell(q):
    imp = ("IA" if q["aspect"] == NULL else "EA") + "-" + ("IO" if q["opinion"] == NULL else "EO")
    return f"{q['category']}|{q['polarity']}|{imp}"


def quad_region(q):
    offs = [o for o in (q.get("aspect_offsets"), q.get("opinion_offsets")) if o]
    return (min(o[0] for o in offs), max(o[1] for o in offs)) if offs else None


def candidate_cuts(text):
    """Positions where a new piece may start: after punctuation, or before a connective."""
    cuts = set()
    for m in PUNCT_RE.finditer(text):
        cuts.add(m.end())
    for m in CONN_RE.finditer(text):
        if m.start() > 0:
            cuts.add(m.start())
    return sorted(c for c in cuts if 0 < c < len(text))


def decompose(item):
    """Return a list of units {"text", "quads", "source_id"}; a single unit means the review was not split."""
    text, quads = item["text"], item["quads"]
    regions = [quad_region(q) for q in quads]
    if any(r is None for r in regions):
        return [dict(text=text, quads=quads, source_id=item["id"], split=False)]
    allowed = [c for c in candidate_cuts(text) if not any(s < c < e for s, e in regions)]
    bounds = [0] + allowed + [len(text)]
    units = []
    for s, e in zip(bounds, bounds[1:]):
        inside = [q for q, (rs, re_) in zip(quads, regions) if s <= rs and re_ <= e]
        if not inside:
            continue
        seg = text[s:e]
        # strip edge punctuation/space and leading connectives, tracking the left shift
        left = len(seg) - len(seg.lstrip(STRIP_CHARS))
        seg2 = seg[left:]
        m = re.match(r"(" + "|".join(re.escape(c) for c in CONNECTIVES) + r")(?!\w)\s*", seg2)
        if m and not any(rs < s + left + m.end() for rs, _ in (quad_region(q) for q in inside)):
            left += m.end()
            seg2 = seg2[m.end():]
        seg2 = seg2.rstrip(STRIP_CHARS)
        off = s + left
        new_quads = []
        for q in inside:
            nq = dict(q)
            for f in ("aspect", "opinion"):
                o = q.get(f + "_offsets")
                nq[f + "_offsets"] = [o[0] - off, o[1] - off] if o else None
                if o:
                    assert seg2[o[0] - off:o[1] - off] == q[f], (item["id"], q[f], seg2)
            new_quads.append(nq)
        units.append(dict(text=seg2, quads=new_quads, source_id=item["id"]))
    covered = sum(len(u["quads"]) for u in units)
    assert covered == len(quads), (item["id"], covered, len(quads))
    for u in units:
        u["split"] = len(units) > 1
    return units


def report(items, units_by_item, rng):
    n = len(items)
    split = sum(1 for us in units_by_item if len(us) > 1)
    all_units = [u for us in units_by_item for u in us if u["split"]]
    nq = collections.Counter(len(u["quads"]) for u in all_units)
    orig_cells = collections.Counter(cell(q) for it in items for q in it["quads"])
    unit_cells = collections.Counter(cell(q) for u in all_units for q in u["quads"])
    single = collections.Counter(cell(u["quads"][0]) for u in all_units if len(u["quads"]) == 1)
    lens = [len(u["text"].split()) for u in all_units]
    print(f"reviews: {n}; split into >= 2 units: {split} ({100 * split / n:.1f}%); not split: {n - split}")
    print(f"units from split reviews: {len(all_units)}; quads per unit: {dict(sorted(nq.items()))}; "
          f"single-quad units: {nq[1]}; mean unit length {sum(lens) / max(1, len(lens)):.1f} syllables")
    print(f"quads that ended up in units of split reviews: {sum(unit_cells.values())} / {sum(orig_cells.values())} "
          f"({100 * sum(unit_cells.values()) / sum(orig_cells.values()):.1f}%)")
    print("\nper category × polarity (all implicit types pooled): quads in train | quads in units | single-quad units")
    pooled = lambda c: collections.Counter({k: v for k, v in ((key.rsplit('|', 1)[0], val) for key, val in c.items())})  # noqa: E731
    po, pu, ps = (collections.defaultdict(int) for _ in range(3))
    for src, dst in ((orig_cells, po), (unit_cells, pu), (single, ps)):
        for k, v in src.items():
            dst[k.rsplit("|", 1)[0]] += v
    for k in sorted(po, key=lambda k: po[k]):
        print(f"  {k:36s} {po[k]:5d} | {pu[k]:5d} ({100 * pu[k] / po[k]:5.1f}%) | {ps[k]:5d}")
    print("\nimplicit types: quads in train | quads in units")
    for t in ("EA-EO", "EA-IO", "IA-EO", "IA-IO"):
        o = sum(v for k, v in orig_cells.items() if k.endswith(t))
        u = sum(v for k, v in unit_cells.items() if k.endswith(t))
        print(f"  {t}: {o:5d} | {u:5d} ({100 * u / max(1, o):.1f}%)")
    print("\n30 random units:")
    for u in rng.sample(all_units, min(30, len(all_units))):
        qs = "; ".join(f"{q['category']},{q['aspect']},{q['polarity']},{q['opinion']}" for q in u["quads"])
        print(f"  [{u['source_id']}] {u['text']}  ->  {qs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    items = load_jsonl(a.path)
    units_by_item = [decompose(it) for it in items]
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for us in units_by_item:
                for u in us:
                    f.write(json.dumps(u, ensure_ascii=False) + "\n")
    if a.report:
        report(items, units_by_item, random.Random(a.seed))


if __name__ == "__main__":
    main()
