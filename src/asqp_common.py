"""Shared data handling, target format and scoring for every ASQP experiment.

All models and augmentation methods use these functions, so numbers are comparable:
- data: JSONL from scripts/prepare_data.py ({"id", "text", "quads": [{category, aspect, polarity, opinion, ...}]})
- quads are compared as (category, aspect, opinion, polarity) tuples; implicit spans are the string "null"
- primary metric: exact-match micro P/R/F1 over quads pooled over all sentences (in percent)
"""
import collections
import json
import re
import unicodedata

NULL = "null"
CATEGORY_VI = {
    "FOOD#QUALITY": "chất lượng đồ ăn",
    "FOOD#STYLE&OPTIONS": "lựa chọn đồ ăn",
    "FOOD#PRICES": "giá đồ ăn",
    "DRINKS#QUALITY": "chất lượng đồ uống",
    "DRINKS#STYLE&OPTIONS": "lựa chọn đồ uống",
    "DRINKS#PRICES": "giá đồ uống",
    "SERVICE#GENERAL": "dịch vụ",
    "AMBIENCE#GENERAL": "không gian",
    "LOCATION#GENERAL": "địa điểm",
    "RESTAURANT#GENERAL": "nhà hàng nói chung",
    "RESTAURANT#PRICES": "giá nhà hàng",
    "RESTAURANT#MISCELLANEOUS": "nhà hàng khác",
}
POLARITY_VI = {"positive": "tích cực", "negative": "tiêu cực", "neutral": "trung lập"}
VI_CATEGORY = {v: k for k, v in CATEGORY_VI.items()}
VI_POLARITY = {v: k for k, v in POLARITY_VI.items()}
SEP = " [SSEP] "
QUAD_RE = re.compile(r"\[A\](.*?)\[O\](.*?)\[C\](.*?)\[S\](.*)$")
ORDER = "AOCS"  # element order of the default target; other orders are the views of --views (multi-view)
# Views in a fixed order chosen before any run: view 0 = the default target, then orders that move each element
# to the front at least once and differ in the position of the aspect-opinion pair.
VIEWS = ["AOCS", "OASC", "CSAO", "SCOA", "ACSO", "OCAS"]


def view_prefix(order):
    """Input prefix that tells the model which element order to generate, e.g. "[O] [A] [S] [C]"."""
    return " ".join(f"[{e}]" for e in order)


def _quad_re(order):
    return re.compile("".join(rf"\[{e}\](.*?)" for e in order[:-1]) + rf"\[{order[-1]}\](.*)$")


def clean_span(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def quad_tuple(q):
    return (q["category"], clean_span(q["aspect"]), clean_span(q["opinion"]), q["polarity"])


# ---------- Paraphrase target (marked template, cf. "PARAPHRASE + Marked Template" in GenDA) ----------

def quads_to_target(quads, order=ORDER):
    """[A] aspect [O] opinion [C] category words [S] polarity word, quads joined by [SSEP] (gold order).
    order permutes the four marked elements (default AOCS = the template above)."""
    parts = []
    for q in quads:
        v = {"A": q["aspect"], "O": q["opinion"], "C": CATEGORY_VI[q["category"]], "S": POLARITY_VI[q["polarity"]]}
        parts.append(" ".join(f"[{e}] {v[e]}" for e in order))
    return SEP.join(parts)


def target_to_quads(text, order=ORDER):
    """Parse a generated target. Returns (set of quad tuples, number of unparsable segments)."""
    quads, invalid = set(), 0
    rx = QUAD_RE if order == ORDER else _quad_re(order)
    for seg in text.split("[SSEP]"):
        seg = seg.strip()
        if not seg:
            continue
        m = rx.match(seg)
        if not m:
            invalid += 1
            continue
        g = dict(zip(order, (clean_span(x) for x in m.groups())))
        a, o, c, s = g["A"], g["O"], g["C"], g["S"]
        if c not in VI_CATEGORY or s not in VI_POLARITY or not a or not o:
            invalid += 1
            continue
        quads.add((VI_CATEGORY[c], a, o, VI_POLARITY[s]))
    return quads, invalid


# ---------- scoring ----------

def prf(tp, fp, fn):
    p = 100 * tp / (tp + fp) if tp + fp else 0.0
    r = 100 * tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _overlap(x, y):
    if x == NULL or y == NULL:
        return x == y
    a, b = set(x.split()), set(y.split())
    return bool(a & b)


def implicit_type(q):
    return ("IA" if q[1] == NULL else "EA") + "-" + ("IO" if q[2] == NULL else "EO")


def score(records):
    """records: list of {"gold": [tuple], "pred": [tuple], "n_invalid": int}. Returns a metrics dict.

    Raw tp/fp/fn counts are stored for every breakdown so any later grouping (head/tail, …) can be recomputed.
    """
    tot = collections.Counter()
    pairs = {k: collections.Counter() for k in ("aspect_category_polarity", "aspect_opinion", "category_polarity",
                                                 "aspect", "opinion")}
    keyf = {
        "aspect_category_polarity": lambda q: (q[0], q[1], q[3]),
        "aspect_opinion": lambda q: (q[1], q[2]),
        "category_polarity": lambda q: (q[0], q[3]),
        "aspect": lambda q: q[1],
        "opinion": lambda q: q[2],
    }
    by_cat, by_imp, by_nq, by_pol = (collections.defaultdict(collections.Counter) for _ in range(4))
    relaxed = collections.Counter()
    n_invalid = 0
    for r in records:
        g, p = set(map(tuple, r["gold"])), set(map(tuple, r["pred"]))
        n_invalid += r.get("n_invalid", 0)
        tot["tp"] += len(g & p); tot["fp"] += len(p - g); tot["fn"] += len(g - p)
        for k, f in keyf.items():
            gk, pk = {f(q) for q in g}, {f(q) for q in p}
            pairs[k]["tp"] += len(gk & pk); pairs[k]["fp"] += len(pk - gk); pairs[k]["fn"] += len(gk - pk)
        for groups, f in ((by_cat, lambda q: q[0]), (by_imp, implicit_type), (by_pol, lambda q: q[3])):
            for q in g:
                groups[f(q)]["tp" if q in p else "fn"] += 1
            for q in p - g:
                groups[f(q)]["fp"] += 1
        nq = str(min(len(g), 5)) + ("+" if len(g) >= 5 else "")
        by_nq[nq]["tp"] += len(g & p); by_nq[nq]["fp"] += len(p - g); by_nq[nq]["fn"] += len(g - p)
        # relaxed: same category and polarity, aspect and opinion overlap by at least one syllable (1-to-1 greedy)
        unmatched = list(p)
        for q in g:
            for i, x in enumerate(unmatched):
                if q[0] == x[0] and q[3] == x[3] and _overlap(q[1], x[1]) and _overlap(q[2], x[2]):
                    relaxed["tp"] += 1
                    unmatched.pop(i)
                    break
            else:
                relaxed["fn"] += 1
        relaxed["fp"] += len(unmatched)
    out = {"quad": prf(tot["tp"], tot["fp"], tot["fn"]), "relaxed_quad": prf(relaxed["tp"], relaxed["fp"], relaxed["fn"]),
           "n_sentences": len(records), "n_invalid_segments": n_invalid}
    out["pairs"] = {k: prf(v["tp"], v["fp"], v["fn"]) for k, v in pairs.items()}
    for name, groups in (("by_category", by_cat), ("by_implicit_type", by_imp), ("by_polarity", by_pol),
                         ("by_num_gold_quads", by_nq)):
        out[name] = {k: prf(v["tp"], v["fp"], v["fn"]) for k, v in sorted(groups.items())}
    return out
