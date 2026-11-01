#!/usr/bin/env python3
"""What makes a quad or a review hard for the model on unseen data, measured on train out-of-fold predictions.

    python3 scripts/error_factors.py [--oof outputs/kfold] [--json out.json]

The out-of-fold (OOF) predictions of the 5-fold runs (outputs/kfold/fold<k>/seed_42/test_predictions.jsonl) are
predictions on training reviews the model did not see, so they show what does not transfer from the training part
to new reviews without touching dev or test. Every factor is computed against the training part of the fold
(seen terms, seen pairs, category ambiguity). For each bucket: OOF F1 and the share of gold quads (or reviews)
in it; the share of the same bucket in dev and test is printed as description only (no model output of dev/test).
Quad factors: (aspect, opinion) pair seen, aspect/opinion term seen, aspect category ambiguity, span lengths,
words between aspect and opinion, opinion before aspect, aspect with several opinions in the review.
Review factors: length, number of quads, slang (abbreviations other than "ok"), emoji / emoticons, missing
diacritics (a plain word whose accented form is >= 5x more frequent in train, e.g. "qua" for "quá"), English words,
commas. Style factors correlate with review length; --by_length compares them within length buckets. Also annotation consistency of train: the same (aspect, opinion) pair with different category or polarity.
"""
import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from asqp_common import NULL, implicit_type, load_jsonl, quad_tuple  # noqa: E402

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
SLANG = {"ko", "k", "dc", "đc", "j", "mn", "hok", "hông", "kh", "bt", "mik", "mk", "vs", "nc", "ng", "nv", "ok", "oke"}
ENGLISH = {"ok", "oke", "good", "nice", "view", "menu", "service", "staff", "free", "size", "order", "check", "in",
           "chill", "style", "decor", "wifi", "sale", "combo", "topping", "set", "buffet", "food", "drink", "price",
           "the", "is", "very", "so", "and", "but", "not", "ship", "fan", "vibe", "cool", "best", "top"}


def strip_marks(w):
    return "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn").replace("đ", "d")


class Diacritics:
    """Plain words written without the diacritics their usual (>= 5x more frequent) accented form in train has."""

    def __init__(self, train):
        self.vocab = collections.Counter(w for x in train for w in x["text"].split())
        self.accented = collections.defaultdict(collections.Counter)
        for w, n in self.vocab.items():
            if strip_marks(w) != w:
                self.accented[strip_marks(w)][w] += n

    def missing(self, w):
        if not (w.isascii() and w.isalpha()) or w in ENGLISH or w not in self.accented:
            return False
        top = self.accented[w].most_common(1)[0][1]
        return top >= 5 and top >= 5 * self.vocab[w]


def bucket_num(v, edges, unit=""):
    for e in edges:
        if v <= e:
            return f"<= {e}{unit}"
    return f"> {edges[-1]}{unit}"


def counts(items):
    c = {"A": collections.Counter(), "O": collections.Counter(), "pair": collections.Counter(),
         "acat": collections.defaultdict(collections.Counter)}
    for x in items:
        for q in x["quads"]:
            c["A"][q["aspect"]] += 1; c["O"][q["opinion"]] += 1; c["pair"][(q["aspect"], q["opinion"])] += 1
            c["acat"][q["aspect"]][q["category"]] += 1
    return c


def quad_factors(q, item, c):
    """Factors of one gold or predicted quad tuple; offsets come from the gold quad of the item when it matches."""
    cat, a, o, pol = q
    f = {}
    f["pair (A,O) seen"] = "seen" if c["pair"][(a, o)] else "unseen"
    if a != NULL:
        f["aspect term"] = "unseen" if c["A"][a] == 0 else "1-2x" if c["A"][a] <= 2 else ">=3x"
        cats = c["acat"].get(a)
        if cats:
            top = cats.most_common(1)[0][1] / sum(cats.values())
            f["aspect category ambiguity"] = "one category" if top == 1 else ">= 80% one" if top >= 0.8 else "mixed"
        f["aspect words"] = bucket_num(len(a.split()), [1, 2, 3])
    if o != NULL:
        f["opinion term"] = "unseen" if c["O"][o] == 0 else "1-2x" if c["O"][o] <= 2 else ">=3x"
        f["opinion words"] = bucket_num(len(o.split()), [1, 2, 3])
    g = next((g for g in item["quads"] if quad_tuple(g) == q), None)
    if g and g.get("aspect_offsets") and g.get("opinion_offsets"):
        (s1, e1), (s2, e2) = g["aspect_offsets"], g["opinion_offsets"]
        gap = item["text"][min(e1, e2):max(s1, s2)] if max(s1, s2) > min(e1, e2) else ""
        f["words between A and O"] = bucket_num(len(gap.split()), [0, 2, 5])
        f["order"] = "opinion before aspect" if s2 < s1 else "aspect before opinion"
    f["implicit type"] = implicit_type(q)
    n_op = sum(1 for g2 in item["quads"] if g2["aspect"] == a and a != NULL)
    if a != NULL:
        f["opinions of this aspect in review"] = "1" if n_op <= 1 else ">= 2"
    return f


def review_factors(x, dia):
    w = x["text"].split()
    return {"review words": bucket_num(len(w), [12, 20, 30, 45]),
            "quads in review": bucket_num(len(x["quads"]), [2, 3, 4]),
            "slang": "yes" if any(t in SLANG - {"ok", "oke"} for t in w) else "no",
            "emoji / emoticon": "yes" if EMOJI.search(x["text"]) or re.search(r"[:;]\)|:d|:v|=\)", x["text"]) else "no",
            "missing diacritics": "yes" if any(dia.missing(t) for t in w) else "no",
            "english words": "yes" if any(t in ENGLISH for t in w) else "no",
            "commas": "yes" if "," in x["text"] else "no"}


def f1(v):
    tp, fp, fn = v
    return 200 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(ROOT / "outputs/kfold"))
    ap.add_argument("--json")
    ap.add_argument("--by_length", action="store_true", help="style factors within review-length buckets")
    ap.add_argument("--exclude_fold", default="", help="e.g. fold0: leave the inner dev fold out of the analysis")
    a = ap.parse_args()
    train = load_jsonl(ROOT / "data/restaurant_v1/train.jsonl")
    dia = Diacritics(train)
    by_id = {x["id"]: x for x in train}
    dev = load_jsonl(ROOT / "data/restaurant_v1/dev.jsonl"); test = load_jsonl(ROOT / "data/restaurant_v1/test.jsonl")
    folds = sorted(Path(a.oof).glob("fold*/seed_*/test_predictions.jsonl"))
    if a.exclude_fold:
        folds = [f for f in folds if f.parts[-3] not in a.exclude_fold.split(",")]
    qf, rf = collections.defaultdict(lambda: [0, 0, 0]), collections.defaultdict(lambda: [0, 0, 0])
    gold_n, rev_n = collections.Counter(), collections.Counter()
    for fp_ in folds:
        rows = load_jsonl(fp_)
        held = {r["id"] for r in rows}
        c = counts([x for x in train if x["id"] not in held])
        for r in rows:
            item = by_id[r["id"]]
            g, p = {tuple(q) for q in r["gold"]}, {tuple(q) for q in r["pred"]}
            for q in g | p:
                for k, v in quad_factors(q, item, c).items():
                    s = qf[(k, v)]
                    s[0] += q in g and q in p; s[1] += q in p and q not in g; s[2] += q in g and q not in p
                    gold_n[(k, v)] += q in g
            rfs = review_factors(item, dia)
            if a.by_length:
                L = rfs["review words"]
                rfs.update({f"{k} [{L} words]": v for k, v in rfs.items() if k in ("slang", "emoji / emoticon",
                                                                                   "missing diacritics", "english words")})
            for k, v in rfs.items():
                s = rf[(k, v)]
                s[0] += len(g & p); s[1] += len(p - g); s[2] += len(g - p); rev_n[(k, v)] += 1
    # descriptive shares in dev / test (gold only, factors against the full train)
    call = counts(train)

    def shares(items):
        qn, rn = collections.Counter(), collections.Counter()
        for x in items:
            for q in x["quads"]:
                for k, v in quad_factors(quad_tuple(q), x, call).items():
                    qn[(k, v)] += 1
            for k, v in review_factors(x, dia).items():
                rn[(k, v)] += 1
        return qn, rn
    (dq, dr), (tq, trr) = shares(dev), shares(test)
    nq_tr = sum(len(x["quads"]) for x in train)
    nq = {"dev": sum(len(x["quads"]) for x in dev), "test": sum(len(x["quads"]) for x in test)}
    out = {"quad": {}, "review": {}}
    print(f"{'quad factor':34s} {'bucket':24s} {'OOF F1':>7s} {'train%':>7s} {'dev%':>6s} {'test%':>6s}")
    for k, v in sorted(qf, key=lambda kv: (kv[0], kv[1])):
        row = {"oof_f1": f1(qf[(k, v)]), "train_share": gold_n[(k, v)] / nq_tr,
               "dev_share": dq[(k, v)] / nq["dev"], "test_share": tq[(k, v)] / nq["test"]}
        out["quad"][f"{k} | {v}"] = row
        print(f"{k:34s} {v:24s} {row['oof_f1']:7.1f} {100 * row['train_share']:7.1f} {100 * row['dev_share']:6.1f} {100 * row['test_share']:6.1f}")
    print(f"\n{'review factor':34s} {'bucket':24s} {'OOF F1':>7s} {'train%':>7s} {'dev%':>6s} {'test%':>6s}")
    for k, v in sorted(rf, key=lambda kv: (kv[0], kv[1])):
        row = {"oof_f1": f1(rf[(k, v)]), "train_share": rev_n[(k, v)] / len(train),
               "dev_share": dr[(k, v)] / len(dev), "test_share": trr[(k, v)] / len(test)}
        out["review"][f"{k} | {v}"] = row
        print(f"{k:34s} {v:24s} {row['oof_f1']:7.1f} {100 * row['train_share']:7.1f} {100 * row['dev_share']:6.1f} {100 * row['test_share']:6.1f}")
    # annotation consistency inside train
    lab = collections.defaultdict(collections.Counter)
    for x in train:
        for q in x["quads"]:
            if q["aspect"] != NULL and q["opinion"] != NULL:
                lab[(q["aspect"], q["opinion"])][(q["category"], q["polarity"])] += 1
    multi = {k: v for k, v in lab.items() if sum(v.values()) >= 2}
    incons = {k: v for k, v in multi.items() if len(v) > 1}
    pol_incons = {k: v for k, v in multi.items() if len({p for _, p in v}) > 1}
    print(f"\ntrain annotation consistency: {len(multi)} explicit (aspect, opinion) pairs occur >= 2 times; "
          f"{len(incons)} ({100 * len(incons) / len(multi):.1f}%) get more than one (category, polarity), "
          f"{len(pol_incons)} ({100 * len(pol_incons) / len(multi):.1f}%) more than one polarity")
    for k, v in sorted(pol_incons.items(), key=lambda kv: -sum(kv[1].values()))[:8]:
        print(f"   {k}: {dict(v)}")
    out["consistency"] = {"pairs_ge2": len(multi), "label_inconsistent": len(incons), "polarity_inconsistent": len(pol_incons)}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
