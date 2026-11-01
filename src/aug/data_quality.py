"""Distribution drift of augmented / synthetic data against the training set (CPU, local).

    python src/aug/data_quality.py data/aug/llm_para.jsonl data/aug/syn20k_cons.jsonl ... [--json out.json]

Every candidate set is compared with train on the same measures, and dev (real held-out reviews) is compared with
train the same way: dev-vs-train is the natural gap between two real samples, so a candidate is "drifted" when it is
much further from train than dev is. Dev is only measured here, never used to build or select data.
Text: word-length distribution (KS statistic), unigram Jensen-Shannon divergence, share of tokens unseen in train,
distinct-2 and mean max word-trigram Jaccard to other reviews of the same set (diversity), comma / emoji / slang
shares, novelty (1 - mean max word-trigram Jaccard of a review with any training review other than an identical
one: 0 = copies of train), and a classifier two-sample test: cross-validated ROC-AUC of a char 2-4-gram logistic regression telling
the set from train (0.5 = indistinguishable).
Labels (if the set has quads): quads per review, JSD of the category, polarity and implicit-type distributions,
share of reviews with mixed polarity, mean opinion length, and the share of opinion spans among the 50 most frequent
training opinions ("stereotyped" labels: high = easier, more ideal data than the real reviews).
Size-dependent measures (unigram JSD, distinct-2, self overlap, C2ST) are averaged over --draws random subsets of
the dev size (221) of each set, against train subsets of the same size that exclude texts identical to the set's
texts (a copy of train would otherwise sit in both classes of the C2ST); rates use the whole set.
"""
import argparse
import collections
import json
import random
import warnings
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import NULL, implicit_type, load_jsonl, quad_tuple  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
SLANG = {"ko", "k", "dc", "đc", "j", "mn", "hok", "hông", "kh", "bt", "mik", "mk", "vs", "nc", "ng"}


def jsd(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    P = np.array([p.get(k, 0) / sp for k in keys]); Q = np.array([q.get(k, 0) / sq for k in keys])
    M = (P + Q) / 2

    def kl(a, b):
        m = a > 0
        return float(np.sum(a[m] * np.log2(a[m] / b[m])))
    return (kl(P, M) + kl(Q, M)) / 2


def tri(t):
    w = t.split()
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def novelty(texts, train_texts):
    sets = [tri(t) for t in train_texts]
    index = collections.defaultdict(list)
    for i, s in enumerate(sets):
        for g in s:
            index[g].append(i)
    vals = []
    for t in texts:
        s = tri(t)
        if not s:
            continue
        shared = collections.Counter(j for g in s for j in index.get(g, ()) if train_texts[j] != t)
        vals.append(max((n / len(s | sets[j]) for j, n in shared.items()), default=0.0))
    return 1 - statistics.mean(vals) if vals else 1.0


def self_overlap(texts):
    """Mean over reviews of the max word-trigram Jaccard with another review of the same set (inverted index)."""
    sets = [tri(t) for t in texts]
    index = collections.defaultdict(list)
    for i, s in enumerate(sets):
        for g in s:
            index[g].append(i)
    vals = []
    for i, s in enumerate(sets):
        if not s:
            vals.append(0.0)
            continue
        shared = collections.Counter(j for g in s for j in index[g] if j != i)
        vals.append(max((n / len(s | sets[j]) for j, n in shared.items()), default=0.0))
    return statistics.mean(vals)


def c2st(texts, ref, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    X = texts + ref
    y = [1] * len(texts) + [0] * len(ref)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    return float(np.mean(cross_val_score(make_pipeline(vec, clf), X, y, scoring="roc_auc",
                                         cv=StratifiedKFold(5, shuffle=True, random_state=seed))))


def sized(texts, train_texts, m, rng, seed):
    """Size-dependent measures on a subset of m texts vs m train texts not identical to any of the set's texts."""
    sub = rng.sample(texts, min(m, len(texts)))
    own = set(texts)
    pool = [t for t in train_texts if t not in own]
    ref = rng.sample(pool if len(pool) >= len(sub) else train_texts, len(sub))
    uni, ref_uni = collections.Counter(w for t in sub for w in t.split()), collections.Counter(w for t in ref for w in t.split())
    bigr = [(t.split()[i], t.split()[i + 1]) for t in sub for i in range(len(t.split()) - 1)]
    return {"unigram_JSD": jsd(uni, ref_uni), "distinct2": len(set(bigr)) / max(len(bigr), 1),
            "self_trigram_overlap": self_overlap(sub),
            "C2ST_AUC": c2st(sub, ref, seed) if len(pool) >= len(sub) else None}  # a copy of train: undefined


def measure(items, train, train_vocab, top_ops, a, m, rng):
    texts = [x["text"] for x in items]
    toks = [w for t in texts for w in t.split()]
    train_texts = [x["text"] for x in train]
    draws = [sized(texts, train_texts, m, rng, a.seed + d) for d in range(a.draws)]
    m = {"n": len(items),
         "words_mean": statistics.mean(len(t.split()) for t in texts),
         "len_KS": ks_2samp([len(t.split()) for t in texts], [len(x["text"].split()) for x in train]).statistic,
         **{k: (statistics.mean(d[k] for d in draws) if draws[0][k] is not None else None) for k in draws[0]},
         "unseen_token_share": sum(1 for w in toks if w not in train_vocab) / max(len(toks), 1),
         "novelty": novelty(rng.sample(texts, min(1000, len(texts))), train_texts),
         "comma_share": sum("," in t for t in texts) / len(texts),
         "emoji_share": sum(bool(EMOJI.search(t)) for t in texts) / len(texts),
         "slang_share": sum(any(w in t.split() for w in SLANG) for t in texts) / len(texts)}
    if any(x.get("quads") for x in items):
        qs = [quad_tuple(q) for x in items for q in x["quads"]]
        tq = [quad_tuple(q) for x in train for q in x["quads"]]
        ops = [q[2] for q in qs if q[2] != NULL]
        m.update({
            "quads_per_review": len(qs) / len(items),
            "category_JSD": jsd(collections.Counter(q[0] for q in qs), collections.Counter(q[0] for q in tq)),
            "polarity_JSD": jsd(collections.Counter(q[3] for q in qs), collections.Counter(q[3] for q in tq)),
            "implicit_JSD": jsd(collections.Counter(map(implicit_type, qs)), collections.Counter(map(implicit_type, tq))),
            "implicit_share": sum(implicit_type(q) != "EA-EO" for q in qs) / max(len(qs), 1),
            "multi_quad_share": sum(len(x["quads"]) > 1 for x in items) / len(items),
            "mixed_polarity_share": sum(len({q["polarity"] for q in x["quads"]}) > 1 for x in items) / len(items),
            "opinion_words": statistics.mean(len(o.split()) for o in ops) if ops else 0.0,
            "stereotyped_opinion_share": sum(o in top_ops for o in ops) / max(len(ops), 1)})
    return m


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="JSONL files (augmented / synthetic data)")
    ap.add_argument("--draws", type=int, default=5, help="random subsets for the size-dependent measures")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--json", help="write all measures here")
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    ap.add_argument("--dev_path", default=str(ROOT / "data/restaurant_v1/dev.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    train = load_jsonl(a.train_path)
    train_vocab = {w for x in train for w in x["text"].split()}
    top_ops = {o for o, _ in collections.Counter(q["opinion"] for x in train for q in x["quads"]
                                                  if q["opinion"] != NULL).most_common(50)}
    dev = load_jsonl(a.dev_path)
    m = len(dev)
    rows = {}
    # reference rows: dev vs train = the natural gap between two real samples; a train half vs the other half = floor
    half_ids = set(rng.sample(range(len(train)), len(train) // 2))
    half, rest = [x for i, x in enumerate(train) if i in half_ids], [x for i, x in enumerate(train) if i not in half_ids]
    rows["(dev, real)"] = measure(dev, train, train_vocab, top_ops, a, m, rng)
    rows["(train half)"] = measure(half, rest, {w for x in rest for w in x["text"].split()}, top_ops, a, m, rng)
    for p in a.sets:
        rows[Path(p).stem] = measure(load_jsonl(p), train, train_vocab, top_ops, a, m, rng)
    cols = list(dict.fromkeys(k for r in rows.values() for k in r))
    w = max(len(k) for k in rows)
    print(" " * w + " " + " ".join(f"{c[:12]:>12s}" for c in cols))
    for k, r in rows.items():
        print(f"{k:{w}s} " + " ".join(f"{r[c]:12.3f}" if isinstance(r.get(c), float) else f"{'' if r.get(c) is None else r[c]:>12}" for c in cols))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
