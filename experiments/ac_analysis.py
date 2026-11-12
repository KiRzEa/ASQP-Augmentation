"""Round ac (English ACOS, pre-registered 2026-11-11): audit every run, then the confirmatory tests.
Per dataset: H1 M1 > M0, H2 M1 > M2; two-sided Welch t-test on per-seed test quad F1, Holm over the 2.
Secondary, uncorrected: P/R, implicit types, M2 vs M0, the 5-teacher vote ensemble on test (M0 seeds A).
    python3 experiments/ac_analysis.py
"""
import collections
import json
import statistics as S
import sys

sys.path.insert(0, "src")
from asqp_common import load_jsonl, score, quad_tuple  # noqa: E402
from scipy import stats  # noqa: E402

A, B = [42, 123, 2024, 7, 99], [11, 13, 17, 19, 23]
SIZES = {"rest": 583, "laptop": 816}
RUNS = {"none": A + B, "ens_pos": A, "pos": A + B}
NAME = {"none": "M0 plain", "ens_pos": "M1 method", "pos": "M2 text order"}


def holm(ps):
    order, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0] * len(ps), 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


P, bad = {}, []
for d, n in SIZES.items():
    test = load_jsonl(f"data/acos/{d}/test.jsonl")
    gold = {x["id"]: {quad_tuple(q) for q in x["quads"]} for x in test}
    for m, seeds in RUNS.items():
        for s in seeds:
            p = f"outputs/ac_{d}_{m}/seed_{s}"
            try:
                rows = load_jsonl(f"{p}/test_predictions.jsonl")
                a = json.load(open(f"{p}/run_config.json"))["args"]
            except FileNotFoundError:
                bad.append(f"missing {p}")
                continue
            ok = (len(rows) == n and {r["id"] for r in rows} == set(gold)
                  and all({tuple(q) for q in r["gold"]} == gold[r["id"]] for r in rows)
                  and a["test_path"] == f"data/acos/{d}/test.jsonl" and a["train_path"] == f"data/acos/{d}/train.jsonl"
                  and float(a["lr"]) == 1e-4 and a["model_name"] == "google-t5/t5-base" and a.get("lang") == "en"
                  and a.get("tuple_order") == ("gold" if m == "none" else "text")
                  and bool(a.get("aug_path")) == (m == "ens_pos")
                  and (m != "ens_pos" or f"ac_{d}_ens.jsonl" in str(a["aug_path"])))
            if not ok or sum(not r["pred"] for r in rows) > 0.02 * n:
                bad.append(f"bad {p}")
            P[(d, m, s)] = rows
print("AUDIT:", "clean" if not bad else bad, f"({len(P)} runs)")
if bad:
    sys.exit(1)

out = {}
for d in SIZES:
    M = {k[1:]: score(v) for k, v in P.items() if k[0] == d}
    by_id = [{r["id"]: r for r in P[(d, "none", s)]} for s in A]
    ens = []
    for r in P[(d, "none", 42)]:
        c = collections.Counter(q for t in by_id for q in {tuple(x) for x in t[r["id"]]["pred"]})
        ens.append({"gold": r["gold"], "pred": [list(q) for q, k in c.items() if k >= 3]})
    vote = score(ens)
    f = lambda m, k="f1": [M[(m, s)]["quad"][k] for s in RUNS[m]]  # noqa: E731
    imp = lambda m, t: S.mean(M[(m, s)]["by_implicit_type"].get(t, {"f1": 0})["f1"] for s in RUNS[m])  # noqa: E731
    print(f"\n=== {d} (test {SIZES[d]} reviews): quad F1 mean ± sd | P / R | implicit F1 EA-EO EA-IO IA-EO IA-IO")
    for m in RUNS:
        x = f(m)
        print(f"{NAME[m]:14s} n={len(x):2d} {S.mean(x):6.2f} ± {S.stdev(x):.2f} | P {S.mean(f(m, 'precision')):.1f} "
              f"R {S.mean(f(m, 'recall')):.1f} | " + " ".join(f"{imp(m, t):5.1f}" for t in ("EA-EO", "EA-IO", "IA-EO", "IA-IO"))
              + f" | {[round(v, 2) for v in x]}")
    q = vote["quad"]
    print(f"{'5-teacher vote':14s}       {q['f1']:6.2f}        | P {q['precision']:.1f} R {q['recall']:.1f}")
    H = [("H1", "ens_pos", "none"), ("H2", "ens_pos", "pos")]
    res = []
    for _, a, b in H:
        t = stats.ttest_ind(f(a), f(b), equal_var=False)
        res.append((S.mean(f(a)) - S.mean(f(b)), t.pvalue))
    print("PRIMARY (Welch two-sided, Holm over 2)")
    for (h, a, b), (dif, p), pa in zip(H, res, holm([p for _, p in res])):
        print(f"  {h}: {NAME[a]} - {NAME[b]} = {dif:+.2f}  p {p:.2e}  Holm {pa:.2e}  -> "
              f"{'REJECT H0' if pa < 0.05 else 'not rejected'}")
        out[f"{d}:{h}"] = {"diff": dif, "p": p, "holm": pa}
    t = stats.ttest_ind(f("pos"), f("none"), equal_var=False)
    print(f"  secondary: M2 - M0 = {S.mean(f('pos')) - S.mean(f('none')):+.2f}  p {t.pvalue:.4f} (uncorrected)")
    for k, v in M.items():
        out[f"{d}:{k[0]}:{k[1]}"] = v["quad"]
    out[f"{d}:teacher_vote5"] = q
json.dump(out, open("outputs/ac_quad_scores.json", "w"), indent=1)
