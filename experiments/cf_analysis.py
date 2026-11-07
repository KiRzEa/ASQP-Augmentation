import json, sys, collections, statistics as S
sys.path.insert(0, "src")
from asqp_common import load_jsonl, score, quad_tuple
from scipy import stats
A, B = [42, 123, 2024, 7, 99], [11, 13, 17, 19, 23]
RUNS = {"cf_none": A + B, "cf_pos": A + B, "cf_ens_full_pos": A, "cf_ada_joint": A, "cf_concat": A, "cf_eda": A,
        "cf_genda": A, "cf_rc": A, "cf_mvp": A, "cf_single_full_pos": A, "cf_ens_full": [42, 123, 2024]}
test = load_jsonl("data/restaurant_v1/test.jsonl")
gold = {x["id"]: {quad_tuple(q) for q in x["quads"]} for x in test}
P, bad = {}, []
for run, seeds in RUNS.items():
    for s in seeds:
        d = f"outputs/{run}/seed_{s}"
        try:
            rows = load_jsonl(f"{d}/test_predictions.jsonl")
        except FileNotFoundError:
            bad.append(f"missing {d}"); continue
        a = json.load(open(f"{d}/run_config.json"))["args"]
        ok = (len(rows) == 583 and {r["id"] for r in rows} == set(gold)
              and all({tuple(q) for q in r["gold"]} == gold[r["id"]] for r in rows)
              and str(a["test_path"]).endswith("test.jsonl") and float(a["lr"]) == 1e-4)
        if not ok or sum(not r["pred"] for r in rows) > 5: bad.append(f"bad {d}")
        P[(run, s)] = rows
for m in ("uniform", "greedy"):
    rows = load_jsonl(f"outputs/cf_ens_full_pos_soup/{m}/test_predictions.jsonl")
    if len(rows) != 583: bad.append(f"soup {m}")
    P[("soup_" + m, 0)] = rows
print("AUDIT:", "clean" if not bad else bad, f"({len(P)} prediction files)")
if bad: sys.exit(1)
# teacher ensemble: quads predicted by >= 3 of the 5 teachers (cf_none seeds A), per test review
by_id = [{r["id"]: r for r in P[("cf_none", s)]} for s in A]
ens = []
for r in P[("cf_none", 42)]:
    c = collections.Counter(tuple(q) for t in by_id for q in {tuple(x) for x in t[r["id"]]["pred"]})
    ens.append({"gold": r["gold"], "pred": [list(q) for q, n in c.items() if n >= 3]})
M = {k: score(v) for k, v in P.items()}
M[("teacher_vote5", 0)] = score(ens)
f = lambda run, k="f1": [M[(run, s)]["quad"][k] for s in RUNS[run]]
imp = lambda run, t: S.mean(M[(run, s)]["by_implicit_type"].get(t, {"f1": 0})["f1"] for s in RUNS[run])
print("\nTEST quad F1 (mean ± sd over seeds), P / R, implicit F1 EA-EO EA-IO IA-EO IA-IO")
for run in RUNS:
    x = f(run)
    print(f"{run:20s} n={len(x):2d} {S.mean(x):6.2f} ± {S.stdev(x):.2f} | P {S.mean(f(run,'precision')):.1f} R {S.mean(f(run,'recall')):.1f} | "
          + " ".join(f"{imp(run,t):5.1f}" for t in ("EA-EO", "EA-IO", "IA-EO", "IA-IO")) + f" | seeds {[round(v,2) for v in x]}")
for k in ("soup_uniform", "soup_greedy", "teacher_vote5"):
    q = M[(k, 0)]["quad"]; print(f"{k:20s}       {q['f1']:6.2f}        | P {q['precision']:.1f} R {q['recall']:.1f}")
sm = json.load(open("outputs/cf_ens_full_pos_soup/greedy/soup.json")); print("greedy soup members", len(sm["members"]), "of", len(sm["ingredients"]))
ing = [M[("cf_ens_full_pos", s)]["quad"]["f1"] for s in (2024, 7, 99)]; print("soup ingredient mean (2024/7/99)", round(S.mean(ing), 2))
def welch(a, b):
    x, y = f(a), f(b); t = stats.ttest_ind(x, y, equal_var=False); return S.mean(x) - S.mean(y), t.pvalue
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj, m, run = [0]*len(ps), len(ps), 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj
print("\nPRIMARY (Holm over 2, Welch two-sided)")
H = [("H1", "cf_ens_full_pos", "cf_none"), ("H2", "cf_ens_full_pos", "cf_pos")]
res = [welch(a, b) for _, a, b in H]
for (h, a, b), (d, p), pa in zip(H, res, holm([p for _, p in res])):
    print(f"{h}: {a} - {b} = {d:+.2f}  p {p:.2e}  Holm {pa:.2e}  -> {'REJECT H0' if pa < 0.05 else 'not rejected'}")
print("\nABLATIONS (secondary family, Holm over 2, vs M1)")
AB = [("A1", "cf_single_full_pos"), ("A2", "cf_ens_full")]
res = [welch(r, "cf_ens_full_pos") for _, r in AB]
for (h, r), (d, p), pa in zip(AB, res, holm([p for _, p in res])):
    print(f"{h}: {r} - M1 = {d:+.2f}  p {p:.4f}  Holm {pa:.4f}")
print("\nSECONDARY, uncorrected: M1 vs each literature baseline; and each baseline vs plain")
for r in ("cf_ada_joint", "cf_concat", "cf_eda", "cf_genda", "cf_rc", "cf_mvp"):
    d, p = welch("cf_ens_full_pos", r); d0, p0 = welch(r, "cf_none")
    print(f"M1 - {r:13s} {d:+.2f} p {p:.4f} | {r} - none {d0:+.2f} p {p0:.4f}")
d, p = welch("cf_pos", "cf_none"); print(f"text order alone - none {d:+.2f} p {p:.4f}")
d, p = welch("cf_single_full_pos", "cf_none"); print(f"single teacher - none {d:+.2f} p {p:.4f}")
json.dump({f"{k[0]}:{k[1]}": v["quad"] for k, v in M.items()}, open("outputs/cf_quad_scores.json", "w"), indent=1)
