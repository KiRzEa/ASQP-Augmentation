"""Aggregate seeds of one or more run folders: mean ± sample std (ddof=1), recomputed from predictions.

    python src/summarize_runs.py outputs/para_none outputs/para_ada ... [--split test] [--json out.json]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import load_jsonl, score  # noqa: E402


def ms(xs):
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--json")
    a = ap.parse_args()
    table = {}
    for run in a.runs:
        seeds = sorted(Path(run).glob(f"seed_*/{a.split}_predictions.jsonl"))
        per_seed = {}
        for p in seeds:
            m = score(load_jsonl(p))  # recomputed from saved predictions, never from cached numbers
            per_seed[p.parent.name] = {"f1": m["quad"]["f1"], "precision": m["quad"]["precision"],
                                       "recall": m["quad"]["recall"], "relaxed_f1": m["relaxed_quad"]["f1"],
                                       **{f"f1_{k}": v["f1"] for k, v in m["by_implicit_type"].items()}}
        if not per_seed:
            print(f"{run}: no predictions"); continue
        keys = sorted({k for v in per_seed.values() for k in v})
        table[run] = {"n_seeds": len(per_seed), "per_seed": per_seed,
                      "mean_std": {k: ms([v[k] for v in per_seed.values() if k in v]) for k in keys}}
        s = table[run]["mean_std"]
        print(f"{Path(run).name:28s} n={len(per_seed)}  F1 {s['f1'][0]:6.2f} ± {s['f1'][1]:4.2f}  "
              f"P {s['precision'][0]:6.2f}  R {s['recall'][0]:6.2f}  relaxed {s['relaxed_f1'][0]:6.2f}  "
              + "  ".join(f"{k[3:]} {s[k][0]:5.1f}" for k in keys if k.startswith("f1_")))
    if a.json:
        json.dump(table, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
