#!/usr/bin/env python3
"""Slice F1 of runs: did an augmentation fix what it was designed for? (mean over seeds, recomputed from predictions)

    python3 scripts/slice_report.py outputs/dv_none outputs/dv_infill_t ... --ref data/restaurant_v1/inner_train.jsonl

Slices are the factors of scripts/error_factors.py (seen/unseen (aspect, opinion) pair and terms, implicit type,
order, distance, polarity), computed against --ref (the training set of the runs: inner_train for pilots).
Predictions are read from <run>/seed_*/test_predictions.jsonl; for pilots that file holds inner_dev predictions
(the official test is never decoded for pilots). Offsets come from --gold (the gold file of the predicted split).
"""
import argparse
import collections
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from asqp_common import load_jsonl  # noqa: E402
from error_factors import counts, f1, quad_factors  # noqa: E402

KEEP = ("pair (A,O) seen", "aspect term", "opinion term", "implicit type", "order", "words between A and O")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=str(ROOT / "data/restaurant_v1/inner_train.jsonl"))
    ap.add_argument("--gold", default=str(ROOT / "data/restaurant_v1/inner_dev.jsonl"))
    a = ap.parse_args()
    c = counts(load_jsonl(a.ref))
    gold = {x["id"]: x for x in load_jsonl(a.gold)}
    table = {}
    for run in a.runs:
        per_seed = []
        for p in sorted(Path(run).glob("seed_*/test_predictions.jsonl")):
            s = collections.defaultdict(lambda: [0, 0, 0])
            for r in load_jsonl(p):
                g, pr = {tuple(q) for q in r["gold"]}, {tuple(q) for q in r["pred"]}
                s[("all", "")][0] += len(g & pr); s[("all", "")][1] += len(pr - g); s[("all", "")][2] += len(g - pr)
                for q in g | pr:
                    for k, v in quad_factors(q, gold[r["id"]], c).items():
                        if k in KEEP:
                            t = s[(k, v)]
                            t[0] += q in g and q in pr; t[1] += q in pr and q not in g; t[2] += q in g and q not in pr
                    t = s[("polarity", q[3])]
                    t[0] += q in g and q in pr; t[1] += q in pr and q not in g; t[2] += q in g and q not in pr
            per_seed.append({k: f1(v) for k, v in s.items()})
        table[Path(run).name] = (len(per_seed), {k: statistics.mean(d.get(k, 0) for d in per_seed) for k in per_seed[0]})
    keys = sorted({k for _, d in table.values() for k in d})
    names = list(table)
    print(f"{'slice':48s} " + " ".join(f"{n[:14]:>14s}" for n in names))
    print(f"{'(seeds)':48s} " + " ".join(f"{table[n][0]:>14d}" for n in names))
    for k in keys:
        print(f"{(k[0] + ' | ' + k[1]).strip(' |'):48s} " + " ".join(f"{table[n][1].get(k, 0):14.2f}" for n in names))


if __name__ == "__main__":
    main()
