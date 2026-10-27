"""Select exactly N augmented reviews from QUADRA candidates (same added amount for every method).

Uses the verifier predictions saved by verify_model.py (<out>.verify_preds.jsonl), so every selection mode
relies on the same verifier and needs no GPU.

    python src/aug/select_aug.py --in data/aug/quadra_gpt.jsonl --mode none   --out data/aug/quadra_gpt_none.jsonl
    python src/aug/select_aug.py --in data/aug/quadra_gpt.jsonl --mode target --preds data/aug/quadra_gpt_v.verify_preds.jsonl \
        --out data/aug/quadra_gpt_target.jsonl

Modes
  none    no model verification (rule checks were already applied when generating); sample N stratified by
          target cell, so the per-cell budget is preserved.
  target  keep a review if the verifier recovers at least one quad of its TARGET cell exactly (the cell the
          review was generated for); other quads may be missed. Sample N stratified by target cell among the
          accepted; if a cell has too few accepted, fill from its rejected reviews with the highest quad F1.
  f1      the s2 filter (verifier quad F1 >= --threshold over all quads), stratified, filled the same way.
A report with per-cell budget, acceptance and composition is written to <out>.select_report.json.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import load_jsonl, quad_tuple, target_to_quads  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["none", "target", "f1"], required=True)
    ap.add_argument("--preds", help="verify_preds.jsonl from verify_model.py (modes target, f1)")
    ap.add_argument("--n", type=int, default=0, help="number to keep; default = size of train")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    items = load_jsonl(a.inp)
    n = a.n or sum(1 for _ in open(a.train_path, encoding="utf-8"))
    preds = {p["id"]: p for p in load_jsonl(a.preds)} if a.preds else {}
    if a.mode != "none" and not preds:
        sys.exit("--preds is required for modes target and f1")

    def score(it):
        """(accepted, quad F1, target recovered) of one candidate under the chosen mode."""
        if a.mode == "none":
            return True, None, None
        p = preds[it["id"]]
        pred, _ = target_to_quads(p["generation"])
        gold = {quad_tuple(q) for q in it["quads"]}
        tc = it["meta"]["target_cell"]
        target_hit = any(f"{q[0]}|{q[3]}" == tc and q in pred for q in gold)
        f1 = p["f1"]
        return (target_hit if a.mode == "target" else f1 >= a.threshold), f1, target_hit

    by_cell = collections.defaultdict(list)
    for it in items:
        acc, f1, hit = score(it)
        by_cell[it["meta"]["target_cell"]].append((it, acc, f1 if f1 is not None else 0.0, hit))
    # per-cell quota proportional to the generated budget (which already encodes frequency x difficulty)
    total = len(items)
    quota = {c: int(n * len(v) / total) for c, v in by_cell.items()}
    for c in sorted(by_cell, key=lambda c: -(n * len(by_cell[c]) / total - quota[c]))[: n - sum(quota.values())]:
        quota[c] += 1
    chosen, report = [], {}
    for c in sorted(by_cell):
        cands = by_cell[c]
        acc = [x for x in cands if x[1]]
        rej = sorted([x for x in cands if not x[1]], key=lambda x: -x[2])
        rng.shuffle(acc)
        take = acc[:quota[c]]
        filled = rej[:quota[c] - len(take)]
        chosen += [x[0] for x in take + filled]
        report[c] = {"generated": len(cands), "accepted": len(acc), "acceptance": len(acc) / len(cands),
                     "quota": quota[c], "kept_accepted": len(take), "kept_filled_from_rejected": len(filled)}
    rng.shuffle(chosen)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in chosen:
            f.write(json.dumps({**it, "meta": {**it["meta"], "select_mode": a.mode}}, ensure_ascii=False) + "\n")
    n_acc = sum(r["accepted"] for r in report.values())
    n_fill = sum(r["kept_filled_from_rejected"] for r in report.values())
    summary = {"input": a.inp, "preds": a.preds, "mode": a.mode, "threshold": a.threshold if a.mode == "f1" else None,
               "n_candidates": total, "n_accepted": n_acc, "n_written": len(chosen), "n_filled_from_rejected": n_fill,
               "quads_written": sum(len(it["quads"]) for it in chosen), "per_cell": report}
    json.dump(summary, open(Path(a.out).with_suffix(".select_report.json"), "w"), indent=1)
    print(f"{a.mode}: accepted {n_acc}/{total} ({100 * n_acc / total:.1f}%), wrote {len(chosen)} "
          f"({n_fill} filled from rejected), {summary['quads_written']} quads -> {a.out}")


if __name__ == "__main__":
    main()
