"""QUADRA step 4b — keep augmented reviews whose quads a baseline model re-extracts.

    python src/aug/verify_model.py --ckpt outputs/verifier/seed_42/checkpoint \
        --in data/aug/quadra_gpt.jsonl --out data/aug/quadra_gpt_verified.jsonl --threshold 0.8

The verifier is Paraphrase (ViT5-base) trained on the original train set only (scripts/kaggle_quadra_prep.sh).
A review is kept if quad F1 between its intended quads and the re-extracted quads is >= threshold.
Acceptance per target cell is written to <out>.verify_report.json.
"""
import argparse
import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import load_jsonl, prf  # noqa: E402
from train_paraphrase import predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.ckpt).to(device)
    items = load_jsonl(a.inp)
    args = SimpleNamespace(eval_batch_size=a.batch_size, max_source_len=256, gen_max_len=384, num_beams=1)
    preds = predict(model, tok, items, args, device)
    kept, per_cell = [], collections.defaultdict(collections.Counter)
    with open(Path(a.out).with_suffix(".verify_preds.jsonl"), "w", encoding="utf-8") as f:
        for it, p in zip(items, preds):
            g, pr = {tuple(q) for q in p["gold"]}, {tuple(q) for q in p["pred"]}
            f1 = prf(len(g & pr), len(pr - g), len(g - pr))["f1"] / 100
            ok = f1 >= a.threshold
            c = it.get("meta", {}).get("target_cell", "?")
            per_cell[c]["kept" if ok else "rejected"] += 1
            f.write(json.dumps({"id": it["id"], "f1": f1, "kept": ok, "generation": p["generation"]}, ensure_ascii=False) + "\n")
            if ok:
                kept.append({**it, "meta": {**it.get("meta", {}), "verify_f1": f1}})
    with open(a.out, "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    rep = {"input": a.inp, "n_in": len(items), "n_kept": len(kept), "threshold": a.threshold, "ckpt": a.ckpt,
           "per_cell": {c: {**v, "acceptance": v["kept"] / (v["kept"] + v["rejected"])} for c, v in sorted(per_cell.items())}}
    json.dump(rep, open(Path(a.out).with_suffix(".verify_report.json"), "w"), indent=1)
    print(f"kept {len(kept)} / {len(items)} ({100 * len(kept) / max(1, len(items)):.1f}%) at F1 >= {a.threshold}")


if __name__ == "__main__":
    main()
