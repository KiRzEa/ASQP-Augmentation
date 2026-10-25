"""QUADRA step 4b — keep augmented reviews whose quads a baseline model re-extracts.

    python src/aug/verify_model.py --ckpt outputs/verifier/seed_42/checkpoint \
        --in data/aug/quadra_gpt.jsonl --out data/aug/quadra_gpt_verified.jsonl --threshold 0.8

The verifier is Paraphrase (ViT5-base) trained on the original train set only (scripts/kaggle_quadra_prep.sh).
A review is kept if quad F1 between its intended quads and the re-extracted quads is >= threshold.
Acceptance per target cell is written to <out>.verify_report.json.
"""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import load_jsonl, prf  # noqa: E402
from train_paraphrase import predict  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verifier_card(ckpt):
    """Everything needed to identify and report the verifier: weights hash + its training run's record."""
    ckpt = Path(ckpt)
    weights = next((ckpt / n for n in ("model.safetensors", "pytorch_model.bin") if (ckpt / n).exists()), None)
    card = {"ckpt_path": str(ckpt), "weights_file": weights.name if weights else None,
            "weights_sha256": sha256_file(weights) if weights else None}
    run_dir = ckpt.parent
    if (run_dir / "run_config.json").exists():
        c = json.load(open(run_dir / "run_config.json"))
        card.update({k: c.get(k) for k in ("seed", "best_epoch", "optimizer_steps", "epochs_run", "n_train", "n_aug",
                                           "data_sha256", "gpu", "versions", "seconds_total", "hostname")})
        card["train_args"] = {k: c["args"].get(k) for k in ("model_name", "lr", "batch_size", "epochs", "patience",
                                                             "train_path", "aug_path")}
    if (run_dir / "metrics.json").exists():
        m = json.load(open(run_dir / "metrics.json"))
        card.update({"dev_quad_f1": m["dev"]["quad"]["f1"], "test_quad_f1": m["test"]["quad"]["f1"],
                     "dev_relaxed_f1": m["dev"]["relaxed_quad"]["f1"]})
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--target_n", type=int, default=0,
                    help="after filtering, randomly keep exactly N reviews (same added amount for every method); 0 = keep all")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--card_only", action="store_true", help="only print the verifier card (no verification)")
    a = ap.parse_args()
    if a.card_only:
        print(json.dumps(verifier_card(a.ckpt), indent=1))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
    n_passed = len(kept)
    if a.target_n:
        if n_passed < a.target_n:
            print(f"WARNING: only {n_passed} reviews passed, fewer than --target_n {a.target_n}; generate more")
        else:
            kept = random.Random(a.seed).sample(kept, a.target_n)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    rep = {"verifier": verifier_card(a.ckpt), "input": a.inp, "input_sha256": sha256_file(a.inp),
           "output_sha256": sha256_file(a.out), "n_in": len(items), "n_passed": n_passed, "n_kept": len(kept), "target_n": a.target_n, "threshold": a.threshold, "ckpt": a.ckpt,
           "per_cell": {c: {**v, "acceptance": v["kept"] / (v["kept"] + v["rejected"])} for c, v in sorted(per_cell.items())}}
    json.dump(rep, open(Path(a.out).with_suffix(".verify_report.json"), "w"), indent=1)
    print(f"passed {n_passed} / {len(items)} ({100 * n_passed / max(1, len(items)):.1f}%) at F1 >= {a.threshold}; wrote {len(kept)}")


if __name__ == "__main__":
    main()
