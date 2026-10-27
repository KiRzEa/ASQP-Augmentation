"""Quad-to-text augmentation with a fine-tuned ViT5 generator (GenDA-style). Needs a GPU: run on Kaggle.

    python -u src/aug/genda.py --out data/aug/genda.jsonl

1. Train ViT5-base to write a review from its quads (input = the marked-template target used by Paraphrase,
   output = the review text) on train; the epoch with the lowest dev loss is kept.
2. Generate from the sampled quad sets of src/aug/quadsets.py (seed 13: the same sets, in the same order, as
   the GPT quad2text baseline) with nucleus sampling, --num_return candidates per set; keep the first
   candidate in which every explicit span is found (relocate / grounded), until exactly --n reviews.
Writes <out>, <out>.report.json (generator card: dev loss, epochs, data hashes, versions, yield).
"""
import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import load_jsonl, quads_to_target  # noqa: E402
from quadra import relocate  # noqa: E402
from quadsets import sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="default |train|")
    ap.add_argument("--model_name", default="VietAI/vit5-base")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42, help="generator training seed")
    ap.add_argument("--set_seed", type=int, default=13, help="quad-set sampling seed (as llm_aug.py)")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--num_return", type=int, default=4)
    ap.add_argument("--gen_batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: tiny data")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    ap.add_argument("--dev_path", default=str(ROOT / "data/restaurant_v1/dev.jsonl"))
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else " — NO GPU"), flush=True)
    t0 = time.time()
    train, dev = load_jsonl(a.train_path), load_jsonl(a.dev_path)
    n = a.n or len(train)
    if a.limit:
        train, dev, n = train[:a.limit], dev[:a.limit], min(n, a.limit)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model_name).to(device)

    def collate(batch):
        enc = tok([quads_to_target(x["quads"]) for x in batch], max_length=384, truncation=True, padding=True, return_tensors="pt")
        lab = tok([x["text"] for x in batch], max_length=a.max_len, truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": lab}

    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(a.seed))
    dev_loader = DataLoader(dev, batch_size=32, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = get_linear_schedule_with_warmup(opt, 0, len(loader) * a.epochs)
    history, best, best_loss = [], None, float("inf")
    for ep in range(1, a.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            dl = float(np.mean([model(**{k: v.to(device) for k, v in b.items()}).loss.item() for b in dev_loader]))
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "dev_loss": dl})
        print(f"generator epoch {ep}: train loss {history[-1]['train_loss']:.4f}, dev loss {dl:.4f}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if dl < best_loss:
            best_loss, best = dl, {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    model.eval()
    best_epoch = min(history, key=lambda h: h["dev_loss"])["epoch"]

    sets = sample(load_jsonl(a.train_path), 4 * n, a.set_seed, prefix="genda")
    out, used, tried = [], 0, 0
    with torch.no_grad():
        while len(out) < n and used < len(sets):
            chunk = sets[used:used + a.gen_batch]
            used += len(chunk)
            enc = tok([quads_to_target(s["quads"]) for s in chunk], max_length=384, truncation=True, padding=True,
                      return_tensors="pt").to(device)
            gen = model.generate(**enc, do_sample=True, top_p=a.top_p, num_return_sequences=a.num_return,
                                 max_new_tokens=a.max_len)
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for i, s in enumerate(chunk):
                tried += 1
                for t in texts[i * a.num_return:(i + 1) * a.num_return]:
                    new = relocate(" ".join(t.split()), s)
                    if new:
                        out.append({"id": f"genda-{len(out):05d}", "text": new["text"], "quads": new["quads"],
                                    "meta": {"method": "genda", "source": s["template"], "set": s["id"]}})
                        break
                if len(out) >= n:
                    break
            if used % (a.gen_batch * 10) == 0:
                print(f"  generated {len(out)}/{n} from {tried} quad sets, {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    report = {"n": len(out), "quad_sets_tried": tried, "yield": len(out) / max(tried, 1),
              "quads": sum(len(it["quads"]) for it in out), "args": vars(a), "best_epoch": best_epoch,
              "best_dev_loss": best_loss, "history": history,
              "data_sha256": {p: sha(p) for p in (a.train_path, a.dev_path)},
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "transformers": __import__("transformers").__version__},
              "minutes": (time.time() - t0) / 60}
    json.dump(report, open(Path(a.out).with_suffix(".report.json"), "w"), indent=1)
    print(f"genda: {len(out)} reviews from {tried} quad sets (yield {report['yield']:.2f}), "
          f"generator best epoch {best_epoch}, {report['minutes']:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
