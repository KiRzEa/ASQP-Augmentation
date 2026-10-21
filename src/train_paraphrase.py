"""Paraphrase (Zhang et al., 2021) ASQP model on ViT5-base, marked-template target, one or more seeds.

Every augmentation method is evaluated with this same script: only --aug_path changes.

    python src/train_paraphrase.py --output_dir outputs/para_none --seeds 42,123,2024,7,99
    python src/train_paraphrase.py --output_dir outputs/para_ada --aug_path data/aug/ada.jsonl

Per seed, <output_dir>/seed_<s>/ gets:
  test_predictions.jsonl, dev_predictions.jsonl  id, text, gold, pred, raw generation, invalid segments
  metrics.json      dev/test metrics of the selected checkpoint (exact + relaxed F1, breakdowns, raw counts)
  history.json      per epoch: train loss, dev F1, learning rate, seconds
  run_config.json   all arguments, data hashes and sizes, package versions, GPU, timings, peak memory,
                    best epoch, optimiser steps, truncation counts
  train.log
A seed whose metrics.json and test_predictions.jsonl exist is skipped (resume). Checkpoints are not kept
unless --keep_ckpt.
"""
import argparse
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import load_jsonl, quad_tuple, quads_to_target, score, target_to_quads  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    ap.add_argument("--aug_path", action="append", default=[], help="augmented JSONL added to train; repeatable")
    ap.add_argument("--dev_path", default=str(ROOT / "data/restaurant_v1/dev.jsonl"))
    ap.add_argument("--test_path", default=str(ROOT / "data/restaurant_v1/test.jsonl"))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seeds", default="42,123,2024,7,99")
    ap.add_argument("--model_name", default="VietAI/vit5-base")
    # Paraphrase defaults (Zhang et al., 2021): AdamW, lr 3e-4, batch 16, 20 epochs, linear schedule
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=0, help="if > 0, train exactly this many optimiser steps (step-matched control)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--max_source_len", type=int, default=256)
    ap.add_argument("--max_target_len", type=int, default=384)
    ap.add_argument("--gen_max_len", type=int, default=384)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--patience", type=int, default=5, help="early stop after N epochs without dev F1 gain; 0 = off")
    ap.add_argument("--eval_every", type=int, default=1, help="evaluate on dev every N epochs")
    ap.add_argument("--fp16", action="store_true", help="mixed precision (T4 has no bf16; T5 can overflow in fp16)")
    ap.add_argument("--keep_ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only the first N examples of each split")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_collate(tok, args):
    def collate(batch):
        enc = tok([b["text"] for b in batch], max_length=args.max_source_len, truncation=True, padding=True,
                  return_tensors="pt")
        lab = tok([b["target"] for b in batch], max_length=args.max_target_len, truncation=True, padding=True,
                  return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": lab}
    return collate


@torch.no_grad()
def predict(model, tok, items, args, device):
    model.eval()
    out = []
    for i in range(0, len(items), args.eval_batch_size):
        chunk = items[i:i + args.eval_batch_size]
        enc = tok([x["text"] for x in chunk], max_length=args.max_source_len, truncation=True, padding=True,
                  return_tensors="pt").to(device)
        gen = model.generate(**enc, max_new_tokens=args.gen_max_len, num_beams=args.num_beams)
        for x, g in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
            pred, n_invalid = target_to_quads(g)
            out.append({"id": x["id"], "text": x["text"], "gold": [list(quad_tuple(q)) for q in x["quads"]],
                        "pred": [list(q) for q in sorted(pred)], "generation": g, "n_invalid": n_invalid})
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_seed(seed, args, data, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"seed{seed}")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(out_dir / "train.log", mode="w", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(logging.Formatter(f"%(asctime)s [seed {seed}] %(message)s"))
        log.addHandler(h)
    set_seed(seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    train = [{"text": x["text"], "target": quads_to_target(x["quads"])} for x in data["train_all"]]
    src_trunc = sum(len(tok(x["text"]).input_ids) > args.max_source_len for x in train)
    tgt_trunc = sum(len(tok(x["target"]).input_ids) > args.max_target_len for x in train)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=gen, collate_fn=make_collate(tok, args))
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    epochs = args.epochs if args.max_steps <= 0 else -(-total_steps // steps_per_epoch)
    no_decay = ("bias", "layer_norm.weight", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log.info(f"train {len(data['train'])} + aug {len(data['train_all']) - len(data['train'])} = {len(train)} examples, "
             f"{steps_per_epoch} steps/epoch, {total_steps} steps, truncated src {src_trunc} tgt {tgt_trunc}")

    history, best, best_f1, best_epoch, bad, step = [], None, -1.0, 0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        te, losses = time.time(), []
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            losses.append(loss.item() * args.grad_accum)
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                step += 1
                if step >= total_steps:
                    break
        rec = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
               "train_seconds": time.time() - te}
        if epoch % args.eval_every == 0 or step >= total_steps or epoch == epochs:
            td = time.time()
            dev_m = score(predict(model, tok, data["dev"], args, device))
            rec.update(dev_f1=dev_m["quad"]["f1"], dev_invalid=dev_m["n_invalid_segments"], dev_seconds=time.time() - td)
            if dev_m["quad"]["f1"] > best_f1:
                best_f1, best_epoch, bad = dev_m["quad"]["f1"], epoch, 0
                best = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            else:
                bad += 1
        history.append(rec)
        log.info(json.dumps(rec))
        if step >= total_steps or (args.patience and args.max_steps <= 0 and bad >= args.patience):
            break

    model.load_state_dict(best)
    tp = time.time()
    dev_pred = predict(model, tok, data["dev"], args, device)
    test_pred = predict(model, tok, data["test"], args, device)
    metrics = {"seed": seed, "best_epoch": best_epoch, "dev": score(dev_pred), "test": score(test_pred)}
    write_jsonl(out_dir / "dev_predictions.jsonl", dev_pred)
    write_jsonl(out_dir / "test_predictions.jsonl", test_pred)
    json.dump(history, open(out_dir / "history.json", "w"), indent=1)
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=1)
    cfg = {
        "args": vars(args), "seed": seed, "best_epoch": best_epoch, "optimizer_steps": step,
        "epochs_run": len(history), "steps_per_epoch": steps_per_epoch,
        "n_train": len(data["train"]), "n_aug": len(data["train_all"]) - len(data["train"]), "n_train_total": len(train),
        "truncated_source": src_trunc, "truncated_target": tgt_trunc,
        "data_sha256": data["hashes"], "seconds_total": time.time() - t0, "seconds_final_predict": time.time() - tp,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": __import__("transformers").__version__},
        "command": " ".join(sys.argv), "hostname": platform.node(),
    }
    json.dump(cfg, open(out_dir / "run_config.json", "w"), indent=1)
    if args.keep_ckpt:
        model.save_pretrained(out_dir / "checkpoint"); tok.save_pretrained(out_dir / "checkpoint")
    log.info(f"done: best epoch {best_epoch}, dev F1 {metrics['dev']['quad']['f1']:.2f}, "
             f"test F1 {metrics['test']['quad']['f1']:.2f}, {cfg['seconds_total'] / 60:.1f} min")
    del model, best
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    data = {k: load_jsonl(getattr(args, f"{k}_path")) for k in ("train", "dev", "test")}
    aug = [x for p in args.aug_path for x in load_jsonl(p)]
    if args.limit:
        data = {k: v[:args.limit] for k, v in data.items()}
        aug = aug[:args.limit]
    data["train_all"] = data["train"] + aug
    data["hashes"] = {p: sha256(p) for p in [args.train_path, args.dev_path, args.test_path, *args.aug_path]}
    out = Path(args.output_dir)
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        d = out / f"seed_{seed}"
        if (d / "metrics.json").exists() and (d / "test_predictions.jsonl").exists():
            print(f"seed {seed}: already done, skipped")
            continue
        run_seed(seed, args, data, device, d)


if __name__ == "__main__":
    main()
