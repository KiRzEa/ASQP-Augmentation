"""Self-training with a pseudo-label scorer (Zhang et al., ACL 2024), without external data.

The paper pseudo-labels external in-domain reviews (Yelp / Amazon) and trains its scorer on comparison data
annotated by humans or ChatGPT. Here the unlabelled reviews are the synthetic texts of data/aug/syn_infill.jsonl
(GPT infilling of training reviews, no labels) and the comparison data come from the gold training labels:
candidates of cross-validation models on their held-out fold are the negatives, the gold label is the positive.
Model-generating steps need a GPU (Kaggle jobs); pool, comparison and select run on CPU.

    python src/aug/st_scorer.py pool --train_path T --out data/aug/st_pool_X.jsonl          # CPU: unlabelled texts allowed for T
    python -u src/aug/st_scorer.py candidates --ckpt C --in F --out O [--shard i/n]           # GPU: beam candidates
    python src/aug/st_scorer.py comparison --cands a.jsonl,b.jsonl --out comp.jsonl            # CPU
    python -u src/aug/st_scorer.py train --comp comp.jsonl --out_dir outputs/scorer_X          # GPU: the scorer
    python -u src/aug/st_scorer.py score --scorer S --in O1,O2 --out scored.jsonl              # GPU: score pseudo-labels
    python src/aug/st_scorer.py select --in scored.jsonl --out data/aug/st_X.jsonl             # CPU: filters

candidates: --beams 4 candidates per review (beam search, num_return_sequences), each with its quads and the
length-normalised beam score; the top one also gets the paper's confidence (minimum token probability).
comparison: one sample per review: positive = the gold target, negatives = candidates whose quad set differs.
train: generative scorer s(x, y) = sum of log p(y_t | y_<t, x) (Eq. 1), listwise loss over the positive and the
negatives (Eq. 2-3) plus alpha x the NLL of the positive (Eq. 4-6); init from the pre-trained model; 10 epochs,
batch 10 (paper); 10 % of the reviews held out, the epoch with the best top-1 ranking accuracy on them is kept.
select: confidence >= --conf (0.7, Eq. 7), then reviews whose scorer score lies in the --band of the ranking
(top 10 % to 40 %, Eq. 8), at most --n (10,000) drawn at random. Pseudo-labelled quads whose explicit spans are
not in the text are dropped first (and reviews left without quads).
Reranking at inference (paper Sec. 4.3) is --rerank_scorer of src/train_paraphrase.py.
"""
import argparse
import collections
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import NULL, load_jsonl, quad_tuple, quads_to_target, target_to_quads  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def as_quads(tuples):
    return [{"category": c, "aspect": a, "opinion": o, "polarity": p} for c, a, o, p in sorted(tuples)]


def found(span, text):
    return span == NULL or re.search(r"(?<!\w)" + re.escape(span) + r"(?!\w)", text) is not None


# ---------- CPU steps ----------

def bigrams(t):
    w = t.split()
    return set(zip(w, w[1:]))


def cmd_pool(a):
    """Unlabelled texts usable with training file T: the synthetic text's most similar real review (share of its
    word bigrams) is in T, and no real review outside T (dev, test, the inner dev fold) shares >= 0.3 of them."""
    train_ids = {x["id"] for x in load_jsonl(a.train_path)}
    real = [x for p in a.real.split(",") for x in load_jsonl(p)]
    rb = [bigrams(x["text"]) for x in real]
    idx = collections.defaultdict(set)
    for i, b in enumerate(rb):
        for g in b:
            idx[g].add(i)
    out, stats = [], collections.Counter()
    for x in load_jsonl(a.syn):
        b = bigrams(x["text"])
        c = collections.Counter(i for g in b for i in idx.get(g, ()))
        share = {i: n / max(1, len(b)) for i, n in c.items()}
        ranked = sorted(share.items(), key=lambda kv: -kv[1])
        best = ranked[0] if ranked else (None, 0.0)
        outside = max((s for i, s in share.items() if real[i]["id"] not in train_ids), default=0.0)
        if best[0] is None or real[best[0]]["id"] not in train_ids:
            stats["source outside train"] += 1
        elif outside >= 0.3:
            stats["overlaps a review outside train"] += 1
        else:
            stats["kept"] += 1
            out.append({"id": x["id"], "text": x["text"], "quads": [],
                        "meta": {**x.get("meta", {}), "source": real[best[0]]["id"], "source_share": round(best[1], 3)}})
    write_jsonl(a.out, out)
    json.dump({"syn": a.syn, "train_path": a.train_path, "stats": dict(stats)},
              open(Path(a.out).with_suffix(".report.json"), "w"), indent=1)
    print(dict(stats))


def cmd_comparison(a):
    rows = [r for p in a.cands.split(",") for r in load_jsonl(p)]
    out, n_neg = [], collections.Counter()
    for r in rows:
        gold = {quad_tuple(q) for q in r["quads"]}
        negs, seen = [], {frozenset(gold)}
        for c in r["candidates"]:
            k = frozenset(tuple(q) for q in c["quads"])
            if k not in seen and c["quads"]:
                seen.add(k)
                negs.append(quads_to_target(as_quads(k)))
        n_neg[len(negs)] += 1
        out.append({"id": r["id"], "text": r["text"], "pos": quads_to_target(r["quads"]), "negs": negs})
    write_jsonl(a.out, out)
    print(f"{len(out)} comparison samples; negatives per sample {dict(sorted(n_neg.items()))}")


def cmd_select(a):
    rng = random.Random(a.seed)
    rows = [r for p in a.inp.split(",") for r in load_jsonl(p)]
    st = collections.Counter(n=len(rows))
    keep = []
    for r in rows:
        quads = [q for q in r["candidates"][0]["quads"] if found(q[1], r["text"]) and found(q[2], r["text"])]
        if not quads:
            st["no grounded quad"] += 1
            continue
        if r["confidence"] < a.conf:
            st["confidence < %.2f" % a.conf] += 1
            continue
        keep.append({**r, "quads_kept": quads})
    keep.sort(key=lambda r: -r["score"])
    lo, hi = (float(v) for v in a.band.split(","))
    band = keep[int(lo * len(keep)):int(hi * len(keep))]
    st["after confidence"] = len(keep)
    st["in band"] = len(band)
    if len(band) > a.n:
        band = rng.sample(band, a.n)
    out = [{"id": f"st-{i:05d}", "text": r["text"], "quads": as_quads(map(tuple, r["quads_kept"])),
            "meta": {"method": "st_scorer", "source": r["id"], "confidence": r["confidence"], "score": r["score"]}}
           for i, r in enumerate(band)]
    write_jsonl(a.out, out)
    rep = {"in": a.inp, "conf": a.conf, "band": a.band, "n_max": a.n, "n_out": len(out),
           "quads_out": sum(len(x["quads"]) for x in out), "stats": dict(st)}
    json.dump(rep, open(Path(a.out).with_suffix(".report.json"), "w"), indent=1)
    print(json.dumps(rep))


# ---------- GPU steps ----------

def load(ckpt, device):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    return tok, AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device)


def seq_logprob(model, tok, texts, targets, device, grad=False):
    """Sum and count of log p(y_t | y_<t, x) per (text, target) pair (teacher forcing)."""
    import torch
    enc = tok(texts, max_length=256, truncation=True, padding=True, return_tensors="pt").to(device)
    lab = tok(targets, max_length=384, truncation=True, padding=True, return_tensors="pt").input_ids.to(device)
    mask = lab != tok.pad_token_id
    with torch.set_grad_enabled(grad):
        logits = model(**enc, labels=lab.masked_fill(~mask, -100)).logits
        lp = torch.log_softmax(logits.float(), -1).gather(-1, lab.unsqueeze(-1)).squeeze(-1) * mask
    return lp.sum(1), mask.sum(1)


def generate_candidates(model, tok, texts, beams, device, batch_size=16):
    """Beam candidates per text: [[(generation, length-normalised beam score)], ...] and the min token prob of the
    top candidate."""
    import torch
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    cands, conf = [None] * len(texts), [None] * len(texts)
    model.eval()
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([texts[i] for i in idx], max_length=256, truncation=True, padding=True,
                      return_tensors="pt").to(device)
            g = model.generate(**enc, max_new_tokens=384, num_beams=beams, num_return_sequences=beams,
                               output_scores=True, return_dict_in_generate=True)
            dec = tok.batch_decode(g.sequences, skip_special_tokens=True)
            sc = g.sequences_scores.tolist()
            top = g.sequences[::beams][:, 1:]  # drop the decoder start token
            mask = (top != tok.pad_token_id)
            logits = model(**enc, labels=top.masked_fill(~mask, -100)).logits
            p = torch.softmax(logits.float(), -1).gather(-1, top.unsqueeze(-1)).squeeze(-1)
            mins = p.masked_fill(~mask, 1.0).min(1).values.tolist()
            for j, i in enumerate(idx):
                cands[i] = [(dec[j * beams + k], sc[j * beams + k]) for k in range(beams)]
                conf[i] = mins[j]
    return cands, conf


def cmd_candidates(a):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_jsonl(a.inp)[:a.limit or None]
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        rows = rows[i::n]
    tok, model = load(a.ckpt, device)
    t0 = time.time()
    cands, conf = generate_candidates(model, tok, [r["text"] for r in rows], a.beams, device, a.batch_size)
    out = [{"id": r["id"], "text": r["text"], "quads": r["quads"], "confidence": c,
            "candidates": [{"generation": g, "beam_score": s, "quads": [list(q) for q in sorted(target_to_quads(g)[0])]}
                           for g, s in cs]}
           for r, cs, c in zip(rows, cands, conf)]
    write_jsonl(a.out, out)
    print(f"{len(out)} reviews, {a.beams} candidates each, {(time.time() - t0) / 60:.1f} min -> {a.out}")


def cmd_train(a):
    import numpy as np
    import torch
    from transformers import get_linear_schedule_with_warmup
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_jsonl(a.comp)[:a.limit or None]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    n_dev = max(1, len(rows) // 10)
    dev, train = [r for r in rows[:n_dev] if r["negs"]], rows[n_dev:]
    tok, model = load(a.model_name, device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    steps = -(-len(train) // a.batch_size) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, 0, steps)

    def accuracy():
        model.eval()
        hit = 0
        for r in dev:
            s, _ = seq_logprob(model, tok, [r["text"]] * (1 + len(r["negs"])), [r["pos"]] + r["negs"], device)
            hit += int(s.argmax().item() == 0)
        return 100 * hit / max(1, len(dev))

    out_dir = Path(a.out_dir)
    best, history = -1.0, []
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    log(f"scorer: {len(train)} training reviews ({sum(bool(r['negs']) for r in train)} with negatives), "
        f"{len(dev)} held-out, {steps} steps, accuracy before training {accuracy():.1f}")
    for ep in range(1, a.epochs + 1):
        model.train()
        rng.shuffle(train)
        l1s, l2s = [], []
        for b in range(0, len(train), a.batch_size):
            opt.zero_grad(set_to_none=True)
            batch = train[b:b + a.batch_size]
            for r in batch:  # one review (positive + its negatives) per forward pass
                s, n = seq_logprob(model, tok, [r["text"]] * (1 + len(r["negs"])), [r["pos"]] + r["negs"],
                                   device, grad=True)
                l2 = -s[0] / n[0]
                loss = a.alpha * l2
                if r["negs"]:
                    l1 = -(s[0] - torch.logsumexp(s, 0))
                    loss = loss + l1
                    l1s.append(l1.item())
                l2s.append(l2.item())
                (loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        acc = accuracy()
        history.append({"epoch": ep, "l1": float(np.mean(l1s)) if l1s else None, "l2": float(np.mean(l2s)),
                        "dev_top1": acc})
        log(json.dumps(history[-1]))
        if acc > best:
            best = acc
            model.save_pretrained(out_dir / "checkpoint"); tok.save_pretrained(out_dir / "checkpoint")
    json.dump({"args": vars(a), "history": history, "best_dev_top1": best, "n_train": len(train), "n_dev": len(dev)},
              open(out_dir / "scorer.json", "w"), indent=1)


def cmd_score(a):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load(a.scorer, device)
    model.eval()
    rows = [r for p in a.inp.split(",") for r in load_jsonl(p)]
    t0 = time.time()
    for b in range(0, len(rows), a.batch_size):
        batch = rows[b:b + a.batch_size]
        s, _ = seq_logprob(model, tok, [r["text"] for r in batch],
                           [quads_to_target(as_quads(map(tuple, r["candidates"][0]["quads"]))) for r in batch], device)
        for r, v in zip(batch, s.tolist()):
            r["score"] = v
    write_jsonl(a.out, rows)
    print(f"{len(rows)} pseudo-labels scored, {(time.time() - t0) / 60:.1f} min -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--train_path", required=True)
    p.add_argument("--syn", default="data/aug/syn_infill.jsonl")
    p.add_argument("--real", default="data/restaurant_v1/train.jsonl,data/restaurant_v1/dev.jsonl,data/restaurant_v1/test.jsonl")
    p.add_argument("--out", required=True)
    p = sub.add_parser("candidates")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--shard", default="")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("comparison")
    p.add_argument("--cands", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--comp", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_name", default="VietAI/vit5-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("score")
    p.add_argument("--scorer", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p = sub.add_parser("select")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.7)
    p.add_argument("--band", default="0.1,0.4")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    {"pool": cmd_pool, "candidates": cmd_candidates, "comparison": cmd_comparison, "train": cmd_train,
     "score": cmd_score, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
