"""Consensus relabelling of augmented reviews: several trained ASQP models label each review, a quad is kept if at
least --vote_min models predict it. Needs a GPU: run on Kaggle (job in rounds.json).

    python -u src/aug/relabel.py --ckpts outputs/teacher_none/seed_42/checkpoint,outputs/teacher_none/seed_123/checkpoint,... \
        --in data/aug/llm_para.jsonl,data/aug/llm_q2t.jsonl --out data/aug/llm_para_cons.jsonl,data/aug/llm_q2t_cons.jsonl

The labels the augmentation method gave a review (copied from the source review, or the sampled quad set) are
replaced by the consensus. Quads whose explicit aspect or opinion is not found in the review (word boundaries) are
dropped; a review with no consensus quad left is dropped. Teachers are Paraphrase models trained on train only
(dev used only for their checkpoint selection), so dev and test are never seen.
Writes <out> and <out>.report.json: sizes, drops, and how far the method's own labels are from the consensus
(quad P/R/F1 of the original labels against the consensus, share of reviews where both are identical).
"""
import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import NULL, load_jsonl, prf, quad_tuple, target_to_quads  # noqa: E402


@torch.no_grad()
def label(ckpt, texts, a, device):
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device).eval()
    out = []
    for i in range(0, len(texts), a.batch_size):
        enc = tok(texts[i:i + a.batch_size], max_length=256, truncation=True, padding=True, return_tensors="pt").to(device)
        gen = model.generate(**enc, max_new_tokens=384, num_beams=1)
        out += [target_to_quads(g)[0] for g in tok.batch_decode(gen, skip_special_tokens=True)]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def found(span, text):
    return span == NULL or re.search(r"(?<!\w)" + re.escape(span) + r"(?!\w)", text) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint folders (teachers)")
    ap.add_argument("--in", dest="inp", required=True, help="comma-separated augmented JSONL files")
    ap.add_argument("--out", required=True, help="comma-separated outputs, one per --in")
    ap.add_argument("--vote_min", type=int, default=0, help="teachers that must agree; 0 = majority")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N reviews of each file")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    ckpts, ins, outs = a.ckpts.split(","), a.inp.split(","), a.out.split(",")
    assert len(ins) == len(outs), "--in and --out need the same number of files"
    vote_min = a.vote_min or len(ckpts) // 2 + 1
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    t0 = time.time()
    sets = [load_jsonl(p)[:a.limit or None] for p in ins]
    texts = [x["text"] for s in sets for x in s]
    votes = [collections.Counter() for _ in texts]
    for c in ckpts:  # one teacher at a time, all files at once
        for v, pred in zip(votes, label(c, texts, a, device)):
            v.update(pred)
        print(f"labelled {len(texts)} reviews with {c}, {(time.time() - t0) / 60:.1f} min", flush=True)
    k = 0
    for src, inp, out in zip(sets, ins, outs):
        kept, st, agree = [], collections.Counter(), collections.Counter()
        for x in src:
            v = votes[k]; k += 1
            cons = {q for q, n in v.items() if n >= vote_min}
            grounded = {q for q in cons if found(q[1], x["text"]) and found(q[2], x["text"])}
            st["quads_ungrounded"] += len(cons) - len(grounded)
            orig = {quad_tuple(q) for q in x["quads"]}
            agree["tp"] += len(orig & grounded); agree["fp"] += len(orig - grounded); agree["fn"] += len(grounded - orig)
            st["identical"] += orig == grounded
            if not grounded:
                st["dropped_no_consensus"] += 1
                continue
            quads = [{"category": c, "aspect": asp, "opinion": o, "polarity": p} for c, asp, o, p in sorted(grounded)]
            kept.append({"id": f"{x['id']}-cons", "text": x["text"], "quads": quads,
                         "meta": {**x.get("meta", {}), "relabel": "consensus", "orig_quads": len(orig)}})
        with open(out, "w", encoding="utf-8") as f:
            for it in kept:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        report = {"in": inp, "out": out, "n_in": len(src), "n_out": len(kept), "teachers": ckpts, "vote_min": vote_min,
                  "quads_in": sum(len(x["quads"]) for x in src), "quads_out": sum(len(x["quads"]) for x in kept),
                  "identical_share": st["identical"] / max(len(src), 1), "stats": dict(st),
                  "orig_vs_consensus": prf(agree["tp"], agree["fp"], agree["fn"]),
                  "minutes": (time.time() - t0) / 60}
        json.dump(report, open(Path(out).with_suffix(".report.json"), "w"), indent=1)
        print(json.dumps({k2: v2 for k2, v2 in report.items() if k2 != "teachers"}), flush=True)


if __name__ == "__main__":
    main()
