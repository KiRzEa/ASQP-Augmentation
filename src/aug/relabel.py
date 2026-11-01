"""Consensus relabelling of augmented or unlabeled reviews: several trained ASQP models label each review, a quad is
kept if at least --vote_min models predict it. Labelling needs a GPU: run on Kaggle (job in rounds.json).

    python -u src/aug/relabel.py --ckpts outputs/teacher_none/seed_42/checkpoint,outputs/teacher_none/seed_123/checkpoint,... \
        --in data/aug/llm_para.jsonl,data/aug/llm_q2t.jsonl --out data/aug/llm_para_cons.jsonl,data/aug/llm_q2t_cons.jsonl
    python src/aug/relabel.py --from_votes data/aug/syn_cons.votes.jsonl --vote_min 4 --out data/aug/syn_cons4.jsonl   # CPU
    --shard 0/2 and --shard 1/2 label every other review on two GPUs; --from_votes a.votes.jsonl,b.votes.jsonl with a
    single --out merges the shards.

The labels a review already has (copied from a source review, a sampled quad set, or none for synthetic reviews)
are replaced by the consensus. Quads whose explicit aspect or opinion is not found in the review (word boundaries)
are dropped; a review with no consensus quad left is dropped, and with --max_ambiguous m a review is also dropped
when more than m quads were predicted by some teachers but by fewer than --vote_min (teachers disagree on it).
--min_quads / --min_polarities drop reviews with fewer consensus quads / distinct polarities: the training set has
>= 2 quads and >= 2 polarities in 2,134 of its 2,135 reviews, so both at 2 keep that structure.
Teachers are Paraphrase models trained on train only (dev used only for their checkpoint selection), so dev and
test are never seen. Every labelling run also writes <out>.votes.jsonl (all predicted quads with their vote
counts), from which --from_votes rebuilds the output for another threshold without a GPU.
Writes <out> and <out>.report.json: sizes, drops, vote histogram, and how far the review's own labels (if any)
are from the consensus (quad P/R/F1 of the original labels against the consensus, share of identical reviews).
"""
import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import NULL, load_jsonl, prf, quad_tuple, target_to_quads  # noqa: E402


def label(ckpt, texts, a, device):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device).eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch: less padding
    out = [None] * len(texts)
    with torch.no_grad():
        for b in range(0, len(order), a.batch_size):
            idx = order[b:b + a.batch_size]
            enc = tok([texts[i] for i in idx], max_length=256, truncation=True, padding=True,
                      return_tensors="pt").to(device)
            gen = model.generate(**enc, max_new_tokens=384, num_beams=1)
            for i, g in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = target_to_quads(g)[0]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def found(span, text):
    return span == NULL or re.search(r"(?<!\w)" + re.escape(span) + r"(?!\w)", text) is not None


def build(rows, n_teachers, a, out, src_name):
    """rows: [{"id", "text", "quads" (own labels), "meta", "votes": [[quad, count], ...]}] -> write out + report."""
    vote_min = a.vote_min or n_teachers // 2 + 1
    kept, st, agree, hist = [], collections.Counter(), collections.Counter(), collections.Counter()
    for x in rows:
        votes = {tuple(q): n for q, n in x["votes"]}
        hist.update(votes.values())
        cons = {q for q, n in votes.items() if n >= vote_min}
        grounded = {q for q in cons if found(q[1], x["text"]) and found(q[2], x["text"])}
        st["quads_ungrounded"] += len(cons) - len(grounded)
        ambiguous = sum(1 for n in votes.values() if 0 < n < vote_min)
        orig = {quad_tuple(q) for q in x["quads"]}
        if orig:
            st["with_own_labels"] += 1
            agree["tp"] += len(orig & grounded); agree["fp"] += len(orig - grounded); agree["fn"] += len(grounded - orig)
            st["identical"] += orig == grounded
        if not grounded:
            st["dropped_no_consensus"] += 1
            continue
        if a.max_ambiguous >= 0 and ambiguous > a.max_ambiguous:
            st["dropped_ambiguous"] += 1
            continue
        if len(grounded) < a.min_quads or len({q[3] for q in grounded}) < a.min_polarities:
            st["dropped_structure"] += 1
            continue
        quads = [{"category": c, "aspect": asp, "opinion": o, "polarity": p} for c, asp, o, p in sorted(grounded)]
        kept.append({"id": f"{x['id']}-cons", "text": x["text"], "quads": quads,
                     "meta": {**x.get("meta", {}), "relabel": "consensus", "orig_quads": len(orig),
                              "ambiguous_quads": ambiguous}})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    report = {"in": src_name, "out": out, "n_in": len(rows), "n_out": len(kept), "n_teachers": n_teachers,
              "vote_min": vote_min, "max_ambiguous": a.max_ambiguous, "min_quads": a.min_quads,
              "min_polarities": a.min_polarities,
              "quads_in": sum(len(x["quads"]) for x in rows), "quads_out": sum(len(x["quads"]) for x in kept),
              "quads_per_review_out": sum(len(x["quads"]) for x in kept) / max(len(kept), 1),
              "vote_histogram": {str(k): hist[k] for k in sorted(hist)}, "stats": dict(st),
              "identical_share": st["identical"] / st["with_own_labels"] if st["with_own_labels"] else None,
              "orig_vs_consensus": prf(agree["tp"], agree["fp"], agree["fn"]) if st["with_own_labels"] else None}
    json.dump(report, open(Path(out).with_suffix(".report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", help="comma-separated checkpoint folders (teachers)")
    ap.add_argument("--in", dest="inp", help="comma-separated JSONL files to label")
    ap.add_argument("--out", required=True, help="comma-separated outputs, one per --in (or per --from_votes)")
    ap.add_argument("--from_votes", help="comma-separated <out>.votes.jsonl of an earlier run: rebuild, no model")
    ap.add_argument("--vote_min", type=int, default=0, help="teachers that must agree; 0 = majority")
    ap.add_argument("--max_ambiguous", type=int, default=-1,
                    help="drop a review with more than this many quads below the threshold but predicted; -1 = off")
    ap.add_argument("--min_quads", type=int, default=0)
    ap.add_argument("--min_polarities", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: label only reviews with index %% n == i")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N reviews of each file")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    outs = a.out.split(",")
    if a.from_votes:
        vps = a.from_votes.split(",")
        if len(outs) == 1 and len(vps) > 1:  # merge shards into one output
            rows = [r for vp in vps for r in load_jsonl(vp)]
            build(rows, rows[0]["n_teachers"] if rows else 0, a, outs[0], a.from_votes)
            return
        assert len(vps) == len(outs), "--from_votes and --out need the same number of files"
        for vp, out in zip(vps, outs):
            rows = load_jsonl(vp)
            build(rows, rows[0]["n_teachers"] if rows else 0, a, out, vp)
        return
    import torch
    ckpts, ins = a.ckpts.split(","), a.inp.split(",")
    assert len(ins) == len(outs), "--in and --out need the same number of files"
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    t0 = time.time()
    si, sn = map(int, a.shard.split("/"))
    sets = [[x for i, x in enumerate(load_jsonl(p)) if i % sn == si][:a.limit or None] for p in ins]
    texts = [x["text"] for s in sets for x in s]
    votes = [collections.Counter() for _ in texts]
    for c in ckpts:  # one teacher at a time, all files at once
        for v, pred in zip(votes, label(c, texts, a, device)):
            v.update(pred)
        print(f"labelled {len(texts)} reviews with {c}, {(time.time() - t0) / 60:.1f} min", flush=True)
    k = 0
    for src, inp, out in zip(sets, ins, outs):
        rows = []
        for x in src:
            rows.append({"id": x["id"], "text": x["text"], "quads": x.get("quads", []), "meta": x.get("meta", {}),
                         "n_teachers": len(ckpts), "votes": [[list(q), n] for q, n in sorted(votes[k].items())]})
            k += 1
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(Path(out).with_suffix(".votes.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        build(rows, len(ckpts), a, out, inp)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
