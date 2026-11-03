"""Multi-view consensus distillation: an MvP teacher (src/train_paraphrase.py --order_mode mvp --keep_ckpt) labels
unlabelled synthetic texts with every view; the majority vote becomes the pseudo-label of an augmented set.
Ensemble distillation is the same with several single-view teachers (one label run each): select merges the rows of
its inputs by id, so every teacher's output becomes one view of the vote.

Self-training with a single-view teacher (src/aug/st_scorer.py, p2) adds nothing because the teacher is no better
than the student. The MvP vote is better than any of its views (inner_dev), most of all on implicit quads, so its
labels carry what a single-view student does not get from the gold data alone.

    python -u src/aug/mvd.py label --ckpt outputs/dv_mvp/seed_7/checkpoint --in data/aug/st_pool_inner.jsonl \\
        --shard 0/2 --out data/aug/mvd_inner/views_0.jsonl                                   # GPU
    python src/aug/mvd.py select --in views_0.jsonl,views_1.jsonl --out data/aug/dv_mvd.jsonl --consensus --n 2015  # CPU

label: greedy generation of every view of the teacher (view orders from the run_config.json next to the checkpoint),
written as the "views" field of src/train_paraphrase.py predictions (so its *_predictions.jsonl can be selected too;
a single-view prediction counts as one view).
select: rows with the same id in the inputs (teachers) are merged, their views concatenated (shards of one
teacher have disjoint ids). Pseudo-label = quads predicted by >= --vote_min views (default majority); quads whose explicit spans are not
in the text are dropped, then reviews left without quads. --consensus also drops every review where some quad got
votes but fewer than --vote_min (the views disagree on it). At most --n reviews drawn at random (--seed).
"""
import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import implicit_type, load_jsonl  # noqa: E402
from aug.st_scorer import as_quads, found, halving_on_oom, load, write_jsonl  # noqa: E402


def views(r):
    return r["views"] if "views" in r else [r["pred"]]


def votes(r):
    return collections.Counter(tuple(q) for v in r["views"] for q in {tuple(x) for x in v})


def cmd_label(a):
    import torch
    from train_paraphrase import predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_jsonl(a.inp)[:a.limit or None]
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        rows = rows[i::n]
    cfg = json.load(open(a.run_config or Path(a.ckpt).parent / "run_config.json"))
    args = argparse.Namespace(view_orders=cfg["view_orders"], target_orders=None, vote_min=0, num_beams=1,
                              eval_batch_size=a.batch_size, max_source_len=cfg["args"]["max_source_len"],
                              gen_max_len=cfg["args"]["gen_max_len"])
    tok, model = load(a.ckpt, device)
    items = [{"id": r["id"], "text": r["text"], "quads": []} for r in rows]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["text"]))
    out = [None] * len(items)

    def run(idx):
        args.eval_batch_size = len(idx)
        for i, rec in zip(idx, predict(model, tok, [items[i] for i in idx], args, device)):
            out[i] = {"id": rec["id"], "text": rec["text"], "pred": rec["pred"],
                      "generations": rec.get("generations", [rec["generation"]]), "views": views(rec),
                      "views_invalid": rec.get("views_invalid", [rec["n_invalid"]])}

    t0 = time.time()
    for b in range(0, len(order), a.batch_size):
        halving_on_oom(run, order[b:b + a.batch_size])
    write_jsonl(a.out, out)
    print(f"{len(out)} reviews, {len(args.view_orders)} views {args.view_orders}, {(time.time() - t0) / 60:.1f} min"
          f" -> {a.out}")


def cmd_select(a):
    rng = random.Random(a.seed)
    merged = {}
    for p in a.inp.split(","):
        for r in load_jsonl(p):
            m = merged.setdefault(r["id"], {"id": r["id"], "text": r["text"], "views": []})
            m["views"] += views(r)
    rows = list(merged.values())
    n_views = collections.Counter(len(r["views"]) for r in rows)
    if len(n_views) > 1:
        raise SystemExit(f"reviews have different numbers of views: {dict(n_views)}")
    st = collections.Counter(n=len(rows))
    keep = []
    for r in rows:
        c = votes(r)
        k = a.vote_min or len(r["views"]) // 2 + 1
        quads = [q for q, n in c.items() if n >= k and found(q[1], r["text"]) and found(q[2], r["text"])]
        if not quads:
            st["no grounded majority quad"] += 1
            continue
        if a.consensus and any(n < k for n in c.values()):
            st["views disagree"] += 1
            continue
        keep.append((r, quads, c))
    st["kept"] = len(keep)
    if len(keep) > a.n:
        keep = rng.sample(keep, a.n)
    out = [{"id": f"mvd-{i:05d}", "text": r["text"], "quads": as_quads(quads),
            "meta": {"method": "mvp_consensus", "source": r["id"], "votes": [c[q] for q in sorted(quads)]}}
           for i, (r, quads, c) in enumerate(keep)]
    write_jsonl(a.out, out)
    types = collections.Counter(implicit_type(tuple(q.values())) for x in out for q in x["quads"])
    rep = {"in": a.inp, "vote_min": a.vote_min or "majority", "consensus": a.consensus, "n_max": a.n,
           "n_views": next(iter(n_views)), "n_out": len(out), "quads_out": sum(types.values()), "stats": dict(st),
           "implicit_types": dict(types), "seed": a.seed}
    json.dump(rep, open(Path(a.out).with_suffix(".report.json"), "w"), indent=1)
    print(json.dumps(rep))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("label")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run_config", help="default: run_config.json next to the checkpoint directory")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--shard", default="")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("select")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vote_min", type=int, default=0, help="views that must predict a quad; 0 = majority")
    p.add_argument("--consensus", action="store_true", help="drop reviews with a quad below --vote_min votes")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    {"label": cmd_label, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
