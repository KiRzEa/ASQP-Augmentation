"""Score the 24 element orders of the target with the pre-trained (not fine-tuned) model, for order-based baselines.

    python src/aug/order_scores.py --train_path data/restaurant_v1/train.jsonl --out data/aug/order_scores_train.json
    python src/aug/order_scores.py --lang en --model_name google-t5/t5-base --train_path data/acos/rest/train.jsonl \
        --out data/aug/acos_rest_order_scores.json                      (English ACOS label words)

For every training review and every permutation of (A, O, C, S), the target is written in that order with the
element markers replaced by spaces (quads still joined by [SSEP]) and fed to the decoder with teacher forcing:
- entropy: mean over target tokens of the entropy of the output distribution (DLO / ILO, Hu et al. 2022, Eq. 5;
  smaller = better);
- logprob: mean log-probability of the target tokens (MvP, Gou et al. 2023, Eq. 1, conditional generation
  score; larger = better).
Output: {"orders": [...], "dataset": {"entropy": {order: mean}, "logprob": {order: mean}},
         "instance": {id: {"entropy": [24], "logprob": [24]}}} (lists in the order of "orders").
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from asqp_common import CAT_WORDS, POL_WORDS, SEP, load_jsonl, set_label_words  # noqa: E402

ORDERS = ["".join(p) for p in itertools.permutations("AOCS")]


def plain_target(quads, order):
    """Target in the given order without element markers (as both papers score it)."""
    v = lambda q: {"A": q["aspect"], "O": q["opinion"], "C": CAT_WORDS[q["category"]], "S": POL_WORDS[q["polarity"]]}
    return SEP.join(" ".join(v(q)[e] for e in order) for q in quads)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default="VietAI/vit5-base")
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lang", choices=["vi", "en"], default="vi",
                    help="label words of the target: vi (default) or en (English ACOS words, as train_paraphrase --lang en)")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model_name).to(device).eval()
    items = load_jsonl(a.train_path)[:a.limit or None]
    set_label_words(a.lang, sorted({q["category"] for x in items for q in x["quads"]}))
    pairs = [(x, o) for x in items for o in ORDERS]
    ent, lp = [], []
    for i in range(0, len(pairs), a.batch_size):
        b = pairs[i:i + a.batch_size]
        enc = tok([x["text"] for x, _ in b], max_length=256, truncation=True, padding=True, return_tensors="pt").to(device)
        lab = tok([plain_target(x["quads"], o) for x, o in b], max_length=384, truncation=True, padding=True,
                  return_tensors="pt").input_ids.to(device)
        mask = lab != tok.pad_token_id
        logits = model(**enc, labels=lab.masked_fill(~mask, -100)).logits.float()
        logp = torch.log_softmax(logits, -1)
        tok_ent = -(logp.exp() * logp).sum(-1)
        tok_lp = logp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
        n = mask.sum(1)
        ent += ((tok_ent * mask).sum(1) / n).tolist()
        lp += ((tok_lp * mask).sum(1) / n).tolist()
        if i // a.batch_size % 100 == 0:
            print(f"{i + len(b)}/{len(pairs)}", flush=True)
    k = len(ORDERS)
    inst = {x["id"]: {"entropy": ent[j * k:(j + 1) * k], "logprob": lp[j * k:(j + 1) * k]} for j, x in enumerate(items)}
    mean = lambda key, o: sum(v[key][o] for v in inst.values()) / len(inst)
    out = {"model": a.model_name, "train_path": a.train_path, "lang": a.lang, "orders": ORDERS,
           "dataset": {key: {o: mean(key, j) for j, o in enumerate(ORDERS)} for key in ("entropy", "logprob")},
           "instance": inst}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))
    for key, rev in (("entropy", False), ("logprob", True)):
        r = sorted(ORDERS, key=lambda o: out["dataset"][key][o], reverse=rev)
        print(key, "best first:", " ".join(f"{o}:{out['dataset'][key][o]:.3f}" for o in r[:6]))


if __name__ == "__main__":
    main()
