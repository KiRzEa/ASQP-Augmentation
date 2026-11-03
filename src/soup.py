"""Weight averaging ("model soup", Wortsman et al., 2022) of fine-tuned checkpoints of src/train_paraphrase.py.

    python -u src/soup.py --ckpts outputs/X/seed_42/checkpoint,outputs/X/seed_123/checkpoint,... --out_dir outputs/X_soup \\
        --test_path data/restaurant_v1/inner_dev.jsonl

uniform: the mean of every floating-point tensor of all checkpoints.
greedy: ingredients sorted by their own dev F1 (metrics.json next to each checkpoint); one is added to the soup only if
the soup's dev F1 does not drop (dev = --dev_path, the checkpoint-selection set; the test set is not used to choose).
Decoding settings (views, lengths) come from the run_config.json next to the first checkpoint. Per mode,
<out_dir>/<mode>/ gets dev_predictions.jsonl, test_predictions.jsonl, metrics.json (as a training run) and soup.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import load_jsonl, score  # noqa: E402
from train_paraphrase import predict, single_view, write_jsonl  # noqa: E402
from aug.st_scorer import load  # noqa: E402


def average(states):
    return {k: (sum(s[k].float() for s in states) / len(states)).to(v.dtype) if v.is_floating_point() else v
            for k, v in states[0].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint directories")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dev_path", default="data/restaurant_v1/dev.jsonl")
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--modes", default="uniform,greedy")
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N reviews of each set")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = [Path(c) for c in a.ckpts.split(",")]
    cfg = json.load(open(ckpts[0].parent / "run_config.json"))
    args = argparse.Namespace(**{**cfg["args"], "eval_batch_size": a.eval_batch_size, "target_orders": None})
    args.view_orders = cfg["view_orders"]
    data = {"dev": load_jsonl(a.dev_path)[:a.limit or None], "test": load_jsonl(a.test_path)[:a.limit or None]}
    tok, model = load(str(ckpts[0]), device)
    states = [{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}]  # a copy: the model is reloaded below
    for c in ckpts[1:]:
        states.append(load(str(c), "cpu")[1].state_dict())
    own_dev = [json.load(open(c.parent / "metrics.json"))["dev"]["quad"]["f1"] for c in ckpts]

    def dev_f1(state):
        model.load_state_dict(state)
        return score(predict(model, tok, data["dev"], args, device))["quad"]["f1"]

    for mode in a.modes.split(","):
        t0 = time.time()
        if mode == "uniform":
            members, trace = list(range(len(ckpts))), []
        else:
            ranked = sorted(range(len(ckpts)), key=lambda i: -own_dev[i])
            members, best = [ranked[0]], own_dev[ranked[0]]
            trace = [{"add": str(ckpts[ranked[0]]), "dev_f1": best, "kept": True}]
            for i in ranked[1:]:
                f = dev_f1(average([states[j] for j in members + [i]]))
                trace.append({"add": str(ckpts[i]), "dev_f1": f, "kept": f >= best})
                if f >= best:
                    members, best = members + [i], f
        model.load_state_dict(average([states[j] for j in members]))
        out = Path(a.out_dir) / mode
        out.mkdir(parents=True, exist_ok=True)
        metrics = {"mode": mode}
        for split in ("dev", "test"):
            rows = predict(model, tok, data[split], args, device)
            write_jsonl(out / f"{split}_predictions.jsonl", rows)
            metrics[split] = score(rows)
            if len(args.view_orders) > 1:
                write_jsonl(out / f"{split}_view0_predictions.jsonl", single_view(rows))
                metrics[f"{split}_view0"] = score(single_view(rows))
        json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
        json.dump({"mode": mode, "ingredients": [str(c) for c in ckpts], "ingredient_dev_f1": own_dev,
                   "members": [str(ckpts[j]) for j in members], "greedy_trace": trace,
                   "dev_path": a.dev_path, "test_path": a.test_path, "minutes": (time.time() - t0) / 60},
                  open(out / "soup.json", "w"), indent=1)
        print(f"{mode} soup of {len(members)}/{len(ckpts)}: dev F1 {metrics['dev']['quad']['f1']:.2f}, "
              f"test F1 {metrics['test']['quad']['f1']:.2f}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
