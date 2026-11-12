"""Run the whole study on one GPU, step by step, resumable: a step whose output exists is skipped, a failed step
skips only the steps that need it. Logs in <root>/_logs/<step>.log, times in <root>/_status.tsv.

    python llm/pipeline.py plan                          # print the steps and what is already done
    python llm/pipeline.py run                           # everything (default: vi, rest, laptop x 5 seeds)
    python llm/pipeline.py run --smoke                   # whole chain on a few examples (~20-30 min on an A100)
    python llm/pipeline.py run --datasets vi --only vi_none   # a subset (step names start with the given prefix)
    python llm/pipeline.py pack                          # results_llm_<host>_<time>.zip (no model weights)

Per dataset d (in this order):
  d_none   seed s  M0 = INLG Gemma-7B baseline (gold order); its LoRA adapter is kept: the 5 teachers
  d_label  seed s  the teacher greedy-labels the unlabelled pool       -> data/aug/llm/d_s<seed>.jsonl
  d_select         quads with >= 3 of 5 votes and grounded spans        -> data/aug/llm/d_ens.jsonl (CPU)
  d_ens_pos seed s M1 = gold + voted pool, text order (the method)
  d_pos    seed s  M2 = gold, text order only
Model options (--profile, --model, --cache_dir, --backend) are passed to every step; defaults from the environment
variables ASQP_PROFILE, ASQP_MODEL, ASQP_CACHE_DIR, ASQP_BACKEND (see llm/README.md). --profile qwen writes to
outputs/llm_qwen and data/aug/llm_qwen, so a Gemma study and a Qwen study can live side by side.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SEEDS = [42, 123, 2024, 7, 99]


def steps(a):
    out, aug = Path(a.root), Path(a.aug_root)
    lim_train = ["--limit_train", "16", "--limit_test", "8", "--epochs", "1"] if a.smoke else []
    lim_pool = ["--limit", "16"] if a.smoke else []
    model = ["--profile", a.profile, "--backend", a.backend, "--offline", str(a.offline)] + \
        (["--model", a.model] if a.model else []) + (["--cache_dir", a.cache_dir] if a.cache_dir else [])
    train = [PY, "-u", "llm/inlg.py", "train"] + model + lim_train + a.train_args
    res = []
    for d in a.datasets:
        labels = []
        for s in a.seeds:
            run = out / f"{d}_none" / f"seed_{s}"
            res.append({"name": f"{d}_none:s{s}", "after": [], "done": run / "metrics.json",
                        "cmd": train + ["--dataset", d, "--seed", str(s), "--out_dir", str(run), "--keep_adapter"]})
            lab = aug / f"{d}_s{s}.jsonl"
            labels.append(str(lab))
            res.append({"name": f"{d}_label:s{s}", "after": [f"{d}_none:s{s}"], "done": lab,
                        "cmd": [PY, "-u", "llm/inlg.py", "label"] + model + lim_pool +
                        ["--dataset", d, "--adapter", str(run / "adapter"), "--out", str(lab)]})
        ens = aug / f"{d}_ens.jsonl"
        res.append({"name": f"{d}_select", "after": [f"{d}_label:s{s}" for s in a.seeds], "done": ens,
                    "cmd": [PY, "-u", "src/aug/mvd.py", "select", "--in", ",".join(labels), "--n", "100000",
                            "--out", str(ens)]})
        for s in a.seeds:
            run = out / f"{d}_ens_pos" / f"seed_{s}"
            res.append({"name": f"{d}_ens_pos:s{s}", "after": [f"{d}_select"], "done": run / "metrics.json",
                        "cmd": train + ["--dataset", d, "--seed", str(s), "--out_dir", str(run), "--tuple_order",
                                        "text", "--aug_path", str(ens)]})
        for s in a.seeds:
            run = out / f"{d}_pos" / f"seed_{s}"
            res.append({"name": f"{d}_pos:s{s}", "after": [], "done": run / "metrics.json",
                        "cmd": train + ["--dataset", d, "--seed", str(s), "--out_dir", str(run), "--tuple_order",
                                        "text"]})
    if a.only:
        keep = [s for s in res if any(s["name"].startswith(p) for p in a.only.split(","))]
        names = {s["name"] for s in keep}
        for s in keep:  # dependencies outside the subset must already be done
            s["after"] = [x for x in s["after"] if x in names or not next(
                y for y in res if y["name"] == x)["done"].exists()]
        res = keep
    return res


def cmd_plan(a):
    os.chdir(ROOT)
    for s in steps(a):
        print(f"{'done' if (ROOT / s['done']).exists() else 'todo'}  {s['name']:<22} after {s['after'] or '-'}")


def cmd_run(a):
    os.chdir(ROOT)
    st = steps(a)
    logs = Path(a.root) / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = Path(a.root) / "_status.tsv"
    failed, t_all = set(), time.time()
    print(f"{len(st)} steps; root {a.root}; profile {a.profile}, model {a.model or 'profile default'} ({a.backend}); "
          f"host {socket.gethostname()}", flush=True)
    for i, s in enumerate(st, 1):
        if Path(s["done"]).exists():
            print(f"[{i}/{len(st)}] {s['name']}: done before, skipped", flush=True)
            continue
        bad = [x for x in s["after"] if x in failed or not Path(next(y for y in st if y["name"] == x)["done"]).exists()]
        if bad:
            failed.add(s["name"])
            print(f"[{i}/{len(st)}] {s['name']}: SKIPPED, needs {bad}", flush=True)
            continue
        log = logs / f"{s['name'].replace(':', '_')}.log"
        print(f"[{i}/{len(st)}] {s['name']} ... (log: {log})", flush=True)
        t0 = time.time()
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(s['cmd'])}\n")
            f.flush()
            rc = subprocess.call(s["cmd"], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        minutes = (time.time() - t0) / 60
        ok = rc == 0 and Path(s["done"]).exists()
        with open(status, "a") as f:
            f.write(f"{s['name']}\t{rc}\t{minutes:.1f}\t{time.strftime('%Y-%m-%d %H:%M')}\n")
        if not ok:
            failed.add(s["name"])
            print(f"    FAILED (exit {rc}) after {minutes:.1f} min; last lines of the log:", flush=True)
            print("".join(open(log, encoding="utf-8").readlines()[-15:]), flush=True)
        else:
            print(f"    ok, {minutes:.1f} min", flush=True)
    print(f"\nfinished in {(time.time() - t_all) / 3600:.2f} h; failed or skipped: {sorted(failed) or 'none'}", flush=True)
    if not a.no_pack:
        cmd_pack(a)
    sys.exit(1 if failed else 0)


def cmd_pack(a):
    """Zip run outputs and label files, without adapters, resume states or weights."""
    os.chdir(ROOT)
    tag = "" if a.profile == "gemma" else f"_{a.profile}"
    name = ROOT / f"results_llm{tag}_{socket.gethostname()}_{time.strftime('%Y%m%d_%H%M')}.zip"
    n = 0
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
        for base in (Path(a.root), Path(a.aug_root)):
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if p.is_file() and not {"adapter", "resume"} & set(p.parts) and p.suffix not in (".safetensors", ".pt"):
                    z.write(p, p.relative_to(ROOT) if p.is_absolute() else p)
                    n += 1
    print(f"{name.name}: {n} files, {name.stat().st_size / 2 ** 20:.1f} MB -> send this file back", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "pack"])
    ap.add_argument("--datasets", default="vi,rest,laptop")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--only", default="", help="comma-separated step name prefixes, e.g. vi_none,vi_label")
    ap.add_argument("--smoke", action="store_true", help="tiny run of the whole chain into outputs/llm_smoke")
    ap.add_argument("--profile", choices=["gemma", "qwen"], default=os.environ.get("ASQP_PROFILE", "gemma"),
                    help="gemma = INLG Gemma-7B (default); qwen = Qwen2.5-14B-Instruct (LLM_QWen.ipynb)")
    ap.add_argument("--model", default=os.environ.get("ASQP_MODEL", ""), help="default: the profile's model")
    ap.add_argument("--cache_dir", default=os.environ.get("ASQP_CACHE_DIR", ""))
    ap.add_argument("--backend", default=os.environ.get("ASQP_BACKEND", "hf"))
    ap.add_argument("--offline", type=int, default=int(os.environ.get("ASQP_OFFLINE", "1")))
    ap.add_argument("--no_pack", action="store_true")
    a, extra = ap.parse_known_args()
    a.train_args = extra  # anything else goes to every train step (e.g. --micro_batch 4)
    a.datasets = a.datasets.split(",")
    a.seeds = [int(s) for s in a.seeds.split(",")]
    if a.smoke:
        a.seeds = a.seeds[:1]
    tag = "" if a.profile == "gemma" else f"_{a.profile}"  # gemma keeps the original folders
    a.root = f"outputs/llm{tag}_smoke" if a.smoke else f"outputs/llm{tag}"
    a.aug_root = f"data/aug/llm{tag}_smoke" if a.smoke else f"data/aug/llm{tag}"
    {"plan": cmd_plan, "run": cmd_run, "pack": cmd_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
