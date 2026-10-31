#!/usr/bin/env python3
"""Experiment queue: rounds -> jobs (one run x one seed) -> Kaggle sessions packed under a time budget.

The targets of every round live in experiments/rounds.json. A job is one training of one configuration with one
seed (outputs/<run>/seed_<s>/). `plan` leaves out jobs that already have results locally, estimates their time
from measured times, and packs them into sessions of two GPU lanes each.

  python3 scripts/kaggle_queue.py list   [--round r3]                      inventory: done / to run / estimated minutes
  python3 scripts/kaggle_queue.py plan   --round r3 [--budget-hours 10.5]   -> experiments/run_plan.json (commit + push)
  python3 scripts/kaggle_queue.py run    --session 1 [--budget-hours 10.5]  on Kaggle (2 x T4), from the repo root
  python3 scripts/kaggle_queue.py ingest                                     unzip runs_inbox/*.zip into the repo

`run` binds one worker to each GPU, starts the longest jobs first, never starts a job whose estimate no longer fits
the budget (it stays for the next session), streams every job's output with a prefix, writes it to
outputs/_logs/<job>.log, appends `job  minutes  rc  session` to outputs/_job_times/<round>_s<N>.tsv and re-packs
results_<round>_s<N>.zip after every finished job, so a cut session loses at most the jobs in flight.
`ingest` unzips each zip in runs_inbox/ into the repo root, moves it to outputs/_zips_archive/ and records it in
outputs/_ingested.txt; re-ingesting a zip does nothing. Estimates use minutes per training example, calibrated
from all outputs/_job_times/*.tsv.
"""
import argparse
import json
import os
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUNDS = ROOT / "experiments/rounds.json"
PLAN = ROOT / "experiments/run_plan.json"
TIMES = ROOT / "outputs/_job_times"
INBOX = ROOT / "runs_inbox"
DEFAULT_MIN_PER_EXAMPLE = 0.019  # measured in r1/r2 on one T4: ~42 min for 2,135 examples


# ---------------- jobs ----------------

def n_lines(path):
    return sum(1 for _ in open(ROOT / path, encoding="utf-8")) if path and (ROOT / path).exists() else 0


def expand(round_id, rounds):
    """Jobs of one round: {id: job}. Targets: {"run", "aug"?, "args"?, "seeds"?} or {"id", "command", "est_min", "outputs"}."""
    seeds = rounds["seeds"]
    jobs = {}
    n_train = n_lines("data/restaurant_v1/train.jsonl")
    for t in rounds["rounds"][round_id]["targets"]:
        if "command" in t:  # generic one-off job (e.g. k-fold, verifier)
            jobs[t["id"]] = {"id": t["id"], "command": t["command"], "est_min": t.get("est_min", 60),
                             "outputs": t.get("outputs", []), "done": t.get("done", ""), "examples": None}
            continue
        aug = t.get("aug")
        views = re.search(r"--views (\d+)", t.get("args", ""))
        examples = (n_train + n_lines(aug)) * (int(views.group(1)) if views else 1)  # each view is one more example
        for s in t.get("seeds", seeds):
            jid = f"{t['run']}:s{s}"
            out = f"outputs/{t['run']}"
            cmd = f"python -u src/train_paraphrase.py --output_dir {out} --seeds {s}"
            if aug:
                cmd += f" --aug_path {aug}"
            if t.get("args"):
                cmd += " " + t["args"]
            jobs[jid] = {"id": jid, "run": t["run"], "seed": s, "aug": aug, "command": cmd, "examples": examples,
                         "outputs": [f"{out}/seed_{s}"], "done": f"{out}/seed_{s}/test_predictions.jsonl"}
    return jobs


def calibrate():
    """Minutes per training example from measured times (median over successful training jobs)."""
    ratios = []
    rounds = json.load(open(ROUNDS))
    examples = {}
    for r in rounds["rounds"]:
        for j in expand(r, rounds).values():
            examples[j["id"]] = j["examples"]
    for f in sorted(TIMES.glob("*.tsv")) if TIMES.exists() else []:
        for line in open(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2] == "0" and examples.get(parts[0]):
                ratios.append(float(parts[1]) / examples[parts[0]])
    return (statistics.median(ratios), len(ratios)) if ratios else (DEFAULT_MIN_PER_EXAMPLE, 0)


def estimate(job, mpe):
    return job["est_min"] if job.get("examples") is None else round(mpe * job["examples"], 1)


# ---------------- commands ----------------

def cmd_list(a):
    rounds = json.load(open(ROUNDS))
    mpe, n = calibrate()
    print(f"estimate: {mpe:.4f} min per training example ({'from ' + str(n) + ' measured jobs' if n else 'default'})")
    for r in ([a.round] if a.round else rounds["rounds"]):
        jobs = expand(r, rounds)
        done = [j for j in jobs.values() if j["done"] and (ROOT / j["done"]).exists()]
        todo = [j for j in jobs.values() if j not in done]
        print(f"\n{r}: {rounds['rounds'][r]['goal']}\n  OpenAI key: {rounds['rounds'][r].get('openai_key', '?')}"
              f"\n  done {len(done)}, to run {len(todo)}, "
              f"~{sum(estimate(j, mpe) for j in todo) / 60:.1f} GPU-h")
        for j in todo:
            print(f"    {j['id']:40s} ~{estimate(j, mpe):5.0f} min")


def cmd_plan(a):
    rounds = json.load(open(ROUNDS))
    mpe, n = calibrate()
    jobs = {k: v for k, v in expand(a.round, rounds).items() if not (v["done"] and (ROOT / v["done"]).exists())}
    for j in jobs.values():
        j["est_min"] = estimate(j, mpe)
    cap = a.budget_hours * 60
    sessions = []  # each: {"lanes": [minutes per GPU], "jobs": [...]}
    for j in sorted(jobs.values(), key=lambda j: -j["est_min"]):  # longest first, first fit
        for s in sessions:
            lane = min(range(a.gpus), key=lambda i: s["lanes"][i])
            if s["lanes"][lane] + j["est_min"] <= cap:
                s["lanes"][lane] += j["est_min"]
                s["jobs"].append(j["id"])
                break
        else:
            lanes = [0.0] * a.gpus
            lanes[0] = j["est_min"]
            sessions.append({"lanes": lanes, "jobs": [j["id"]]})
    plan = {"round": a.round, "budget_hours": a.budget_hours, "gpus": a.gpus, "min_per_example": mpe,
            "n_calibrated": n, "created": time.strftime("%Y-%m-%d %H:%M"),
            "sessions": [{"session": i + 1, "jobs": s["jobs"], "est_hours": round(max(s["lanes"]) / 60, 2)}
                         for i, s in enumerate(sessions)],
            "jobs": jobs}
    PLAN.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    print(f"{a.round}: {len(jobs)} jobs in {len(sessions)} session(s) of <= {a.budget_hours} h on {a.gpus} GPUs "
          f"({mpe:.4f} min/example, {n} measured jobs)")
    for s in plan["sessions"]:
        print(f"  session {s['session']}: {len(s['jobs'])} jobs, ~{s['est_hours']} h  ->  bash scripts/run_kaggle.sh {s['session']}")
    print(f"wrote {PLAN.relative_to(ROOT)}; commit and push it before running on Kaggle")


def pack(tag, files, lock):
    with lock:
        zpath = ROOT / f"results_{tag}.zip"
        tmp = zpath.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in sorted(set(files)):
                p = ROOT / rel
                paths = [p] if p.is_file() else (sorted(p.rglob("*")) if p.exists() else [])
                for q in paths:
                    if q.is_file() and "checkpoint" not in q.relative_to(ROOT).parts:
                        z.write(q, q.relative_to(ROOT))
        tmp.replace(zpath)
        print(f"[pack] {zpath.name} ({zpath.stat().st_size / 1e6:.1f} MB)", flush=True)


def cmd_run(a):
    plan = json.load(open(PLAN))
    tag = f"{plan['round']}_s{a.session}"
    sess = next((s for s in plan["sessions"] if s["session"] == a.session), None)
    if not sess:
        sys.exit(f"no session {a.session} in {PLAN}")
    try:
        import torch
        ngpu = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(ngpu)]
    except Exception:
        ngpu, names = 0, []
    print(f"[gpu] {ngpu} CUDA device(s): {names}", flush=True)
    if ngpu == 0 and not os.environ.get("ALLOW_CPU"):
        sys.exit("ERROR: no GPU. Kaggle: Settings -> Accelerator -> GPU T4 x2 (also for Save & Run All).")
    key_rule = json.load(open(ROUNDS))["rounds"][plan["round"]].get("openai_key", "no")
    print(f"[openai] round {plan['round']} needs OPENAI_API_KEY: {key_rule}; "
          f"set in this session: {'yes' if os.environ.get('OPENAI_API_KEY') else 'no'}", flush=True)
    if key_rule.startswith("yes") and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: this round needs OPENAI_API_KEY. Kaggle: Add-ons -> Secrets -> OPENAI_API_KEY, "
                 "then export it before run_kaggle.sh (see RUNNING.md).")
    budget = (a.budget_hours or plan["budget_hours"]) * 60
    sess_dir = ROOT / f"outputs/_sessions/{tag}"
    sess_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/_logs").mkdir(parents=True, exist_ok=True)
    TIMES.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    (sess_dir / "session.json").write_text(json.dumps({"tag": tag, "commit": commit, "jobs": sess["jobs"],
                                                       "gpus": names, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    times_file = TIMES / f"{tag}.tsv"
    files = [f"outputs/_sessions/{tag}", f"outputs/_job_times/{tag}.tsv"]
    lock, t0 = threading.Lock(), time.time()
    q = queue.Queue()
    for jid in sorted(sess["jobs"], key=lambda j: -plan["jobs"][j]["est_min"]):
        q.put(plan["jobs"][jid])
    lanes = max(1, ngpu) if ngpu else 1

    def worker(gpu):
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            elapsed = (time.time() - t0) / 60
            if elapsed + job["est_min"] > budget:
                print(f"[gpu{gpu}] defer {job['id']}: ~{job['est_min']:.0f} min does not fit "
                      f"({elapsed:.0f}/{budget:.0f} min used)", flush=True)
                continue
            log = ROOT / f"outputs/_logs/{re.sub(r'[^A-Za-z0-9_.-]', '_', job['id'])}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)} if ngpu else dict(os.environ)
            print(f"[gpu{gpu}] start {job['id']} (~{job['est_min']:.0f} min) at {elapsed:.0f} min", flush=True)
            js = time.time()
            with open(log, "w") as lf:
                p = subprocess.Popen(job["command"], shell=True, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    lf.write(line)
                    print(f"[gpu{gpu} {job['id']}] {line}", end="", flush=True)
                rc = p.wait()
            mins = (time.time() - js) / 60
            with lock:
                with open(times_file, "a") as tf:
                    tf.write(f"{job['id']}\t{mins:.1f}\t{rc}\t{tag}\n")
                files.extend(job["outputs"] + [str(log.relative_to(ROOT))])
            print(f"[gpu{gpu}] end {job['id']} rc={rc} after {mins:.1f} min", flush=True)
            pack(tag, files, lock)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(lanes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(sess_dir / "session.json") as f:
        info = json.load(f)
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["minutes"] = round((time.time() - t0) / 60, 1)
    (sess_dir / "session.json").write_text(json.dumps(info, indent=1))
    pack(tag, files, lock)
    print(f"session {tag} done after {info['minutes']} min; download results_{tag}.zip into runs_inbox/", flush=True)


def cmd_ingest(a):
    INBOX.mkdir(exist_ok=True)
    ledger = ROOT / "outputs/_ingested.txt"
    seen = set(ledger.read_text().split()) if ledger.exists() else set()
    arch = ROOT / "outputs/_zips_archive"
    arch.mkdir(parents=True, exist_ok=True)
    zips = sorted(INBOX.glob("*.zip"))
    if not zips:
        print("runs_inbox/ is empty")
    for z in zips:
        if z.name in seen:
            print(f"{z.name}: already ingested, moved to archive")
        else:
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                zf.extractall(ROOT)
            runs = sorted({"/".join(n.split("/")[:3]) for n in names if re.match(r"outputs/[^_][^/]*/seed_\d+/", n)})
            print(f"{z.name}: {len(names)} files; seeds: {', '.join(runs) or '-'}")
            with open(ledger, "a") as f:
                f.write(z.name + "\n")
        shutil.move(str(z), arch / z.name)
    print("next: python3 src/report_rounds.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--round")
    p = sub.add_parser("plan"); p.add_argument("--round", required=True)
    p.add_argument("--budget-hours", type=float, default=10.5); p.add_argument("--gpus", type=int, default=2)
    p = sub.add_parser("run"); p.add_argument("--session", type=int, required=True)
    p.add_argument("--budget-hours", type=float, default=0)
    sub.add_parser("ingest")
    a = ap.parse_args()
    {"list": cmd_list, "plan": cmd_plan, "run": cmd_run, "ingest": cmd_ingest}[a.cmd](a)


if __name__ == "__main__":
    main()
