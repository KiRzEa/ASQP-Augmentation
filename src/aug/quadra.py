"""QUADRA steps 2-4: budget per cell, recompose units into new reviews, optional LLM fusion, rule checks.

    python src/aug/quadra.py --out data/aug/quadra_rule.jsonl                       # rules only
    python src/aug/quadra.py --out data/aug/quadra_gpt.jsonl --llm_model <model>    # + LLM fusion (OpenAI)
    python src/aug/quadra.py ... --difficulty data/aug/difficulty.json              # frequency × difficulty budget

Cells are category|polarity. Budget (cf. ADA's condition function): cells sorted by train frequency, ceiling
c(pos) = kappa * n1 * (exp(-gamma * pos) + eta), quota = max(0, ceiling - n), weighted by difficulty
w = 1 + lam * (1 - F1_cell / 100) when out-of-fold F1 is given; quotas are scaled so that the number of new
reviews equals ratio × |train|. Each new review is anchored on a unit of its target cell and joined with 1-2
related units (same entity or an under-quota cell), with optional same-cell span substitution. All spans keep
character offsets; every output is checked to be grounded at word boundaries. The model-based check (step 4b)
is src/aug/verify_model.py.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import NULL, load_jsonl  # noqa: E402
from decompose import decompose  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def cell(q):
    return f"{q['category']}|{q['polarity']}"


def bounded(text, span):
    return [(m.start(), m.end()) for m in re.finditer(r"(?<!\w)" + re.escape(span) + r"(?!\w)", text)]


def grounded(item):
    for q in item["quads"]:
        for f in ("aspect", "opinion"):
            o = q.get(f + "_offsets")
            if (q[f] == NULL) != (o is None):
                return False
            if o and (item["text"][o[0]:o[1]] != q[f] or (o[0], o[1]) not in bounded(item["text"], q[f])):
                return False
    return True


# ---------------- step 2: budget ----------------

def budget(train, total, gamma, eta, kappa, lam, difficulty):
    n = collections.Counter(cell(q) for it in train for q in it["quads"])
    order = [c for c, _ in n.most_common()]
    n1 = n[order[0]]
    raw = {}
    for pos, c in enumerate(order):
        ceiling = kappa * n1 * (math.exp(-gamma * pos) + eta)
        w = 1.0 + lam * (1.0 - difficulty.get(c, 100.0) / 100.0) if difficulty else 1.0
        raw[c] = max(0.0, ceiling - n[c]) * w
    s = sum(raw.values()) or 1.0
    alloc = {c: int(round(total * v / s)) for c, v in raw.items()}
    return alloc, n


# ---------------- step 3: recompose ----------------

def substitute(unit, q_idx, field, new_span):
    """Replace one explicit span of one quad, shifting offsets of everything after it."""
    q = unit["quads"][q_idx]
    s, e = q[field + "_offsets"]
    delta = len(new_span) - (e - s)
    text = unit["text"][:s] + new_span + unit["text"][e:]
    quads = []
    for i, x in enumerate(unit["quads"]):
        y = dict(x)
        for f in ("aspect", "opinion"):
            o = x.get(f + "_offsets")
            if not o:
                continue
            if o == [s, e]:  # the replaced span (possibly shared by several quads)
                y[f], y[f + "_offsets"] = new_span, [s, s + len(new_span)]
            elif o[0] >= e:
                y[f + "_offsets"] = [o[0] + delta, o[1] + delta]
            elif o[1] > s:  # overlaps the replaced span partially: give up
                return None
        quads.append(y)
    return {**unit, "text": text, "quads": quads}


def join(units, rng):
    text, quads = "", []
    for i, u in enumerate(units):
        if i:
            prev_pol = units[i - 1]["quads"][0]["polarity"]
            pol = u["quads"][0]["polarity"]
            text += " nhưng " if prev_pol != pol and rng.random() < 0.7 else rng.choice([", ", " và ", ", "])
        off = len(text)
        text += u["text"]
        for q in u["quads"]:
            y = dict(q)
            for f in ("aspect", "opinion"):
                o = q.get(f + "_offsets")
                y[f + "_offsets"] = [o[0] + off, o[1] + off] if o else None
            quads.append(y)
    return {"text": text, "quads": quads}


def recompose(train, alloc, args, rng):
    units = [u for it in train for u in decompose(it) if u["split"]]
    by_cell, by_entity = collections.defaultdict(list), collections.defaultdict(list)
    for i, u in enumerate(units):
        for c in {cell(q) for q in u["quads"]}:
            by_cell[c].append(i)
        for e in {q["category"].split("#")[0] for q in u["quads"]}:
            by_entity[e].append(i)
    spans = collections.defaultdict(lambda: collections.defaultdict(set))
    for u in units:
        for q in u["quads"]:
            for f in ("aspect", "opinion"):
                if q[f] != NULL:
                    spans[(cell(q), f)][q[f]].add(1)
    span_list = {k: sorted(v) for k, v in spans.items()}
    under = sorted(c for c, k in alloc.items() if k > 0)  # sorted: set order depends on PYTHONHASHSEED
    seen = {it["text"] for it in train}
    out, stats = [], collections.Counter()
    for c, k in sorted(alloc.items(), key=lambda kv: -kv[1]):
        if not by_cell.get(c):
            stats["cells_without_units"] += 1
            continue
        made, tries = 0, 0
        while made < k and tries < 50 * k:
            tries += 1
            anchor = units[rng.choice(by_cell[c])]
            ops = []
            if rng.random() < args.p_sub:
                cand = [(i, f) for i, q in enumerate(anchor["quads"]) if cell(q) == c for f in ("aspect", "opinion")
                        if q[f] != NULL and len(span_list[(c, f)]) > 1]
                if cand:
                    i, f = rng.choice(cand)
                    new = rng.choice([s for s in span_list[(c, f)] if s != anchor["quads"][i][f]])
                    sub = substitute(anchor, i, f, new)
                    if sub:
                        anchor, ops = sub, ops + [f"sub_{f}"]
            r = rng.random()
            n_partners = 0 if r < args.p_single else (1 if r < args.p_single + 0.5 * (1 - args.p_single) else 2)
            entity = c.split("#")[0]
            pool = by_entity[entity] if rng.random() < 0.5 else [i for x in under for i in by_cell[x]]
            partners = []
            for _ in range(n_partners):
                if not pool:
                    break
                p = units[rng.choice(pool)]
                if p["source_id"] != anchor["source_id"] and all(p["source_id"] != x["source_id"] for x in partners):
                    partners.append(p)
            parts = [anchor] + partners
            if sum(len(p["quads"]) for p in parts) > args.max_quads:
                continue
            asp = [q["aspect"] for p in parts for q in p["quads"] if q["aspect"] != NULL]
            owners = collections.defaultdict(set)
            for i, p in enumerate(parts):
                for q in p["quads"]:
                    if q["aspect"] != NULL:
                        owners[q["aspect"]].add(i)
            if any(len(v) > 1 for v in owners.values()):  # same aspect in two pieces: risk of contradiction
                stats["rejected_shared_aspect"] += 1
                continue
            rng.shuffle(parts)
            new_item = join(parts, rng)
            if new_item["text"] in seen or not grounded(new_item):
                stats["rejected_duplicate_or_ungrounded"] += 1
                continue
            seen.add(new_item["text"])
            new_item.update(id=f"quadra-{len(out):05d}", meta={"target_cell": c, "sources": [p["source_id"] for p in parts],
                                                                "ops": ops + [f"join_{len(parts)}"]})
            out.append(new_item)
            made += 1
        stats["short_of_quota"] += k - made
    return out, stats


# ---------------- optional LLM fusion (OpenAI, cached) ----------------

PROMPT = """Bạn là người Việt hay viết review quán ăn trên mạng. Đoạn dưới đây được ghép từ vài nhận xét ngắn nên đọc còn rời rạc. Hãy sửa thành MỘT câu review liền mạch, tự nhiên, giọng văn thân mật, viết thường toàn bộ, không dấu chấm cuối.
Chỉ được phép: thêm hoặc đổi từ nối (như "mà", "nhưng", "còn", "với lại", "nói chung", "thì", "nha", "á"), thêm dấu phẩy, đổi thứ tự các ý.
Tuyệt đối KHÔNG được:
- thêm tính từ, nhận xét, cảm xúc hay đánh giá mới (ví dụ không thêm "ấm cúng", "chật chội", "dễ chịu")
- đổi ý khen/chê của bất kỳ nhận xét nào
- sửa, tách hay thêm chữ vào các cụm sau, phải giữ NGUYÊN VĂN: {spans}
Chỉ trả về câu review.

Đoạn gốc: {text}"""


def load_env(path):
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
            if m and m.group(1) not in os.environ:
                os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


def openai_chat(model, prompt, temperature):
    body = json.dumps({"model": model, "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                                          "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(2 ** attempt * 3)
                continue
            raise


def fuse(items, args, cache_path):
    cache = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[r["key"]] = r
    out, stats = [], collections.Counter()
    jobs = []
    for it in items:
        span_set = sorted({q[f] for q in it["quads"] for f in ("aspect", "opinion") if q[f] != NULL})
        prompt = PROMPT.format(spans="; ".join(f'"{s}"' for s in span_set), text=it["text"])
        key = hashlib.sha256(f"{args.llm_model}|{args.llm_temperature}|{prompt}".encode()).hexdigest()
        jobs.append((it, prompt, key))
    todo = {key: prompt for _, prompt, key in jobs if key not in cache}

    def call(kp):
        key, prompt = kp
        resp = openai_chat(args.llm_model, prompt, args.llm_temperature)
        return {"key": key, "model": args.llm_model, "resolved_model": resp.get("model"), "prompt": prompt,
                "output": resp["choices"][0]["message"]["content"].strip(), "usage": resp.get("usage", {}),
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    print(f"LLM: {len(jobs)} reviews, {len(jobs) - len(todo)} cached, {len(todo)} API calls with {args.llm_workers} workers",
          flush=True)
    with open(cache_path, "a", encoding="utf-8") as cf, ThreadPoolExecutor(args.llm_workers) as pool:
        for i, rec in enumerate(pool.map(call, todo.items()), 1):
            cache[rec["key"]] = rec
            cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cf.flush()
            stats["api_calls"] += 1
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} calls", flush=True)
    for it, prompt, key in jobs:
        if True:
            rec = cache[key]
            for k in ("prompt_tokens", "completion_tokens"):
                stats[k] += rec["usage"].get(k, 0)
            text = re.sub(r"\s+", " ", rec["output"].strip().strip('"')).strip()
            fused = None
            if len(text.split()) > args.max_len_ratio * len(it["text"].split()):  # added content = likely new opinions
                stats["fusion_rejected_too_long"] += 1
                out.append({**it, "meta": {**it["meta"], "llm": "failed"}})
                continue
            for cand in (text, text[:1].lower() + text[1:], text.lower()):  # the data are lowercase
                fused = relocate(cand, it)
                if fused:
                    break
            if fused is None:
                stats["fusion_failed_kept_rule_version"] += 1
                out.append({**it, "meta": {**it["meta"], "llm": "failed"}})
            else:
                stats["fusion_ok"] += 1
                out.append({**fused, "id": it["id"], "meta": {**it["meta"], "llm": args.llm_model, "pre_fusion": it["text"]}})
    return out, stats


def relocate(text, item):
    """Find every span of the item in the fused text (word boundaries); choose closest aspect/opinion pairs."""
    quads = []
    for q in item["quads"]:
        occ = {f: (bounded(text, q[f]) if q[f] != NULL else [None]) for f in ("aspect", "opinion")}
        if not occ["aspect"] or not occ["opinion"]:
            return None
        best, dist = (occ["aspect"][0], occ["opinion"][0]), None
        for a in occ["aspect"]:
            for o in occ["opinion"]:
                if a and o:
                    d = abs(a[0] - o[0])
                    if dist is None or d < dist:
                        best, dist = (a, o), d
        quads.append({**q, "aspect_offsets": list(best[0]) if best[0] else None,
                      "opinion_offsets": list(best[1]) if best[1] else None})
    new = {"text": text, "quads": quads}
    return new if grounded(new) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=1.0, help="new reviews = ratio × |train|")
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0, help="difficulty weight strength (needs --difficulty)")
    ap.add_argument("--difficulty", help="JSON {cell: out-of-fold F1} from src/aug/difficulty.py")
    ap.add_argument("--p_sub", type=float, default=0.5)
    ap.add_argument("--p_single", type=float, default=0.2, help="share of new reviews that are a single unit")
    ap.add_argument("--max_quads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--llm_model", help="OpenAI model id (pin a dated snapshot); omit for rules only")
    ap.add_argument("--llm_temperature", type=float, default=0.3)
    ap.add_argument("--max_len_ratio", type=float, default=1.4, help="reject fusions longer than this × the rule text (in syllables)")
    ap.add_argument("--llm_limit", type=int, default=0, help="fuse only the first N samples (test)")
    ap.add_argument("--llm_workers", type=int, default=8, help="parallel API calls")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    train = load_jsonl(a.train_path)
    diff = json.load(open(a.difficulty)) if a.difficulty else None
    total = int(round(a.ratio * len(train)))
    alloc, n = budget(train, total, a.gamma, a.eta, a.kappa, a.lam, diff)
    items, stats = recompose(train, alloc, a, rng)
    if a.llm_model:
        load_env(ROOT / ".env")
        sel = items[:a.llm_limit] if a.llm_limit else items
        fused, fstats = fuse(sel, a, Path(a.out).with_suffix(".llm_cache.jsonl"))
        items = fused + items[len(sel):] if a.llm_limit else fused
        stats.update(fstats)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            assert grounded(it)
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    gen_cells = collections.Counter(cell(q) for it in items for q in it["quads"])
    report = {"args": vars(a), "n_train": len(train), "n_generated": len(items), "stats": dict(stats),
              "cells": {c: {"train": n[c], "allocated_reviews": alloc.get(c, 0), "generated_quads": gen_cells[c]}
                        for c in sorted(n, key=lambda c: n[c])}}
    json.dump(report, open(Path(a.out).with_suffix(".report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in ("n_train", "n_generated", "stats")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
