"""Prepare the Vietnamese restaurant ASQP/ACOS data: clean, add span offsets, write JSONL.

    python3 scripts/prepare_data.py <src_dir> <out_dir>

<src_dir> holds Train_.csv, Dev_.csv, Test_.csv (columns input_raw, input, output, output_raw;
quads in output_raw as {CATEGORY,aspect,polarity,opinion}; "null" = implicit).
Writes <out_dir>/{train,dev,test}.jsonl, changes.csv (every edit) and stats.json.
The source files are never modified. Cleaning rules (C1-C10) are documented in notes/PAPER_NOTES.md.
"""
import collections
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SPLITS = {"train": "Train", "dev": "Dev", "test": "Test"}
QRE = re.compile(r"\{([^{}]*)\}")
NULL = "null"

# Manual decisions (split, csv line number as shown in a spreadsheet, i.e. header = line 1).
# Each was inspected by hand; see notes/PAPER_NOTES.md, "Cleaning".
MANUAL_SPAN = {
    # "hơi bị ổn" is a typo for "ồn" (noisy); the annotated "ồn" only matched inside "mất hồn".
    ("train", 1373, "AMBIENCE#GENERAL", "không gian", "negative", "ồn"): ("opinion", "ổn"),
    # "so với giá giá": the opinion duplicated the aspect; the price judgement is implicit.
    ("train", 579, "FOOD#PRICES", "giá", "negative", "giá"): ("opinion", NULL),
}
# Same (category, aspect, opinion) annotated with two polarities: keep the one listed here.
MANUAL_CONFLICT = {("train", 23, "LOCATION#GENERAL", NULL, "ngay trung tâm"): "positive"}

changes = []


def log(split, line, rule, before, after):
    changes.append({"split": split, "line": line, "rule": rule, "before": before, "after": after})


def norm_key(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", s).lower())).strip()


def bounded(text, span):
    """Character offsets of span occurrences that start and end on word boundaries."""
    return [(m.start(), m.end()) for m in re.finditer(r"(?<!\w)" + re.escape(span) + r"(?!\w)", text)]


def extend_to_words(text, span):
    """C1: a span cut inside a word is extended to the whole word(s) it touches."""
    i = text.find(span)
    if i < 0:
        return span
    s, e = i, i + len(span)
    while s > 0 and text[s - 1].isalnum():
        s -= 1
    while e < len(text) and text[e].isalnum():
        e += 1
    return text[s:e]


def parse(raw):
    quads = []
    for m in QRE.findall(raw):
        p = [x.strip() for x in m.split(",")]
        quads.append({"category": p[0], "aspect": p[1], "polarity": p[2], "opinion": ",".join(p[3:]).strip()})
    return quads


def pick_offsets(text, q):
    """C8: choose the occurrences of aspect and opinion that lie closest to each other."""
    occ = {k: (bounded(text, q[k]) if q[k] != NULL else [None]) for k in ("aspect", "opinion")}
    best, dist = (occ["aspect"][0], occ["opinion"][0]), None
    for a in occ["aspect"]:
        for o in occ["opinion"]:
            if a is None or o is None:
                continue
            d = abs((a[0] + a[1]) / 2 - (o[0] + o[1]) / 2)
            if dist is None or d < dist:
                best, dist = (a, o), d
    return best


data = {}
for split, name in SPLITS.items():
    rows = list(csv.DictReader(open(SRC / f"{name}_.csv", encoding="utf-8")))
    items = []
    for idx, r in enumerate(rows):
        line = idx + 2
        text = unicodedata.normalize("NFC", r["input_raw"])
        clean = re.sub(r"\s+", " ", text).strip()  # C7
        if clean != text:
            log(split, line, "C7 whitespace", text, clean)
        quads = []
        for q in parse(r["output_raw"]):
            key = (split, line, q["category"], q["aspect"], q["polarity"], q["opinion"])
            if key in MANUAL_SPAN:  # C1/C4 manual
                field, value = MANUAL_SPAN[key]
                log(split, line, "C1/C4 manual span fix", f"{field}={q[field]}", f"{field}={value}")
                q[field] = value
            for field in ("aspect", "opinion"):  # C1 automatic
                sp = q[field]
                if sp != NULL and sp in clean and not bounded(clean, sp):
                    new = extend_to_words(clean, sp)
                    log(split, line, f"C1 truncated {field}", sp, new)
                    q[field] = new
            quads.append(q)
        # C2 duplicate quads
        seen, uniq = set(), []
        for q in quads:
            k = (q["category"], q["aspect"], q["polarity"], q["opinion"])
            if k in seen:
                log(split, line, "C2 duplicate quad", str(k), "removed")
                continue
            seen.add(k)
            uniq.append(q)
        # C3 conflicting polarity
        groups = collections.defaultdict(list)
        for q in uniq:
            groups[(q["category"], q["aspect"], q["opinion"])].append(q)
        final = []
        for k, qs in groups.items():
            if len(qs) > 1:
                keep = MANUAL_CONFLICT.get((split, line) + k)
                if keep is None:
                    raise SystemExit(f"unresolved polarity conflict at {split}:{line} {k}")
                log(split, line, "C3 conflicting polarity", str([q["polarity"] for q in qs]), keep)
                qs = [q for q in qs if q["polarity"] == keep]
            final.extend(qs)
        # spans must now be grounded in the text
        for q in final:
            for field in ("aspect", "opinion"):
                if q[field] != NULL and not bounded(clean, q[field]):
                    raise SystemExit(f"span not grounded at {split}:{line}: {q[field]!r}")
        for q in final:  # C8 offsets
            a, o = pick_offsets(clean, q)
            q["aspect_offsets"], q["opinion_offsets"] = (list(a) if a else None), (list(o) if o else None)
        items.append({"id": f"{split}-{line:04d}", "text": clean, "quads": final, "src_line": line})
    data[split] = items

# C5 duplicate sentences inside train (after span repair)
by_text = collections.defaultdict(list)
for it in data["train"]:
    by_text[norm_key(it["text"])].append(it)
drop = set()
for t, its in by_text.items():
    if len(its) < 2:
        continue
    labels = {json.dumps(sorted((q["category"], q["aspect"], q["polarity"], q["opinion"]) for q in it["quads"])) for it in its}
    lines = [it["src_line"] for it in its]
    if len(labels) == 1:
        drop.update(it["id"] for it in its[1:])
        log("train", lines, "C5 duplicate sentence, same labels", t[:80], f"kept line {lines[0]}")
    else:
        drop.update(it["id"] for it in its)
        log("train", lines, "C5 duplicate sentence, DIFFERENT labels", t[:80], "all copies removed")

# C6 train sentences that also occur in dev or test
evals = {norm_key(it["text"]) for s in ("dev", "test") for it in data[s]}
for it in data["train"]:
    if it["id"] not in drop and norm_key(it["text"]) in evals:
        drop.add(it["id"])
        log("train", it["src_line"], "C6 sentence also in dev/test", it["text"][:80], "removed from train")
data["train"] = [it for it in data["train"] if it["id"] not in drop]

OUT.mkdir(parents=True, exist_ok=True)
stats = {}
for split, items in data.items():
    with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = collections.Counter()
    for it in items:
        for q in it["quads"]:
            c["quads"] += 1
            c[("IA" if q["aspect"] == NULL else "EA") + "-" + ("IO" if q["opinion"] == NULL else "EO")] += 1
    stats[split] = {"sentences": len(items), **c}
with open(OUT / "changes.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["split", "line", "rule", "before", "after"])
    w.writeheader()
    w.writerows(changes)
rule_counts = collections.Counter((c["split"], c["rule"]) for c in changes)
stats["changes"] = {f"{s}: {r}": n for (s, r), n in sorted(rule_counts.items())}
json.dump(stats, open(OUT / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
