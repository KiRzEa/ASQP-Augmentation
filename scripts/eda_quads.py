"""EDA and data-quality audit for the Vietnamese restaurant quad data (read-only).

python3 scripts/eda_quads.py <data_dir>   # folder with Train_.csv, Dev_.csv, Test_.csv
"""
import collections
import csv
import re
import sys
import unicodedata
from pathlib import Path

D = Path(sys.argv[1])
SPLITS = ["Train", "Dev", "Test"]
QRE = re.compile(r"\{([^{}]*)\}")
CATS = {"FOOD#QUALITY", "FOOD#STYLE&OPTIONS", "FOOD#PRICES", "DRINKS#QUALITY", "DRINKS#STYLE&OPTIONS", "DRINKS#PRICES",
        "SERVICE#GENERAL", "AMBIENCE#GENERAL", "LOCATION#GENERAL", "RESTAURANT#GENERAL", "RESTAURANT#PRICES", "RESTAURANT#MISCELLANEOUS"}
POLS = {"positive", "negative", "neutral"}


def parse(s):
    out, bad = [], []
    for m in QRE.findall(s):
        p = [x.strip() for x in m.split(",")]
        if len(p) != 4:
            bad.append(m)
        out.append(tuple(p[:3]) + (",".join(p[3:]),) if len(p) >= 4 else tuple(p))
    leftover = QRE.sub("", s).replace(";", "").strip()
    return out, bad, leftover


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", s).lower())).strip()


rows = {s: list(csv.DictReader(open(D / f"{s}_.csv", encoding="utf-8"))) for s in SPLITS}
issues = collections.defaultdict(list)

print("=" * 70, "\n1. IMPLICIT / EXPLICIT BREAKDOWN (ACOS convention)\n" + "=" * 70)
print(f"{'split':6s} {'quads':>6s} {'EA-EO':>12s} {'EA-IO':>12s} {'IA-EO':>12s} {'IA-IO':>12s} | sents with ≥1 implicit")
for s in SPLITS:
    c, n, sent_imp = collections.Counter(), 0, 0
    for r in rows[s]:
        q, _, _ = parse(r["output_raw"])
        has = False
        for x in q:
            if len(x) < 4:
                continue
            k = ("IA" if x[1] == "null" else "EA") + "-" + ("IO" if x[3] == "null" else "EO")
            c[k] += 1; n += 1; has |= k != "EA-EO"
        sent_imp += has
    cells = " ".join(f"{c[k]:5d} ({100*c[k]/n:4.1f}%)" for k in ["EA-EO", "EA-IO", "IA-EO", "IA-IO"])
    print(f"{s:6s} {n:6d} {cells} | {sent_imp}/{len(rows[s])} ({100*sent_imp/len(rows[s]):.0f}%)")

print("\nImplicit share by category (train):")
cc = collections.defaultdict(collections.Counter)
for r in rows["Train"]:
    for x in parse(r["output_raw"])[0]:
        if len(x) == 4:
            cc[x[0]]["n"] += 1; cc[x[0]]["IA"] += x[1] == "null"; cc[x[0]]["IO"] += x[3] == "null"
for k, v in sorted(cc.items(), key=lambda kv: -kv[1]["n"]):
    print(f"  {k:26s} n={v['n']:5d}  implicit aspect {100*v['IA']/v['n']:5.1f}%  implicit opinion {100*v['IO']/v['n']:5.1f}%")

print("\nImplicit share by polarity (train):")
pc = collections.defaultdict(collections.Counter)
for r in rows["Train"]:
    for x in parse(r["output_raw"])[0]:
        if len(x) == 4:
            pc[x[2]]["n"] += 1; pc[x[2]]["IA"] += x[1] == "null"; pc[x[2]]["IO"] += x[3] == "null"
for k, v in pc.items():
    print(f"  {k:9s} n={v['n']:5d}  implicit aspect {100*v['IA']/v['n']:5.1f}%  implicit opinion {100*v['IO']/v['n']:5.1f}%")

print("\n" + "=" * 70, "\n2. DATA-QUALITY AUDIT\n" + "=" * 70)
for s in SPLITS:
    for i, r in enumerate(rows[s]):
        rid, text = f"{s}:{i+2}", r["input_raw"]
        q, bad, leftover = parse(r["output_raw"])
        if bad: issues["malformed quad (≠4 fields)"].append((rid, bad[0]))
        if leftover: issues["text outside braces in label"].append((rid, leftover[:40]))
        if not q: issues["no quads"].append((rid, text[:60]))
        if text != unicodedata.normalize("NFC", text): issues["text not Unicode NFC"].append((rid, text[:60]))
        if text != text.strip() or "  " in text: issues["extra whitespace in text"].append((rid, repr(text[:40])))
        if re.search(r"(.)\1{2,}", text): issues["repeated characters (e.g. 'ngonnn')"].append((rid, re.search(r"\S*(.)\1{2,}\S*", text).group(0)))
        if re.search(r"https?://|www\.", text): issues["URL in text"].append((rid, text[:60]))
        if re.search(r"[\U0001F300-\U0001FAFF☀-➿]", text): issues["emoji in text"].append((rid, text[:60]))
        if any(ch.isupper() for ch in text): issues["uppercase letters in text"].append((rid, text[:50]))
        seen = collections.Counter(x for x in q if len(x) == 4)
        for x, k in seen.items():
            if k > 1: issues["duplicate quad in one sentence"].append((rid, x))
        for x in q:
            if len(x) != 4: continue
            c, a, p, o = x
            if c not in CATS: issues["unknown category"].append((rid, c))
            if p not in POLS: issues["unknown polarity"].append((rid, p))
            for name, sp in (("aspect", a), ("opinion", o)):
                if sp == "null": continue
                if sp.lower() in {"none", "nan", ""}: issues[f"{name} looks like a null variant"].append((rid, sp))
                if sp != sp.strip(" ,.!?;:"): issues[f"{name} span has edge punctuation/space"].append((rid, repr(sp)))
                if sp not in text:
                    if sp.lower() in text.lower(): issues[f"{name} span matches only case-insensitively"].append((rid, sp))
                    else: issues[f"{name} span not in text"].append((rid, sp))
                elif len(re.findall(re.escape(sp), text)) > 1: issues[f"{name} span occurs more than once (ambiguous position)"].append((rid, sp))
                if sp in text and re.search(r"\w" + re.escape(sp) + r"|" + re.escape(sp) + r"\w", text) and not re.search(r"(?<!\w)" + re.escape(sp) + r"(?!\w)", text):
                    issues[f"{name} span cuts inside a word"].append((rid, sp))
            if a != "null" and a == o: issues["aspect == opinion"].append((rid, a))
        # same (aspect, opinion) with different polarity, or same (aspect, opinion, category) twice with other polarity
        ao = collections.defaultdict(set)
        for x in q:
            if len(x) == 4: ao[(x[0], x[1], x[3])].add(x[2])
        for k, v in ao.items():
            if len(v) > 1: issues["conflicting polarity for same (cat, aspect, opinion)"].append((rid, k, v))
        both = [x for x in q if len(x) == 4 and x[1] == "null" and x[3] == "null"]
        cats_null = collections.Counter(x[0] for x in both)
        for c, k in cats_null.items():
            if k > 1: issues["same category twice with both spans implicit"].append((rid, c))

# duplicates
for s in SPLITS:
    c = collections.Counter(norm(r["input_raw"]) for r in rows[s])
    for t, k in c.items():
        if k > 1:
            labs = {r["output_raw"] for r in rows[s] if norm(r["input_raw"]) == t}
            issues[f"duplicate sentence within {s}" + (" (DIFFERENT labels)" if len(labs) > 1 else " (same labels)")].append((s, t[:60]))
sets = {s: {norm(r["input_raw"]) for r in rows[s]} for s in SPLITS}
for a, b in [("Train", "Dev"), ("Train", "Test"), ("Dev", "Test")]:
    for t in sets[a] & sets[b]:
        issues[f"sentence in both {a} and {b}"].append((t[:60],))

# consistency of the two file versions
for s in SPLITS:
    other = list(csv.DictReader(open(D / f"{s}.csv", encoding="utf-8")))
    diff = sum(1 for x, y in zip(rows[s], other) if x["output_raw"] != y["output_raw"] or x["input_raw"] != y["input_raw"])
    if diff or len(other) != len(rows[s]): issues[f"{s}.csv and {s}_.csv disagree"].append((diff, len(other), len(rows[s])))

for k in sorted(issues, key=lambda k: -len(issues[k])):
    v = issues[k]
    print(f"\n[{len(v):5d}] {k}")
    for e in v[:4]:
        print("        e.g.", e)
