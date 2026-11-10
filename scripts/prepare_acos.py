"""Convert the English ACOS data (Cai et al., ACL 2021) to the JSONL format of data/restaurant_v1.

    python3 scripts/prepare_acos.py data/acos/raw data/acos

Reads <raw>/Restaurant-ACOS/rest16_quad_{train,dev,test}.tsv and <raw>/Laptop-ACOS/laptop_quad_*.tsv.
Each line: text<TAB>quad<TAB>quad..., quad = "a_start,a_end CATEGORY sentiment o_start,o_end" with
token offsets (end exclusive, -1,-1 = implicit) and sentiment 0/1/2 = negative/neutral/positive.
Writes <out>/{rest,laptop}/{train,dev,test}.jsonl and <out>/stats.json.

No cleaning: published ACOS numbers use these files as released, so every sentence and quad is kept
(exact duplicate quads included). Problems are only counted in stats.json. The one exception is an
empty span (start == end), which has no text and is written as null.
"""
import collections
import json
import sys
from pathlib import Path

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DOMAINS = {"rest": ("Restaurant-ACOS", "rest16"), "laptop": ("Laptop-ACOS", "laptop")}
POL = {"0": "negative", "1": "neutral", "2": "positive"}
NULL = "null"


def token_spans(text):
    """Character (start, end) of each whitespace token."""
    spans, i = [], 0
    for tok in text.split():
        i = text.index(tok, i)
        spans.append((i, i + len(tok)))
        i += len(tok)
    return spans


def span(text, spans, field, st):
    s, e = (int(x) for x in field.split(","))
    if s == -1:
        return NULL, None
    if s == e:  # empty span (one case: rest train line 451, opinion "3,3"): no text, so treat as implicit
        st["empty span -> null"] += 1
        return NULL, None
    if not (0 <= s < e <= len(spans)):
        raise ValueError(f"bad offsets {field} for {len(spans)} tokens")
    a, b = spans[s][0], spans[e - 1][1]
    return text[a:b], [a, b]


def implicit_type(q):
    return ("IA" if q["aspect"] == NULL else "EA") + "-" + ("IO" if q["opinion"] == NULL else "EO")


stats = {}
for dom, (folder, prefix) in DOMAINS.items():
    (OUT / dom).mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        src = RAW / folder / f"{prefix}_quad_{split}.tsv"
        rows, st = [], collections.Counter()
        cats = collections.Counter()
        for ln, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            text, *qfields = line.split("\t")
            text = text.strip()
            spans = token_spans(text)
            quads = []
            for qf in qfields:
                if not qf.strip():
                    continue
                a_off, cat, pol, o_off = qf.split()
                aspect, aoff = span(text, spans, a_off, st)
                opinion, ooff = span(text, spans, o_off, st)
                q = {"category": cat, "aspect": aspect, "polarity": POL[pol], "opinion": opinion,
                     "aspect_offsets": aoff, "opinion_offsets": ooff}
                quads.append(q)
                st[implicit_type(q)] += 1
                cats[cat] += 1
            keys = [tuple(sorted(q.items(), key=lambda kv: kv[0])) for q in quads]
            st["duplicate quads (kept)"] += len(keys) - len(set(map(str, keys)))
            if not quads:
                st["sentences without quads"] += 1
            rows.append({"id": f"{dom}-{split}-{ln:04d}", "text": text, "quads": quads, "src_line": ln})
        with open(OUT / dom / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        nq = sum(len(r["quads"]) for r in rows)
        stats[f"{dom}/{split}"] = {"sentences": len(rows), "quads": nq,
                                   "quads_per_sentence": round(nq / len(rows), 2),
                                   "categories": len(cats), **dict(st)}

(OUT / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(stats, indent=1))
