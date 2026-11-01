"""Label-aware infilling of training reviews (GPT, runs locally, key in .env).

    python src/aug/infill_labelled.py --train_path data/restaurant_v1/inner_train.jsonl --mode targeted \
        --out data/aug/dv_infill_t.jsonl            # pilot: only the inner training split is used
    python src/aug/infill_labelled.py --mode targeted --out data/aug/infill_t.jsonl          # confirmatory: full train

A training review keeps its real wording except for 1-2 hidden opinion units. A unit is one quad that shares no span
with another quad, from its first span to the next labelled span (so an implicit opinion's wording is hidden too);
units are added until ~--mask_ratio of the words are hidden. Each blank replaces one quad and is filled with exactly
one new quad, so the number of quads per review is that of the source review. The prompt states the annotation
convention of the dataset with --shots labelled training reviews as examples. Labels:
  - visible quads keep their gold label, offsets moved with the visible text (the visible text must stay verbatim);
  - every filled quad is declared by the model (aspect / opinion span or null, category, polarity) and accepted only
    if it follows its directive and its explicit spans are found inside its own blank.
Directives per filled quad:
  --mode untargeted  category, polarity and implicit type of the hidden quad (new content, same structure);
  --mode targeted    as untargeted, plus what the model gets wrong on unseen reviews (train out-of-fold analysis,
                     scripts/error_factors.py: unseen (aspect, opinion) pairs and terms, opinion before aspect): a new
                     aspect term (not one of the category's 20 most frequent training aspects) with p --p_new_aspect,
                     a new opinion wording (not one of the 15 most frequent for the category and polarity) with p
                     --p_new_opinion, opinion before aspect with p --p_opinion_first (about 2x its training rate).
                     Polarity and implicit type stay those of the hidden quad: in the pilot, directives asking for more
                     neutral or implicit quads produced none that passed the blind verification.
Blind verification: a second call of the model sees only the filled reviews (not the directives) and annotates
them under the same convention; a filled quad is kept only if the blind annotation has a quad with the same polarity
and matching spans (equal, or overlapping by a word; null with null). The category is taken from the blind
annotation (the generator was asked for a category; the annotator reads it from the text), counted in the report. Sources are drawn with weights
1 / (share of reviews with that number of quads whose blanks can be planned), so the planning step does not shift
the number of quads per review away from train.
Selection: --overgen x |out| verified reviews are collected, then |out| are drawn without replacement with weights
target(key) / available(key) averaged over their filled quads, key = (polarity, implicit type), target = the
training distribution; the report gives target and achieved shares. Verification removes hard cases more often (annotators disagree on neutral and
implicit quads); the resampling restores them as far as the verified pool allows.
Convention checks on every filled quad: aspect and opinion do not overlap, opinion <= 4 words, aspect <= 5 words, a
leading degree adverb (hơi, khá, rất, quá, siêu, cực, cũng) is left out of the opinion span as in the training data.
Reviews with a fully implicit (IA-IO) quad, or whose units cannot be hidden cleanly, are not used as sources.
Filters: exact duplicates, near-duplicates (word-trigram Jaccard >= 0.5) of the --heldout reviews. Calls are cached.
Writes <out> and <out>.report.json (drops by reason, label distributions of train vs output, novelty of new terms).
"""
import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import CATEGORY_VI, NULL, POLARITY_VI, VI_CATEGORY, VI_POLARITY, implicit_type, load_jsonl, quad_tuple  # noqa: E402
from quadra import load_env, openai_chat  # noqa: E402
from synth_reviews import NearDup, merge, norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

PROMPT = """Quy ước gán nhãn của bộ dữ liệu: aspect là cụm danh từ chỉ đối tượng được nhận xét (món, đồ uống, nhân viên, giá, không gian…); opinion là cụm đánh giá NGẮN NHẤT (thường 1-3 từ, không chứa aspect, không gồm trạng từ mức độ như hơi, khá, rất, quá); null nếu không được nêu ra. Ví dụ gán nhãn đúng từ dữ liệu:
{shots}

Dưới đây là các review quán ăn thật, mỗi review bị che 1-2 đoạn bằng [1], [2]. Với mỗi review, hãy điền vào từng chỗ che một nội dung MỚI theo đúng yêu cầu của chỗ đó, sao cho cả câu tự nhiên và cùng giọng văn với phần còn lại (cùng mức độ thân mật, viết tắt, lủng củng như người viết gốc; dùng từ đời thường, tránh từ ngữ trang trọng, sáo rỗng). Giữ NGUYÊN VĂN mọi phần không bị che, viết thường, không dấu chấm cuối.
Mỗi yêu cầu là một ý cần có trong chỗ che: khía cạnh (category), cảm xúc, đối tượng được nhắc ("aspect", hoặc ngầm hiểu không nêu tên) và cụm nhận xét ("opinion", hoặc nói ngầm không dùng từ đánh giá trực tiếp).

{items}

Trả về DUY NHẤT một mảng JSON, mỗi phần tử ứng với một review theo thứ tự:
{{"text": "<review đã điền đầy đủ>", "fills": [[{{"aspect": "<cụm đúng nguyên văn trong câu hoặc null>", "opinion": "<cụm đúng nguyên văn trong câu hoặc null>", "category": "<tên khía cạnh như trong yêu cầu>", "polarity": "<tích cực|tiêu cực|trung lập>"}}, ...], ...]}}
"fills" có một danh sách cho mỗi chỗ che theo thứ tự [1], [2]; mỗi danh sách có đúng 1 ý, gán nhãn theo đúng quy ước ở trên."""

DEGREE = ("hơi", "khá", "rất", "quá", "siêu", "cực", "cũng")

VERIFY = """Quy ước gán nhãn của bộ dữ liệu: aspect là cụm danh từ chỉ đối tượng được nhận xét (món, đồ uống, nhân viên, giá, không gian…); opinion là cụm đánh giá NGẮN NHẤT (thường 1-3 từ, không chứa aspect, không gồm trạng từ mức độ như hơi, khá, rất, quá); null nếu không được nêu ra. Khía cạnh là một trong: {cats}. Cảm xúc: tích cực, tiêu cực, trung lập. Ví dụ gán nhãn đúng từ dữ liệu:
{shots}

Gán nhãn đầy đủ mọi ý trong từng review sau theo đúng quy ước trên:
{items}

Trả về DUY NHẤT một mảng JSON, mỗi phần tử là danh sách nhãn của review tương ứng: [[{{"aspect": "...", "opinion": "...", "category": "...", "polarity": "..."}}, ...], ...]"""


def span_match(x, y):
    if x == NULL or y == NULL:
        return x == y
    return x == y or bool(set(x.split()) & set(y.split()))



def groups(x):
    """Quads joined when they share a span (same offsets); returns [(interval start, interval end, [quad index])]."""
    qs = x["quads"]
    parent = list(range(len(qs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    spans = collections.defaultdict(list)
    for i, q in enumerate(qs):
        for f in ("aspect", "opinion"):
            if q.get(f + "_offsets"):
                spans[tuple(q[f + "_offsets"])].append(i)
    for idx in spans.values():
        for j in idx[1:]:
            parent[find(j)] = find(idx[0])
    out = collections.defaultdict(list)
    for i in range(len(qs)):
        out[find(i)].append(i)
    return list(out.values())


def plan_masks(x, ratio, rng):
    """Hidden intervals (merged) and, per interval, the indices of the quads it removes; None if not clean."""
    text, qs = x["text"], x["quads"]
    if any(not q.get("aspect_offsets") and not q.get("opinion_offsets") for q in qs):
        return None  # IA-IO quad: no span to hide or keep
    starts = sorted({q[f + "_offsets"][0] for q in qs for f in ("aspect", "opinion") if q.get(f + "_offsets")})
    units = []
    for g in groups(x):
        if len(g) != 1:
            continue  # quads sharing a span stay visible
        offs = [qs[i][f + "_offsets"] for i in g for f in ("aspect", "opinion") if qs[i].get(f + "_offsets")]
        s, e = min(o[0] for o in offs), max(o[1] for o in offs)
        e = next((b for b in starts if b >= e), len(text))
        while s > 0 and not text[s - 1].isspace():
            s -= 1
        units.append((s, e, g))
    rng.shuffle(units)
    n, chosen = len(text.split()), []
    for u in units:
        cov = sum(len(text[a:b].split()) for a, b in merge([c[:2] for c in chosen] + [u[:2]]))
        if chosen and cov > (ratio + 0.2) * n:
            continue
        chosen.append(u)
        if cov >= ratio * n:
            break
    hidden = []
    for a_, b in merge([c[:2] for c in chosen]):
        if hidden and not text[hidden[-1][1]:a_].strip():
            hidden[-1][1] = b
        else:
            hidden.append([a_, b])
    if not hidden or len(hidden) > 2 or sum(len(text[a:b].split()) for a, b in hidden) >= n:
        return None
    removed = [[] for _ in hidden]
    for i, q in enumerate(qs):
        offs = [q[f + "_offsets"] for f in ("aspect", "opinion") if q.get(f + "_offsets")]
        inside = [next((k for k, (a, b) in enumerate(hidden) if a <= o[0] and o[1] <= b), None) for o in offs]
        partial = [any(a < o[1] and o[0] < b for a, b in hidden) for o in offs]
        if all(k is not None for k in inside) and len(set(inside)) == 1:
            removed[inside[0]].append(i)
        elif any(partial):
            return None  # a quad would be half hidden
    if any(len(r) != 1 for r in removed):
        return None
    return hidden, removed


class Directives:
    def __init__(self, train, mode, a):
        self.mode, self.a = mode, a
        qs = [quad_tuple(q) for x in train for q in x["quads"]]
        self.top_a = {c: [t for t, _ in collections.Counter(q[1] for q in qs if q[0] == c and q[1] != NULL).most_common(20)]
                      for c in CATEGORY_VI}
        self.top_o = {(c, p): [t for t, _ in collections.Counter(q[2] for q in qs if q[0] == c and q[3] == p
                                                                  and q[2] != NULL).most_common(15)]
                      for c in CATEGORY_VI for p in POLARITY_VI}

    def make(self, q, rng):
        cat, pol, typ = q[0], q[3], implicit_type(q)
        d = {"category": cat, "polarity": pol, "type": typ, "new_aspect": False, "new_opinion": False, "opinion_first": False}
        if self.mode == "targeted":
            d["new_aspect"] = d["type"] != "IA-EO" and rng.random() < self.a.p_new_aspect
            d["new_opinion"] = d["type"] != "EA-IO" and rng.random() < self.a.p_new_opinion
            d["opinion_first"] = d["type"] == "EA-EO" and rng.random() < self.a.p_opinion_first
        return d

    def describe(self, d):
        cat, pol = CATEGORY_VI[d["category"]], POLARITY_VI[d["polarity"]]
        a = ("đối tượng ngầm hiểu, KHÔNG nêu tên (aspect = null)" if d["type"] == "IA-EO" else
             "một đối tượng cụ thể MỚI, ít gặp (tên món, đồ uống, chi tiết cụ thể…), không dùng các từ: "
             + ", ".join(self.top_a[d["category"]][:12]) if d["new_aspect"] else "một đối tượng được nêu tên")
        o = ("nói ngầm, KHÔNG dùng từ đánh giá trực tiếp (opinion = null)" if d["type"] == "EA-IO" else
             "cụm nhận xét bằng cách nói mới, không dùng các từ: " + ", ".join(self.top_o[(d["category"], d["polarity"])][:10])
             if d["new_opinion"] else "một cụm nhận xét ngắn, bình thường")
        order = "; nhận xét đứng TRƯỚC đối tượng" if d["opinion_first"] else ""
        return f"{cat}, {pol}; {a}; {o}{order}"


def ws(t):
    return re.sub(r"\s+", " ", t).strip()


def found_in(span, region):
    return re.search(r"(?<!\w)" + re.escape(span) + r"(?!\w)", region)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["targeted", "untargeted"], required=True)
    ap.add_argument("--n", type=int, default=0, help="reviews to keep; default |train|")
    ap.add_argument("--mask_ratio", type=float, default=0.4)
    ap.add_argument("--p_new_aspect", type=float, default=0.5)
    ap.add_argument("--p_new_opinion", type=float, default=0.5)
    ap.add_argument("--p_opinion_first", type=float, default=0.07)
    ap.add_argument("--per_call", type=int, default=6)
    ap.add_argument("--shots", type=int, default=4, help="labelled training reviews shown as the convention")
    ap.add_argument("--llm_model", default="gpt-4.1-mini-2025-04-14")
    ap.add_argument("--llm_temperature", type=float, default=0.9)
    ap.add_argument("--llm_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_calls", type=int, default=0, help="smoke test")
    ap.add_argument("--overgen", type=float, default=1.3, help="verified pool size / output size")
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    ap.add_argument("--heldout", default=",".join(str(ROOT / f"data/restaurant_v1/{s}.jsonl")
                                                  for s in ("dev", "test", "inner_dev")),
                    help="reviews that must not be near-duplicated (contamination check only)")
    a = ap.parse_args()
    load_env(ROOT / ".env")
    train = load_jsonl(a.train_path)
    train_ids = {x["id"] for x in train}
    held = [x["text"] for p in a.heldout.split(",") if p for x in load_jsonl(p) if x["id"] not in train_ids]
    nd_held = NearDup([norm(t) for t in held])
    seen = {norm(x["text"]) for x in train}
    n_final = a.n or len(train)
    n = int(round(n_final * a.overgen))  # verified pool, resampled to n_final at the end
    dirs = Directives(train, a.mode, a)
    train_a = collections.Counter(q["aspect"] for x in train for q in x["quads"])
    train_o = collections.Counter(q["opinion"] for x in train for q in x["quads"])
    cache_path = Path(a.out).with_suffix(".llm_cache.jsonl")
    cache = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[r["key"]] = r

    plan_ok = collections.defaultdict(list)  # planning success per number of quads, for the source weights
    for x in train:
        plan_ok[len(x["quads"])].append(plan_masks(x, a.mask_ratio, random.Random(0)) is not None)
    src_w = [1 / max(sum(plan_ok[len(x["quads"])]) / len(plan_ok[len(x["quads"])]), 0.05) for x in train]

    def build(i):
        """Deterministic content of call i: per item the source, hidden intervals, removed quads and directives."""
        rng = random.Random(a.seed * 1_000_003 + i)
        items = []
        while len(items) < a.per_call:
            x = rng.choices(train, src_w)[0]
            pm = plan_masks(x, a.mask_ratio, rng)
            if pm is None:
                continue
            hidden, removed = pm
            d = [[dirs.make(quad_tuple(x["quads"][qi]), rng) for qi in r] for r in removed]
            items.append({"src": x, "hidden": hidden, "removed": removed, "dirs": d})
        shots = rng.sample(train, a.shots)
        return items, shots

    def prompt(items, shots):
        shot_txt = "\n".join(
            f'- "{x["text"]}" -> ' + json.dumps([{"aspect": q["aspect"], "opinion": q["opinion"],
                                                  "category": CATEGORY_VI[q["category"]],
                                                  "polarity": POLARITY_VI[q["polarity"]]} for q in x["quads"]],
                                                ensure_ascii=False) for x in shots)
        blocks = []
        for j, it in enumerate(items):
            text, pos, parts = it["src"]["text"], 0, []
            for k, (s, e) in enumerate(it["hidden"]):
                parts.append(text[pos:s].strip()); parts.append(f"[{k + 1}]"); pos = e
            parts.append(text[pos:].strip())
            req = "\n".join(f"   [{k + 1}] " + dirs.describe(ds[0]) for k, ds in enumerate(it["dirs"]))
            blocks.append(f"Review {j + 1}: {' '.join(p for p in parts if p)}\n{req}")
        return PROMPT.format(shots=shot_txt, items="\n\n".join(blocks))

    def call(i):
        p = prompt(*build(i))
        key = hashlib.sha256(f"{a.llm_model}|{a.llm_temperature}|{i}|{p}".encode()).hexdigest()
        if key in cache:
            return cache[key]
        resp = openai_chat(a.llm_model, p, a.llm_temperature)
        return {"key": key, "call": i, "model": a.llm_model, "resolved_model": resp.get("model"), "prompt": p,
                "output": resp["choices"][0]["message"]["content"], "usage": resp.get("usage", {}),
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    def accept(it, res, st):
        """Validate one filled review; returns the labelled item or None (reason counted in st)."""
        src, stext = it["src"], it["src"]["text"]
        text = ws(str(res.get("text", ""))).rstrip(".").strip()
        fills = res.get("fills")
        if not isinstance(fills, list) or len(fills) != len(it["hidden"]):
            st["drop_fills_shape"] += 1; return None
        # visible chunks of the source (between hidden intervals) must appear verbatim and in order
        cuts = [0] + [b for ab in it["hidden"] for b in ab] + [len(stext)]
        chunks = [(cuts[2 * k], cuts[2 * k + 1]) for k in range(len(it["hidden"]) + 1)]
        segs = [ws(stext[a_:b]) for a_, b in chunks]
        segs[-1] = segs[-1].rstrip(".").strip()
        cur, place = 0, []
        for seg in segs:
            if not seg:
                place.append(None); continue
            k = text.find(seg, cur)
            if k < 0:
                st["drop_visible_changed"] += 1; return None
            place.append(k); cur = k + len(seg)
        ends = [(p_, p_ + len(sg)) if p_ is not None else None for p_, sg in zip(place, segs)]
        regions = []
        for k in range(len(it["hidden"])):
            r0 = ends[k][1] if ends[k] else 0
            r1 = ends[k + 1][0] if ends[k + 1] else len(text)
            if r1 - r0 < 2:
                st["drop_empty_fill"] += 1; return None
            regions.append((r0, r1))
        removed = {i for r in it["removed"] for i in r}
        quads = []
        for qi, q in enumerate(src["quads"]):  # visible gold quads: same span, offsets moved with their chunk
            if qi in removed:
                continue
            nq = dict(q)
            for f in ("aspect", "opinion"):
                o = q.get(f + "_offsets")
                if not o:
                    continue
                ci = next(c for c, (a_, b) in enumerate(chunks) if a_ <= o[0] and o[1] <= b)
                raw = stext[chunks[ci][0]:chunks[ci][1]]
                pat = r"(?<!\w)" + re.escape(q[f]) + r"(?!\w)"
                nth = [m.start() for m in re.finditer(pat, raw)].index(o[0] - chunks[ci][0])
                hits = [m.start() for m in re.finditer(pat, segs[ci])]
                if place[ci] is None or nth >= len(hits):
                    st["drop_visible_offsets"] += 1; return None
                start = place[ci] + hits[nth]
                nq[f + "_offsets"] = [start, start + len(q[f])]
            quads.append(nq)
        for k, (ds, fl) in enumerate(zip(it["dirs"], fills)):  # filled quads, declared by the model
            if not isinstance(fl, list) or len(fl) != len(ds):
                st["drop_fill_count"] += 1; return None
            r0, r1 = regions[k]
            for d, f in zip(ds, fl):
                if not isinstance(f, dict):
                    st["drop_fill_format"] += 1; return None
                cat = VI_CATEGORY.get(str(f.get("category", "")).strip().lower())
                pol = VI_POLARITY.get(str(f.get("polarity", "")).strip().lower())
                asp, op = (ws(str(f.get(x) or NULL)) for x in ("aspect", "opinion"))
                asp = NULL if asp.lower() in ("null", "none", "") else asp
                op = NULL if op.lower() in ("null", "none", "") else op
                for adv in DEGREE:  # convention: the degree adverb is not part of the opinion span
                    if op != NULL and op.startswith(adv + " ") and len(op.split()) > 1:
                        op = op[len(adv) + 1:]
                if op != NULL and len(op.split()) > 4:
                    st["drop_convention_long_opinion"] += 1; return None
                if asp != NULL and len(asp.split()) > 5:
                    st["drop_convention_long_aspect"] += 1; return None
                if asp != NULL and op != NULL and (asp in op or op in asp):
                    st["drop_convention_overlap"] += 1; return None
                if cat != d["category"] or pol != d["polarity"]:
                    st["drop_label_not_as_asked"] += 1; return None
                if implicit_type((cat, asp, op, pol)) != d["type"]:
                    st["drop_type_not_as_asked"] += 1; return None
                nq = {"category": cat, "aspect": asp, "polarity": pol, "opinion": op,
                      "aspect_offsets": None, "opinion_offsets": None}
                for fld, span in (("aspect", asp), ("opinion", op)):
                    if span != NULL:
                        m = found_in(span, text[r0:r1])
                        if not m:
                            st["drop_span_not_in_fill"] += 1; return None
                        nq[fld + "_offsets"] = [r0 + m.start(), r0 + m.end()]
                if d["opinion_first"] and nq["opinion_offsets"][0] > nq["aspect_offsets"][0]:
                    st["drop_order_not_as_asked"] += 1; return None
                quads.append(nq)
        if len({quad_tuple(q) for q in quads}) != len(quads):
            st["drop_duplicate_quad"] += 1; return None
        for q in quads:  # every explicit span sits at its offsets
            for f in ("aspect", "opinion"):
                if q.get(f + "_offsets") and text[slice(*q[f + "_offsets"])] != q[f]:
                    st["drop_offsets_check"] += 1; return None
        return {"text": text, "quads": quads}

    vcache_path = Path(a.out).with_suffix(".verify_cache.jsonl")
    vcache = {}
    if vcache_path.exists():
        for line in open(vcache_path, encoding="utf-8"):
            r = json.loads(line)
            vcache[r["key"]] = r

    def verify(batch):
        """Blind annotation of a batch of filled reviews (text only); returns one list of quad tuples per review."""
        rng = random.Random(hashlib.sha256("|".join(t for t, _ in batch).encode()).hexdigest())
        shots = "\n".join(f'- "{x["text"]}" -> ' + json.dumps(
            [{"aspect": q["aspect"], "opinion": q["opinion"], "category": CATEGORY_VI[q["category"]],
              "polarity": POLARITY_VI[q["polarity"]]} for q in x["quads"]], ensure_ascii=False)
            for x in rng.sample(train, a.shots))
        p = VERIFY.format(cats=", ".join(CATEGORY_VI.values()), shots=shots,
                          items="\n".join(f"{j + 1}. {t}" for j, (t, _) in enumerate(batch)))
        key = hashlib.sha256(f"{a.llm_model}|0|verify|{p}".encode()).hexdigest()
        if key not in vcache:
            resp = openai_chat(a.llm_model, p, 0.0)
            vcache[key] = {"key": key, "prompt": p, "output": resp["choices"][0]["message"]["content"],
                           "usage": resp.get("usage", {}), "new": True}
        raw = vcache[key]["output"]
        try:
            res = json.loads(raw[raw.index("["):raw.rindex("]") + 1])
        except (ValueError, json.JSONDecodeError):
            return [None] * len(batch), vcache[key]
        if not isinstance(res, list) or len(res) != len(batch):
            return [None] * len(batch), vcache[key]
        out_ = []
        for lst in res:
            qs = []
            for f in lst if isinstance(lst, list) else []:
                if not isinstance(f, dict):
                    continue
                cat = VI_CATEGORY.get(str(f.get("category", "")).strip().lower())
                pol = VI_POLARITY.get(str(f.get("polarity", "")).strip().lower())
                asp, op = (ws(str(f.get(x) or NULL)) for x in ("aspect", "opinion"))
                qs.append((cat, NULL if asp.lower() in ("null", "none", "") else asp,
                           NULL if op.lower() in ("null", "none", "") else op, pol))
            out_.append(qs)
        return out_, vcache[key]

    out, st, nxt = [], collections.Counter(), 0
    vf = open(vcache_path, "a", encoding="utf-8")
    with open(cache_path, "a", encoding="utf-8") as cf, ThreadPoolExecutor(a.llm_workers) as pool:
        while len(out) < n and (not a.max_calls or nxt < a.max_calls):
            m = max(1, min(int((n - len(out)) / a.per_call * 1.6) + 1, 300))
            if a.max_calls:
                m = min(m, a.max_calls - nxt)
            ids = list(range(nxt, nxt + m))
            nxt += m
            cands = []
            for i, rec in zip(ids, pool.map(call, ids)):
                if rec["key"] not in cache:
                    cache[rec["key"]] = rec
                    cf.write(json.dumps(rec, ensure_ascii=False) + "\n"); cf.flush()
                    st["api_calls"] += 1
                st["calls"] += 1
                for k in ("prompt_tokens", "completion_tokens"):
                    st[k] += rec["usage"].get(k, 0)
                items, _ = build(i)
                raw = rec["output"]
                try:
                    res = json.loads(raw[raw.index("["):raw.rindex("]") + 1])
                except (ValueError, json.JSONDecodeError):
                    st["drop_call_not_json"] += 1; continue
                if not isinstance(res, list) or len(res) != len(items):
                    st["drop_call_length"] += 1; continue
                for it, r in zip(items, res):
                    st["items"] += 1
                    if not isinstance(r, dict):
                        st["drop_fill_format"] += 1; continue
                    lab = accept(it, r, st)
                    if lab is None:
                        continue
                    t = lab["text"]
                    if t in seen:
                        st["drop_exact_duplicate"] += 1; continue
                    if nd_held.max_jaccard(t) >= 0.5:
                        st["drop_near_heldout"] += 1; continue
                    seen.add(t)
                    cands.append((t, {"lab": lab, "it": it, "call": i}))
            batches = [cands[b:b + 8] for b in range(0, len(cands), 8)]
            for batch, (vres, vrec) in zip(batches, pool.map(verify, batches)):
                if vrec.pop("new", False):
                    vf.write(json.dumps(vrec, ensure_ascii=False) + "\n"); vf.flush()
                    st["verify_api_calls"] += 1
                    for k in ("prompt_tokens", "completion_tokens"):
                        st["verify_" + k] += vrec["usage"].get(k, 0)
                for (t, c), vq in zip(batch, vres):
                    k = c["it"]["dirs"]
                    filled = c["lab"]["quads"][-len(k):]
                    if vq is None:
                        st["drop_verify_format"] += 1; continue
                    ok = True
                    for q in filled:
                        f = quad_tuple(q)
                        match = [v for v in vq if v[3] == f[3] and span_match(v[1], f[1]) and span_match(v[2], f[2])
                                 and v[0] in CATEGORY_VI]
                        if not match:
                            near = [v for v in vq if span_match(v[1], f[1]) or (f[1] == NULL and span_match(v[2], f[2]))]
                            why = ("no_matching_target" if not near else
                                   "polarity" if all(v[3] != f[3] for v in near) else "spans")
                            st[f"drop_verify_{why}" + ("_implicit" if implicit_type(f) != "EA-EO" else "")] += 1
                            ok = False
                            break
                        cat = next((v[0] for v in match if v[0] == f[0]), match[0][0])
                        if cat != f[0]:
                            q["category"] = cat
                            st["category_from_verifier"] += 1
                    if not ok:
                        continue
                    if len({quad_tuple(q) for q in c["lab"]["quads"]}) != len(c["lab"]["quads"]):
                        st["drop_duplicate_quad"] += 1; continue
                    if len(out) >= n:
                        continue
                    out.append({"id": f"infill_{a.mode}-{len(out):05d}", "text": t, "quads": c["lab"]["quads"],
                                "meta": {"method": f"infill_{a.mode}", "llm": a.llm_model,
                                         "source": c["it"]["src"]["id"], "call": c["call"], "new_quads": len(k),
                                         "directives": [d for ds in k for d in ds]}})
            print(f"  {len(out)}/{n} kept after {st['calls']} calls ({st['api_calls']} new), "
                  f"tokens {st['prompt_tokens']}+{st['completion_tokens']}", flush=True)
    vf.close()
    # stratified resampling of the verified pool on the filled quads' (polarity, implicit type)
    key = lambda q: (q["polarity"], implicit_type(quad_tuple(q)))
    tq = [q for x in train for q in x["quads"] if implicit_type(quad_tuple(q)) != "IA-IO"]
    tgt = collections.Counter(key(q) for q in tq)
    tot = sum(tgt.values())
    tgt = {k: v / tot for k, v in tgt.items()}
    pool_q = collections.Counter(key(q) for x in out for q in x["quads"][-x["meta"]["new_quads"]:])
    ptot = sum(pool_q.values()) or 1
    w = {k: tgt.get(k, 0) / (pool_q[k] / ptot) for k in pool_q}
    rng = random.Random(a.seed)
    weights = [sum(w[key(q)] for q in x["quads"][-x["meta"]["new_quads"]:]) / x["meta"]["new_quads"] for x in out]
    chosen = []
    idx = list(range(len(out)))
    for _ in range(min(n_final, len(out))):  # weighted sampling without replacement
        j = rng.choices(range(len(idx)), [weights[i] for i in idx])[0]
        chosen.append(idx.pop(j))
    pool_size = len(out)
    out = [out[i] for i in sorted(chosen)]
    for k_, x in enumerate(out):
        x["id"] = f"infill_{a.mode}-{k_:05d}"
    got = collections.Counter(key(q) for x in out for q in x["quads"][-x["meta"]["new_quads"]:])
    gtot = sum(got.values()) or 1
    selection = {"pool": pool_size, "kept": len(out),
                 "target_filled_shares": {f"{k[0]}|{k[1]}": round(v, 4) for k, v in sorted(tgt.items())},
                 "pool_filled_shares": {f"{k[0]}|{k[1]}": round(v / ptot, 4) for k, v in sorted(pool_q.items())},
                 "kept_filled_shares": {f"{k[0]}|{k[1]}": round(v / gtot, 4) for k, v in sorted(got.items())}}
    with open(a.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    def dist(items, key):
        c = collections.Counter(key(quad_tuple(q)) for x in items for q in x["quads"])
        s = sum(c.values()) or 1
        return {str(k): round(v / s, 4) for k, v in sorted(c.items())}
    newterms = [(q["aspect"], q["opinion"]) for x in out for q in x["quads"][len(x["quads"]) - x["meta"]["new_quads"]:]]
    report = {"n": len(out), "mode": a.mode, "args": vars(a), "stats": dict(st), "selection": selection,
              "quads_per_review": {"train": sum(len(x["quads"]) for x in train) / len(train),
                                   "out": sum(len(x["quads"]) for x in out) / max(len(out), 1)},
              "polarity": {"train": dist(train, lambda q: q[3]), "out": dist(out, lambda q: q[3])},
              "implicit_type": {"train": dist(train, implicit_type), "out": dist(out, implicit_type)},
              "category": {"train": dist(train, lambda q: q[0]), "out": dist(out, lambda q: q[0])},
              "filled_quads": len(newterms),
              "filled_aspect_unseen_in_train": sum(a_ != NULL and train_a[a_] == 0 for a_, _ in newterms) / max(len(newterms), 1),
              "filled_opinion_unseen_in_train": sum(o != NULL and train_o[o] == 0 for _, o in newterms) / max(len(newterms), 1)}
    json.dump(report, open(Path(a.out).with_suffix(".report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "args"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
