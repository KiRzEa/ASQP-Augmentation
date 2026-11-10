"""Unlabeled synthetic restaurant reviews in the style of the training set (GPT, runs locally, key in .env).

    python src/aug/synth_reviews.py --n 20000 --out data/aug/syn20k.jsonl

Only the training set is used. Every requested review is tied to one random training review (its "profile"):
  --mode free     --shots random training reviews as examples; the review copies the profile's surface (word count,
                  commas, emoji, the profile's own slang words).
  --mode plan     as free, plus a content plan: the profile's categories and polarities, each with an aspect term
                  drawn from the training aspects of that category (by frequency; "null" = no explicit target).
  --mode restyle  the profile review itself is the style model (wording, connectives, slang, length) and the new
                  review follows a content plan as in plan mode (other targets, same category/polarity structure).
  --mode infill   the profile review with whole opinion units hidden (a unit = from the first span of a training quad
                  to the next labelled span after its last one, so an implicit opinion's wording is hidden too), units
                  added in random order until ~--mask_ratio of the words are hidden; the model fills every blank with
                  new content (other dish, aspect, opinion); the visible words must stay verbatim.
No dev/test text is shown to the model, and no label is taken from the output: reviews carry no quads and are
labelled afterwards by the teacher consensus (src/aug/relabel.py, Kaggle).
Filters: lines that are too short/long, exact duplicates (also of train), near-duplicates (word-trigram Jaccard
>= --max_jaccard) of a train/dev/test review; dev/test are used only for this contamination check, and the
report counts every drop. Calls are cached in <out>.llm_cache.jsonl (resumable).
Writes <out> and <out>.report.json (sizes, drops, tokens, length/emoji statistics vs train).
--lang en (English ACOS data, infill only): the same procedure with an English prompt; the outputs are re-tokenised
the ACOS way (lower case, every non-alphanumeric character a separate token, e.g. "it ' s") before every check, and
profiles are drawn only from reviews with at least one explicit span (a review without one has nothing to hide, so
its output could only be dropped as a copy). --lang vi (default) is unchanged.
"""
import argparse
import collections
import hashlib
import json
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import CATEGORY_VI, NULL, POLARITY_VI, load_jsonl  # noqa: E402
from quadra import load_env, openai_chat  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

PROMPT = """Bạn là người Việt hay viết review quán ăn, quán cà phê trên mạng. Dưới đây là vài review thật:
{shots}

Hãy viết {k} review MỚI, mỗi review một dòng, như do những khách khác nhau viết về những quán khác nhau (món, đồ uống, giá, không gian, phục vụ, vị trí… khác với các ví dụ). Giữ phong cách của các ví dụ: viết thường toàn bộ, không dấu chấm cuối, câu tự nhiên kiểu mạng xã hội, kể cụ thể như người thật (món gì, giá bao nhiêu, chuyện gì xảy ra), tránh câu sáo rỗng. Review thứ i phải theo đúng yêu cầu thứ i:
{specs}
Không đánh số, không giải thích, chỉ trả về {k} dòng."""

RESTYLE = """Bạn là người Việt hay viết review quán ăn, quán cà phê trên mạng. Với mỗi câu mẫu dưới đây (review thật), hãy viết MỘT review mới của một khách khác về một quán khác: nội dung theo dàn ý đi kèm, còn cách viết bắt chước sát câu mẫu (từ nối, cách xưng hô, viết tắt, dấu câu, emoji, độ dài, mức độ lủng củng). Không chép lại các cụm của câu mẫu, viết thường toàn bộ, không dấu chấm cuối.
{specs}
Trả về đúng {k} dòng theo thứ tự, mỗi dòng một review, không đánh số, không giải thích."""

INFILL = """Dưới đây là các review quán ăn thật, mỗi review bị che vài đoạn bằng [___]. Với mỗi review, hãy điền vào từng chỗ [___] một nội dung MỚI khác với nội dung gốc có thể có (món khác, khía cạnh khác, nhận xét khác, có thể đổi khen thành chê), sao cho cả câu tự nhiên, đúng giọng văn của phần còn lại, độ dài mỗi chỗ điền tương đương số từ ghi trong ngoặc. Giữ NGUYÊN VĂN mọi phần không bị che, viết thường, không dấu chấm cuối.
{specs}
Trả về đúng {k} dòng theo thứ tự, mỗi dòng là review đã điền đầy đủ (không còn [___]), không đánh số, không giải thích."""


INFILL_EN = """Below are real {domain} reviews; in each, some passages are hidden as [___]. For each review, fill every [___] with NEW content that differs from what the original probably said (another dish or part, another aspect, another opinion; praise may become criticism), so that the whole review is natural and keeps the voice of the visible text; each filled passage should be about as long as the word count in brackets. Keep EVERY visible part EXACTLY as written, all lower case.
{specs}
Return exactly {k} lines in order, each line one fully filled review (no [___] left), no numbering, no explanations."""

ACOS_TOKEN = re.compile(r"[^\W_]+|\S")


def acos_tokenise(t):
    """ACOS style: lower case, runs of letters/digits, every other non-space character a token of its own."""
    return " ".join(ACOS_TOKEN.findall(t.lower()))


def mask(x, ratio, rng, unit="từ"):
    """Hide whole opinion units of a training review until ~ratio of its words are hidden.
    Returns (template with [___ (n từ)] blanks, visible chunks that must appear verbatim in the filled review)."""
    text = x["text"]
    starts = sorted({q[f + "_offsets"][0] for q in x["quads"] for f in ("aspect", "opinion") if q.get(f + "_offsets")})
    units = []
    for q in x["quads"]:
        offs = [q[f + "_offsets"] for f in ("aspect", "opinion") if q.get(f + "_offsets")]
        if not offs:
            continue
        s, e = min(o[0] for o in offs), max(o[1] for o in offs)
        e = next((b for b in starts if b >= e), len(text))  # up to the next labelled span
        while s > 0 and not text[s - 1].isspace():
            s -= 1
        units.append((s, e))
    rng.shuffle(units)
    n, hidden = len(text.split()), []
    for u in units:  # a unit that would hide more than ratio + 0.2 is skipped unless nothing is hidden yet
        cov = sum(len(text[a:b].split()) for a, b in merge(hidden + [u]))
        if hidden and cov > (ratio + 0.2) * n:
            continue
        hidden.append(u)
        if cov >= ratio * n:
            break
    hidden = merge(hidden)
    out, chunks, pos = [], [], 0
    for a, b in hidden:
        if text[pos:a].strip():
            out.append(text[pos:a].strip()); chunks.append(text[pos:a].strip())
        out.append(f"[___ ({max(1, len(text[a:b].split()))} {unit})]")
        pos = b
    if text[pos:].strip():
        out.append(text[pos:].strip()); chunks.append(text[pos:].strip())
    return " ".join(out), chunks


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
SLANG = {"ko", "k", "dc", "đc", "j", "mn", "hok", "hông", "kh", "bt", "mik", "mk", "vs", "nc", "ng"}


def spec(t):
    """Surface profile of a training review, as an instruction."""
    w = t.split()
    n = t.count(",")
    parts = [f"khoảng {len(w)} từ",
             f"BẮT BUỘC có {n} dấu phẩy (,) ngắt giữa các ý" if n else "không dấu phẩy"]
    if EMOJI.search(t):
        parts.append("có emoji")
    slang = [x for x in dict.fromkeys(w) if x in SLANG]
    if slang:
        parts.append("có viết tắt kiểu mạng như " + ", ".join(slang))
    return ", ".join(parts)


def plan(x, aspects, rng):
    """Content plan of a profile review: its category/polarity structure with aspect terms drawn from train."""
    items = []
    for q in x["quads"]:
        terms, weights = aspects[q["category"]]
        a = rng.choices(terms, weights)[0]
        target = f'"{a}"' if a != NULL else "không nêu tên đối tượng cụ thể"
        items.append(f"{CATEGORY_VI[q['category']]} ({target}): {POLARITY_VI[q['polarity']]}")
    return "; ".join(items)


def norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def trigrams(t):
    w = t.split()
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


class NearDup:
    """Word-trigram Jaccard against a reference set, with an inverted index on trigrams."""

    def __init__(self, texts):
        self.sets = [trigrams(t) for t in texts]
        self.index = collections.defaultdict(list)
        for i, s in enumerate(self.sets):
            for g in s:
                self.index[g].append(i)

    def max_jaccard(self, t):
        s = trigrams(t)
        if not s:
            return 0.0
        shared = collections.Counter(i for g in s for i in self.index.get(g, ()))
        return max((n / len(s | self.sets[i]) for i, n in shared.items()), default=0.0)


def stats(texts):
    w = [len(t.split()) for t in texts]
    return {"n": len(texts), "words_mean": statistics.mean(w), "words_median": statistics.median(w),
            "words_p10": sorted(w)[len(w) // 10], "words_p90": sorted(w)[9 * len(w) // 10],
            "emoji_share": sum(bool(EMOJI.search(t)) for t in texts) / len(texts),
            "comma_share": sum("," in t for t in texts) / len(texts),
            "commas_per_review": sum(t.count(",") for t in texts) / len(texts),
            "slang_share": sum(any(x in t.split() for x in SLANG) for t in texts) / len(texts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000, help="reviews to keep")
    ap.add_argument("--mode", choices=["free", "plan", "restyle", "infill"], default="infill")
    ap.add_argument("--lang", choices=["vi", "en"], default="vi", help="en: English ACOS data (infill only)")
    ap.add_argument("--domain", default="restaurant", help="en: review domain named in the prompt")
    ap.add_argument("--mask_ratio", type=float, default=0.4, help="infill: share of the words hidden")
    ap.add_argument("--max_jaccard_train", type=float, default=0.0,
                    help="near-duplicate threshold against train; 0 = --max_jaccard (infill default 0.8: it keeps "
                         "most source words by design, only near-copies are dropped)")
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--per_call", type=int, default=10)
    ap.add_argument("--llm_model", default="gpt-4.1-mini-2025-04-14")
    ap.add_argument("--llm_temperature", type=float, default=1.0)
    ap.add_argument("--llm_workers", type=int, default=8)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--max_words", type=int, default=80)
    ap.add_argument("--max_jaccard", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_calls", type=int, default=0, help="stop after this many calls (smoke test); 0 = until --n")
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    ap.add_argument("--dev_path", default=str(ROOT / "data/restaurant_v1/dev.jsonl"))
    ap.add_argument("--test_path", default=str(ROOT / "data/restaurant_v1/test.jsonl"))
    a = ap.parse_args()
    if a.lang == "en" and a.mode != "infill":
        raise SystemExit("--lang en supports --mode infill only")
    global norm
    if a.lang == "en":
        norm = acos_tokenise
    load_env(ROOT / ".env")
    train_items = load_jsonl(a.train_path)
    train = [x["text"] for x in train_items]
    acount = collections.defaultdict(collections.Counter)
    for x in train_items:
        for q in x["quads"]:
            acount[q["category"]][q["aspect"]] += 1
    aspects = {c: (list(cn), list(cn.values())) for c, cn in acount.items()}
    heldout = [x["text"] for p in (a.dev_path, a.test_path) for x in load_jsonl(p)]
    profile_items = train_items if a.lang == "vi" else [
        x for x in train_items if any(q.get(f + "_offsets") for q in x["quads"] for f in ("aspect", "opinion"))]
    nd_train, nd_held = NearDup([norm(t) for t in train]), NearDup([norm(t) for t in heldout])
    seen = {norm(t) for t in train}
    jac_train = a.max_jaccard_train or (0.8 if a.mode == "infill" else a.max_jaccard)
    cache_path = Path(a.out).with_suffix(".llm_cache.jsonl")
    cache = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[r["key"]] = r

    def prompt(i):
        rng = random.Random(a.seed * 1_000_003 + i)
        profiles = rng.sample(profile_items, a.per_call)
        if a.mode == "infill":
            ms = [mask(x, a.mask_ratio, rng, "từ" if a.lang == "vi" else "words") for x in profiles]
            specs = "\n".join(f"{j + 1}. {m[0]}" for j, m in enumerate(ms))
            tpl = INFILL.format(k=a.per_call, specs=specs) if a.lang == "vi" else \
                INFILL_EN.format(k=a.per_call, specs=specs, domain=a.domain)
            return tpl, [m[1] for m in ms]
        if a.mode == "restyle":
            return RESTYLE.format(k=a.per_call, specs="\n".join(
                f"{j + 1}. câu mẫu: {x['text']}\n   dàn ý: {plan(x, aspects, rng)}" for j, x in enumerate(profiles))), None
        shots = rng.sample(train, a.shots)
        specs = [spec(x["text"]) + (f"; nội dung: {plan(x, aspects, rng)}" if a.mode == "plan" else "") for x in profiles]
        return PROMPT.format(shots="\n".join(f"- {t}" for t in shots), k=a.per_call,
                             specs="\n".join(f"{j + 1}. {sp}" for j, sp in enumerate(specs))), None

    def call(i):
        p, _ = prompt(i)
        key = hashlib.sha256(f"{a.llm_model}|{a.llm_temperature}|{a.mode}|{i}|{p}".encode()).hexdigest()
        if key in cache:
            return cache[key]
        resp = openai_chat(a.llm_model, p, a.llm_temperature)
        return {"key": key, "call": i, "model": a.llm_model, "resolved_model": resp.get("model"), "prompt": p,
                "output": resp["choices"][0]["message"]["content"], "usage": resp.get("usage", {}),
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    out, st, nxt = [], collections.Counter(), 0
    with open(cache_path, "a", encoding="utf-8") as cf, ThreadPoolExecutor(a.llm_workers) as pool:
        while len(out) < a.n and (not a.max_calls or nxt < a.max_calls):
            m = max(1, min(int((a.n - len(out)) / a.per_call * 1.15) + 1, 400))
            if a.max_calls:
                m = min(m, a.max_calls - nxt)
            ids = list(range(nxt, nxt + m))
            nxt += m
            for rec in pool.map(call, ids):
                if rec["key"] not in cache:
                    cache[rec["key"]] = rec
                    cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cf.flush()
                    st["api_calls"] += 1
                st["calls"] += 1
                for k in ("prompt_tokens", "completion_tokens"):
                    st[k] += rec["usage"].get(k, 0)
                lines = [ln for ln in rec["output"].splitlines() if ln.strip()]
                visible = prompt(rec["call"])[1]
                if visible is not None and len(lines) != len(visible):
                    st["drop_call_line_mismatch"] += 1
                    continue
                for li, line in enumerate(lines):
                    t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
                    t = re.sub(r"[\"“”]", "", t)
                    t = norm(t).rstrip(".").strip() if a.lang == "vi" else norm(t)
                    if not t:
                        continue
                    st["lines"] += 1
                    if not a.min_words <= len(t.split()) <= a.max_words:
                        st["drop_length"] += 1
                    elif t in seen:
                        st["drop_exact_duplicate"] += 1
                    elif nd_held.max_jaccard(t) >= a.max_jaccard:
                        st["drop_near_dev_test"] += 1
                    elif visible is not None and (
                            "[ _ _ _" in t or "[___" in t or
                            not all((norm(c).rstrip(".") if a.lang == "vi" else norm(c)) in t for c in visible[li])):
                        st["drop_infill_changed_visible"] += 1
                    elif nd_train.max_jaccard(t) >= jac_train:
                        st["drop_near_train"] += 1
                    elif len(out) < a.n:
                        seen.add(t)
                        out.append({"id": f"syn-{len(out):06d}", "text": t, "quads": [],
                                    "meta": {"method": f"synthetic_{a.mode}", "llm": a.llm_model, "call": rec["call"]}})
            print(f"  {len(out)}/{a.n} kept after {st['calls']} calls ({st['api_calls']} new), "
                  f"tokens {st['prompt_tokens']}+{st['completion_tokens']}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    report = {"n": len(out), "args": vars(a), "stats": dict(st),
              "texts": stats([x["text"] for x in out]) if out else {}, "train_texts": stats(train)}
    json.dump(report, open(Path(a.out).with_suffix(".report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "args"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
