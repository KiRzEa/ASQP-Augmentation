"""LLM augmentation baselines with GPT (OpenAI API, key in .env; runs locally, no GPU). Cached per prompt.

    python src/aug/llm_aug.py paraphrase --out data/aug/llm_para.jsonl
    python src/aug/llm_aug.py quad2text  --out data/aug/llm_q2t.jsonl

paraphrase  rewrite a training review in other words, keeping every labelled span verbatim (same quads).
quad2text   write a new review from a sampled quad set (src/aug/quadsets.py, same sets as the ViT5
            quad-to-text baseline); explicit spans must appear verbatim, implicit ones are described.
Every output is checked: all explicit spans are found with word boundaries (relocate / grounded), length
<= --max_words (paraphrase also <= --max_len_ratio x source); quotes and a final period are removed. Failed outputs are dropped and new sources are taken until exactly --n (default |train|).
"""
import argparse
import collections
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import CATEGORY_VI, NULL, POLARITY_VI, load_jsonl  # noqa: E402
from quadra import load_env, openai_chat, relocate  # noqa: E402
from quadsets import sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

PARA_PROMPT = """Bạn là người Việt hay viết review quán ăn trên mạng. Hãy viết lại câu review dưới đây theo cách diễn đạt khác (đổi từ ngữ, cấu trúc câu, thứ tự các ý), giữ nguyên ý nghĩa và cảm xúc, giọng văn thân mật, viết thường toàn bộ, không dấu chấm cuối.
Bắt buộc giữ NGUYÊN VĂN các cụm sau: {spans}
Không thêm khía cạnh hay nhận xét mới, không đổi ý khen/chê. Chỉ trả về câu review.

Câu gốc: {text}"""

Q2T_PROMPT = """Bạn là người Việt hay viết review quán ăn trên mạng. Hãy viết MỘT câu review ngắn, tự nhiên, giọng văn thân mật, viết thường toàn bộ, không dấu chấm cuối, thể hiện đúng và đủ các ý sau, không thêm ý khen/chê nào khác:
{items}
Các cụm trong ngoặc kép phải xuất hiện NGUYÊN VĂN trong câu (không ghi dấu ngoặc kép). Chỉ trả về câu review."""


def describe(q):
    cat, pol = CATEGORY_VI[q["category"]], POLARITY_VI[q["polarity"]]
    a, o = q["aspect"], q["opinion"]
    if a != NULL and o != NULL:
        return f'- "{a}" được nhận xét là "{o}" (về {cat}, {pol})'
    if a == NULL and o != NULL:
        return f'- nhận xét "{o}" về {cat} nói chung, không nêu tên đối tượng cụ thể ({pol})'
    if a != NULL and pol == POLARITY_VI["neutral"]:
        return f'- có nhắc tới "{a}" (về {cat}) nhưng không khen cũng không chê'
    if a != NULL:
        return f'- nhắc tới "{a}" (về {cat}) và thể hiện ý {pol} một cách ngầm, không dùng từ đánh giá trực tiếp'
    return f"- thể hiện ý {pol} về {cat} một cách ngầm, không nêu tên đối tượng và không dùng từ đánh giá trực tiếp"


def prompt_for(method, it):
    if method == "paraphrase":
        spans = sorted({q[f] for q in it["quads"] for f in ("aspect", "opinion") if q[f] != NULL})
        return PARA_PROMPT.format(spans="; ".join(f'"{s}"' for s in spans), text=it["text"])
    return Q2T_PROMPT.format(items="\n".join(describe(q) for q in it["quads"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["paraphrase", "quad2text"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="default |train|")
    ap.add_argument("--llm_model", default="gpt-4.1-mini-2025-04-14")
    ap.add_argument("--llm_temperature", type=float, default=0.7)
    ap.add_argument("--llm_workers", type=int, default=8)
    ap.add_argument("--max_words", type=int, default=80)
    ap.add_argument("--max_len_ratio", type=float, default=1.5, help="paraphrase: max words relative to the source")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit_calls", type=int, default=0, help="smoke test: stop after this many candidates")
    ap.add_argument("--train_path", default=str(ROOT / "data/restaurant_v1/train.jsonl"))
    a = ap.parse_args()
    load_env(ROOT / ".env")
    train = load_jsonl(a.train_path)
    n = a.n or len(train)
    if a.method == "paraphrase":
        import random
        order = list(train)
        random.Random(a.seed).shuffle(order)
        cands = [order[i % len(order)] for i in range(4 * n)]  # cycle; a source can be paraphrased again
        cands = [{**c, "_rep": i // len(order)} for i, c in enumerate(cands)]
    else:
        cands = sample(train, 4 * n, a.seed, prefix="q2t")
    if a.limit_calls:
        cands, n = cands[:a.limit_calls], min(n, a.limit_calls)
    cache_path = Path(a.out).with_suffix(".llm_cache.jsonl")
    cache = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[r["key"]] = r

    def key_of(it):
        rep = it.get("_rep", 0)  # a repeated paraphrase source gets its own call
        return hashlib.sha256(f"{a.llm_model}|{a.llm_temperature}|{rep}|{prompt_for(a.method, it)}".encode()).hexdigest()

    def call(it):
        k, p = key_of(it), prompt_for(a.method, it)
        if k in cache:
            return cache[k]
        resp = openai_chat(a.llm_model, p, a.llm_temperature)
        return {"key": k, "model": a.llm_model, "resolved_model": resp.get("model"), "prompt": p,
                "output": resp["choices"][0]["message"]["content"].strip(), "usage": resp.get("usage", {}),
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    out, stats, pos = [], collections.Counter(), 0
    with open(cache_path, "a", encoding="utf-8") as cf, ThreadPoolExecutor(a.llm_workers) as pool:
        while len(out) < n and pos < len(cands):
            batch = cands[pos:pos + int((n - len(out)) * 1.3) + 8]
            pos += len(batch)
            for it, rec in zip(batch, pool.map(call, batch)):
                if rec["key"] not in cache:
                    cache[rec["key"]] = rec
                    cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cf.flush()
                    stats["api_calls"] += 1
                for k in ("prompt_tokens", "completion_tokens"):
                    stats[k] += rec["usage"].get(k, 0)
                if len(out) >= n:
                    continue
                text = re.sub(r"[\"“”]", "", rec["output"])
                text = re.sub(r"\s+", " ", text).strip().rstrip(".").strip()
                if len(text.split()) > a.max_words or (
                        a.method == "paraphrase" and len(text.split()) > a.max_len_ratio * len(it["text"].split())):
                    stats["rejected_too_long"] += 1
                    continue
                new = None
                for cand in (text, text[:1].lower() + text[1:], text.lower()):  # the data are lowercase
                    new = relocate(cand, it)
                    if new:
                        break
                if not new or (a.method == "paraphrase" and new["text"] == it["text"]):
                    stats["rejected_spans_not_found"] += 1
                    continue
                stats["kept"] += 1
                meta = {"method": f"llm_{a.method}", "llm": a.llm_model, "source": it.get("template", it["id"])}
                out.append({"id": f"llm_{a.method}-{len(out):05d}", "text": new["text"], "quads": new["quads"],
                            "meta": meta})
            print(f"  {len(out)}/{n} kept after {pos} candidates ({stats['api_calls']} new API calls)", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    report = {"method": a.method, "llm_model": a.llm_model, "temperature": a.llm_temperature, "seed": a.seed,
              "n": len(out), "candidates_used": pos, "quads": sum(len(it["quads"]) for it in out), "stats": dict(stats)}
    json.dump(report, open(Path(a.out).with_suffix(".report.json"), "w"), indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
