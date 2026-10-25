"""Rebuild the rules-only QUADRA set from an LLM-fused QUADRA file, so both variants share the same samples.

Every fused review stores its rule-based text in meta.pre_fusion (reviews whose fusion failed already hold the
rule text). The quads are identical; offsets are relocated in the rule text. The result differs from the fused
set only by LLM fusion, which is the comparison the ablation needs.

    python src/aug/pre_fusion.py data/aug/quadra_gpt.jsonl data/aug/quadra_rule.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quadra import grounded, relocate  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    out, bad = [], 0
    for line in open(src, encoding="utf-8"):
        it = json.loads(line)
        meta = dict(it["meta"])
        text = meta.pop("pre_fusion", it["text"])
        meta.pop("llm", None)
        new = relocate(text, it) if text != it["text"] else {"text": it["text"], "quads": it["quads"]}
        if new is None or not grounded(new):
            bad += 1
            continue
        out.append({"text": new["text"], "quads": new["quads"], "id": it["id"], "meta": {**meta, "derived_from": src}})
    with open(dst, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(out)} rule-based reviews written to {dst} ({bad} could not be grounded)")


if __name__ == "__main__":
    main()
