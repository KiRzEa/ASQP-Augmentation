"""Sample new quad sets for quad-to-text augmentation (shared by the GPT and ViT5 generators, so both write
from exactly the same labels).

A set copies the structure of a random training review (number of quads, and per quad its category,
polarity and implicit type) and fills each slot with a quad drawn from all training quads of that
(category, polarity, implicit aspect, implicit opinion) slot, so the label distribution follows train.
"""
import collections
import random

from asqp_common import NULL


def slot(q):
    return (q["category"], q["polarity"], q["aspect"] == NULL, q["opinion"] == NULL)


def sample(train, n, seed=13, prefix="qs"):
    rng = random.Random(seed)
    pool = collections.defaultdict(list)
    for it in train:
        for q in it["quads"]:
            pool[slot(q)].append({k: q[k] for k in ("category", "aspect", "opinion", "polarity")})
    out = []
    while len(out) < n:
        tpl = rng.choice(train)
        quads, seen = [], set()
        for q in tpl["quads"]:
            for _ in range(10):
                c = rng.choice(pool[slot(q)])
                key = (c["aspect"], c["opinion"])
                if key not in seen or key == (NULL, NULL):
                    break
            if key in seen and key != (NULL, NULL):
                continue
            seen.add(key)
            quads.append(dict(c))
        if len(quads) >= 1:
            out.append({"id": f"{prefix}-{len(out):05d}", "template": tpl["id"], "quads": quads})
    return out
