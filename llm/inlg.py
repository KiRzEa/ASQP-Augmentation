"""Gemma-7B + LoRA, ASQP as code generation (the INLG 2025 recipe), and the steps of the ensemble-distillation
augmentation (teachers label an unlabelled pool, majority vote, student on gold + voted pool).
The model is a profile (--profile, env ASQP_PROFILE): gemma (default, everything below) or qwen
(Qwen/Qwen2.5-14B-Instruct as in the authors' LLM_QWen.ipynb: "json_format" prompt, one JSON list per review with
ASCII escapes, "null" for implicit spans, effective batch 8; LoRA, lr, epochs, NEFTune and beam 5 as for gemma).

    python llm/inlg.py preflight --model google/gemma-7b --cache_dir /path/to/models      # checks only
    python llm/inlg.py train --dataset vi --seed 42 --out_dir outputs/llm/vi_none/seed_42 --keep_adapter
    python llm/inlg.py label --dataset vi --adapter outputs/llm/vi_none/seed_42/adapter --out data/aug/llm/vi_s42.jsonl
    python src/aug/mvd.py select --in a.jsonl,b.jsonl,... --n 100000 --out data/aug/llm/vi_ens.jsonl     # CPU vote
    python llm/inlg.py train --dataset vi --seed 42 --aug_path data/aug/llm/vi_ens.jsonl --tuple_order text ...

Recipe (from the authors' notebook LLM_Gemma.ipynb, prompt "coding_format"): google/gemma-7b (base) in bf16; LoRA r 8,
alpha 16, rsLoRA, dropout 0.1 on every linear layer; AdamW lr 2e-4, weight decay 0.01, cosine schedule, 4 warmup
steps, effective batch 16 (INLG: 4 x grad accum 4), NEFTune alpha 5, 5 epochs, last checkpoint (dev not used); loss
on the whole prompt + target sequence (TRL SFTTrainer on a text field); test decoded with beam 5.
Differences, none of which changes the recipe: own training loop (no TRL); sdpa attention instead of eager; no
deterministic-algorithm flags; each batch of 16 is run as forward passes of <= 8 examples and <= 4096 padded
tokens (an OOM halves both), so the effective batch stays 16; batches grouped by length; loss averaged over all target tokens of the
16 examples; the docstring of the prompt names the real domain (INLG wrote "restaurant" also for Laptop).
The pool is labelled with greedy decoding (as in the ViT5 study).

Target, one line per quad (gold order, or --tuple_order text = by position in the review):
    quadruplet_list.append({"category": "laptop price", "aspect": "unit", "sentiment": "negative", "opinion": "NULL"})
English ACOS categories in lower case with "#" as a space (underscores kept, as in INLG); Vi-AbSQA categories as in
the data ("FOOD#QUALITY"); implicit aspect / opinion = "NULL".

Outputs of train (same schema as src/train_paraphrase.py): run_config.json, history.json, test_predictions.jsonl,
metrics.json, train.log; adapter/ with --keep_adapter. A run resumes from its last finished epoch (resume/).
The test score is written to metrics.json and never printed.
"""
import argparse
import ast
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asqp_common import NULL, clean_span, load_jsonl, quad_tuple, score, text_order  # noqa: E402

PINNED = {"transformers": "4.57.6", "peft": "0.17.1", "accelerate": "1.10.1"}
DATASETS = {
    "vi": {"dir": "data/restaurant_v1", "pool": "data/aug/st_pool_train.jsonl", "lang": "vi", "domain": "restaurant",
           "sizes": {"train": 2135, "dev": 221, "test": 583, "pool": 15907}},
    "rest": {"dir": "data/acos/rest", "pool": "data/aug/acos_rest_pool_train.jsonl", "lang": "en",
             "domain": "restaurant", "sizes": {"train": 1530, "dev": 171, "test": 583, "pool": 11511}},
    "laptop": {"dir": "data/acos/laptop", "pool": "data/aug/acos_laptop_pool_train.jsonl", "lang": "en",
               "domain": "laptop", "sizes": {"train": 2934, "dev": 326, "test": 816, "pool": 21095}},
}
PROMPT_GEMMA = '''<start_of_turn>user
def extract_quadruplet_list_from_review(input_review):
    """Extract list of quadruplet sentiments from the input review for the {domain} domain. """

    input_review = "{review}"

    quadruplet_list = []
    for phrase in extract_important_phrase(input_review):
        category = classify_category(phrase)
        aspect = extract_aspect_for_category(phrase, category)  # mentioned in review or implicit aspect
        sentiment = classify_sentiment_for_aspect(phrase, aspect, category)  # positive, negative, or neutral
        opinion = extract_opinion_for_sentiment(phrase, sentiment)  # mentioned in review or implicit opinion
        quadruplet_list.append({{
            "category": category,
            "aspect": aspect,
            "sentiment": sentiment,
            "opinion": opinion
        }})
    return quadruplet_list

# Run an example
input_review= "{review}"
results = extract_quadruplet_list_from_review(input_review)
for quad in results:
    print(f"quadruplet_list.append({{quad}})")<end_of_turn>
<start_of_turn>model
'''
# LLM_QWen.ipynb, prompt "json_format", rebuilt from the prompt the notebook prints (no role names in the chat turn)
PROMPT_QWEN = '''<|im_start|>
Extract all aspect-opinion quadruples from the following {language} {domain} review and present them in this exact format:
AT: aspect term AC: aspect category SP: sentiment polarity OT: opinion term

+ Input review: "{review}".
+ Output:<|im_end|>
'''
PROFILES = {
    # INLG 2025 (LLM_Gemma.ipynb): the default; every setting as before the profiles were added
    "gemma": {"model": "google/gemma-7b", "prompt": PROMPT_GEMMA, "end": "<end_of_turn>",
              "chat_tokens": ("<start_of_turn>", "<end_of_turn>"), "target": "code", "null": "NULL",
              "json_ascii": False, "batch": 16, "micro_tokens": 4096, "min_gpu_gb": 40},
    # LLM_QWen.ipynb (not in the paper): Qwen2.5-14B-Instruct, one JSON list per review, json.dumps with its default
    # ASCII escapes (as the notebook prints), "null" for implicit spans, effective batch 2 x grad accum 4 = 8
    "qwen": {"model": "Qwen/Qwen2.5-14B-Instruct", "prompt": PROMPT_QWEN, "end": "<|im_end|>",
             "chat_tokens": ("<|im_start|>", "<|im_end|>"), "target": "json", "null": "null",
             "json_ascii": True, "batch": 8, "micro_tokens": 2048, "min_gpu_gb": 70},
}
PROFILE = dict(PROFILES["gemma"], name="gemma")
END = PROFILE["end"]
LANGUAGE = {"vi": "Vietnamese", "en": "English"}
LINE_RE = re.compile(r"quadruplet_list\.append\((\{.*\})\)\s*$")
POLARITIES = ("positive", "negative", "neutral")


# ---------------------------------------------------------------- data and target format

def use_profile(name, json_ascii=None):
    """Select the model profile (prompt, target format, end token, defaults) for this process."""
    global END
    PROFILE.clear()
    PROFILE.update(PROFILES[name], name=name)
    if json_ascii is not None and json_ascii >= 0:
        PROFILE["json_ascii"] = bool(json_ascii)
    END = PROFILE["end"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ds_path(d, split):
    return ROOT / DATASETS[d]["dir"] / f"{split}.jsonl"


class Labels:
    """Category <-> target words for one dataset (inventory = categories of train + dev + test)."""

    def __init__(self, d):
        self.lang = DATASETS[d]["lang"]
        cats = sorted({q["category"] for s in ("train", "dev", "test") for x in load_jsonl(ds_path(d, s))
                       for q in x["quads"]})
        self.words = {c: (c.lower().replace("#", " ") if self.lang == "en" else c) for c in cats}
        if len(set(self.words.values())) != len(self.words):
            raise SystemExit(f"{d}: two categories map to the same words")
        self.cat = {w: c for c, w in self.words.items()}

    def target(self, quads):
        nul = PROFILE["null"]
        objs = [{"category": self.words[q["category"]], "aspect": nul if q["aspect"] == NULL else q["aspect"],
                 "sentiment": q["polarity"], "opinion": nul if q["opinion"] == NULL else q["opinion"]} for q in quads]
        if PROFILE["target"] == "json":
            return json.dumps(objs, ensure_ascii=PROFILE["json_ascii"])
        return "\n".join("quadruplet_list.append(" + json.dumps(o, ensure_ascii=False) + ")" for o in objs)

    def _objects(self, generation):
        """-> (list of candidate dicts or None per item, number of unparsable items)."""
        if PROFILE["target"] == "json":
            g = generation.strip()
            try:
                obj = json.loads(g)
                if isinstance(obj, list):
                    return obj, 0
            except ValueError:
                pass
            items, bad = [], 0  # broken list (e.g. cut off): keep every complete {...} object
            for m in re.finditer(r"\{[^{}]*\}", g):
                try:
                    items.append(json.loads(m.group(0)))
                except ValueError:
                    bad += 1
            rest = re.sub(r"\{[^{}]*\}", "", g).strip(" \n\t[],")
            return items, bad + (1 if rest else 0)
        items = []
        for line in generation.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.search(line)
            obj = None
            if m:
                try:
                    obj = json.loads(m.group(1))
                except ValueError:
                    try:
                        obj = ast.literal_eval(m.group(1))
                    except (ValueError, SyntaxError):
                        obj = None
            items.append(obj)
        return items, 0

    def parse(self, generation):
        """-> (list of (category, aspect, opinion, polarity) tuples, number of invalid non-empty lines)."""
        out = []
        items, bad = self._objects(generation)
        for obj in items:
            if not isinstance(obj, dict) or not {"category", "aspect", "sentiment", "opinion"} <= set(obj):
                bad += 1
                continue
            c, s = self.cat.get(str(obj["category"]).strip()), str(obj["sentiment"]).strip()
            if c is None or s not in POLARITIES:
                bad += 1
                continue
            a, o = (clean_span(str(obj[k])) for k in ("aspect", "opinion"))
            nul = PROFILE["null"]
            t = (c, NULL if a in (nul, "") else a, NULL if o in (nul, "") else o, s)
            if t not in out:
                out.append(t)
        return out, bad


def prompt(d, text):
    return PROFILE["prompt"].format(domain=DATASETS[d]["domain"], review=text, language=LANGUAGE[DATASETS[d]["lang"]])


def end_ids(tok):
    """Token ids that close a target: the profile's end token, then eos if it is a different token."""
    e = tok.convert_tokens_to_ids(END)
    return [e] if tok.eos_token_id in (None, e) else [e, tok.eos_token_id]


def round_trip(d, labels=None):
    """Every gold target parses back to its quads (as sets of tuples). Returns (n reviews, n mismatches, example)."""
    labels = labels or Labels(d)
    bad, ex, n = 0, None, 0
    for split in ("train", "dev", "test"):
        for x in load_jsonl(ds_path(d, split)):
            n += 1
            for order in ("gold", "text"):
                quads = text_order(x["text"], x["quads"]) if order == "text" else x["quads"]
                got, inv = labels.parse(labels.target(quads))
                if set(got) != {quad_tuple(q) for q in x["quads"]} or inv:
                    bad += 1
                    ex = ex or (x["id"], labels.target(quads), got)
    return n, bad, ex


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- model

def env_offline(a):
    if a.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if getattr(a, "backend", "hf") == "unsloth":
        import unsloth  # noqa: F401  (must be imported before transformers to patch it)


def dtype_of(a):
    import torch
    if a.dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    return {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[a.dtype]


def load_tokenizer(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=a.cache_dir, local_files_only=a.offline)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(a, device):
    import torch
    dt = dtype_of(a)
    if a.backend == "unsloth":
        from unsloth import FastLanguageModel
        kw = {"cache_dir": a.cache_dir} if a.cache_dir else {}
        model, _ = FastLanguageModel.from_pretrained(model_name=a.model, max_seq_length=a.max_len or 2048, dtype=dt,
                                                     load_in_4bit=False, **kw)
        return model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation=a.attn, cache_dir=a.cache_dir,
                                                 local_files_only=a.offline)
    return model.to(device)


def add_lora(model, a):
    if a.backend == "unsloth":
        from unsloth import FastLanguageModel
        return FastLanguageModel.get_peft_model(
            model, r=a.rank, lora_alpha=a.rank * 2, lora_dropout=a.lora_dropout, bias="none", use_rslora=True,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth" if a.grad_ckpt else False, random_state=a.seed)
    from peft import LoraConfig, TaskType, get_peft_model
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=a.rank, lora_alpha=a.rank * 2, lora_dropout=a.lora_dropout,
                     bias="none", target_modules="all-linear", use_rslora=True)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return get_peft_model(model, cfg)


def add_neftune(model, alpha):
    """NEFTune as in transformers.Trainer: uniform noise of magnitude alpha / sqrt(seq_len * dim) on the input
    embeddings, in training mode only."""
    import torch

    def hook(module, inp, out):
        if module.training and alpha > 0:
            mag = alpha / math.sqrt(out.size(1) * out.size(2))
            out = out + torch.zeros_like(out).uniform_(-mag, mag)
        return out
    return model.get_input_embeddings().register_forward_hook(hook)


def versions():
    import importlib.metadata as md
    v = {"python": platform.python_version()}
    for p in ("torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors", "unsloth"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return v


# ---------------------------------------------------------------- generation

def generate(model, tok, prompts, a, beams, batch_size, device):
    """Decoded continuations (text after the prompt, up to <end_of_turn>), in input order; sorted by length into
    batches, a batch is halved on OOM."""
    import torch
    tok.padding_side = "left"
    eos = end_ids(tok)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    out = [None] * len(prompts)
    model.eval()

    def run(idx):
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=dtype_of(a), enabled=device.type == "cuda"):
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, num_beams=beams, do_sample=False,
                                 eos_token_id=eos, pad_token_id=tok.pad_token_id,
                                 early_stopping=True if beams > 1 else False)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)
        for i, t in zip(idx, texts):
            t = t.split(END)[0].replace(tok.eos_token or END, "").replace(tok.pad_token or END, "")
            out[i] = t.strip()

    def safe(idx):
        try:
            run(idx)
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()
            log(f"OOM at generation batch {len(idx)}: halving")
            safe(idx[:len(idx) // 2])
            safe(idx[len(idx) // 2:])

    t0, done = time.time(), 0
    for b in range(0, len(order), batch_size):
        safe(order[b:b + batch_size])
        done += len(order[b:b + batch_size])
        if (b // batch_size) % 10 == 0:
            log(f"generated {done}/{len(order)} ({(time.time() - t0) / 60:.1f} min)")
    return out


# ---------------------------------------------------------------- train

def build_examples(d, a, labels):
    train = load_jsonl(ds_path(d, "train"))[:a.limit_train or None]
    aug = []
    for p in a.aug_path or []:
        aug += load_jsonl(ROOT / p if not Path(p).is_absolute() else p)
    if a.limit_train and aug:
        aug = aug[:a.limit_train]
    items = train + aug
    if a.tuple_order == "text":
        items = [{**x, "quads": text_order(x["text"], x["quads"])} for x in items]
    return [(prompt(d, x["text"]), labels.target(x["quads"])) for x in items], len(train), len(aug)


def encode(tok, examples, a):
    """Token ids of prompt + target + end token (+ eos if different; bos if the tokenizer adds one); loss mask = all tokens
    (--loss_on all, INLG) or the target only."""
    rows, trunc = [], 0
    for p, t in examples:
        ids = tok(p + t + END, add_special_tokens=True)["input_ids"] + end_ids(tok)[1:]
        n_prompt = len(tok(p, add_special_tokens=True)["input_ids"]) if a.loss_on == "target" else 0
        if a.max_len and len(ids) > a.max_len:
            ids, trunc = ids[:a.max_len], trunc + 1
        rows.append((ids, n_prompt))
    return rows, trunc


def batches(rows, seed, epoch, size, group):
    """Optimizer batches of `size` examples: random order per (seed, epoch); with group, examples of similar
    length share a batch (sorted inside windows of 50 batches, batch order shuffled again)."""
    rng = random.Random(seed * 1000 + epoch)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    if group:
        w = size * 50
        idx = [i for s in range(0, len(idx), w) for i in sorted(idx[s:s + w], key=lambda i: len(rows[i][0]))]
    bs = [idx[s:s + size] for s in range(0, len(idx), size)]
    if group:
        rng.shuffle(bs)
    return bs


def micro_chunks(rows, idx, max_n, max_tok):
    """Split one optimizer batch into forward passes of <= max_n examples and <= max_tok padded tokens
    (longest first), so long batches do not run out of memory and short ones still run many examples at once.
    The loss is normalised by the batch's token count, so the gradient does not depend on the split."""
    idx = sorted(idx, key=lambda i: -len(rows[i][0]))
    out, cur = [], []
    for i in idx:
        if cur and (len(cur) + 1 > max_n or (len(cur) + 1) * len(rows[cur[0]][0]) > max_tok):
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


def collate(rows, idx, pad_id, device):
    import torch
    n = max(len(rows[i][0]) for i in idx)
    ids = torch.full((len(idx), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(idx), n), dtype=torch.long)
    lab = torch.full((len(idx), n), -100, dtype=torch.long)
    for r, i in enumerate(idx):
        x, n_prompt = rows[i]
        ids[r, :len(x)] = torch.tensor(x)
        att[r, :len(x)] = 1
        lab[r, n_prompt:len(x)] = torch.tensor(x[n_prompt:])
    return ids.to(device), att.to(device), lab.to(device)


def cmd_train(a):
    import torch
    env_offline(a)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() and not a.force:
        log(f"{out} already finished (metrics.json); skip")
        return
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = a.dataset
    labels = Labels(d)
    tok = load_tokenizer(a)
    tok.padding_side = "right"
    examples, n_train, n_aug = build_examples(d, a, labels)
    if not a.max_len:  # INLG: max prompt + max target length of the training data (no truncation)
        a.max_len = max(len(tok(p + t + END)["input_ids"]) + 1 for p, t in examples)
    rows, trunc = encode(tok, examples, a)
    if not a.max_new_tokens:
        a.max_new_tokens = max(len(tok(t + END, add_special_tokens=False)["input_ids"]) for _, t in examples) + 32
    steps_per_epoch = math.ceil(len(rows) / a.batch)
    total = steps_per_epoch * a.epochs
    log(f"{d} seed {a.seed}: {n_train} gold + {n_aug} aug = {len(rows)} examples, max_len {a.max_len} "
        f"(truncated {trunc}), {steps_per_epoch} steps/epoch x {a.epochs}, backend {a.backend}, device {device}")

    t_load = time.time()
    model = add_lora(load_base(a, device), a)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"model loaded in {time.time() - t_load:.0f} s; trainable {sum(p.numel() for p in trainable) / 1e6:.1f} M")
    hook = add_neftune(model, a.neftune)
    from transformers import get_cosine_schedule_with_warmup
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.999), eps=1e-8,
                            fused=device.type == "cuda")
    sched = get_cosine_schedule_with_warmup(opt, a.warmup_steps, total)
    history, start_epoch, micro, micro_tok = [], 0, a.micro_batch, a.micro_tokens
    res = out / "resume" / "state.pt"
    if res.exists():
        st = torch.load(res, map_location="cpu", weights_only=False)
        model.load_state_dict(st["lora"], strict=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        history, start_epoch, micro = st["history"], st["epoch"], st["micro"]
        micro_tok = st.get("micro_tok", micro_tok)
        log(f"resumed after epoch {start_epoch}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    oom = 0
    for epoch in range(start_epoch, a.epochs):
        model.train()
        te, loss_sum, tok_sum = time.time(), 0.0, 0
        for step, bidx in enumerate(batches(rows, a.seed, epoch, a.batch, not a.no_group)):
            n_tok = sum(len(rows[i][0]) - max(rows[i][1], 1) for i in bidx)  # predicted (shifted) tokens
            while True:
                failed = False
                try:
                    opt.zero_grad(set_to_none=True)
                    batch_loss = 0.0
                    for chunk in micro_chunks(rows, bidx, micro, micro_tok):
                        ids, att, lab = collate(rows, chunk, tok.pad_token_id, device)
                        k = int((lab[:, 1:] != -100).sum())
                        with torch.autocast(device.type, dtype=dtype_of(a), enabled=device.type == "cuda"):
                            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                        (loss * k / n_tok).backward()
                        batch_loss += float(loss.detach()) * k
                        del ids, att, lab, loss
                except torch.cuda.OutOfMemoryError:
                    if micro == 1:
                        raise
                    failed = True
                if not failed:
                    break
                # recover outside the except block: its traceback keeps the failed step's activations alive
                ids = att = lab = loss = None
                opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                micro, micro_tok, oom = max(1, micro // 2), max(256, micro_tok // 2), oom + 1
                log(f"OOM: micro batch -> {micro} examples / {micro_tok} tokens (effective batch stays {a.batch})")
            torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm)
            opt.step()
            sched.step()
            loss_sum, tok_sum = loss_sum + batch_loss, tok_sum + n_tok
            if step % a.log_every == 0:
                log(f"epoch {epoch + 1} step {step}/{steps_per_epoch} loss {batch_loss / n_tok:.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - te) / 60:.1f} min")
            if a.max_steps and step + 1 >= a.max_steps:
                break
        history.append({"epoch": epoch + 1, "step": (epoch + 1) * steps_per_epoch, "train_loss": loss_sum / tok_sum,
                        "lr": sched.get_last_lr()[0], "train_seconds": time.time() - te, "micro_batch": micro,
                        "micro_tokens": micro_tok, "oom": oom})
        log(f"epoch {epoch + 1} done: loss {loss_sum / tok_sum:.4f}, {(time.time() - te) / 60:.1f} min")
        if epoch + 1 < a.epochs:
            res.parent.mkdir(exist_ok=True)
            lora = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
            torch.save({"lora": lora, "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history,
                        "epoch": epoch + 1, "micro": micro, "micro_tok": micro_tok}, res.with_suffix(".tmp"))
            os.replace(res.with_suffix(".tmp"), res)
    hook.remove()
    train_seconds = sum(h["train_seconds"] for h in history)
    if a.keep_adapter:
        model.save_pretrained(out / "adapter")

    # test, decoded once at the end (beam 5 as INLG); the score goes to metrics.json only
    test = load_jsonl(ds_path(d, "test"))[:a.limit_test or None]
    t1 = time.time()
    if a.backend == "unsloth":
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    gens = generate(model, tok, [prompt(d, x["text"]) for x in test], a, a.test_beams, a.test_batch, device)
    recs = []
    for x, g in zip(test, gens):
        pred, bad = labels.parse(g)
        recs.append({"id": x["id"], "text": x["text"], "gold": [list(quad_tuple(q)) for q in x["quads"]],
                     "pred": [list(t) for t in pred], "generation": g, "n_invalid": bad})
    with open(out / "test_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dev_metrics = None
    if a.eval_dev:  # sanity check of an implementation only (INLG does not use dev); score printed
        dev = load_jsonl(ds_path(d, "dev"))[:a.limit_test or None]
        dgens = generate(model, tok, [prompt(d, x["text"]) for x in dev], a, a.test_beams, a.test_batch, device)
        drecs = []
        for x, g in zip(dev, dgens):
            pred, bad = labels.parse(g)
            drecs.append({"id": x["id"], "text": x["text"], "gold": [list(quad_tuple(q)) for q in x["quads"]],
                          "pred": [list(t) for t in pred], "generation": g, "n_invalid": bad})
        with open(out / "dev_predictions.jsonl", "w", encoding="utf-8") as f:
            for r in drecs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dev_metrics = score([{"gold": [tuple(q) for q in r["gold"]], "pred": [tuple(q) for q in r["pred"]],
                              "n_invalid": r["n_invalid"]} for r in drecs])
        log(f"dev quad F1 {dev_metrics['quad']['f1']:.2f} ({len(drecs)} reviews, "
            f"{sum(r['n_invalid'] > 0 for r in drecs)} with invalid lines)")
    cfg = {"args": vars(a), "seed": a.seed, "n_train": n_train, "n_aug": n_aug, "n_train_total": len(rows),
           "steps_per_epoch": steps_per_epoch, "optimizer_steps": total, "truncated": trunc,
           "data_sha256": {str(p): sha256(ROOT / p if not Path(p).is_absolute() else p)
                           for p in [str(ds_path(d, "train").relative_to(ROOT))] + list(a.aug_path or [])},
           "profile": dict((k, v) for k, v in PROFILE.items() if k != "prompt"),
           "prompt_example": prompt(d, test[0]["text"]) + labels.target(test[0]["quads"]) + END,
           "seconds_train": train_seconds, "seconds_test": time.time() - t1, "oom_fallbacks": oom,
           "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None,
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "versions": versions(),
           "command": " ".join(sys.argv), "hostname": socket.gethostname()}
    json.dump(cfg, open(out / "run_config.json", "w"), indent=1, ensure_ascii=False)
    json.dump(history, open(out / "history.json", "w"), indent=1)
    json.dump({"seed": a.seed, "dev": dev_metrics, "test": score([{"gold": [tuple(q) for q in r["gold"]],
                                               "pred": [tuple(q) for q in r["pred"]], "n_invalid": r["n_invalid"]}
                                              for r in recs])}, open(out / "metrics.json", "w"), indent=1)
    shutil.rmtree(out / "resume", ignore_errors=True)
    log(f"done: train {train_seconds / 60:.1f} min, test decoded ({len(recs)} rows, "
        f"{sum(r['n_invalid'] > 0 for r in recs)} with invalid lines) in {(time.time() - t1) / 60:.1f} min -> {out}")


# ---------------------------------------------------------------- label the pool

def cmd_label(a):
    import torch
    env_offline(a)
    if Path(a.out).exists() and not a.force:
        log(f"{a.out} exists; skip")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = a.dataset
    labels = Labels(d)
    rows = load_jsonl(ROOT / (a.inp or DATASETS[d]["pool"]))[:a.limit or None]
    cfg = json.load(open(Path(a.adapter).parent / "run_config.json"))
    t_args = cfg["args"]
    if t_args.get("profile", "gemma") != PROFILE["name"]:
        raise SystemExit(f"the teacher was trained with --profile {t_args.get('profile', 'gemma')}; "
                         f"label with the same profile (got {PROFILE['name']})")
    if t_args.get("json_ascii", -1) not in (-1, None):
        use_profile(PROFILE["name"], t_args["json_ascii"])
    a.max_new_tokens = a.max_new_tokens or cfg["args"]["max_new_tokens"]
    tok = load_tokenizer(a)
    if a.backend == "unsloth":
        from unsloth import FastLanguageModel
        kw = {"cache_dir": a.cache_dir} if a.cache_dir else {}
        model, _ = FastLanguageModel.from_pretrained(model_name=a.adapter, max_seq_length=2048, dtype=dtype_of(a),
                                                     load_in_4bit=False, **kw)
        FastLanguageModel.for_inference(model)
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(load_base(a, device), a.adapter).to(device)
    t0 = time.time()
    gens = generate(model, tok, [prompt(d, x["text"]) for x in rows], a, 1, a.batch_size, device)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(a.out) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for x, g in zip(rows, gens):
            pred, bad = labels.parse(g)
            f.write(json.dumps({"id": x["id"], "text": x["text"], "pred": [list(t) for t in pred], "generation": g,
                                "n_invalid": bad}, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    log(f"labelled {len(rows)} texts in {(time.time() - t0) / 60:.1f} min -> {a.out}")


# ---------------------------------------------------------------- preflight

def cmd_preflight(a):
    """Environment, model files, data and target format. Exit code 1 if a check fails."""
    env_offline(a)
    res = []

    def check(name, ok, info=""):
        res.append((name, ok, info))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}", flush=True)

    v = sys.version_info
    check("python >= 3.9", v >= (3, 9), platform.python_version())
    check("profile", True, f"{PROFILE['name']}: model {a.model}, target {PROFILE['target']}")
    try:
        import torch
        check("torch >= 2.4", tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 4),
              torch.__version__)
        cuda = torch.cuda.is_available()
        if a.cpu_ok:
            check("device", True, "cuda" if cuda else "cpu (allowed by --cpu_ok)")
        else:
            check("CUDA GPU", cuda, torch.cuda.get_device_name(0) if cuda else "torch.cuda.is_available() is False")
        if cuda:
            mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            check(f"GPU memory >= {PROFILE['min_gpu_gb']} GB ({PROFILE['name']})", mem >= PROFILE["min_gpu_gb"],
                  f"{mem:.0f} GB")
            check("bf16 supported", torch.cuda.is_bf16_supported())
            x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            check("bf16 matmul on GPU", bool(torch.isfinite(x @ x).all()))
    except ImportError as e:
        check("torch import", False, str(e))
    vs = versions()
    for p, want in PINNED.items():
        check(f"{p} == {want}", vs.get(p) == want, vs.get(p, "not installed"))
    if a.backend == "unsloth":
        try:
            import unsloth  # noqa: F401
            check("unsloth import", True, vs.get("unsloth", "?"))
        except Exception as e:  # noqa: BLE001
            check("unsloth import", False, repr(e)[:300])
    free = shutil.disk_usage(ROOT).free / 2 ** 30
    check("free disk >= 20 GB (repo)", free >= 20, f"{free:.0f} GB")
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(a.model, cache_dir=a.cache_dir, local_files_only=a.offline)
        check("model config", True, f"{a.model}: {cfg.model_type}, hidden {cfg.hidden_size}, "
                                    f"layers {cfg.num_hidden_layers}")
        tok = load_tokenizer(a)
        ids = [tok.convert_tokens_to_ids(t) for t in PROFILE["chat_tokens"]]
        check("chat tokens are single tokens", all(i is not None and i != tok.unk_token_id for i in ids), str(ids))
        if a.weights:
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(a.model, cache_dir=a.cache_dir, local_files_only=True,
                                       allow_patterns=["*.safetensors", "*.json"])) if not Path(a.model).is_dir() \
                else Path(a.model)
            n = sum(f.stat().st_size for f in p.glob("*.safetensors")) / 2 ** 30
            check("model weights (safetensors)", n > 1, f"{n:.1f} GB in {p}")
    except Exception as e:  # noqa: BLE001
        check("model files", False, repr(e)[:300])
        tok = None
    manifest = json.load(open(ROOT / "llm" / "MANIFEST.json")) if (ROOT / "llm" / "MANIFEST.json").exists() else {}
    for d, info in DATASETS.items():
        for split in ("train", "dev", "test", "pool"):
            p = ROOT / info["pool"] if split == "pool" else ds_path(d, split)
            if not p.exists():
                check(f"{d}/{split}", False, f"missing {p}")
                continue
            n = sum(1 for line in open(p, encoding="utf-8") if line.strip())
            rel = str(p.relative_to(ROOT))
            ok = n == info["sizes"][split] and (rel not in manifest or manifest[rel] == sha256(p))
            check(f"{d}/{split}", ok, f"{n} rows (expected {info['sizes'][split]})"
                  + ("" if rel not in manifest else ", sha256 " + ("ok" if manifest[rel] == sha256(p) else "DIFFERS")))
        n, bad, ex = round_trip(d)
        check(f"{d} target round trip", bad == 0, f"{n} reviews x 2 orders, {bad} mismatches" + (f": {ex}" if ex else ""))
        if tok is not None:
            lab = Labels(d)
            lens = [len(tok(prompt(d, x["text"]) + lab.target(x["quads"]) + END)["input_ids"])
                    for x in load_jsonl(ds_path(d, "train"))]
            check(f"{d} sequence length", max(lens) <= 2048, f"max {max(lens)}, mean {sum(lens) / len(lens):.0f} tokens")
    n_fail = sum(not ok for _, ok, _ in res)
    print(f"\n{len(res) - n_fail} passed, {n_fail} failed", flush=True)
    sys.exit(1 if n_fail else 0)


# ---------------------------------------------------------------- CLI

def add_model_args(p):
    p.add_argument("--profile", choices=sorted(PROFILES), default=os.environ.get("ASQP_PROFILE", "gemma"),
                   help="model profile: prompt, target format and defaults (gemma = INLG Gemma-7B, the default; "
                        "qwen = Qwen2.5-14B-Instruct as in LLM_QWen.ipynb); env ASQP_PROFILE")
    p.add_argument("--json_ascii", type=int, default=-1,
                   help="json target only: 1 = ASCII escapes (notebook), 0 = characters as they are; -1 = profile")
    p.add_argument("--model", default=os.environ.get("ASQP_MODEL"),
                   help="HF id (with --cache_dir) or a local directory; env ASQP_MODEL; default: the profile's model")
    p.add_argument("--cache_dir", default=os.environ.get("ASQP_CACHE_DIR"), help="HF cache dir; env ASQP_CACHE_DIR")
    p.add_argument("--offline", type=int, default=int(os.environ.get("ASQP_OFFLINE", "1")),
                   help="1 = local files only (default; the server has no Internet)")
    p.add_argument("--backend", choices=["hf", "unsloth"], default=os.environ.get("ASQP_BACKEND", "hf"))
    p.add_argument("--dtype", choices=["auto", "bf16", "fp32", "fp16"], default="auto")
    p.add_argument("--attn", default="sdpa", help="sdpa (default) or eager (INLG)")
    p.add_argument("--max_new_tokens", type=int, default=0, help="0 = longest training target + 32 tokens")
    p.add_argument("--max_len", type=int, default=0)
    p.add_argument("--force", action="store_true")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    add_model_args(p)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--aug_path", nargs="*", default=[])
    p.add_argument("--tuple_order", choices=["gold", "text"], default="gold")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=0, help="effective batch; 0 = profile (gemma 16 = INLG 4 x grad "
                                                       "accum 4; qwen 8 = notebook 2 x 4)")
    p.add_argument("--micro_batch", type=int, default=8, help="max examples per forward pass")
    p.add_argument("--micro_tokens", type=int, default=0,
                   help="max padded tokens per forward pass; 0 = profile (A100 80GB: gemma 4096, qwen 2048)")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--neftune", type=float, default=5.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--loss_on", choices=["all", "target"], default="all")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--no_group", action="store_true", help="plain random batches (no length grouping)")
    p.add_argument("--test_beams", type=int, default=5)
    p.add_argument("--test_batch", type=int, default=16)
    p.add_argument("--keep_adapter", action="store_true")
    p.add_argument("--eval_dev", action="store_true", help="also decode dev and print its F1 (sanity check)")
    p.add_argument("--limit_train", type=int, default=0)
    p.add_argument("--limit_test", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0, help="steps per epoch cap (smoke / timing only)")
    p.add_argument("--log_every", type=int, default=25)
    p = sub.add_parser("label")
    add_model_args(p)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--in", dest="inp", default="", help="default: the dataset's pool")
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("preflight")
    add_model_args(p)
    p.add_argument("--cpu_ok", action="store_true")
    p.add_argument("--weights", type=int, default=1, help="1 = also check the safetensors files are present")
    a = ap.parse_args()
    use_profile(a.profile, a.json_ascii)
    a.model = a.model or PROFILE["model"]
    if a.cmd == "train":
        a.batch = a.batch or PROFILE["batch"]
        a.micro_tokens = a.micro_tokens or PROFILE["micro_tokens"]
    {"train": cmd_train, "label": cmd_label, "preflight": cmd_preflight}[a.cmd](a)


if __name__ == "__main__":
    main()
