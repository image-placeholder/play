"""
15M-OPTIMIZED DATASET BUILDER
Designed for small instruction-tuned models (10M–50M params)

Key principles:
- short inputs
- short outputs
- consistent format
- high repetition
"""

import argparse, json, pathlib
import numpy as np
from tqdm import tqdm


# ── Tokenizer ───────────────────────────────────────────────────────────────

def load_tokenizer(tok_path: str):
    tok_path = pathlib.Path(tok_path)
    if tok_path.exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(tok_path))
        return lambda text: sp.encode(text, out_type=int)
    else:
        from transformers import LlamaTokenizer
        tok = LlamaTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer", use_fast=False)
        return lambda text: tok.encode(text, add_special_tokens=False)


# ── Filters (VERY IMPORTANT) ────────────────────────────────────────────────

MAX_INPUT_CHARS = 400
MAX_OUTPUT_CHARS = 200

def clean_text(x):
    return " ".join(x.strip().split())


def is_good_sample(instr, inp, out):
    if not instr or not out:
        return False

    if len(inp) > MAX_INPUT_CHARS:
        return False

    if len(out) > MAX_OUTPUT_CHARS:
        return False

    # kill long explanations (bad for 15M)
    if len(out.split()) > 40:
        return False

    return True


# ── Dataset loaders (trimmed for 15M usefulness) ────────────────────────────

def _iter_code_alpaca(max_examples):
    from datasets import load_dataset
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break

        instr = clean_text(ex.get("instruction", ""))
        inp   = clean_text(ex.get("input", ""))
        out   = clean_text(ex.get("output", ""))

        # compress long outputs → first sentence only
        if "." in out:
            out = out.split(".")[0]

        if not is_good_sample(instr, inp, out):
            continue

        yield instr, inp, out


def _iter_csn(max_examples):
    from datasets import load_dataset
    ds = load_dataset("code_search_net", "python", split="train")

    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break

        code = clean_text(ex.get("func_code_string", ""))
        doc  = clean_text(ex.get("func_documentation_string", ""))

        if not code or len(code) < 20:
            continue

        # convert to MICRO explanation
        summary = doc.split(".")[0] if doc else ""
        if len(summary) < 5:
            continue

        instr = "What does this code do?"
        inp   = code[:MAX_INPUT_CHARS]
        out   = summary[:MAX_OUTPUT_CHARS]

        if not is_good_sample(instr, inp, out):
            continue

        yield instr, inp, out


def _iter_mbpp(max_examples):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split="train")

    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break

        instr = clean_text(ex.get("text", ""))
        code  = clean_text(ex.get("code", ""))

        if not instr or not code:
            continue

        # convert to SHORT task
        out = code.split("\n")[0]

        if not is_good_sample(instr, "", out):
            continue

        yield instr, "", out


SOURCES = {
    "code_alpaca": _iter_code_alpaca,
    "csn": _iter_csn,
    "mbpp": _iter_mbpp,
}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="code_alpaca", choices=SOURCES.keys())
    ap.add_argument("--out_dir", default="llama2.c/data/inst15m")
    ap.add_argument("--tokenizer", default="llama2.c/tokenizer.model")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--shard_size", type=int, default=5_000_000)
    ap.add_argument("--max_seq_len", type=int, default=512)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encode = load_tokenizer(args.tokenizer)

    BOS, EOS = 1, 2
    MAX_OUTPUT_TOKENS = 80

    shard_idx = 0
    shard_toks = []
    total_toks = 0
    n_examples = 0

    def flush(toks, idx):
        arr = np.array(toks, dtype=np.uint16)
        path = out_dir / f"data{idx:02d}.bin"
        arr.tofile(str(path))
        print(f"shard {idx}: {len(arr)} tokens")
        return idx + 1

    print("Building 15M-optimized dataset...")

    for instr, inp, out in tqdm(SOURCES[args.source](args.max_examples or None)):

        text = f"""### Instruction:
{instr}

### Input:
{inp}

### Response:
"""

        try:
            prompt_tokens = encode(text)
            output_tokens = encode(out)

            # 🔥 CRITICAL: truncate outputs hard
            if len(output_tokens) > MAX_OUTPUT_TOKENS:
                output_tokens = output_tokens[:MAX_OUTPUT_TOKENS]

            seq = [BOS] + prompt_tokens + output_tokens + [EOS]

            if len(seq) > args.max_seq_len:
                continue  # skip instead of truncate (better for quality)

        except Exception:
            continue

        shard_toks.extend(seq)
        total_toks += len(seq)
        n_examples += 1

        if len(shard_toks) >= args.shard_size:
            shard_idx = flush(shard_toks, shard_idx)
            shard_toks = []

    if shard_toks:
        shard_idx = flush(shard_toks, shard_idx)

    meta = {
        "examples": n_examples,
        "tokens": total_toks,
        "format": "15M optimized instruction dataset",
        "max_output_tokens": MAX_OUTPUT_TOKENS
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDONE: {n_examples} examples, {total_toks} tokens")


if __name__ == "__main__":
    main()
