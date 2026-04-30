"""
scripts/build_explain_code_dataset.py
──────────────────────────────────────────────────────────────────────────────
Builds a pretokenized explain_code dataset from HuggingFace in the exact
.bin shard format that karpathy/llama2.c's train.py expects.

Sources (selectable via --source flag):
  csn           code_search_net Python
  code_alpaca   sahil2801/CodeAlpaca-20k (Updated to include instructions)
  mbpp          google-research-datasets/mbpp

Each example is serialised as:
    Task: explain_code
    Instruction: <instruction>
    Input: <code>
    Output: <explanation/result>
    <EOS>
"""

import argparse, json, os, pathlib, struct, sys
import numpy as np
from tqdm import tqdm


# ── Tokenizer loader ────────────────────────────────────────────────────────

def load_tokenizer(tok_path: str):
    """Returns a callable encode(text) → list[int]."""
    tok_path = pathlib.Path(tok_path)
    if tok_path.exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(tok_path))
        return lambda text: sp.encode(text, out_type=int)
    else:
        print(f"tokenizer.model not found at {tok_path}, using HF mirror …")
        from transformers import LlamaTokenizer
        tok = LlamaTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer", use_fast=False)
        return lambda text: tok.encode(text, add_special_tokens=False)


# ── Dataset loaders — each yields (instruction, code_str, explanation_str) ───

def _iter_csn(max_examples):
    """code_search_net Python split."""
    from datasets import load_dataset
    ds = load_dataset("code_search_net", "python", split="train",
                      trust_remote_code=True)
    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        code = (ex.get("func_code_string") or "").strip()
        doc  = (ex.get("func_documentation_string") or "").strip()
        if not code or len(code) < 20:
            continue
        summary = doc.split("\n")[0].split(".")[0].strip() if doc else ""
        if len(summary) < 8:
            summary = doc[:150].strip() if doc else "No description."
        # CSN doesn't have a separate instruction, so we use a generic one
        yield "Explain the following Python code.", code[:800], summary


def _iter_code_alpaca(max_examples):
    """sahil2801/CodeAlpaca-20k — instruction/input/output triplets."""
    from datasets import load_dataset
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        instruction = (ex.get("instruction") or "").strip()
        context = (ex.get("input") or "").strip()
        response = (ex.get("output") or "").strip()
        
        if not instruction or not response:
            continue
        
        yield instruction, context, response


def _iter_self_instruct(max_examples):
    """bigcode/self-oss-instruct-sc2-exec-filter-50k"""
    from datasets import load_dataset
    ds = load_dataset("bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train")
    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        instruction = (ex.get("instruction") or "").strip()
        response = (ex.get("response") or "").strip()
        
        if not instruction or not response:
            continue
        
        yield instruction, "", response

def _iter_mbpp(max_examples):
    """google-research-datasets/mbpp — task description + Python solution."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split="train",
                      trust_remote_code=True)
    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        code = (ex.get("code") or "").strip()
        text = (ex.get("text") or "").strip()
        if not code or not text:
            continue
        # MBPP 'text' is the instruction; we leave Input empty as it's built into code
        yield text, "", code


SOURCES = {
    "csn":         _iter_csn,
    "code_alpaca": _iter_code_alpaca,
    "mbpp":        _iter_mbpp,
    "self_instruct": _iter_self_instruct
}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="self_instruct",
                    choices=list(SOURCES.keys()),
                    help="HuggingFace dataset source")
    ap.add_argument("--out_dir", default="llama2.c/data/explain_code",
                    help="Output directory for .bin shards")
    ap.add_argument("--tokenizer", default="llama2.c/tokenizer.model",
                    help="Path to llama2 tokenizer.model")
    ap.add_argument("--max_examples", type=int, default=0,
                    help="Cap on examples (0 = all)")
    ap.add_argument("--shard_size", type=int, default=10_000_000,
                    help="Max tokens per shard")
    ap.add_argument("--max_seq_len", type=int, default=512,
                    help="Truncate sequences longer than this")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encode = load_tokenizer(args.tokenizer)
    BOS, EOS = 1, 2

    shard_idx  = 0
    shard_toks = []
    total_toks = 0
    n_examples = 0

    def flush(toks, idx):
        arr  = np.array(toks, dtype=np.uint16)
        path = out_dir / f"data{idx:02d}.bin"
        arr.tofile(str(path))
        print(f"  shard {idx:02d}: {len(arr):,} tokens → {path}")
        return idx + 1

    iter_fn = SOURCES[args.source]
    max_ex  = args.max_examples if args.max_examples > 0 else None

    print(f"Building explain_code dataset  source={args.source}  out={out_dir}")

      for instruction, input_data, output in tqdm(iter_fn(max_ex)):
        # 1. Construct parts. Use input_data if it exists, otherwise just the instruction.
        # This keeps the format consistent for a 15M model.
        full_instruction = f"{instruction}\nInput: {input_data}" if input_data else instruction
        
        instruction_part = f"I: {full_instruction}\nO: "
        output_part = f"{output}"
        
        try:
            # 2. Encode pieces separately
            # We encode them individually so we don't accidentally double-BOS
            encoded_prompt = encode(instruction_part)
            encoded_output = encode(output_part)
            
            # 3. Assemble: [BOS] + Prompt + Output + [EOS]
            seq = [BOS] + encoded_prompt + encoded_output + [EOS]
            
            # 4. Truncate if it exceeds max_seq_len
            # We save 1 slot for the EOS token to ensure the model learns to stop
            if len(seq) > args.max_seq_len:
                seq = seq[:args.max_seq_len - 1] + [EOS]
                
        except Exception as e:
            print(f"  Tokenization error (skipping): {e}")
            continue

        # 5. Record keeping and shard management
        shard_toks.extend(seq)
        total_toks += len(seq)
        n_examples += 1

        # 6. Periodic flush to .bin file
        if len(shard_toks) >= args.shard_size:
            shard_idx = flush(shard_toks, shard_idx)
            shard_toks = []

    # Final flush for the remaining tokens
    if shard_toks:
        shard_idx = flush(shard_toks, shard_idx) 

    meta = {
        "source":        args.source,
        "examples":      n_examples,
        "total_tokens":  total_toks,
        "shards":        shard_idx,
        "vocab_size":    32000,
        "vocab_source":  "llama2",
        "format":        "uint16 token ids, BOS-prefixed, EOS-terminated",
        "prompt_format": "Task: explain_code\nInstruction: <instr>\nInput: <in>\nOutput: <out>",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✓  {n_examples:,} examples  {total_toks:,} tokens  {shard_idx} shards")
    print(f"    Metadata: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
