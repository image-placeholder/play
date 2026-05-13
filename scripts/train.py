import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
import torch.nn.functional as F

from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task
from export import model_export


# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = "scratch"

wandb_log = False
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# data
batch_size = 64
max_seq_len = 512
vocab_source = "llama2"
vocab_size = 32000

# model (~15M–50MB target)
dim = 384
n_layers = 8
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0

# optimizer
gradient_accumulation_steps = 4
learning_rate = 3e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr decay
decay_lr = True
warmup_iters = 1000

# system
device = "cuda"
dtype = "bfloat16"
compile = True

# -----------------------------------------------------------------------------

config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}

lr_decay_iters = max_iters
min_lr = 0.0

assert vocab_source in ["llama2", "custom"]
assert vocab_source == "custom" or vocab_size == 32000


# -----------------------------------------------------------------------------
# RESPONSE TOKEN SETUP (for loss masking)

def get_response_tokens(tokenizer_path="llama2.c/tokenizer.model"):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    return sp.encode("### Response:\n", out_type=int)

RESPONSE_TOKENS = get_response_tokens()


# -----------------------------------------------------------------------------
# DDP setup

ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len

if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# -----------------------------------------------------------------------------
# data loader

iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

# -----------------------------------------------------------------------------
# model

iter_num = 0
best_val_loss = 1e9

model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)

print("Initializing a new model from scratch")
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("compiling the model...")
    model = torch.compile(model)

if ddp:
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model


# -----------------------------------------------------------------------------
# eval

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)

                logits = logits[:, :-1, :]
                targets = Y[:, 1:]

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="mean"
                )

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -----------------------------------------------------------------------------
# LR schedule

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# TRAIN

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)

t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    for micro_step in range(gradient_accumulation_steps):

        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

        with ctx:
            logits = model(X, Y)

            logits = logits[:, :-1, :]
            targets = Y[:, 1:]

            mask = torch.zeros_like(targets, dtype=torch.float32)

            for b in range(targets.size(0)):
                seq = X[b].tolist()

                for i in range(len(seq) - len(RESPONSE_TOKENS)):
                    if seq[i:i+len(RESPONSE_TOKENS)] == RESPONSE_TOKENS:
                        start = i + len(RESPONSE_TOKENS)
                        mask[b, start:] = 1.0
                        break

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none"
            )

            loss = loss.view_as(mask)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            loss = loss / gradient_accumulation_steps

        X, Y = next(train_batch_iter)

        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps

        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break


if ddp:
    destroy_process_group()
