"""
Usage:
    python3 bench_v1.py                       # distilgpt2, sane defaults
    python3 bench_v1.py gpt2                   # any HF model id
    python3 bench_v1.py gpt2 --n 30 --device mps --dtype fp16
    python3 bench_v1.py meta-llama/Llama-3-70B --skip-run   # just the memory report
"""

import argparse
import random
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
OVERHEAD = 1.2  # rough allowance for KV cache, activations, runtime buffers
GPUS = [
    ("T4", 16), ("L4", 24), ("A10G", 24), ("RTX 4090", 24), ("L40S", 48),
    ("A100 40GB", 40), ("A100 80GB", 80), ("H100", 80), ("H200", 141), ("B200", 192),
]


PROMPTS = [
    "The future of AI inference is",
    "Once upon a time",
    "Explain how a neural network learns to recognize images of cats and dogs, "
    "starting from the raw pixels",
    "Hello",
    "In the year 2050, cities will",
    "The most important thing to know about GPUs is",
    "Water boils at",
]

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


DEFAULT_MAX_RUN_PARAMS = 2_000_000_000


def pick_device():
    """Best device available, GPU first."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    """CPU has no fast half-precision path, so default to fp32 there."""
    return "fp32" if device == "cpu" else "bf16"


def sync(device):
    """GPU work is asynchronous; wait for it to finish before reading the clock."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def count_params(model_name):
    """Parameter count from the config alone, no weights downloaded."""
    config = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):  # tensors with shapes but no memory behind them
        model = AutoModelForCausalLM.from_config(config)
    return sum(p.numel() for p in model.parameters())


def print_memory_report(model_name, n_params):
    print(f"\n{'=' * 60}\nMODEL: {model_name}\n{'=' * 60}")
    print(f"Parameters: {n_params:,.0f} ({n_params / 1e9:.2f}B)")

    precisions = list(BYTES_PER_PARAM)
    weights = {p: n_params * BYTES_PER_PARAM[p] / 1e9 for p in precisions}
    needed = {p: weights[p] * OVERHEAD for p in precisions}

    print(f"\n{'':<18}" + "".join(f"{p:>10}" for p in precisions))
    print(f"{'weights (GB)':<18}" + "".join(f"{weights[p]:>10.2f}" for p in precisions))
    print(f"{'+overhead (GB)':<18}" + "".join(f"{needed[p]:>10.2f}" for p in precisions))

    print(f"\nFits on ONE GPU? (weights + {round((OVERHEAD - 1) * 100)}% overhead)")
    print(f"{'GPU':<18}" + "".join(f"{p:>10}" for p in precisions))
    for name, vram in GPUS:
        cells = "".join(f"{'yes' if needed[p] <= vram else '-':>10}" for p in precisions)
        print(f"{name + f' ({vram}GB)':<18}{cells}")
    return needed


class TokenTimestamps(BaseStreamer):
    """Records the time each generated token arrives.

    generate() calls put() once with the prompt, then once per new token.
    """

    def __init__(self):
        self.times = []
        self._seen_prompt = False

    def put(self, value):
        if not self._seen_prompt:  # first call is the prompt, not a generated token
            self._seen_prompt = True
            return
        self.times.append(time.perf_counter())

    def end(self):
        pass


def run_one(prompt, n_tokens, tokenizer, model, device):
    """Generate exactly n_tokens; return per-request timing metrics."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TokenTimestamps()

    start = time.perf_counter()
    with torch.no_grad():
        model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=n_tokens,
            min_new_tokens=n_tokens,  # don't stop early, so every run has a known length
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    sync(device)

    t = streamer.times
    return {
        "ttft": t[0] - start,
        "itls": [b - a for a, b in zip(t, t[1:])],
        "total": t[-1] - start,
        "n_tokens": len(t),
        "tps": (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else float("nan"),
        "start": start,
        "end": t[-1],
    }


def print_latency_row(name, values, unit_scale=1000, unit="ms"):
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    print(
        f"{name:<26}{np.mean(values) * unit_scale:>9.1f}{p50 * unit_scale:>9.1f}"
        f"{p95 * unit_scale:>9.1f}{p99 * unit_scale:>9.1f}   {unit}"
    )


def batched_throughput(batch_size, tokenizer, model, device, n_tokens=32):
    """Generate n_tokens for a batch of prompts at once; return (wall seconds, total TPS)."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch_size)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            min_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    sync(device)
    wall = time.perf_counter() - start
    return wall, batch_size * n_tokens / wall


def run_benchmark(model_name, device, dtype, n_gens, min_tokens, max_tokens, warmup, batch_sizes):
    print(f"\nLoading {model_name} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # so new tokens follow the prompt in a padded batch

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE_MAP[dtype])
    model.to(device)
    model.eval()

    for _ in range(warmup):  # pay one-time setup costs before measuring
        run_one("Hello", 8, tokenizer, model, device)

    rng = random.Random(0)
    print(f"Running {n_gens} generations...")
    runs = []
    for i in range(n_gens):
        prompt = PROMPTS[i % len(PROMPTS)]
        n_tokens = rng.randint(min_tokens, max_tokens)
        runs.append(run_one(prompt, n_tokens, tokenizer, model, device))

    ttfts = [r["ttft"] for r in runs]
    totals = [r["total"] for r in runs]
    tps_values = [r["tps"] for r in runs]
    all_itls = [gap for r in runs for gap in r["itls"]]

    print(f"\n{'':<26}{'mean':>9}{'P50':>9}{'P95':>9}{'P99':>9}")
    print_latency_row("TTFT", ttfts)
    print_latency_row("Inter-token latency (ITL)", all_itls)
    print_latency_row("Total response time", totals, unit_scale=1, unit="s")

    tp5, tp1 = np.percentile(tps_values, [5, 1])
    print(f"\nPerceived TPS (per user): mean {np.mean(tps_values):.1f}, "
          f"slowest 1-in-20 (P5) {tp5:.1f}, slowest 1-in-100 (P1) {tp1:.1f}")

    total_tokens = sum(r["n_tokens"] for r in runs)
    wall = runs[-1]["end"] - runs[0]["start"]
    print(f"Total TPS (throughput, one request at a time): "
          f"{total_tokens} tokens / {wall:.1f}s = {total_tokens / wall:.1f} tokens/s")

    if batch_sizes:
        print("\nBatching demo (32 tokens each):")
        print(f"{'batch size':>12}{'wall (s)':>10}{'total TPS':>12}{'per-user TPS':>14}")
        batched_throughput(1, tokenizer, model, device, n_tokens=8)  # warm up batched shapes
        for b in batch_sizes:
            wall_b, total_tps = batched_throughput(b, tokenizer, model, device)
            print(f"{b:>12}{wall_b:>10.2f}{total_tps:>12.1f}{total_tps / b:>14.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", default="distilgpt2", help="Hugging Face model id")
    parser.add_argument("--n", type=int, default=20, help="number of generations to time (default: 20)")
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], help="default: best available")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), help="default: fp32 on cpu, bf16 on gpu")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 4],
                         help="batch sizes for the throughput demo (default: 1 4)")
    parser.add_argument("--skip-run", action="store_true",
                         help="print only the parameter/memory report, don't load or run the model")
    parser.add_argument("--max-run-params", type=float, default=DEFAULT_MAX_RUN_PARAMS,
                         help="skip running (but not the memory report) above this many parameters")
    parser.add_argument("--force", action="store_true", help="run even above --max-run-params")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = args.device or pick_device()
    dtype = args.dtype or pick_dtype(device)

    print(f"Counting parameters for {args.model}...")
    n_params = count_params(args.model)
    print_memory_report(args.model, n_params)

    if args.skip_run:
        print("\n--skip-run set: no generations timed.")
    elif n_params > args.max_run_params and not args.force:
        print(f"\n{n_params / 1e9:.1f}B parameters exceeds --max-run-params "
              f"({args.max_run_params / 1e9:.1f}B); skipping the timed run. Pass --force to run anyway.")
    else:
        run_benchmark(args.model, device, dtype, args.n, args.min_tokens, args.max_tokens,
                      args.warmup, args.batch_sizes)
