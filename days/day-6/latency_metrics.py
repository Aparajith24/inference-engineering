import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

MODEL_NAME = "distilgpt2"
N_GENERATIONS = 50
MIN_TOKENS, MAX_TOKENS = 32, 128   # each request asks for a random length in this range
WARMUP_RUNS = 3
BATCH_SIZES = [1, 4, 16]           # for the throughput demo at the end

PROMPTS = [
    "The future of AI inference is",
    "Once upon a time",
    "Explain how a neural network learns to recognize images of cats and dogs, "
    "starting from the raw pixels",
    "Hello",
    "In the year 2050, cities will",
    "The most important thing to know about GPUs is",
    "Write a short story about a robot who wants to learn how to paint. "
    "The robot lived in a small workshop at the edge of town and",
    "Water boils at",
]


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


def run_one(prompt, n_tokens, tokenizer, model):
    """Generate exactly n_tokens; return per-request metrics."""
    inputs = tokenizer(prompt, return_tensors="pt")
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

    t = streamer.times
    return {
        "ttft": t[0] - start,                          # time to first token
        "itls": [b - a for a, b in zip(t, t[1:])],     # gaps between consecutive tokens
        "total": t[-1] - start,                        # total response time
        "n_tokens": len(t),
        "tps": (len(t) - 1) / (t[-1] - t[0]),          # perceived TPS: tokens/s after the first
        "end": t[-1],
        "start": start,
    }


def percentiles(values, points=(50, 95, 99)):
    return {p: float(np.percentile(values, p)) for p in points}


def print_latency_row(name, values, unit_scale=1000, unit="ms"):
    """Mean, then P50 / P95 / P99, all in the same unit."""
    p = percentiles(values)
    print(
        f"{name:<26}{np.mean(values) * unit_scale:>9.1f}{p[50] * unit_scale:>9.1f}"
        f"{p[95] * unit_scale:>9.1f}{p[99] * unit_scale:>9.1f}   {unit}"
    )


def batched_throughput(batch_size, tokenizer, model, n_tokens=64):
    """Generate n_tokens for a batch of prompts at once; return (wall seconds, total TPS)."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch_size)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    start = time.perf_counter()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            min_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    wall = time.perf_counter() - start
    return wall, batch_size * n_tokens / wall


if __name__ == "__main__":
    torch.manual_seed(0)
    rng = random.Random(0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # batches pad on the left so new tokens follow the prompt
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    for _ in range(WARMUP_RUNS):  # pay one-time setup costs before measuring
        run_one("Hello", 8, tokenizer, model)

    print(f"Running {N_GENERATIONS} generations on {MODEL_NAME}...")
    runs = []
    for i in range(N_GENERATIONS):
        prompt = PROMPTS[i % len(PROMPTS)]
        n_tokens = rng.randint(MIN_TOKENS, MAX_TOKENS)
        runs.append(run_one(prompt, n_tokens, tokenizer, model))

    ttfts = [r["ttft"] for r in runs]
    totals = [r["total"] for r in runs]
    tps_values = [r["tps"] for r in runs]
    all_itls = [gap for r in runs for gap in r["itls"]]  # every gap from every request

    print(f"\n{'':<26}{'mean':>9}{'P50':>9}{'P95':>9}{'P99':>9}")
    print_latency_row("TTFT", ttfts)
    print_latency_row("Inter-token latency (ITL)", all_itls)
    print_latency_row("Total response time", totals, unit_scale=1, unit="s")

    # TPS is a "higher is better" number, so the slow tail is the LOW end: P5 and P1
    tp = np.percentile(tps_values, [1, 5, 50])
    print(f"\nPerceived TPS (per user): mean {np.mean(tps_values):.1f}, "
          f"P50 {tp[2]:.1f}, slowest 1-in-20 (P5) {tp[1]:.1f}, slowest 1-in-100 (P1) {tp[0]:.1f}")
    print(f"Check: 1 / mean ITL = {1 / np.mean(all_itls):.1f} tokens/s per user")

    # Total TPS: everything the service produced divided by the wall-clock time it took
    total_tokens = sum(r["n_tokens"] for r in runs)
    wall = runs[-1]["end"] - runs[0]["start"]
    print(f"\nTotal TPS (throughput): {total_tokens} tokens / {wall:.1f} s = {total_tokens / wall:.1f} tokens/s")
    print("  (one request at a time, so this is close to per-user TPS)")

    # Throughput demo: batching raises total TPS, but each user's share shrinks
    print("\nBatching demo (64 tokens each):")
    print(f"{'batch size':>12}{'wall (s)':>10}{'total TPS':>12}{'per-user TPS':>14}")
    batched_throughput(1, tokenizer, model, n_tokens=8)  # warmup for batched shapes
    for b in BATCH_SIZES:
        wall_b, total_tps = batched_throughput(b, tokenizer, model)
        print(f"{b:>12}{wall_b:>10.2f}{total_tps:>12.1f}{total_tps / b:>14.1f}")
