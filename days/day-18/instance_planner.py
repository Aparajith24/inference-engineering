import argparse
import math

GPUS = {
    "L4": (121, 242, 24, 0.30, False, 0.80),
    "A100": (312, None, 80, 2.04, True, 1.80),
    "H100": (989, 1979, 80, 3.35, True, 2.50),
    "H200": (989, 1979, 141, 4.80, True, 3.50),
    "B200": (2500, 5000, 192, 8.00, True, 5.50),
}

MIG_SLICES = {
    "H100 MIG 1g.10gb": ("H100", 1 / 7, 10, 1 / 8),
    "H100 MIG 3g.40gb": ("H100", 3 / 7, 40, 4 / 8),
}

MODELS = {
    "qwen2.5-1.5b": (1.54, 28, 2, 128),
    "llama-3.1-8b": (8.03, 32, 8, 128),
    "llama-3.3-70b": (70.6, 80, 8, 128),
}

BYTES = {"fp16": 2, "fp8": 1}
KV_BYTES = 2
HEADROOM = 1.5
BW_EFFICIENCY = 0.7
COMPUTE_EFFICIENCY = 0.5
TARGET_UTILIZATION = 0.8
MAX_BATCH = 256
GPU_COUNTS = [1, 2, 4, 8]


def instance_options():
    """List every instance shape to try: 1, 2, 4 or 8 full GPUs, plus MIG slices of an H100."""
    options = []
    for gpu, (fp16, fp8, mem, bw, nvlink, price) in GPUS.items():
        for n in GPU_COUNTS if nvlink else [1]:
            name = f"{n}x {gpu}"
            options.append((name, gpu, n * fp16, n * fp8 if fp8 else None, n * mem, n * bw, n * price))
    for name, (gpu, compute, mem, bw) in MIG_SLICES.items():
        fp16, fp8, _, full_bw, _, price = GPUS[gpu]
        options.append((name, gpu, fp16 * compute, fp8 * compute, mem, full_bw * bw, price * compute))
    return options


def step_seconds(params_b, weights, kv_per_seq, batch, bw, flops):
    """Return the realistic seconds for one decode step of the whole batch."""
    memory_s = (weights + batch * kv_per_seq) / (bw * 1000 * BW_EFFICIENCY)
    compute_s = 2 * params_b * 1e9 * batch / (flops * 1e12 * COMPUTE_EFFICIENCY)
    return max(memory_s, compute_s)


def plan_instance(option, model, dtype, in_tokens, out_tokens, min_tps):
    """Work out how many requests per second one instance can serve while keeping every user above min_tps."""
    name, gpu, fp16, fp8, mem, bw, price = option
    params_b, layers, kv_heads, head_dim = MODELS[model]
    weights = params_b * BYTES[dtype]
    flops = fp8 if dtype == "fp8" and fp8 else fp16
    if weights * HEADROOM > mem:
        return None
    kv_per_seq = 2 * layers * kv_heads * head_dim * KV_BYTES * (in_tokens + out_tokens) / 1e9
    max_batch = min(MAX_BATCH, math.floor((mem - weights) / kv_per_seq))
    batch = 0
    for b in range(1, max_batch + 1):
        if 1 / step_seconds(params_b, weights, kv_per_seq, b, bw, flops) < min_tps:
            break
        batch = b
    if batch == 0:
        return None
    step = step_seconds(params_b, weights, kv_per_seq, batch, bw, flops)
    prefill_s = 2 * params_b * 1e9 * in_tokens / (flops * 1e12 * COMPUTE_EFFICIENCY)
    rps = 1 / (prefill_s + out_tokens * step / batch)
    return {"name": name, "price": price, "batch": batch, "user_tps": 1 / step, "ttft_ms": prefill_s * 1e3,
            "rps": rps}


def plan(model, dtype, rps, in_tokens, out_tokens, min_tps):
    """Print every instance shape that works, sorted from cheapest to most expensive per 1M tokens."""
    tokens_per_hour = rps * (in_tokens + out_tokens) * 3600
    rows = []
    for option in instance_options():
        p = plan_instance(option, model, dtype, in_tokens, out_tokens, min_tps)
        if p is None:
            continue
        p["count"] = math.ceil(rps / (p["rps"] * TARGET_UTILIZATION))
        p["cost_hr"] = p["count"] * p["price"]
        p["per_1m"] = p["cost_hr"] / (tokens_per_hour / 1e6)
        rows.append(p)
    rows.sort(key=lambda r: r["per_1m"])

    print(f"Model: {model} at {dtype}. Traffic: {rps} requests/s, {in_tokens} in + {out_tokens} out tokens each.")
    print(f"Target: every user gets at least {min_tps} tokens/s. Instances run at most "
          f"{TARGET_UTILIZATION:.0%} busy, to leave room for spikes.\n")
    print(f"{'instance':<18}{'$/hr':>7}{'batch':>7}{'user tok/s':>12}{'TTFT ms':>9}{'req/s each':>12}"
          f"{'count':>7}{'$/hr total':>12}{'$ per 1M tok':>14}")
    for r in rows:
        print(f"{r['name']:<18}{r['price']:>7.2f}{r['batch']:>7}{r['user_tps']:>12.0f}{r['ttft_ms']:>9.0f}"
              f"{r['rps']:>12.1f}{r['count']:>7}{r['cost_hr']:>12.2f}{r['per_1m']:>14.3f}")
    if not rows:
        print("No instance can serve this model at this speed target.")
        return
    best = rows[0]
    print(f"\nCheapest: {best['count']} instance(s) of {best['name']}, ${best['cost_hr']:.2f}/hour, "
          f"${best['cost_hr'] * 24:,.0f}/day, ${best['per_1m']:.3f} per 1M tokens.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan instance count and cost for a model and its traffic.")
    parser.add_argument("--model", choices=MODELS, default="llama-3.1-8b")
    parser.add_argument("--dtype", choices=BYTES, default="fp16")
    parser.add_argument("--rps", type=float, default=50)
    parser.add_argument("--input", type=int, default=500)
    parser.add_argument("--output", type=int, default=200)
    parser.add_argument("--min-tps", type=float, default=30)
    args = parser.parse_args()
    plan(args.model, args.dtype, args.rps, args.input, args.output, args.min_tps)
