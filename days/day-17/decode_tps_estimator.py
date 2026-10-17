import argparse
import math

GPUS = {
    "T4": ("Turing", 2018, 65, None, 16, 0.32, False),
    "A10": ("Ampere", 2021, 125, None, 24, 0.60, False),
    "A100": ("Ampere", 2020, 312, None, 80, 2.04, True),
    "L4": ("Lovelace", 2023, 121, 242, 24, 0.30, False),
    "L40": ("Lovelace", 2022, 181, 362, 48, 0.86, False),
    "H100": ("Hopper", 2022, 989, 1979, 80, 3.35, True),
    "H200": ("Hopper", 2024, 989, 1979, 141, 4.80, True),
    "B200": ("Blackwell", 2024, 2500, 5000, 192, 8.00, True),
    "B300": ("Blackwell", 2025, 2500, 5000, 288, 8.00, True),
    "RTX 5090": ("Blackwell", 2025, 209, 419, 32, 1.79, False),
    "M3 Ultra": ("Apple", 2025, 28, None, 512, 0.82, False),
    "M4 (yours)": ("Apple", 2024, 4, None, 16, 0.12, False),
}

MODELS = {
    "qwen2.5-1.5b": (1.54, 28, 2, 128),
    "llama-3.1-8b": (8.03, 32, 8, 128),
    "llama-3.3-70b": (70.6, 80, 8, 128),
}

BYTES = {"fp16": 2, "fp8": 1, "q4": 0.5}
KV_BYTES = 2
HEADROOM = 1.5
BW_EFFICIENCY = 0.7
GPU_COUNTS = [1, 2, 4, 8]
BATCHES = [1, 4, 16, 64, 256, 1024]
SHORT_CONTEXT = 128


def weights_gb(params_b, dtype):
    """Return how many GB the model weights take at this precision."""
    return params_b * BYTES[dtype]


def kv_gb_per_token(model):
    """Return how many GB of KV cache one token takes (day 13's formula, with GQA from day 14)."""
    _, layers, kv_heads, head_dim = MODELS[model]
    return 2 * layers * kv_heads * head_dim * KV_BYTES / 1e9


def gpus_needed(gpu, weights):
    """Return the fewest GPUs that hold the weights plus 50% headroom, or None if it cannot fit."""
    mem, nvlink = GPUS[gpu][4], GPUS[gpu][6]
    for n in GPU_COUNTS if nvlink else [1]:
        if weights * HEADROOM <= n * mem:
            return n
    return None


def tflops(gpu, dtype):
    """Return dense TFLOPS for this precision; fp8 needs fp8 Tensor Cores, q4 runs its math in fp16."""
    fp16, fp8 = GPUS[gpu][2], GPUS[gpu][3]
    return fp8 if dtype == "fp8" and fp8 else fp16


def decode_tps(bw_tbs, weights):
    """Return the ideal tokens/s for one user: every token reads every weight once."""
    return bw_tbs * 1000 / weights


def step_time(params_b, weights, kv_per_seq, batch, bw_tbs, flops_t):
    """Return seconds for one decode step of a batch, and whether memory or compute is the limit."""
    memory_s = (weights + batch * kv_per_seq) / (bw_tbs * 1000)
    compute_s = 2 * params_b * 1e9 * batch / (flops_t * 1e12)
    return max(memory_s, compute_s), "memory" if memory_s >= compute_s else "compute"


def spec_table():
    """Print the GPU spec sheet, with each GPU's ops:byte ratio from day 12."""
    print("PART 1: GPU spec sheet (dense, no sparsity)\n")
    print(f"{'GPU':<11}{'arch':<10}{'year':>5}{'fp16 TF':>9}{'fp8 TF':>8}{'VRAM GB':>9}{'TB/s':>6}"
          f"{'NVLink':>8}{'ops:byte':>10}")
    for name, (arch, year, fp16, fp8, mem, bw, nvlink) in GPUS.items():
        fp8_txt = f"{fp8:>8}" if fp8 else f"{'-':>8}"
        print(f"{name:<11}{arch:<10}{year:>5}{fp16:>9}{fp8_txt}{mem:>9}{bw:>6.2f}{'yes' if nvlink else 'no':>8}"
              f"{fp16 / bw:>10.0f}")
    print()


def per_gpu_table(model, dtype, prompt_tokens):
    """Print decode speed and time to first token for one model on every GPU."""
    params_b = MODELS[model][0]
    weights = weights_gb(params_b, dtype)
    print(f"PART 2: {model} at {dtype} = {weights:.1f} GB of weights, one user, {prompt_tokens}-token prompt\n")
    print(f"{'GPU':<11}{'GPUs':>5}{'ideal tok/s':>13}{'real ~tok/s':>13}{'TTFT ms':>9}")
    for gpu in GPUS:
        n = gpus_needed(gpu, weights)
        if n is None:
            print(f"{gpu:<11}{'does not fit':>18}")
            continue
        bw = GPUS[gpu][5] * n
        ideal = decode_tps(bw, weights)
        ttft_ms = 2 * params_b * 1e9 * prompt_tokens / (tflops(gpu, dtype) * n * 1e12) * 1e3
        print(f"{gpu:<11}{n:>5}{ideal:>13.0f}{ideal * BW_EFFICIENCY:>13.0f}{ttft_ms:>9.1f}")
    print()


def batch_table(model, dtype, gpu, context, part):
    """Print how batching raises total throughput until the GPU becomes compute-bound or runs out of memory."""
    params_b = MODELS[model][0]
    weights = weights_gb(params_b, dtype)
    n = gpus_needed(gpu, weights)
    if n is None:
        print(f"PART {part}: {model} at {dtype} does not fit on {gpu}.\n")
        return
    bw, flops_t, mem = GPUS[gpu][5] * n, tflops(gpu, dtype) * n, GPUS[gpu][4] * n
    kv_per_seq = kv_gb_per_token(model) * context
    max_batch = math.floor((mem - weights) / kv_per_seq)
    print(f"PART {part}: batching {model} at {dtype} on {n}x {gpu}, {context} tokens of context per user")
    print(f"KV cache: {kv_per_seq * 1e3:.0f} MB per user, so at most {max_batch} users fit in memory\n")
    print(f"{'batch':>6}{'per-user tok/s':>16}{'total tok/s':>13}{'limit':>9}")
    for batch in [b for b in BATCHES if b < max_batch] + [max_batch]:
        secs, limit = step_time(params_b, weights, kv_per_seq, batch, bw, flops_t)
        print(f"{batch:>6}{1 / secs:>16.0f}{batch / secs:>13.0f}{limit:>9}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate decode tokens/s from GPU specs.")
    parser.add_argument("--model", choices=MODELS, default="llama-3.1-8b")
    parser.add_argument("--dtype", choices=BYTES, default="fp16")
    parser.add_argument("--gpu", choices=GPUS, default="H100")
    parser.add_argument("--prompt", type=int, default=1000)
    parser.add_argument("--context", type=int, default=2048)
    args = parser.parse_args()

    spec_table()
    per_gpu_table(args.model, args.dtype, args.prompt)
    batch_table(args.model, args.dtype, args.gpu, args.context, 3)
    batch_table(args.model, args.dtype, args.gpu, SHORT_CONTEXT, 4)
    print("Rule of thumb: decode tok/s for one user = memory bandwidth / model size.")
    print("More bandwidth = faster decode. More FLOPS = faster prefill (TTFT).")
