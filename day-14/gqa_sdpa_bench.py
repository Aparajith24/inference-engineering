import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoConfig

WARMUP, ITERS = 3, 10
DECODE_ITERS = 100  # a decode step takes under a millisecond, so average over more calls

# Part 1: naive attention vs SDPA, distilgpt2-sized heads
HEADS, HEAD_DIM = 12, 64
SEQ_LENS = [256, 512, 1024, 2048, 4096]

# Part 2: one decode step with a long KV cache, Llama-3-8B-sized heads
Q_HEADS, BIG_HEAD_DIM = 32, 128
KV_VARIANTS = [("MHA", 32), ("GQA-8", 8), ("MQA", 1)]  # (name, number of K/V heads)
CONTEXTS = [2048, 8192, 32768]

# blue, orange, aqua: the first three slots of the reference palette
COLORS = {"naive": "#eb6834", "SDPA": "#2a78d6", "MHA": "#eb6834", "GQA-8": "#2a78d6", "MQA": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def pick_device():
    """GPU with fp16 if we have one; CPU has no fast fp16 path (day 4.1), so fp32 there."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def sync(device):
    """GPU work is asynchronous. Wait for it to finish before reading the clock."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, iters=ITERS):
    """Average seconds per call, after a warmup."""
    for _ in range(WARMUP):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - start) / iters


def naive_attention(q, k, v):
    """Day 9's attention, step by step. Builds the full (T, T) score grid in memory."""
    T = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, T): written to memory
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    weights = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)  # another (B, H, T, T)
    return weights @ v


def sdpa_attention(q, k, v):
    """The same math in one call. PyTorch picks a fused kernel for the device."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def repeat_kv(k, n_q_heads):
    """Give every query head its own copy of its group's K (or V). Simple, but copies memory."""
    return k.repeat_interleave(n_q_heads // k.shape[1], dim=1)


def gqa_decode_step(q, k, v):
    """One decode step with shared K/V heads, WITHOUT copying the cache.

    q:    (B, Hq, 1, d)   the new token's queries, one per query head
    k, v: (B, Hkv, S, d)  the cache: only Hkv heads, shared by groups of query heads
    """
    B, Hq, _, d = q.shape
    Hkv = k.shape[1]
    group = Hq // Hkv                         # query heads per K/V head
    q = q.view(B, Hkv, group, d)              # line up each group with its shared K/V head
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)  # (B, Hkv, group, S)
    out = scores.softmax(dim=-1) @ v          # (B, Hkv, group, d)
    return out.reshape(B, Hq, 1, d)

def kv_cache_bytes(layers, kv_heads, head_dim, tokens, bytes_per_number=2, batch=1):
    """2 (K and V) x layers x K/V heads x head_dim x bytes x tokens x batch."""
    return 2 * layers * kv_heads * head_dim * bytes_per_number * tokens * batch


def fmt_bytes(n):
    for unit, size in [("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n:.0f} B"


def plot(part1, part2, device, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor(SURFACE)

    for name in ("naive", "SDPA"):
        ax1.plot(SEQ_LENS, [r[name] * 1e3 for r in part1], color=COLORS[name], linewidth=2,
                 marker="o", markersize=6, markeredgecolor=SURFACE, label=name)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(SEQ_LENS)
    ax1.set_xticklabels([str(t) for t in SEQ_LENS])
    ax1.set_xlabel("Sequence length T (prefill, causal)", color=MUTED)
    ax1.set_ylabel("Time per attention call (ms)", color=MUTED)
    ax1.set_title(f"Naive vs SDPA ({HEADS} heads x {HEAD_DIM})", loc="left", color=INK, fontweight="bold")

    for name, _ in KV_VARIANTS:
        ax2.plot(CONTEXTS, [r[name] * 1e3 for r in part2], color=COLORS[name], linewidth=2,
                 marker="o", markersize=6, markeredgecolor=SURFACE, label=name)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(CONTEXTS)
    ax2.set_xticklabels([str(c) for c in CONTEXTS])
    ax2.set_xlabel("Tokens in the KV cache", color=MUTED)
    ax2.set_ylabel("Time for one decode step of attention (ms)", color=MUTED)
    ax2.set_title(f"One decode step, {Q_HEADS} query heads x {BIG_HEAD_DIM}", loc="left",
                  color=INK, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.minorticks_off()
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.tick_params(colors=MUTED, length=0)
        ax.legend(frameon=False, labelcolor=INK)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.suptitle(f"Optimizing attention on {device.upper()}", x=0.01, ha="left",
                 color=INK, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot to {path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    device, dtype = pick_device()
    bpn = torch.tensor([], dtype=dtype).element_size()
    print(f"device: {device}, dtype: {dtype}")

    # ---- Part 1 checks: SDPA gives the same answer as our naive version ----
    q, k, v = (torch.randn(2, HEADS, 128, HEAD_DIM) for _ in range(3))  # fp32 on CPU: tight check
    assert torch.allclose(naive_attention(q, k, v), sdpa_attention(q, k, v), atol=1e-5)
    q, k, v = (torch.randn(1, HEADS, 512, HEAD_DIM, device=device, dtype=dtype) for _ in range(3))
    assert torch.allclose(naive_attention(q, k, v), sdpa_attention(q, k, v), atol=2e-2)
    print("Check passed: SDPA == naive attention (fp32 on CPU, and on this device).")

    print(f"\nPART 1: naive attention vs SDPA (1 sequence, {HEADS} heads x {HEAD_DIM}, causal)\n")
    print(f"{'T':>6}{'naive (ms)':>12}{'SDPA (ms)':>11}{'speedup':>9}"
          f"   naive's scores tensor, all {HEADS} heads (x2 with weights):")
    part1 = []
    for T in SEQ_LENS:
        q, k, v = (torch.randn(1, HEADS, T, HEAD_DIM, device=device, dtype=dtype) for _ in range(3))
        t_naive = timed(lambda: naive_attention(q, k, v), device)
        t_sdpa = timed(lambda: sdpa_attention(q, k, v), device)
        per_head = T * T * bpn       # one (T, T) matrix
        all_heads = HEADS * per_head  # the whole (1, HEADS, T, T) scores tensor for this sequence
        part1.append({"T": T, "naive": t_naive, "SDPA": t_sdpa})
        print(f"{T:>6}{t_naive * 1e3:>12.2f}{t_sdpa * 1e3:>11.2f}{t_naive / t_sdpa:>8.1f}x"
              f"   {fmt_bytes(all_heads)} ({fmt_bytes(per_head)} per head)")

    # ---- Part 2 checks: our GQA step is correct ----
    S, B = 64, 2
    q = torch.randn(B, Q_HEADS, 1, BIG_HEAD_DIM)
    for name, kv_heads in KV_VARIANTS:
        k, v = (torch.randn(B, kv_heads, S, BIG_HEAD_DIM) for _ in range(2))
        ours = gqa_decode_step(q, k, v)
        ref = F.scaled_dot_product_attention(q, repeat_kv(k, Q_HEADS), repeat_kv(v, Q_HEADS))
        builtin = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        assert torch.allclose(ours, ref, atol=1e-5) and torch.allclose(ours, builtin, atol=1e-5)
    print("\nCheck passed: our GQA/MQA step == copying K/V to every head == SDPA(enable_gqa=True).")

    print(f"\nPART 2: one decode step, {Q_HEADS} query heads x {BIG_HEAD_DIM}, "
          f"reading a long KV cache (one layer)\n")
    print(f"{'context':>8}" + "".join(f"{name + ' (ms)':>13}" for name, _ in KV_VARIANTS)
          + "   KV cache read (MHA / GQA-8 / MQA)")
    part2 = []
    for S in CONTEXTS:
        q = torch.randn(1, Q_HEADS, 1, BIG_HEAD_DIM, device=device, dtype=dtype)
        row, sizes = {"S": S}, []
        for name, kv_heads in KV_VARIANTS:
            k, v = (torch.randn(1, kv_heads, S, BIG_HEAD_DIM, device=device, dtype=dtype) for _ in range(2))
            row[name] = timed(lambda: gqa_decode_step(q, k, v), device, DECODE_ITERS)
            sizes.append(fmt_bytes(2 * k.numel() * bpn))
        part2.append(row)
        print(f"{S:>8}" + "".join(f"{row[name] * 1e3:>13.3f}" for name, _ in KV_VARIANTS)
              + "   " + " / ".join(sizes))
    mha_bytes = 2 * Q_HEADS * CONTEXTS[-1] * BIG_HEAD_DIM * bpn
    print(f"\nAt {CONTEXTS[-1]} tokens, MHA reads {fmt_bytes(mha_bytes)} in "
          f"{part2[-1]['MHA'] * 1e3:.2f} ms = {mha_bytes / part2[-1]['MHA'] / 1e9:.0f} GB/s "
          f"(memory-bound: compare day 12's measured bandwidth)")

    # ---- Part 3: KV cache calculator ----
    print("\nPART 3: KV cache size (fp16, 2 bytes per number)\n")
    cfg = AutoConfig.from_pretrained("distilgpt2")
    models = [
        # (name, layers, query heads, K/V heads, head_dim)
        ("distilgpt2 (real config, MHA)", cfg.n_layer, cfg.n_head, cfg.n_head, cfg.n_embd // cfg.n_head),
        ("8B shape as MHA", 32, 32, 32, 128),
        ("8B shape as GQA-8 (Llama 3 8B)", 32, 32, 8, 128),
        ("8B shape as MQA", 32, 32, 1, 128),
        ("70B shape as MHA", 80, 64, 64, 128),
        ("70B shape as GQA-8 (Llama 3 70B)", 80, 64, 8, 128),
    ]
    print(f"{'model':<34}{'KV heads':>9}{'per token':>12}{'8K tokens':>12}{'128K tokens':>13}")
    for name, layers, q_heads, kv_heads, hd in models:
        per_tok = kv_cache_bytes(layers, kv_heads, hd, 1)
        print(f"{name:<34}{kv_heads:>9}{fmt_bytes(per_tok):>12}{fmt_bytes(per_tok * 8192):>12}"
              f"{fmt_bytes(per_tok * 131072):>13}")
    assert kv_cache_bytes(cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, 1, 4) == 36864  # day 13, fp32

    gpu_gb, weights_gb, context = 80, 16, 8192  # an 80 GB GPU holding an 8B model at fp16
    free = (gpu_gb - weights_gb) * 1e9
    print(f"\nHow many 8K-token conversations fit next to an 8B model on an {gpu_gb} GB GPU "
          f"({gpu_gb - weights_gb} GB left after {weights_gb} GB of weights)?")
    for name, kv_heads in KV_VARIANTS:
        per_user = kv_cache_bytes(32, kv_heads, 128, context)
        print(f"  {name:<6} {fmt_bytes(per_user):>9} per conversation -> {int(free // per_user):>4} at once")

    plot(part1, part2, device, Path(__file__).parent / "gqa_sdpa_bench.png")
