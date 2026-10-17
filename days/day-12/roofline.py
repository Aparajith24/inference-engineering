import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import torch

D = 4096                                   # layer width: each weight matrix is D x D
N_LAYERS = 8                               # a stack of 8 different matrices, like 8 model layers
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
LABELED = {1, 2, 4, 8, 16, 32, 64, 256, 4096}
WARMUP, ITERS = 3, 10
H100_OPS_PER_BYTE = 989e12 / 3.35e12       # the book's example GPU, for reference

# blue and orange: the first two slots of the reference palette
MEM_COLOR, COMPUTE_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def pick_device():
    """GPU if we have one, and fp16 there. CPU has no fast fp16 path (day 4.1), so fp32."""
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


def timed(fn, device):
    """Average seconds per call, after a warmup."""
    for _ in range(WARMUP):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    sync(device)
    return (time.perf_counter() - start) / ITERS


def measure_bandwidth(device, dtype):
    """Bytes per second the memory can move: copy a big tensor (read it once, write it once)."""
    n = 256 * 1024 * 1024 // torch.tensor([], dtype=dtype).element_size()  # 256 MB
    src = torch.randn(n, device=device, dtype=dtype)
    dst = torch.empty_like(src)
    secs = timed(lambda: dst.copy_(src), device)
    return 2 * src.numel() * src.element_size() / secs


def measure_peak_compute(device, dtype):
    """Operations per second on a big square matmul, which is heavily compute-bound."""
    n = 4096
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    secs = timed(lambda: a @ b, device)
    return 2 * n**3 / secs


def intensity(T, d, bytes_per_number):
    """Arithmetic intensity of x (T x d) @ W (d x d): operations / bytes moved.

    operations = 2*T*d^2 (one multiply + one add per weight, per token)
    bytes      = (d^2 weights + T*d inputs + T*d outputs) * bytes_per_number
    """
    ops = 2 * T * d * d
    moved = (d * d + 2 * T * d) * bytes_per_number
    return ops / moved


def run_sweep(device, dtype):
    """Push T tokens through the same stack of weights; time it for each T."""
    torch.manual_seed(0)
    weights = [torch.randn(D, D, device=device, dtype=dtype) * 0.01 for _ in range(N_LAYERS)]
    bpn = torch.tensor([], dtype=dtype).element_size()

    def forward(x):
        for W in weights:
            x = x @ W
        return x

    rows = []
    for T in TOKENS:
        x = torch.randn(T, D, device=device, dtype=dtype)
        secs = timed(lambda: forward(x), device)
        ops = 2 * T * D * D * N_LAYERS
        rows.append({"T": T, "secs": secs, "flops": ops / secs, "intensity": intensity(T, D, bpn)})
    return rows


def plot(rows, peak, bandwidth, device, path):
    ratio = peak / bandwidth
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # The roofline: min(compute ceiling, bandwidth x intensity)
    xs = [0.5 * 1.15**i for i in range(65)]
    roof = [min(peak, bandwidth * x) / 1e12 for x in xs]
    ax.plot(xs, roof, color=INK, linewidth=2)
    ax.axvline(ratio, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"ridge: {ratio:.0f} ops/byte\n(balance point)", (ratio, peak / 1e12 * 0.12),
                xytext=(8, 0), textcoords="offset points", color=MUTED, fontsize=9)
    ax.annotate(f"compute ceiling {peak / 1e12:.1f} TFLOPS", (xs[-1], peak / 1e12),
                xytext=(-4, 6), textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    ax.annotate(f"bandwidth ceiling {bandwidth / 1e9:.0f} GB/s", (1.5, bandwidth * 1.5 / 1e12),
                xytext=(-30, 18), textcoords="offset points", color=MUTED, fontsize=9, rotation=33)

    for i, r in enumerate(rows):
        memory_bound = r["intensity"] < ratio
        ax.plot(r["intensity"], r["flops"] / 1e12, "o", markersize=8,
                color=MEM_COLOR if memory_bound else COMPUTE_COLOR,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        if r["T"] in LABELED:  # label a subset so the crowded right side stays readable
            ax.annotate(f"T={r['T']}", (r["intensity"], r["flops"] / 1e12), xytext=(6, -14),
                        textcoords="offset points", color=INK, fontsize=8)

    ax.plot([], [], "o", color=MEM_COLOR, label="memory-bound (below the ridge)")
    ax.plot([], [], "o", color=COMPUTE_COLOR, label="compute-bound (at or past the ridge)")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (operations per byte moved)", color=MUTED)
    ax.set_ylabel("Achieved performance (TFLOPS)", color=MUTED)
    fig.suptitle(f"Roofline on {device.upper()}: T tokens through {D}x{D} weight matrices",
                 x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.01, 0.925, "T=1 is one decode step. Large T is prefill (a whole prompt at once).",
             color=MUTED, fontsize=9)
    ax.grid(color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot to {path}")


if __name__ == "__main__":
    device, dtype = pick_device()
    print(f"device: {device}, dtype: {dtype}")

    bandwidth = measure_bandwidth(device, dtype)
    peak = measure_peak_compute(device, dtype)
    ratio = peak / bandwidth
    print(f"measured memory bandwidth : {bandwidth / 1e9:8.1f} GB/s")
    print(f"measured peak compute     : {peak / 1e12:8.2f} TFLOPS")
    print(f"ops:byte ratio (ridge)    : {ratio:8.1f}   (H100 fp16: {H100_OPS_PER_BYTE:.0f})")

    rows = run_sweep(device, dtype)
    base = rows[0]["secs"]
    print(f"\n{N_LAYERS} layers of {D}x{D} weights ({N_LAYERS * D * D * dtype.itemsize / 1e6:.0f} MB), "
          f"T tokens through all of them:\n")
    print(f"{'T':>6}{'time (ms)':>11}{'vs T=1':>8}{'us/token':>10}{'intensity':>11}"
          f"{'TFLOPS':>9}{'% of peak':>11}   bound")
    for r in rows:
        bound = "memory" if r["intensity"] < ratio else "compute"
        print(f"{r['T']:>6}{r['secs'] * 1e3:>11.2f}{r['secs'] / base:>7.1f}x"
              f"{r['secs'] / r['T'] * 1e6:>10.1f}{r['intensity']:>11.1f}{r['flops'] / 1e12:>9.2f}"
              f"{r['flops'] / peak:>10.0%}   {bound}")

    plot(rows, peak, bandwidth, device, Path(__file__).parent / "roofline.png")
