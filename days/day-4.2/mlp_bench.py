import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

SIZES = [128, 256, 512, 1024, 2048, 4096]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
WARMUP = 3
ITERS = 10
MAX_SIZE = {"cpu": 2048}  # CPU half-precision is slow, cap the size

# blue, orange, aqua: the first three slots of the reference palette
COLORS = {"fp32": "#2a78d6", "fp16": "#eb6834", "bf16": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def available_devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def sync(device):
    """GPU work is asynchronous. Wait for it to finish before reading the clock."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class TwoLayerMLP(nn.Module):
    """The day 3 MLP as a PyTorch module: linear -> ReLU -> linear."""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)  # holds a weight matrix W1 and bias b1
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)  # holds W2 and b2

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def mlp_flops(n):
    """Two matmuls of (n x n) @ (n x n), each ~2*n^3 operations (bias and ReLU are negligible)."""
    return 2 * (2 * n**3)


def bench_mlp(n, dtype, device):
    """Average time in seconds for one forward pass of an MLP on a batch of n inputs."""
    model = TwoLayerMLP(n).to(device=device, dtype=dtype)  # moves and converts every parameter
    model.eval()  # inference mode
    x = torch.randn(n, n, device=device, dtype=dtype)  # batch of n inputs, n features each

    with torch.no_grad():  # we only run forward, so skip gradient tracking
        for _ in range(WARMUP):  # first runs include one-time setup cost
            model(x)
        sync(device)

        start = time.perf_counter()
        for _ in range(ITERS):
            model(x)
        sync(device)
    return (time.perf_counter() - start) / ITERS


def run_benchmarks():
    results = {}  # results[device][dtype_name] = [(n, seconds), ...]
    for device in available_devices():
        results[device] = {name: [] for name in DTYPES}
        for n in SIZES:
            if n > MAX_SIZE.get(device, max(SIZES)):
                continue
            for name, dtype in DTYPES.items():
                try:
                    secs = bench_mlp(n, dtype, device)
                except (RuntimeError, TypeError) as e:
                    print(f"skip {device} {name} n={n}: {e}")
                    continue
                tflops = mlp_flops(n) / secs / 1e12
                print(f"{device:>4} {name} n={n:>5}  {secs * 1000:9.3f} ms  {tflops:7.3f} TFLOPS")
                results[device][name].append((n, secs))
    return results


MARKERS = {"fp32": "o", "fp16": "s", "bf16": "^"}  # distinct shapes so overlapping lines stay visible


def draw_panel(ax, points_by_dtype, device, metric):
    """One panel: `metric` is 'time' (ms, lower is better) or 'tflops' (higher is better)."""
    ax.set_facecolor("#fcfcfb")
    for name, points in points_by_dtype.items():
        if not points:
            continue
        xs = [n for n, _ in points]
        if metric == "time":
            ys = [s * 1000 for _, s in points]
        else:
            ys = [mlp_flops(n) / s / 1e12 for n, s in points]
        ax.plot(xs, ys, color=COLORS[name], linewidth=2, marker=MARKERS[name], markersize=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=name)

    ax.set_xscale("log", base=2)
    if metric == "time":
        ax.set_yscale("log")
        ax.set_ylabel("Time per forward pass (ms), lower is faster", color=MUTED)
        ax.set_title(f"{device.upper()}", loc="left", color=INK, fontsize=12, fontweight="bold")
    else:
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Throughput (TFLOPS), higher is faster", color=MUTED)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.minorticks_off()
    ax.set_xlabel("Layer width N (batch of N, N -> N -> N)", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot(results, path):
    devices = list(results)
    fig, axes = plt.subplots(2, len(devices), figsize=(6.5 * len(devices), 8), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    for col, device in enumerate(devices):
        draw_panel(axes[0][col], results[device], device, "time")
        draw_panel(axes[1][col], results[device], device, "tflops")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, labelcolor=INK)
    fig.suptitle("2-layer MLP (nn.Module) forward pass across sizes and dtypes", x=0.01, ha="left",
                 color=INK, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot to {path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, devices: {available_devices()}\n")
    results = run_benchmarks()
    plot(results, Path(__file__).parent / "mlp_bench.png")
