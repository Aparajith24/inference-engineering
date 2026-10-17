import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STEP_COUNTS = [1, 2, 4, 8, 16, 32, 50]
WARMUP = 2

CHANNELS, PATCH = 4, 2
D_MODEL, N_HEADS, N_LAYERS = 512, 8, 6
LATENT_SIZE = 32
LATENT_SIZES = [16, 32, 64, 128]
FIXED_STEPS = 20

TIME_COLOR, ERROR_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def pick_device():
    """Pick the best device: GPU with fp16 if there is one, otherwise CPU with fp32."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def sync(device):
    """Wait for the GPU to finish its work, so the timing is correct."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def make_ring(h=16, w=32):
    """Make the clean picture: a ring, where +1 is ink and -1 is background."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2 + 0.5) * 2) ** 2 + (xx - w / 2 + 0.5) ** 2)
    return np.where((r > 7) & (r < 12), 1.0, -1.0)


def stand_in_denoiser(x0):
    """Pretend to be a trained model: guess blurry when noisy, sharp when nearly clean."""
    def guess(x, t):
        return (1 - t) * x0 + t * x0.mean()
    return guess


def denoise(noise, guess, n_steps, snapshots=None):
    """Turn noise into a picture in n_steps, moving a little towards the guess each step."""
    x = noise.copy()
    ts = np.linspace(1, 0, n_steps + 1)
    for i in range(n_steps):
        t, t_next = ts[i], ts[i + 1]
        clean_guess = guess(x, t)
        x = x - (t - t_next) * (x - clean_guess) / t
        if snapshots is not None:
            snapshots.append((t_next, x.copy()))
    return x


def ascii_image(x):
    """Turn an image into lines of text characters."""
    chars = " .:-=+*#%@"
    idx = np.clip(((x + 1.5) / 3 * len(chars)).astype(int), 0, len(chars) - 1)
    return ["".join(chars[i] for i in row) for row in idx]


def side_by_side(images, titles):
    """Print several text images next to each other, with a title above each."""
    rows = [ascii_image(x) for x in images]
    print("   ".join(f"{t:<{len(rows[0][0])}}" for t in titles))
    for lines in zip(*rows):
        print("   ".join(lines))


class DenoiserBlock(nn.Module):
    """One transformer block for images: attention with no causal mask, then an MLP."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.out_proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class TinyDenoiser(nn.Module):
    """A small diffusion transformer: cut the image into patches, run the blocks, rebuild the image."""

    def __init__(self, max_tokens):
        super().__init__()
        patch_numbers = CHANNELS * PATCH * PATCH
        self.patch_in = nn.Linear(patch_numbers, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, max_tokens, D_MODEL) * 0.02)
        self.t_emb = nn.Linear(1, D_MODEL)
        self.blocks = nn.ModuleList(DenoiserBlock(D_MODEL, N_HEADS) for _ in range(N_LAYERS))
        self.patch_out = nn.Linear(D_MODEL, patch_numbers)

    def forward(self, x, t):
        B, C, H, W = x.shape
        p = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
        tokens = p.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * PATCH * PATCH)
        h = self.patch_in(tokens) + self.pos_emb[:, :tokens.shape[1]] + self.t_emb(t.view(1, 1, 1))
        for block in self.blocks:
            h = block(h)
        out = self.patch_out(h)
        out = out.view(B, H // PATCH, W // PATCH, C, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5)
        return out.reshape(B, C, H, W)


@torch.no_grad()
def sample(model, size, n_steps, device, dtype):
    """Generate one image: start from noise and run the network once per step."""
    x = torch.randn(1, CHANNELS, size, size, device=device, dtype=dtype)
    ts = torch.linspace(1, 0, n_steps + 1, device=device, dtype=dtype)
    for i in range(n_steps):
        direction = model(x, ts[i])
        x = x - (ts[i] - ts[i + 1]) * direction
    return x


def time_sampling(model, size, n_steps, device, dtype):
    """Measure how many seconds it takes to generate one image."""
    for _ in range(WARMUP):
        sample(model, size, 4, device, dtype)
    sync(device)
    repeats = max(1, 50 // n_steps)
    start = time.perf_counter()
    for _ in range(repeats):
        sample(model, size, n_steps, device, dtype)
    sync(device)
    return (time.perf_counter() - start) / repeats


def flops_per_step(tokens):
    """Roughly count the math operations in one forward pass."""
    per_layer = 2 * tokens * 12 * D_MODEL**2 + 4 * tokens**2 * D_MODEL
    return N_LAYERS * per_layer


def plot(error_rows, time_rows, path):
    """Draw two charts: time vs steps, and error vs steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(SURFACE)

    ax1.plot(STEP_COUNTS, [t for _, t in time_rows], color=TIME_COLOR, linewidth=2, marker="o",
             markersize=6, markeredgecolor=SURFACE)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Denoising steps", color=MUTED)
    ax1.set_ylabel("Time to generate one image (s)", color=MUTED)
    ax1.set_title("Latency grows in a straight line with steps", loc="left", color=INK, fontweight="bold")

    ax2.plot(STEP_COUNTS, [e for _, e in error_rows], color=ERROR_COLOR, linewidth=2, marker="o",
             markersize=6, markeredgecolor=SURFACE)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(STEP_COUNTS)
    ax2.set_xticklabels([str(s) for s in STEP_COUNTS])
    ax2.minorticks_off()
    ax2.set_xlabel("Denoising steps", color=MUTED)
    ax2.set_ylabel("Error vs the clean picture (lower is better)", color=MUTED)
    ax2.set_title("Quality improves with steps (Part 1 toy)", loc="left", color=INK, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.suptitle("Diffusion: more steps = better image, slower image", x=0.01, ha="left",
                 color=INK, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot to {path}")


if __name__ == "__main__":
    clean = make_ring()
    guess = stand_in_denoiser(clean)
    noise = np.random.default_rng(0).standard_normal(clean.shape)

    print("PART 1: denoising a tiny picture, 8 steps\n")
    snaps = []
    denoise(noise, guess, 8, snaps)
    side_by_side([noise, snaps[1][1], snaps[3][1]], ["start: pure noise (t=1.0)", "after 2 steps (t=0.75)",
                                                      "after 4 steps (t=0.5)"])
    print()
    side_by_side([snaps[5][1], snaps[7][1], clean], ["after 6 steps (t=0.25)", "after 8 steps (t=0)",
                                                     "the clean picture"])

    print("\nThe same start, with fewer or more steps:\n")
    results = {n: denoise(noise, guess, n) for n in [1, 4, 32]}
    side_by_side(list(results.values()), [f"{n} step{'s' if n > 1 else ''}" for n in results])

    error_rows = []
    print(f"\n{'steps':>6}{'error':>9}   (average distance from the clean picture)")
    for n in STEP_COUNTS:
        err = float(np.abs(denoise(noise, guess, n) - clean).mean())
        error_rows.append((n, err))
        print(f"{n:>6}{err:>9.3f}")
    assert all(a[1] > b[1] for a, b in zip(error_rows, error_rows[1:]))
    assert np.allclose(denoise(noise, lambda x, t: clean, 1), clean)

    torch.manual_seed(0)
    device, dtype = pick_device()
    max_tokens = (max(LATENT_SIZES) // PATCH) ** 2
    model = TinyDenoiser(max_tokens).to(device=device, dtype=dtype).eval()
    tokens = (LATENT_SIZE // PATCH) ** 2
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nPART 2: a {N_LAYERS}-layer transformer denoiser ({n_params / 1e6:.0f}M parameters, random weights) "
          f"on {device} ({dtype})")
    print(f"A {LATENT_SIZE}x{LATENT_SIZE}x{CHANNELS} latent, cut into {PATCH}x{PATCH} patches = "
          f"{tokens} tokens through the network EVERY step\n")

    time_rows = []
    print(f"{'steps':>6}{'total (s)':>11}{'per step (ms)':>15}")
    for n in STEP_COUNTS:
        secs = time_sampling(model, LATENT_SIZE, n, device, dtype)
        time_rows.append((n, secs))
        print(f"{n:>6}{secs:>11.3f}{secs / n * 1e3:>15.1f}")
    print(f"50 steps took {time_rows[-1][1] / time_rows[0][1]:.0f}x as long as 1 step.")

    print(f"\nSame network, {FIXED_STEPS} steps, bigger latents (more patches per step):\n")
    print(f"{'latent':>9}{'tokens':>8}{'total (s)':>11}{'per step (ms)':>15}{'TFLOPS':>9}")
    for size in LATENT_SIZES:
        t_tokens = (size // PATCH) ** 2
        secs = time_sampling(model, size, FIXED_STEPS, device, dtype)
        tflops = flops_per_step(t_tokens) * FIXED_STEPS / secs / 1e12
        print(f"{size:>4}x{size:<4}{t_tokens:>8}{secs:>11.3f}{secs / FIXED_STEPS * 1e3:>15.1f}{tflops:>9.2f}")
    print("Day 12: this GPU peaks at about 2.8 TFLOPS, and an LLM decode step (1 token) used about 3% of it.")
    print("Here hundreds or thousands of patch tokens share every weight fetch, so each step is")
    print("compute-heavy like prefill, not memory-bound like decode.")

    plot(error_rows, time_rows, Path(__file__).parent / "toy_diffusion_steps.png")
