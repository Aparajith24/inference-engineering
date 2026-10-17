import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

D_MODEL, N_HEADS, N_LAYERS = 768, 12, 4
SHAPES = [("small input: 8 sequences x 1 token (like decode)", 8, 1),
          ("large input: 8 sequences x 512 tokens (like prefill)", 8, 512)]
MODES = ["eager", "default", "reduce-overhead"]
WARMUP, ITERS = 3, 50


class Block(nn.Module):
    """One GPT block like day 10: LayerNorm, causal attention, residual, LayerNorm, MLP with GELU, residual."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


def pick_device():
    """Use the NVIDIA GPU in fp16 if there is one, otherwise the CPU in fp32."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def sync(device):
    """Wait for the GPU to finish its work, so the timing is correct."""
    if device == "cuda":
        torch.cuda.synchronize()


def per_call_ms(fn, x, device):
    """Return the average milliseconds per forward pass, after a warmup."""
    for _ in range(WARMUP):
        fn(x)
    sync(device)
    start = time.perf_counter()
    for _ in range(ITERS):
        fn(x)
    sync(device)
    return (time.perf_counter() - start) / ITERS * 1e3


def count_kernels(fn, x, device):
    """Count how many GPU kernels one forward pass launches."""
    if device != "cuda":
        return None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(x)
        sync(device)
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def run_mode(model, mode, x, device):
    """Prepare the model in one mode, then measure its first call, its steady speed and its kernel count."""
    torch._dynamo.reset()
    fn = model if mode == "eager" else torch.compile(model, mode=mode)
    start = time.perf_counter()
    out = fn(x).clone()
    sync(device)
    first_s = time.perf_counter() - start
    steady = per_call_ms(fn, x, device)
    kernels = count_kernels(fn, x, device) if mode != "reduce-overhead" else None
    return out, first_s, steady, kernels


if __name__ == "__main__":
    torch.manual_seed(0)
    device, dtype = pick_device()
    model = nn.Sequential(*[Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)]).to(device, dtype).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{N_LAYERS} transformer blocks, d_model {D_MODEL}, {n_params / 1e6:.0f}M parameters, "
          f"on {device} ({dtype}), PyTorch {torch.__version__}\n")

    with torch.no_grad():
        for label, batch, tokens in SHAPES:
            x = torch.randn(batch, tokens, D_MODEL, device=device, dtype=dtype)
            print(label)
            print(f"{'mode':<17}{'first call s':>13}{'per call ms':>13}{'speedup':>9}{'kernels':>9}{'max diff':>10}")
            eager_out, eager_ms = None, None
            for mode in MODES:
                out, first_s, steady, kernels = run_mode(model, mode, x, device)
                if mode == "eager":
                    eager_out, eager_ms = out, steady
                diff = (out - eager_out).abs().max().item()
                assert diff < 0.05, f"{mode} output differs from eager by {diff}"
                kernel_txt = f"{kernels:>9}" if kernels is not None else f"{'graph' if device == 'cuda' else '-':>9}"
                print(f"{mode:<17}{first_s:>13.2f}{steady:>13.3f}{eager_ms / steady:>8.2f}x{kernel_txt}{diff:>10.4f}")
            print()

    print("first call: compiling happens here, so compiled modes pay seconds once.")
    print("kernels: GPU kernels launched per forward pass. Fusion lowers this number.")
    print("reduce-overhead also records the kernels into a CUDA graph and replays it with one launch.")
