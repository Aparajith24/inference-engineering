import time

import torch

WARMUP, ITERS = 3, 10
SIZES = [256, 512, 1024, 2048, 4096, 8192]
CPU_MAX = 2048
TRANSFER_MB = [1, 16, 128, 512]
ROUND_TRIP_SIZES = [256, 1024, 4096]


def pick_device():
    """Pick the GPU: CUDA on Colab, MPS on a Mac, CPU only if there is no GPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    """Wait for the GPU to finish its work, so the timing is correct."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def timed(fn, device, iters=ITERS):
    """Run fn a few times after a warmup and return the average seconds per run."""
    for _ in range(WARMUP):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - start) / iters


def describe_gpu(device):
    """Print what the GPU is made of: SMs, VRAM, L2 cache, and how big a model fits."""
    if device != "cuda":
        print(f"Running on {device}. SM count and cache sizes are only reported on CUDA (Colab).\n")
        return
    p = torch.cuda.get_device_properties(0)
    vram_gb = p.total_memory / 1e9
    l2_mb = getattr(p, "L2_cache_size", 0) / 1e6
    max_params_b = vram_gb / 1.5 / 2
    print(f"GPU:                 {p.name}")
    print(f"Streaming MPs (SMs): {p.multi_processor_count}")
    print(f"VRAM:                {vram_gb:.1f} GB")
    print(f"L2 cache:            {l2_mb:.1f} MB" if l2_mb else "L2 cache:            not reported by this PyTorch")
    print(f"Biggest fp16 model with 50% KV cache headroom: about {max_params_b:.1f}B parameters\n")


def matmul_tflops(n, device, dtype):
    """Multiply two n x n matrices and return the speed in TFLOPS."""
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    iters = 3 if device == "cpu" else ITERS
    secs = timed(lambda: a @ b, device, iters)
    return 2 * n**3 / secs / 1e12


def matmul_table(device):
    """Compare matmul speed on the CPU, the GPU in fp32 and the GPU in fp16."""
    print("PART 1: matmul speed in TFLOPS (higher is better)\n")
    print(f"{'n':>6}{'CPU fp32':>10}{'GPU fp32':>10}{'GPU fp16':>10}{'fp16 vs CPU':>13}")
    for n in SIZES:
        cpu = matmul_tflops(n, "cpu", torch.float32) if n <= CPU_MAX else None
        g32 = matmul_tflops(n, device, torch.float32)
        g16 = matmul_tflops(n, device, torch.float16)
        cpu_txt = f"{cpu:>10.2f}" if cpu else f"{'-':>10}"
        ratio_txt = f"{g16 / cpu:>12.1f}x" if cpu else f"{'-':>13}"
        print(f"{n:>6}{cpu_txt}{g32:>10.2f}{g16:>10.2f}{ratio_txt}")
    print()


def transfer_gbps(mb, device, pinned):
    """Copy mb megabytes from CPU RAM to the GPU and return the speed in GB/s."""
    x = torch.empty(mb * 2**20 // 4, dtype=torch.float32)
    if pinned:
        x = x.pin_memory()
    secs = timed(lambda: x.to(device, non_blocking=pinned), device)
    return x.numel() * 4 / secs / 1e9


def vram_gbps(mb, device):
    """Copy mb megabytes inside the GPU's own memory and return the speed in GB/s."""
    x = torch.empty(mb * 2**20 // 4, dtype=torch.float32, device=device)
    secs = timed(lambda: x.clone(), device)
    return 2 * x.numel() * 4 / secs / 1e9


def transfer_table(device):
    """Compare copying CPU to GPU against moving data inside the GPU."""
    print("PART 2: moving data, in GB/s (higher is better)\n")
    pin = device == "cuda"
    header = f"{'size':>8}{'CPU->GPU':>10}"
    header += f"{'pinned':>9}" if pin else ""
    header += f"{'inside GPU':>12}"
    print(header)
    for mb in TRANSFER_MB:
        row = f"{mb:>5} MB{transfer_gbps(mb, device, False):>10.1f}"
        row += f"{transfer_gbps(mb, device, True):>9.1f}" if pin else ""
        row += f"{vram_gbps(mb, device):>12.1f}"
        print(row)
    print()


def round_trip_table(device):
    """Time one matmul on the CPU, on the GPU with data already there, and on the GPU including the copies."""
    print("PART 3: one fp32 matmul, with and without copying the data (ms, lower is better)\n")
    print(f"{'n':>6}{'CPU':>10}{'GPU only':>10}{'GPU + copies':>14}{'time copying':>14}")
    for n in ROUND_TRIP_SIZES:
        a_cpu, b_cpu = torch.randn(n, n), torch.randn(n, n)
        cpu = timed(lambda: a_cpu @ b_cpu, "cpu", 3)
        a_gpu, b_gpu = a_cpu.to(device), b_cpu.to(device)
        gpu_only = timed(lambda: a_gpu @ b_gpu, device)
        full = timed(lambda: (a_cpu.to(device) @ b_cpu.to(device)).cpu(), device)
        share = max(0.0, (full - gpu_only) / full)
        print(f"{n:>6}{cpu * 1e3:>10.2f}{gpu_only * 1e3:>10.2f}{full * 1e3:>14.2f}{share:>13.0%}")
    print()


if __name__ == "__main__":
    torch.manual_seed(0)
    device = pick_device()
    describe_gpu(device)
    matmul_table(device)
    transfer_table(device)
    round_trip_table(device)
    print("Takeaways:")
    print("- The GPU wins big on large matmuls, and fp16 (Tensor Cores) is far faster than fp32.")
    print("- Copying CPU -> GPU is much slower than moving data inside the GPU.")
    print("- So load the weights onto the GPU once and keep them there; never copy them per request.")
