import torch

try:
    import triton
    import triton.language as tl
    import triton.testing
except ImportError:
    raise SystemExit("Triton is not installed. Run this on Colab with a T4 GPU (Runtime -> Change runtime type).")

ADD_BLOCK = 1024
ADD_SIZES = [2**12, 2**16, 2**20, 2**24, 2**26]
SOFTMAX_ROWS = 4096
SOFTMAX_COLS = [256, 512, 1024, 2048, 4096, 8192]


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Each program adds one block of BLOCK numbers: out = x + y."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    """Each program handles one whole row: read it once, max, exp, sum, divide, write it once."""
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    x = tl.load(in_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    out = e / tl.sum(e, axis=0)
    tl.store(out_ptr + row * n_cols + offsets, out, mask=mask)


def triton_add(x, y):
    """Launch add_kernel with enough programs to cover every element."""
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, ADD_BLOCK),)
    add_kernel[grid](x, y, out, n, BLOCK=ADD_BLOCK)
    return out


def triton_softmax(x):
    """Launch softmax_kernel with one program per row."""
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    num_warps = 4 if block <= 1024 else 8 if block <= 4096 else 16
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, n_cols, BLOCK=block, num_warps=num_warps)
    return out


def naive_softmax(x):
    """Softmax as five separate PyTorch steps; each step is its own kernel that reads and writes memory."""
    row_max = x.max(dim=1, keepdim=True).values
    shifted = x - row_max
    e = torch.exp(shifted)
    total = e.sum(dim=1, keepdim=True)
    return e / total


def ms(fn):
    """Time fn on the GPU with Triton's helper, which handles warmup and syncing; returns milliseconds."""
    return triton.testing.do_bench(fn)


def show_blocks(n=10, block=4):
    """Print which numbers each program handles, for a tiny example."""
    print(f"PART 0: how the work is split. {n} numbers, BLOCK = {block}, "
          f"so {triton.cdiv(n, block)} programs\n")
    for pid in range(triton.cdiv(n, block)):
        offsets = [pid * block + i for i in range(block)]
        cells = [f"{o}" if o < n else f"({o})" for o in offsets]
        print(f"  program {pid}: offsets {' '.join(f'{c:>4}' for c in cells)}")
    print("  (numbers in brackets are past the end: the mask switches them off, like `if (i < n)` in CUDA)\n")


def add_table():
    """Check the Triton add against PyTorch, then compare their speed in GB/s."""
    print("PART 1: vector add, out = x + y (fp32)\n")
    x, y = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
    assert torch.equal(triton_add(x, y), x + y)
    print("  check: Triton result matches x + y exactly\n")
    print(f"{'numbers':>12}{'MB moved':>10}{'PyTorch GB/s':>14}{'Triton GB/s':>13}")
    for n in ADD_SIZES:
        x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        moved = 3 * n * 4
        torch_gbs = moved / (ms(lambda: x + y) * 1e-3) / 1e9
        triton_gbs = moved / (ms(lambda: triton_add(x, y)) * 1e-3) / 1e9
        print(f"{n:>12,}{moved / 1e6:>10.1f}{torch_gbs:>14.1f}{triton_gbs:>13.1f}")
    print("  Small arrays: launch overhead dominates, so GB/s is tiny.")
    print("  Big arrays: both approach the T4's real bandwidth (day 16 measured ~233 GB/s).\n")


def softmax_table():
    """Check the fused Triton softmax, then race it against naive and PyTorch softmax."""
    print(f"PART 2: softmax over {SOFTMAX_ROWS:,} rows (fp32)\n")
    x = torch.randn(SOFTMAX_ROWS, 1000, device="cuda")
    expected = torch.softmax(x, dim=1)
    assert torch.allclose(triton_softmax(x), expected, atol=1e-6)
    assert torch.allclose(naive_softmax(x), expected, atol=1e-6)
    print("  check: Triton and naive softmax both match torch.softmax\n")
    print(f"{'columns':>8}{'naive ms':>10}{'torch ms':>10}{'Triton ms':>11}{'Triton GB/s':>13}{'vs naive':>10}")
    for n_cols in SOFTMAX_COLS:
        x = torch.randn(SOFTMAX_ROWS, n_cols, device="cuda")
        naive = ms(lambda: naive_softmax(x))
        builtin = ms(lambda: torch.softmax(x, dim=1))
        fused = ms(lambda: triton_softmax(x))
        gbs = 2 * x.numel() * 4 / (fused * 1e-3) / 1e9
        print(f"{n_cols:>8}{naive:>10.3f}{builtin:>10.3f}{fused:>11.3f}{gbs:>13.1f}{naive / fused:>9.1f}x")
    print("  Naive: 5 kernels, about 8 trips of the data through memory (5 reads, 3 writes).")
    print("  Fused: 1 kernel, 2 trips (1 read, 1 write). Same maths, far less memory traffic.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("No NVIDIA GPU found. On Colab: Runtime -> Change runtime type -> T4 GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}, Triton {triton.__version__}\n")
    show_blocks()
    add_table()
    softmax_table()
