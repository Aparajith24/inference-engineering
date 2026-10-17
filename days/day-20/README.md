# Day 20: CUDA and kernels — your first GPU kernels in Triton

Book chapter: **4.1 CUDA** (kernels, kernel libraries, kernel selection, kernel fusion)

Task: first GPU kernel in Triton: vector add, then fused softmax. Runs on Colab.

## What we built

`triton_softmax.py` has three parts:

0. **How the work is split:** a tiny example (10 numbers, blocks of 4) showing
   which numbers each program handles, and how the mask switches off the extras.
1. **Vector add (`out = x + y`):** our first kernel. It checks the result matches
   PyTorch exactly, then compares speed in GB/s from tiny to huge arrays.
2. **Fused softmax:** one kernel that does all of softmax in a single pass. It
   checks the result, then races three versions:
   - **naive:** softmax written as 5 separate PyTorch steps (5 kernels),
   - **torch.softmax:** PyTorch's own built-in (already fused) kernel,
   - **Triton:** our fused kernel.

## How to run it on Colab

Triton only runs on NVIDIA GPUs, so this cannot run on a Mac.

1. colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. Upload `triton_softmax.py` with the folder icon on the left (or paste it
   into a cell that starts with `%%writefile triton_softmax.py`).
4. Run `!python triton_softmax.py`. It takes under a minute. PyTorch and Triton
   are already installed on Colab.

## What we learned

### What a kernel is

A **kernel** is a function that runs on the GPU, carried out by many workers at
once. In CUDA C++ (chapter 4.1's example), one thread handles one number:

```cpp
__global__ void double_it(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] * 2;
    }
}
```

**Triton** is a Python-like language for writing kernels. Instead of one thread
per number, you write what **one program** does with a **block** of numbers,
and Triton spreads the block over threads for you.

### Our vector add, line by line

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

| Line | Meaning |
|---|---|
| `@triton.jit` | "compile this function into a GPU kernel" |
| `x_ptr, y_ptr, out_ptr` | **pointers**: where each array starts in GPU memory |
| `BLOCK: tl.constexpr` | how many numbers one program handles (1,024); fixed when compiled |
| `pid = tl.program_id(0)` | "which program am I?", like a cook's number tag |
| `offsets = pid * BLOCK + tl.arange(0, BLOCK)` | the positions this program handles. Program 2 gets 2048 to 3071 |
| `mask = offsets < n` | switch off positions past the end, like CUDA's `if (i < n)` |
| `tl.load(...)` | read a block of numbers from VRAM |
| `tl.store(...)` | write the results back to VRAM |

The launch line decides how many programs run:

```python
grid = (triton.cdiv(n, ADD_BLOCK),)
add_kernel[grid](x, y, out, n, BLOCK=ADD_BLOCK)
```

`triton.cdiv(n, 1024)` is "n divided by 1,024, rounded up". 1,000,000 numbers
need 977 programs; the last one is only partly used, so the mask matters.

Part 0 prints this for 10 numbers in blocks of 4:

```
program 0: offsets    0    1    2    3
program 1: offsets    4    5    6    7
program 2: offsets    8    9 (10) (11)     ← (10), (11) are past the end: masked off
```

### Softmax: why fusing it helps

Softmax (day 8) is really three steps on each row: **find the max, take exp and
sum, divide**. Written as normal PyTorch (`naive_softmax`), that is 5 separate
kernels, and each one reads its input from VRAM and writes its output back:

```
max      read row          write max
x - max  read row + max    write shifted        ← the row goes through memory
exp      read shifted      write e                 about 8 times in total
sum      read e            write total
divide   read e + total    write result
```

Our fused kernel loads each row **once**, does all the steps while the numbers
sit in fast on-chip memory, and writes the answer **once**:

```python
x = tl.load(in_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
x = x - tl.max(x, axis=0)
e = tl.exp(x)
out = e / tl.sum(e, axis=0)
tl.store(out_ptr + row * n_cols + offsets, out, mask=mask)
```

- One program per row, so 4,096 rows means 4,096 programs.
- `other=-float("inf")`: positions past the end of the row load as −∞, so
  exp turns them into 0 and they do not change the sum (the same trick as the
  causal mask in day 8).
- About **8 memory trips become 2**, with exactly the same maths. Softmax is
  memory-bound, so fewer trips means faster.

This is the same idea as FlashAttention (day 14): fuse the steps so big
in-between results never go out to VRAM.

### Kernel libraries and kernel selection

Most engineers do not write kernels; they **choose** them:

| Library | What it gives |
|---|---|
| cuBLAS | ready-made matmul (GEMM) kernels, used by every linear layer |
| cuDNN | ready-made neural-network kernels |
| CUTLASS / CuTe | building blocks for writing your own fast kernels |
| FlashInfer | ready-made LLM kernels (attention, sampling) |
| DeepGEMM | DeepSeek's fp8 matmul kernels for Hopper |

Kernels are tuned to one GPU. A kernel tuned for the H100 does not use a
B200's extras, and a B200 kernel may not run on an H100 at all. Selection is
mostly automatic: day 14's SDPA picked FlashAttention on NVIDIA and a different
kernel on the Mac.

## Results (Colab T4)

Paste your output here after running it. What to expect:

- **Vector add:** tiny arrays reach only a few GB/s (launching a kernel costs
  a few microseconds, longer than adding 4,096 numbers). Big arrays approach
  the T4's real bandwidth: day 16 measured about 233 GB/s. Triton and PyTorch
  should be about equal, because adding is simple and both just stream memory.
- **Softmax:** Triton should be **several times faster than naive** (fewer
  memory trips) and about the same as `torch.softmax`, which is already a fused
  kernel written by PyTorch's engineers.

## Connection to earlier days

- Day 4.1 and 16: speed measured in GB/s, compared with day 16's ~233 GB/s.
- Day 8: softmax and the −∞ trick, now inside a kernel.
- Day 12: softmax and vector add do very little maths per byte, so they are
  memory-bound. That is why fusion (fewer bytes) helps them.
- Day 14: FlashAttention is fusion applied to all of attention.
- Day 16: SMs run the programs; the fused kernel keeps data in on-chip SRAM.

## Caveats

- `triton.testing.do_bench` handles warmup and syncing and returns the time in
  milliseconds.
- GB/s for softmax counts the ideal 1 read and 1 write per number.
- Our softmax holds a whole row in one program, so it suits rows up to a few
  thousand numbers. Very long rows need a version that loops over chunks.
- fp32 only, to keep things simple.

## Things to try

1. Change `ADD_BLOCK` to 128 or 4096. Does the speed of big arrays change?
2. Add 16384 to `SOFTMAX_COLS`. What happens to the Triton time?
3. Fuse more: write a kernel that does `softmax(x * 2)` in one pass, and compare
   with `torch.softmax(x * 2, dim=1)` (two kernels).
