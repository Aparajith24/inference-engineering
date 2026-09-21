# Day 4: PyTorch basics (tensors, dtype, device)

Task: benchmark matmul across sizes and dtypes (fp32/fp16/bf16) and plot timings.

## What we built

`matmul_bench.py` multiplies two N x N matrices for N from 128 to 4096, in
fp32, fp16 and bf16, on CPU and on Apple's GPU (MPS). It reports time per
matmul and TFLOPS, and saves a plot to `matmul_bench.png`.

Run it:

```
pip3 install matplotlib   # once, for the plot
python3 matmul_bench.py
```

It takes under a minute. CPU is capped at N=1024 because CPU half-precision
matmul takes seconds per call.

## What we learned

### Tensors

A tensor is a grid of numbers, the basic PyTorch data type. It is like a NumPy
array that can also live on a GPU and track gradients. Inputs, weights and
outputs of a model are all tensors. `a @ b` is the same matmul as day 3.

### dtype

The dtype is how each number is stored.

| dtype | Bits | Exponent / mantissa bits | Trade-off |
|---|---|---|---|
| fp32 | 32 | 8 / 23 | Accurate, the default |
| fp16 | 16 | 5 / 10 | Half the memory, small range (can overflow) |
| bf16 | 16 | 8 / 7 | Half the memory, fp32's range, coarser precision |

Fewer bits means fewer bytes to read from memory, which matters because decode
is memory-bound (day 1). This is the simplest form of quantization (day 2).

### Device

The device is where a tensor lives and where math on it runs: `cpu`, `cuda`
(NVIDIA GPU) or `mps` (Apple GPU). Tensors must be on the same device to be
combined.

### Benchmarking a GPU correctly

- **Synchronize.** GPU calls are asynchronous: `a @ b` only submits the job.
  Call `torch.mps.synchronize()` before reading the clock, or you time the
  submission instead of the math.
- **Warm up.** The first calls include one-time setup cost, so run a few
  untimed multiplications first.
- **Average.** Time several runs and divide, to smooth out noise.
- **TFLOPS.** A matmul does about `2 * N^3` operations, so
  `2 * N^3 / seconds / 1e12` gives throughput independent of size.

## Results (Apple M4)

| Device | Size | fp32 | fp16 | bf16 |
|---|---|---|---|---|
| CPU | 1024 | 1.2 ms | 640 ms | 640 ms |
| MPS | 4096 | 53 ms | 49 ms | 49 ms |

1. **CPU fp16/bf16 is about 500x slower than fp32.** The CPU has fast fp32
   paths but no fast half-precision path. Lower precision is not automatically
   faster; it depends on hardware support.
2. **On MPS, fp16/bf16 are only slightly faster than fp32** (about 8-10% at
   large sizes). Many NVIDIA GPUs gain 2-8x from dedicated half-precision
   hardware.
3. **Small matrices underuse the GPU.** MPS reaches about 0.05 TFLOPS at N=128
   and about 2.8 at N=4096. Overhead dominates small jobs, which is why
   batching helps.
4. **Time grows about 8x each time N doubles**, because the work is `N^3`.
5. **fp16 and bf16 run at the same speed.** Both are 16 bits. They differ in
   range and precision, not speed.
6. **CPU beats the GPU at small sizes**, and the GPU wins only when matrices
   are big enough.

## Connection to days 1-3

- Day 3 did the same `x @ W` in NumPy (CPU, float64). Here we time that
  operation across device and dtype.
- Matmul is where nearly all of a model's compute goes, so matmul speed is
  model speed.
- Device, dtype and matrix size are the three levers that the day 2
  techniques (batching, quantization) pull.

## Caveats

- Numbers vary a little between runs, and other apps on the machine affect
  them.
- This measures speed only, not accuracy. fp16 and bf16 lose precision, so
  results differ slightly from fp32.
- The CPU x-axis shows sizes up to 4096 but the CPU has no data past 1024.

## Things to try

1. Set `ITERS = 1` and see how much noisier the numbers get.
2. Comment out `sync(device)` in `bench_matmul` on MPS. The times look
   impossibly fast, which shows the async behavior.
3. Measure the accuracy cost:
   ```python
   a = torch.randn(1024, 1024); b = torch.randn(1024, 1024)
   exact = a @ b
   approx = (a.bfloat16() @ b.bfloat16()).float()
   print((exact - approx).abs().max())
   ```
   Repeat with `.half()` and compare.
