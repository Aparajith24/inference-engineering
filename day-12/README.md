# Day 12: Bottlenecks — a roofline plot

Book chapter: **2.4 Calculating bottlenecks**

Task: roofline plot showing that prefill is compute-bound and decode is
memory-bandwidth-bound.

## What we built

`roofline.py` measures your machine's two resources, then pushes different
numbers of tokens through the same weights and plots where each one lands:

1. **Memory bandwidth:** copy a 256 MB tensor and time it.
2. **Peak compute:** time a big 4096 x 4096 matmul.
3. **The sweep:** push T tokens (1, 2, 4 ... 4,096) through a stack of 8
   weight matrices, each 4096 x 4096 (268 MB total, like 8 model layers).
   Time each one and calculate its arithmetic intensity.
4. **The plot:** `roofline.png`, with each T as a point on the roofline.

Run it:

```
python3 roofline.py
```

It uses the MPS GPU with fp16 if available (CUDA if present), and CPU with
fp32 otherwise, since CPU has no fast fp16 path (day 4.1).

## What we learned

### Two resources, one bottleneck

| Resource | Measures | Analogy |
|---|---|---|
| Compute | Math operations per second | The chef |
| Memory bandwidth | Bytes fetched from memory per second | The delivery van |

Whichever runs out first is the bottleneck. Speeding up the other one does
nothing.

### Ops:byte ratio (a property of the hardware)

```
ops:byte ratio = peak compute / memory bandwidth
```

The number of operations the GPU can do in the time it fetches one byte. It is
the balance point, the "ridge" on the roofline.

| Hardware | Compute | Bandwidth | Ratio |
|---|---|---|---|
| H100 (fp16, from the book) | 989 TFLOPS | 3.35 TB/s | 295 |
| Apple M4 GPU (measured, fp16) | about 2.8 TFLOPS | about 94 GB/s | about 30 |

### Arithmetic intensity (a property of the work)

```
arithmetic intensity = operations performed / bytes moved
```

- Below the ratio: **memory-bound** (the chef waits for the van).
- Above the ratio: **compute-bound** (the van waits for the chef).

### The formula for one layer

For `x @ W`, with `x` holding T tokens of d numbers and `W` a d x d matrix, at
2 bytes per number:

```
operations = 2 * T * d^2                  (one multiply + one add per weight, per token)
bytes      = 2 * (d^2 + 2 * T * d)        (read the weights once, read x, write the result)
intensity  = T * d / (d + 2T)             (about T while T is much smaller than d)
```

**Intensity is roughly the number of tokens sharing one fetch of the
weights.** It levels off at `d / 2` for very large T.

### The roofline chart

```
performance
    |               ________________  compute ceiling
    |              /
    |             /   compute-bound
    |            /
    |   memory  /
    |   bound  /
    |_________/______________________ arithmetic intensity
             ridge (ops:byte ratio)
```

Plot a point at its intensity. On the slope, it is memory-bound. On the flat
roof, it is compute-bound.

## Results (Apple M4 GPU, fp16)

| T (tokens) | Time | vs T=1 | Intensity | % of peak | Bound |
|---|---|---|---|---|---|
| **1** (decode) | 3.31 ms | 1.0x | 1 | 3% | memory |
| 4 | 3.31 ms | 1.0x | 4 | 12% | memory |
| 16 | 2.83 ms | 0.9x | 16 | 54% | memory |
| 32 | 3.94 ms | 1.2x | 32 | 77% | compute |
| 256 | 25.6 ms | 7.7x | 228 | 95% | compute |
| **4,096** (prefill) | 387 ms | 117x | 1,365 | 101% | compute |

### What it shows

- **16 tokens took the same time as 1.** With T=1, the GPU spends its whole
  3.3 ms fetching 268 MB of weights and uses 3% of its compute. Extra tokens
  use that idle compute for free, because the weights are fetched once either
  way.
- **Past the ridge (about 30 tokens), time grows with tokens.** The GPU is out
  of compute, and everything runs at about 100% of peak.
- On the plot, T = 1 to 16 sit on the bandwidth slope (blue) and T = 32 and up
  sit on the compute ceiling (orange).

## Why this matters for inference

| Workload | Tokens per fetch of the weights | Bound | Controls |
|---|---|---|---|
| Decode, one user | 1 | Memory | TPS |
| Decode, batched | Batch size | Moves toward compute | Total TPS |
| Prefill | Whole prompt | Compute | TTFT |
| Image/video generation | Whole image | Compute | Generation time |

- **Decode reads all the weights to produce one token**, which is why it is
  memory-bound and uses a tiny fraction of the GPU.
- **Batching** fills that idle compute: 16 users cost about the same as 1
  here. This explains day 6's batching result.
- **Quantization** reads fewer bytes, which helps memory-bound decode.
- Fix the bottleneck resource. Optimizing the other one does nothing.

### The book's attention example

For standard attention with N = 4,096 tokens and d = 128 at fp16, the book
works out about 138 MB moved for about 8.6 billion operations, an intensity
of about **62**, well below the H100's 295. Most of that traffic is attention
writing its 4096 x 4096 score matrices to memory and reading them back.
FlashAttention (day 14) avoids exactly that.

## Connection to earlier days

- Day 1: prefill sets TTFT, decode sets TPS.
- Day 4.1: the same `sync()` and warmup rules for timing a GPU.
- Day 6: batching raised total TPS while per-user TPS fell.
- Day 13: the KV cache keeps each decode step at 1 token.

## Caveats

- These are matmuls on random weights, not a full model. Real layers also
  have attention, norms and activations.
- The bandwidth test is a simple copy. Real peak bandwidth can be a little
  higher, which is why a few points land slightly above 100% of "peak".
- The M4 numbers are measured on this machine and vary between runs.
- Small T timings on MPS include some fixed launch overhead.

## Things to try

1. Change `D` to 1024 and see how the ridge and the timings move.
2. Run on CPU (edit `pick_device`) and compare the ops:byte ratio.
3. Work out by hand the intensity for T = 64 with d = 4096, then check the table.
4. Add a point for the book's attention example (intensity 62) and see which
   side of your Mac's ridge it falls on.
