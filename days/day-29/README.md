# Day 29: 4-bit group quantization — perplexity and speed

Book chapter: **5.1 Quantization (continued)**

Task: 4-bit group quantization, then compare perplexity and speed for fp16,
int8 and int4.

## What we built

`int4_group_quant.py` has three parts:

1. **Toy example:** eight weights with one outlier, quantized to 4 bits as one
   group and as two groups. Then **packing**: two 4-bit values per byte, and
   unpacking to check.
2. **Quality:** distilgpt2's **perplexity** with all 24 linear layers in
   each format: fp32, fp16, int8 per-channel (day 28), int4 per-channel, and
   int4 in groups of 128, 64 and 32. It also counts the **real bits per
   weight**, including the stored scales.
3. **Speed:** one big linear layer (4096² and 8192²) on the Mac GPU, timed in
   fp16 and with PyTorch's built-in **int8** and **int4** kernels, at batch 1
   (decode) and batch 16. Each kernel's output is checked against doing the
   same math by hand.

```
python3 days/day-29/int4_group_quant.py
```

About 15 s. Peak memory is under 1.5 GB: the 8192² layer is 256 MB while being
built in fp32, then 128 MB in fp16, 64 MB in int8 and 34 MB in int4.

## What we learned

### 4 bits = only 16 levels

int8 has 255 levels; 4 bits has **16** (0 to 15). That's 16× fewer, so one
scale per channel is no longer enough. The answer is **groups**: one scale for
every 32, 64 or 128 weights along the input. This is the **block level** from
the book (MXFP4 uses blocks of 32, NVFP4 uses 16).

We use **asymmetric** quantization: each group stores its **minimum** and its
**scale**, so 0 maps to the group's minimum and 15 to its maximum. No level is
wasted when a group's values aren't centred on zero.

```python
scale = (max - min) / 15
q = round((w - min) / scale)
w_back = q * scale + min
```

### The toy example

```
Eight weights: [0.02, -0.05, 0.01, 0.04, 1.6, -0.03, 0.06, -0.01]
  groups of 8: 4-bit values [1, 0, 1, 1, 15, 0, 1, 0]
               back = [0.06, -0.05, 0.06, 0.06, 1.6, -0.05, 0.06, -0.05]
  groups of 4: 4-bit values [12, 0, 10, 15, 15, 0, 1, 0]
               back = [0.022, -0.05, 0.01, 0.04, 1.6, -0.03, 0.079, -0.03]
  packed: 8 values in 4 bytes [192, 175, 240, 16], unpacks correctly: True
```

- **One group of 8:** the outlier 1.6 stretches the ruler. Each step is 0.11,
  so all seven small weights collapse to just two values (0.06 and −0.05).
- **Two groups of 4:** the first group has no outlier and comes back almost
  exactly. Only the group holding the outlier stays rough. **The outlier only
  hurts its own group.**
- **Packing:** the first value goes in the high 4 bits of a byte and the second
  in the low 4. `[12, 0]` → `12 × 16 + 0 = 192`.

### Scales cost bits

Every group stores a scale and a minimum, 16 bits each in fp16. Per weight that
adds:

| Group size | Extra bits per weight | Real bits per weight |
|---|---|---|
| whole row (768 or 3,072) | ~0.03 | 4.03 |
| 128 | 32 / 128 = 0.25 | 4.25 |
| 64 | 0.5 | 4.5 |
| 32 | 1.0 | **5.0** |

This is the book's tradeoff: finer granularity gives better quality but more
memory for scales, and more work to apply them.

## Results (Apple M4)

### Quality: perplexity on 267 tokens of text

```
          weights bits/weight linear MB perplexity   change
-----------------------------------------------------------
             fp32       32.00     162.0      44.84    +0.0%
             fp16       16.00      81.0      44.84    +0.0%
             int8        8.02      40.6      44.63    -0.5%
 int4 per-channel        4.03      20.4   17489.30 +38902.3%
        int4 g128        4.25      21.5      48.87    +9.0%
         int4 g64        4.50      22.8      49.10    +9.5%
         int4 g32        5.00      25.3      44.95    +0.3%
```

- **fp16 and int8 are lossless here.** int8's −0.5% is noise: rounding happened
  to help slightly on this text. The book calls this "indistinguishable from
  noise".
- **int4 with one scale per row destroys the model:** perplexity goes from 45 to
  17,489, basically random guessing. 16 levels can't cover a whole row with
  outliers.
- **Groups rescue it.** Groups of 128 and 64 give +9%, a visible quality loss
  for a model this small. **Groups of 32** give +0.3%, almost lossless.
- Small models are **more sensitive** than big ones (book: "the more parameters,
  the less sensitive"). An 8B model in g128 int4 usually loses much less.

### Speed: one linear layer on the Mac GPU

```
       size batch weights     MB      ms   GB/s speedup  max err
----------------------------------------------------------------
  4096x4096     1    fp16     32   0.357     94   1.00x   0.0000
  4096x4096     1    int8     16   0.166    101   2.15x   0.0039
  4096x4096     1    int4      8   0.094     94   3.77x   0.0029
  4096x4096    16    fp16     32   0.431     78   1.00x   0.0000
  4096x4096    16    int8     16   0.457     37   0.94x   0.0039
  4096x4096    16    int4      8   1.170      8   0.37x   0.0039
  8192x8192     1    fp16    128   1.354     99   1.00x   0.0000
  8192x8192     1    int8     64   0.667    101   2.03x   0.0039
  8192x8192     1    int4     34   0.394     90   3.44x   0.0039
  8192x8192    16    fp16    128   1.711     78   1.00x   0.0000
  8192x8192    16    int8     64   1.689     40   1.01x   0.0039
  8192x8192    16    int4     34   4.615      8   0.37x   0.0078
```

- **Batch 1 (decode): time follows bytes.** All three formats read at about
  **90–100 GB/s**, close to the M4's memory bandwidth. int8 reads half the bytes
  and is **2×** faster; int4 reads a quarter and is **3.4–3.8×** faster. This is
  day 12's "decode is memory-bound", shown with real kernels.
- That's better than the book's "30–50% per level". These kernels dequantize
  almost for free at batch 1, and a single layer has none of a full model's
  other costs (attention, layer norms, Python).
- **Batch 16: the gain disappears.** int8 only ties fp16, and **int4 is 2.7×
  slower**. These are **weight-only** kernels: they unpack the 4-bit weights and
  do the math in fp16, so they save bytes but **not** compute. With 16 rows the
  work is less memory-bound, and the unpacking cost dominates.
- That's why the book's prefill gain ("2× the FLOPS") needs **low-precision
  Tensor Cores** (FP8 on Hopper, FP4 on Blackwell) that compute in the small
  format directly. Weight-only int4 is a **decode / small-batch** optimization.
- **max err** is only fp16 rounding: the kernels compute exactly what our
  dequantize-then-multiply reference does.

## Connection to earlier days

- Day 12 and 17: decode TPS ≈ bandwidth / bytes. Here it is, kernel by kernel.
- Day 19: Ollama's `q4_K_M` is this same idea, 4-bit values with per-block
  scales.
- Day 28: per-channel was enough for int8. At 4 bits it fails, and groups are
  needed.

## Caveats

- **Quality uses fake quantization** (quantize, then back to fp32), like day
  28. The math is the same as a real kernel's. **Speed uses real kernels**, but
  on one layer, not the whole model: distilgpt2's layers are too small to time
  well.
- Perplexity on 267 tokens is a small test. Real checks use thousands of tokens
  (e.g. WikiText) plus benchmarks and custom evals (book 5.1.3).
- Plain round-to-nearest. Smarter methods (GPTQ, AWQ) choose roundings that
  minimize the layer's **output** error and do much better at g128.
- These are PyTorch's built-in Mac kernels. CUDA kernels such as Marlin keep
  int4 fast at larger batches.

## Things to try

1. Quantize only the `mlp.c_fc` layers to int4 g128 and keep the rest in int8.
   How much of the 9% comes back?
2. Add group size 16 (NVFP4's block size) and compare bits per weight and
   perplexity.
3. Find the batch size where int4 stops being faster than fp16 (try 2, 4, 8).
