# Day 16: GPU architecture — CPU vs GPU and the cost of copying

Book chapter: **3.1 GPU architecture** (3.1.1 Compute, 3.1.2 Memory and Caches)

Task: CPU vs GPU matmul and the cost of host-to-device transfer, run on Colab.

## What we built

`cpu_vs_gpu.py` has four parts:

1. **What this GPU is made of:** name, number of Streaming Multiprocessors,
   VRAM, L2 cache, and the biggest fp16 model that fits with 50% headroom.
2. **Matmul speed:** the same matmul on the CPU (fp32), the GPU (fp32) and the
   GPU (fp16), in TFLOPS.
3. **Moving data:** copying from CPU RAM to the GPU (normal and pinned memory)
   vs copying inside the GPU's own VRAM, in GB/s.
4. **The round trip:** one matmul timed on the CPU, on the GPU with the data
   already there, and on the GPU including the copies in and out.

## How to run it on Colab

1. colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. Check the GPU: run a cell with `!nvidia-smi`.
4. Upload `cpu_vs_gpu.py` with the folder icon on the left (or paste it into a
   cell that starts with `%%writefile cpu_vs_gpu.py`).
5. Run `!python cpu_vs_gpu.py`. It takes about a minute.

It also runs on a Mac (MPS), but a Mac has unified memory, so it cannot show
the real CPU-to-GPU copy cost. That is why this day uses Colab.

## What we learned

### A GPU is many small workers in teams

```
GPU
 ├── SM 1 ─┬─ Tensor Cores   matrix math (matmul): the part that matters for inference
 │         ├─ CUDA Cores     one number at a time
 │         ├─ SFUs           sin, cos, exp (softmax needs these)
 │         └─ L1 cache       small, very fast, private to this SM
 ├── SM 2 ... (T4: 40 SMs, H100: 132)
 ├── L2 cache                shared by all SMs (T4: 4 MB, H100: 50 MB)
 └── VRAM (HBM)              big but slower (T4: 16 GB, H100: 80 GB)
```

- A Tensor Core does **MMA, Matrix Multiply and Accumulate**: `D = A x B + C`.
  This is day 3's `x @ W + b`, done in hardware.
- A GPU runs **tens of thousands of threads** at once, each doing simple work.
  Matmul is millions of independent multiply-adds, so it fits perfectly.

### Reading FLOPS on a spec sheet

- Use **dense** FLOPS. "Sparse" numbers assume half the values are 0, which is
  not true for normal inference, and are usually 2x higher.
- **Halving the precision roughly doubles the FLOPS** (fp16 to fp8 is about
  2x). Always compare GPUs at the same precision.
- More FLOPS helps **compute-bound** work: prefill, image and video generation.

### Two kinds of memory

| | Where | Size | Speed |
|---|---|---|---|
| **DRAM** (VRAM, HBM) | off the chip | gigabytes | slower |
| **SRAM** (L1, L2 caches) | on the chip | KB to MB | much faster |

- **VRAM size** decides what fits: the weights plus at least 50% extra for
  the KV cache. Not enough means an OOM (out of memory) error.
- **VRAM bandwidth** decides decode speed, because decode reads every weight
  for every token (days 12 and 13). That is why the H200 decodes faster than
  the H100: same compute, more bandwidth.
- Day 14's FlashAttention is fast because it keeps the scores in **SRAM** and
  never writes them to VRAM.

### Our Colab results (Tesla T4)

```
GPU: Tesla T4, 40 SMs, 15.6 GB VRAM, 4.2 MB L2
Biggest fp16 model with 50% headroom: about 5.2B parameters
```

**Matmul speed (TFLOPS):**

| n | CPU fp32 | GPU fp32 | GPU fp16 | fp16 vs CPU |
|---|---|---|---|---|
| 256 | 0.11 | 0.70 | 1.60 | 14x |
| 1024 | 0.12 | 2.59 | 14.54 | 125x |
| 2048 | 0.09 | 2.85 | 17.94 | 204x |
| 4096 | - | 4.25 | 24.44 | - |
| 8192 | - | 4.05 | 23.52 | - |

- **The GPU needs big matrices to shine.** At n=256 there is too little work to
  keep 40 SMs busy (1.6 TFLOPS); at n=4096 it reaches 24 TFLOPS.
- **fp16 is about 6x faster than fp32** on the T4, because only fp16 runs on its
  Tensor Cores. That is the "lower precision = more FLOPS" rule, made extra
  large here.
- We reached about **37% of the T4's 65 TFLOPS spec**. Spec sheets are a
  ceiling, and the T4 is a 70-watt card that slows itself down when hot.
- Colab's CPU is tiny (2 cores): about 0.1 TFLOPS, so the GPU was **over 200x
  faster**.

**Moving data (GB/s):**

| Size | CPU to GPU | CPU to GPU, pinned | Inside the GPU |
|---|---|---|---|
| 1 MB | 2.8 | 11.5 | 175 |
| 128 MB | 6.9 | 12.4 | 233 |
| 512 MB | 6.8 | 12.4 | 233 |

```
CPU RAM ──PCIe, ~12 GB/s──> GPU VRAM ──~233 GB/s──> SMs
```

- Copying **into** the GPU (over the PCIe cable) is about **19x slower** than
  moving data **inside** it.
- **Pinned memory** doubles the copy speed. Normal memory can be moved around by
  the operating system, so it is first copied into a pinned buffer, then sent.
  Pinned memory skips that extra copy.
- We reached 233 of the T4's 320 GB/s spec (73%). Real code never reaches 100%.

**One matmul, with and without the copies (ms):**

| n | CPU | GPU only | GPU + copies | Time spent copying |
|---|---|---|---|---|
| 256 | 0.32 | 0.02 | 0.35 | 93% |
| 1024 | 16.5 | 0.34 | 3.20 | 89% |
| 4096 | 1523 | 33.7 | 93.8 | 64% |

- At n=256, copying the data made the GPU **slower than the CPU** overall.
- Even at n=4096, two thirds of the time was copying.
- **The lesson:** load the model onto the GPU once (`model.to("cuda")`) and keep
  it there. Only small things (token IDs in, token IDs out) should cross the
  PCIe link per request.

### The rule of thumb

| Bottleneck | Workload | Buy more |
|---|---|---|
| Compute | prefill, images, video | FLOPS |
| Memory bandwidth | decode at small and medium batch | GB/s |
| Memory size | big models, long context, many users | GB of VRAM |

## Connection to earlier days

- Day 4.1: the same matmul benchmark, now on a real NVIDIA GPU. The sync before
  timing still matters.
- Day 5: "biggest model that fits" is `VRAM / 1.5 / 2 bytes`.
- Day 12: the T4's ops:byte ratio from its spec is 65 / 0.32 = about 203; from
  our measured numbers, about 24.4 / 0.233 = about 105.
- Day 14: FlashAttention keeps its work in SRAM, the fast on-chip memory.

## Caveats

- Colab gives you whatever T4 is free. Numbers vary between sessions.
- The CPU column only goes up to n=2048, because Colab's CPU is so slow.
- "TFLOPS" counts `2 x n³` operations per matmul, the standard way to count.

## Things to try

1. Add `torch.bfloat16` to the GPU columns. The T4 (Turing) has no fast bf16,
   so expect it to be slow. Newer GPUs (A100 and later) are fast at bf16.
2. Work out decode tokens/s on the T4 for the day 19 model (3.09 GB at fp16)
   with day 17's rule: 233 GB/s / 3.09 GB = about 75 tokens/s.
3. Switch Colab to an A100 or L4 (if you have access) and compare all three
   tables.
