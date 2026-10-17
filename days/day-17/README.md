# Day 17: GPU generations — estimating decode speed from a spec sheet

Book chapter: **3.2 GPU Architecture Generations**

Task: GPU spec table + estimator: decode TPS ≈ memory bandwidth / model size.

## What we built

`decode_tps_estimator.py` prints four tables:

1. **A spec sheet** for 12 GPUs, from the old T4 to the B300, plus two desktop
   options and your Mac.
2. **One model on every GPU:** how many GPUs it needs, decode tokens/s for one
   user, and the time to first token.
3. **Batching on one GPU** with 2,048 tokens of context per user.
4. **The same with short context** (128 tokens), to show when compute becomes
   the limit.

Run it:

```
python3 decode_tps_estimator.py
python3 decode_tps_estimator.py --model llama-3.3-70b --dtype fp8 --gpu H200
```

Options: `--model` (qwen2.5-1.5b, llama-3.1-8b, llama-3.3-70b), `--dtype`
(fp16, fp8, q4), `--gpu`, `--prompt` (prompt tokens), `--context` (tokens per
user). It needs no libraries and runs instantly: it is pure arithmetic.

## What we learned

### Reading a GPU name

**B200 = B (the generation) + 200 (the model within it).** The letter matters
most. A bigger number usually means a bigger, more expensive GPU.

| Letter | Generation | GPUs | What it added |
|---|---|---|---|
| T | Turing (2018) | T4 | Legacy (Colab's free GPU) |
| A | Ampere (2020) | A10, A100 | Legacy, still seen |
| L | Lovelace (2022) | L4, L40 | Cheap, small models. **No NVLink** |
| H | Hopper (2022) | H100, H200 | **FP8**, FlashAttention 3 |
| B | Blackwell (2024) | B200, B300 | **FP4**, FlashAttention 4. The new gold standard |
| R | Rubin (2026) | - | HBM4 memory, a separate **CPX** chip for prefill |
| F | Feynman (2028) | - | Little known yet |

Most real deployments use Lovelace, Hopper or Blackwell.

### The one rule

```
decode tokens/s for one user ≈ memory bandwidth / model size in bytes
```

Each new token reads every weight once (day 12). An 8B model at fp16 is
16 GB, and an H100 reads 3,350 GB/s, so at best 3,350 / 16 = about 209 tokens/s.

### Results: Llama 3.1 8B at fp16 (16.1 GB), one user

| GPU | GPUs needed | Ideal tok/s | Realistic (70%) | TTFT, 1,000-token prompt |
|---|---|---|---|---|
| A100 | 1 | 127 | 89 | 52 ms |
| L40 | 1 | 54 | 37 | 89 ms |
| H100 | 1 | 209 | 146 | 16 ms |
| **H200** | 1 | **299** | 209 | **16 ms** |
| B200 | 1 | 498 | 349 | 6 ms |
| RTX 5090 | 1 | 111 | 78 | 77 ms |
| M3 Ultra | 1 | 51 | 36 | 574 ms |
| T4, A10, L4, your M4 | - | does not fit | | |

- **H100 vs H200 is the cleanest lesson:** the same compute, so the same TTFT
  (prefill is compute-bound), but 43% more bandwidth, so 43% faster decode.
- TTFT is `2 x parameters x prompt tokens / FLOPS`: prefill is all compute.
- **"Fits"** means weights plus 50% headroom for the KV cache (chapter 3.1).
- Halving the precision (fp8) halves the bytes, so it roughly doubles decode
  speed, and on Hopper and Blackwell it also doubles prefill FLOPS.

### Results: batching on an H100

With **2,048 tokens of context**, the KV cache is 268 MB per user, so 238 users
fit:

| Batch | Per-user tok/s | Total tok/s | Limit |
|---|---|---|---|
| 1 | 205 | 205 | memory |
| 16 | 165 | 2,633 | memory |
| 64 | 101 | 6,450 | memory |
| 238 | 42 | 9,973 | memory |

With **128 tokens of context**, many more users fit:

| Batch | Per-user tok/s | Total tok/s | Limit |
|---|---|---|---|
| 1 | 208 | 208 | memory |
| 256 | 165 | 42,132 | memory |
| 1,024 | 60 | 61,582 | **compute** |

- Batching shares one read of the weights across many users (day 12), so total
  throughput climbs a lot while each user slows a little (day 6).
- With long context, **memory runs out before compute does**: every user's KV
  cache must also be read each step. That is why GQA and PagedAttention
  (day 14) matter so much.
- Only with short context and a huge batch does the GPU finally become
  compute-bound.

### Other things from the chapter

- **Grace CPUs** (GH200, GB200) link the CPU and GPU at 900 GB/s, many times
  faster than PCIe (day 16 measured about 12 GB/s on Colab). Useful for keeping
  KV caches or LoRA weights in CPU memory and fetching them fast.
- **Wait for real benchmarks** before trusting a new GPU. Software support
  takes about a year to catch up.

## Connection to earlier days

- Day 5: weights = parameters x bytes.
- Day 12: decode is memory-bound; the ops:byte column is day 12's ratio (H100:
  295).
- Day 13 and 14: KV cache per token = `2 x layers x KV heads x head size x
  bytes`. Llama 8B uses GQA with 8 KV heads, which keeps this small.
- Day 16: bandwidth sets decode speed, FLOPS set prefill speed.
- Day 19 checks this rule against real measurements on your Mac.

## Caveats

- Spec numbers are rounded, dense, and best case. Real software reaches about
  60-80% of the bandwidth, so the "realistic" column uses 70%.
- Splitting across GPUs is treated as perfect. Real multi-GPU setups lose some
  speed to communication (chapter 5.4).
- q4 is treated as weight-only: half a byte per weight, with math done in fp16.
- The KV cache is always fp16 here, and the model has no other overhead.

## Things to try

1. `--model llama-3.3-70b`: how many H100s does it need at fp16? At fp8?
2. `--dtype q4 --gpu "RTX 5090"`: what does a 4-bit 8B model do on a gaming GPU?
3. `--context 32768`: how many users still fit on an H100?
4. Add a new GPU to `GPUS` from a spec sheet you find online.
