# Day 14: Optimizing attention — SDPA, GQA/MQA and PagedAttention

Book chapter: **2.5 Optimizing attention**

Task: naive attention vs `torch.scaled_dot_product_attention`, plus a KV-cache
size calculator for MHA / GQA / MQA. We also added a small PagedAttention
demo, since the chapter covers it.

## What we built

Two files:

- `gqa_sdpa_bench.py`: SDPA, GQA/MQA and the KV cache calculator.
- `paged_attention.py`: a hand-built paged KV cache, and how much memory it saves.

`gqa_sdpa_bench.py` has three parts:

1. **Naive attention vs SDPA:** day 9's step-by-step attention against
   PyTorch's one-call `F.scaled_dot_product_attention`, checked for the same
   answer and timed from 256 to 4,096 tokens.
2. **MHA vs GQA vs MQA:** one decode step reading a long KV cache, with 32, 8
   or 1 K/V heads, checked for correctness and timed at 2K to 32K tokens.
3. **KV cache calculator:** cache size per token, per 8K and per 128K tokens
   for real and example model shapes, and how many conversations fit on one
   GPU.

It saves `gqa_sdpa_bench.png` (Parts 1 and 2 side by side).

Run it:

```
python3 gqa_sdpa_bench.py
```

It takes about 10 seconds. It uses the MPS GPU with fp16 if available (CUDA if
present), and CPU with fp32 otherwise.

`paged_attention.py` has two parts:

1. **A paged KV cache by hand:** one shared pool of 16-token blocks, a block
   table per conversation, and blocks freed and reused. Three conversations
   grow at the same time, one finishes, and a new one takes its blocks. Each
   is checked against a plain `torch.cat` cache like day 13's.
2. **The memory payoff:** 2,000 conversations of varying length at an 8B
   model's size, comparing "reserve the maximum up front" with paged.

```
python3 paged_attention.py
```

It runs in about a second on CPU. It imports the calculator from
`gqa_sdpa_bench.py`, so keep both files in the same folder.

## What we learned

### Part 1: why naive attention is slow

Day 9's attention writes its scratch work to memory:

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(d)   # (B, H, T, T): written to memory
weights = scores.masked_fill(mask, -inf).softmax(-1)  # another (B, H, T, T)
out = weights @ v
```

At T = 4,096 in fp16, one head's `(T, T)` grid is 4096 x 4096 x 2 bytes =
33.6 MB. With 12 heads, each of those tensors (`scores` and `weights`) is about
**400 MB** for one sequence, written out and read straight back. That is the low arithmetic
intensity from the book's attention example (day 12): most of the time goes to
moving scratch data, not math.

**SDPA does the same math in one call:**

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

PyTorch picks a **fused kernel** for the device: it does the multiply, mask,
softmax and second multiply together, and on GPUs that support it
(FlashAttention on NVIDIA) it processes the grid in small tiles that stay in
fast on-chip memory, so the full `(T, T)` grid is never written out. Same
answer, much less memory traffic.

### Part 2: sharing K and V between heads

In day 9's attention (**MHA, multi-head attention**) every head has its own Q,
K and V. The KV cache stores K and V for every head, and each decode step reads
all of it (day 13). That cache is often what limits memory and decode speed.

| Variant | K/V heads (with 32 query heads) | Idea |
|---|---|---|
| **MHA** | 32 | Every query head has its own K and V |
| **GQA** (grouped-query) | 8 | Groups of 4 query heads share one K and V |
| **MQA** (multi-query) | 1 | All 32 query heads share one K and V |

Query heads stay separate, so each head can still look for different things.
Only K and V are shared, which shrinks the cache by `query heads / K-V heads`:
4x for GQA-8, 32x for MQA. Models are **trained** this way (for example,
Llama 3 uses GQA with 8 K/V heads). You cannot switch a trained MHA model to
GQA just by changing code.

**Doing it without copying the cache:**

```python
q = q.view(B, Hkv, group, d)                     # line up each group with its shared K/V head
scores = q @ k.transpose(-2, -1) / math.sqrt(d)  # (B, Hkv, group, S)
out = scores.softmax(dim=-1) @ v
```

The simple way is to copy each K/V head to every query head in its group
(`repeat_interleave`), but that rebuilds the full-size cache and throws away
the saving. Reshaping the queries into groups lets each group read its one
shared K/V head directly.

### Part 3: the KV cache formula

```
KV cache bytes = 2 (K and V) x layers x K/V heads x head_dim x bytes per number x tokens
```

It uses **K/V heads**, not query heads. That is the only thing GQA and MQA
change.

### PagedAttention: storing the KV cache without waste

**The problem.** Every conversation's KV cache grows one token at a time
(day 13), and nobody knows in advance how long a reply will be. If each cache
needs **one unbroken block** of memory, the server must reserve room for the
longest possible conversation up front. Most conversations are much shorter, so
most of that reservation is never used, and fewer conversations fit.

**The idea.** Cut the KV cache memory into small fixed-size **blocks** (pages),
for example 16 tokens each:

- All blocks live in one shared **pool**, allocated once.
- A conversation takes a free block only when its last block is full, from
  wherever a free block happens to be.
- A **block table** records which blocks belong to which conversation, and in
  what order, so they can be read back as one cache.
- When a conversation ends, its blocks go back to the pool for others.

Analogy: a parking lot. Cars park in any free space and the ticket says where
each one is, instead of reserving a whole row for a family that might arrive.

```python
if n % self.block_size == 0:              # last block full: take any free block
    table.append(self.free_blocks.pop(0))
block, offset = table[n // self.block_size], n % self.block_size
self.k_pool[block, :, offset] = k         # write the token's K into its slot
```

```python
blocks = pool[table]                      # the lookup: fetch this conversation's blocks, in order
```

It is **lossless**: the model and the attention math are unchanged, only where
the K and V numbers are stored. It is a serving-engine technique (the core of
vLLM), unlike GQA, which is a model design choice.

**GQA and paging work together:** GQA makes each conversation's cache smaller,
paging stores those caches without waste.

## Results (Apple M4 GPU, fp16)

### Part 1: naive vs SDPA (12 heads x 64, causal prefill)

| T | Naive | SDPA | Speedup | Naive's scores tensor, all 12 heads (per head) |
|---|---|---|---|---|
| 256 | 0.61 ms | 0.33 ms | 1.8x | 1.6 MB (131 KB) |
| 1,024 | 5.85 ms | 2.58 ms | 2.3x | 25 MB (2.1 MB) |
| 4,096 | 70.3 ms | 37.3 ms | 1.9x | 403 MB (33.6 MB) |

The weights tensor after softmax is the same size again, so naive attention
writes twice that much.

SDPA matches naive attention (to 1e-5 in fp32 on CPU, and on the MPS GPU in
fp16) and is about **2x faster**. On NVIDIA GPUs with FlashAttention the gap is
usually larger, and the memory saving is what lets very long contexts fit at
all.

### Part 2: one decode step reading the KV cache (32 query heads x 128, one layer)

| Tokens in cache | MHA | GQA-8 | MQA | Cache read (MHA / GQA-8 / MQA) |
|---|---|---|---|---|
| 2,048 | 0.48 ms | 0.18 ms | 0.13 ms | 34 / 8 / 1 MB |
| 8,192 | 1.60 ms | 0.62 ms | 0.24 ms | 134 / 34 / 4 MB |
| 32,768 | 6.44 ms | 2.21 ms | 0.68 ms | 537 / 134 / 17 MB |

- Our grouped version matches both the copy-everything version and PyTorch's
  `SDPA(enable_gqa=True)`.
- **Less cache to read means a faster decode step.** At 32K tokens GQA-8 is
  about 3x faster than MHA and MQA about 9x.
- MHA at 32K reads 537 MB in about 6 ms, about **85-90 GB/s**, which is day 12's
  measured memory bandwidth. Decode attention is memory-bound: its time is set
  by how many bytes of cache it reads.
- The speedups are smaller than the 4x and 32x cache reductions because small
  steps also pay fixed launch overhead and the query math.

### Part 3: KV cache sizes (fp16)

| Model shape | K/V heads | Per token | 8K tokens | 128K tokens |
|---|---|---|---|---|
| distilgpt2 (real config, MHA) | 12 | 18.4 KB | 151 MB | 2.4 GB |
| 8B shape as MHA | 32 | 524 KB | 4.3 GB | 68.7 GB |
| 8B shape as GQA-8 (Llama 3 8B) | 8 | 131 KB | 1.1 GB | 17.2 GB |
| 8B shape as MQA | 1 | 16 KB | 134 MB | 2.1 GB |
| 70B shape as MHA | 64 | 2.6 MB | 21.5 GB | 344 GB |
| 70B shape as GQA-8 (Llama 3 70B) | 8 | 328 KB | 2.7 GB | 42.9 GB |

The "as MHA" and "as MQA" rows are the same shapes with a different number of
K/V heads, to show the effect. The distilgpt2 row at fp32 is 36,864 bytes per
token, which matches day 13's measured cache.

**How many 8K-token conversations fit next to an 8B model (16 GB of fp16
weights) on an 80 GB GPU (64 GB left)?**

| Variant | Cache per conversation | Conversations at once |
|---|---|---|
| MHA | 4.3 GB | 14 |
| GQA-8 | 1.1 GB | 59 |
| MQA | 134 MB | 476 |

More conversations at once means bigger decode batches, which (day 12) means
much better use of the GPU and lower cost per token.

### PagedAttention (`paged_attention.py`, CPU)

**Part 1: a pool of 12 blocks shared by every conversation.** A (40 tokens),
B (20) and C (33) generate at the same time, so they take blocks in turn:

```
A:  40 tokens -> blocks [0, 3, 6]
B:  20 tokens -> blocks [1, 4]
C:  33 tokens -> blocks [2, 5, 7]
free blocks: [8, 9, 10, 11]
```

B finishes and returns blocks 1 and 4. D then starts and reuses them first:

```
D:  45 tokens -> blocks [1, 4, 8]
```

D's blocks are scattered across the pool, yet its K, V and attention output
match a plain `torch.cat` cache exactly. So does every other conversation's.

**Part 2: the memory payoff.** 2,000 conversations from 32 to 8,192 tokens
long (average about 1,500), for an 8B GQA-8 model (131 KB per token at fp16),
with 64 GB of KV cache memory:

| Approach | Memory wasted | Conversations that fit |
|---|---|---|
| One contiguous block of 8,192 tokens each | 81.5% | 59 |
| Paged, 16-token blocks | 0.5% | **316** |

The contiguous row is the same 59 as Part 3's GQA-8 row above: both assume
every conversation takes a full 8K. Paging only wastes the unused part of each
conversation's last block (at most 15 tokens), so **over 5x more conversations
fit** in the same memory.

## Why this matters for inference

| Technique | Fixes | Helps |
|---|---|---|
| Fused attention (SDPA, FlashAttention) | Writing the `(T, T)` grid to memory | Prefill speed (TTFT), long contexts fitting |
| GQA / MQA | A large KV cache | Decode speed (TPS), users per GPU, long contexts |
| PagedAttention | Memory wasted on reserved, unused KV cache | Users per GPU (bigger batches), lower cost |

- **SDPA is free to use:** same math, one line, and inference engines always
  use a fused kernel.
- **GQA is a model choice made at training time.** Most modern open models use
  it, because MQA can cost some quality and GQA keeps most of the saving.
- **PagedAttention is a serving choice,** lossless, and works with any model.
- All three attack **memory**, because attention is memory-bound in the ways
  that matter (day 12): SDPA cuts memory traffic, GQA shrinks the cache, paging
  stops wasting it.

## Connection to earlier days

- Day 8 and 9: naive attention is our own day 9 code.
- Day 12: the book's attention example had intensity 62 because of the `(T, T)`
  grids; decode is memory-bound.
- Day 13: the KV cache and its 36 KB per token; GQA shrinks exactly that. Day
  13's cache grew with `torch.cat`, copying the whole cache into a new block
  every step: the contiguous-memory problem PagedAttention solves.
- Day 5: the KV cache is part of the memory overhead; here it decides how many
  users fit.

## Caveats

- On MPS, PyTorch's fused kernel is not FlashAttention, so the SDPA gain here
  (about 2x) is smaller than on NVIDIA GPUs.
- Part 2 times one attention layer, not a whole model. A real decode step also
  reads all the weights (day 12).
- Random tensors, not trained weights: this measures speed and correctness,
  not quality. MQA and GQA trade some quality for the saving, which only a
  trained model and evals (chapter 1.3) can show.
- The calculator ignores weights' overhead and fragmentation; real serving fits
  fewer conversations.
- Timings vary between runs, especially the sub-millisecond ones.
- Our paged cache **gathers** a conversation's blocks into one tensor before
  attention, which copies them. Real PagedAttention kernels read the blocks
  directly from the pool, so they never build that copy.
- The paged demo uses one layer and one batch of decode steps. A real engine
  keeps a block table per conversation and pages for every layer.
- Part 2's conversation lengths are a made-up distribution (mostly short, a few
  long), not real traffic. The exact numbers depend on that mix; the 5x gap
  shows the size of the effect, not a measured result.

## Things to try

1. Change `KV_VARIANTS` to add GQA-4 (`("GQA-4", 4)`) and see where it lands.
2. Use `repeat_kv` in Part 2's timing instead of `gqa_decode_step` and see the
   GQA/MQA speed advantage shrink, because the copy rebuilds the full cache.
3. Work out the KV cache for the 70B GQA-8 shape at 32K tokens by hand, then
   check with `kv_cache_bytes`.
4. Push Part 1 to T = 8,192 and watch the naive version's memory: two tensors
   (scores and weights) of about 1.6 GB each across all 12 heads, 134 MB per
   head.
5. In `paged_attention.py`, make the pool only 8 blocks and see the
   `MemoryError` when conversations run out of blocks.
6. Change `BLOCK_SIZE` to 256 and rerun Part 2. Bigger blocks waste more of
   each last block; how much does that cost?
7. Print `cache.k_pool[1]` before and after B is freed and D starts, to see
   D's numbers overwrite B's old ones in the same block.
