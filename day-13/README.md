# Day 13: Adding a KV cache

Book chapter: **2.4 Calculating bottlenecks + the KV cache**

Task: add a KV cache to the day 10 GPT and measure the decode speedup versus
no cache.

## What we built

`kv_cache.py` adds a cached way of running **day 10's model** (with
`distilgpt2`'s trained weights). The model and its weights are unchanged; only
how we run it changes.

```
PREFILL: run the whole prompt once, store K and V for every token, in every layer
DECODE:  run ONLY the new token -> append its K and V to the store
         -> its Q compares against all stored K's, blends all stored V's
```

Run it:

```
python3 kv_cache.py
```

It takes about a minute and a half, mostly for the 512-token no-cache run. It
checks the cache gives identical output, reports the cache size, compares
speed at 16 to 512 new tokens, and saves `kv_cache.png` (time per step).

This file imports day 10, which imports day 9, so keep the folder names
`day-9` and `day-10`.

## What we learned

### The problem

Day 10's loop reruns **the whole sentence** every step:

```
step 1: run "The cat sat"          -> " on"
step 2: run "The cat sat on"       -> " the"
step 3: run "The cat sat on the"   -> " mat"
```

### What the cache stores, and why

| Thing | Stored? | Why |
|---|---|---|
| K of every earlier token | Yes | Never changes, and every new token compares its Q against it |
| V of every earlier token | Yes | Never changes, and every new token blends it |
| Q | No | Used once, at its own token's step |
| Scores, weights, output | No | Depend on the new Q, so only useful right now |

K and V never change because of the causal mask: an earlier token cannot see
later ones, so its K and V are fixed once computed.

### The four lines that matter

```python
k = torch.cat([past[0], k], dim=2)                      # append the new K after the stored ones
v = torch.cat([past[1], v], dim=2)                      # same for V
pos = torch.arange(T_past, T_past + T)                  # positions keep counting, not restart at 0
logits, cache = forward_with_cache(model, next_id, cache)  # feed ONE token, not the whole idx
```

### Shapes during one decode step (6 tokens already stored)

```
new token's q, k, v    (1, 12, 1, 64)     batch, heads, 1 token, numbers per head
after cat with store   k, v: (1, 12, 7, 64)
scores = q @ k^T       (1, 12, 1, 7)      ONE row per head, not a 7 x 7 grid
output                 (1, 12, 1, 64)
```

Day 8's T x T grid becomes a single 1 x T row, which is why decode with a cache
is cheap.

### The causal mask with a cache

```python
mask = torch.triu(torch.ones(T, T_total), diagonal=T_past + 1)
```

The new token at position `T_past + i` may see stored keys 0 to `T_past + i`.

- **Prefill** (`T_past = 0`): the usual causal triangle.
- **One decode step** (`T = 1`): nothing is blocked. The newest token has no
  future.

## Results (distilgpt2, CPU, fp32)

### Same output

- The last token's scores match whether we run all 6 prompt tokens at once or
  5 and then 1 using the store.
- 30, 100 and every speed-test length: **identical tokens** with and without
  the cache.

### Cache size

```
after 106 tokens: 6 layers x (K, V), each (1, 12, 106, 64) = 3.91 MB
2 (K and V) x 6 layers x 768 numbers x 4 bytes = 36,864 bytes = 36 KB per token
```

That is 36 KB per token at fp32, or 18 KB at fp16. The measured size equals the
formula exactly.

### Speed

| New tokens | No cache | With cache | Speedup |
|---|---|---|---|
| 16 | 85 tok/s | 165 tok/s | 1.9x |
| 64 | 74 tok/s | 170 tok/s | 2.3x |
| 256 | 36 tok/s | 168 tok/s | 4.7x |
| 512 | 20 tok/s | 168 tok/s | **8.6x** |

- **With the cache, speed stays flat** however long the text gets.
- **Without it, speed collapses**, because each step reruns a longer sentence.
- So the speedup **grows with length**.

Hugging Face's `generate()` (which has its own cache) ran at 141 tok/s for 512
tokens. Ours is slightly faster because its loop does extra bookkeeping each
step. On day 10, without a cache, we were slower (87 vs about 150).

### Time per step

| | First 10 steps | Last 10 steps |
|---|---|---|
| No cache | 11.7 ms | **89.1 ms** (growing) |
| KV cache | 5.3 ms | **6.2 ms** (flat) |

`kv_cache.png` plots this: the no-cache line climbs, the cache line stays flat.
That is quadratic total work versus linear.

## Why this matters for inference

| | No cache | KV cache |
|---|---|---|
| Work per decode step | Whole sentence | 1 token |
| Attention per step | T x T grid | 1 x T row |
| Total work for N tokens | Quadratic | Linear |
| Extra memory | None | 36 KB per token here |
| Output | Same | Same |

**The cache trades memory for speed.** For `distilgpt2` it is tiny, but a large
model (say 80 layers, 8,192 wide, fp16) needs about 2.6 MB per token. A
100,000-token conversation would need hundreds of GB, per user. That is why:

- The KV cache often decides how many users fit on a GPU (day 5's overhead).
- Day 14's GQA and MQA make heads share K and V to shrink it.
- The book covers storing and reusing it in section 5.3.

It also ties to day 12: each cached decode step processes one token, so it
reads all the weights and the whole cache for very little math. That is why
decode stays memory-bound.

## Connection to earlier days

- Day 8: the causal mask is why K and V of earlier tokens never change.
- Day 9: the cache stores each head's K and V, shape `(batch, heads, tokens, d_head)`.
- Day 10: the same model and weights; only the loop changes.
- Day 12: decode is memory-bound; the cache keeps each step to one token.

## Caveats

- Batch of one, greedy decoding only.
- Generation must stay inside the 1,024-token context window (the code
  asserts this). Real engines handle longer contexts differently.
- The cache grows with `torch.cat` every step, which copies it. Real engines
  preallocate the cache (and use paged memory) to avoid that.
- Timings are on CPU and vary between runs.

## Things to try

1. Print `cache[0][0].shape` after each of the first few decode steps and watch
   the token dimension grow.
2. Change `pos` to always start at 0 and see the output diverge from day 10's.
3. Remove the mask line and check whether decode still matches (it should,
   since `T = 1`), then check whether the 5-then-1 prefill check still passes.
4. Work out the cache size for `gpt2` (12 layers, 768 wide) at 1,024 tokens in
   fp16, then check it by running `load_gpt2_weights("gpt2")`.
