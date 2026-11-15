# Day 32: a prefix cache on a radix tree of token blocks

Book chapter: **5.3 Caching**

Task: keep the KV cache between requests, not just inside one request, and find
the longest prefix a new prompt can reuse. Index it on a radix tree of token
blocks, and report hit rates.

## What we built

`radix_prefix_cache.py` has three parts.

- **`BlockPool`**: a fixed pool of KV blocks. One block holds 16 tokens of K and
  V (vLLM's default page size). Blocks are allocated and freed by id, like
  day 25.
- **`RadixCache`**: the index. A node owns a **run of whole blocks** and
  remembers their tokens, so a prefix match is one walk down from the root:

  ```
  root
    "Weather in" | "Paris ?"     <- first request
    "Weather in" | "Rome ?"      <- second request shares the first node
  ```

  ```python
  lookup(tokens)      -> node to hold, tokens already cached
  insert(keys)        -> store the blocks that were just prefilled
  release(node)       -> the request is done with them
  ```

- **`FlatBlockCache`**: the baseline, day 26's design. One `OrderedDict` of
  chained block hashes, same pool size, same least-recently-used eviction, same
  15% headroom. The only difference is that there is no tree.

The script runs five small walk-throughs and then serves the same chat traffic
through all of them plus two more variants.

```
python3 days/day32-prefix-cache/radix_prefix_cache.py
```

It takes a tenth of a second and needs nothing but the standard library.

## What we learned

### A prefix cache is a trie of prefixes, so a tree is the natural index

Every request has one path from the root. Two requests share a node exactly
when they share the tokens that node stands for, so the tree *is* the set of
cached prefixes, with no separate index to keep in sync.

Inserting `a b c e` after `a b c d` splits the first node in two:

```
    2 block(s): a b | c d          <- before
    1 block(s): a b                <- after
        1 block(s): c d
        1 block(s): c e
```

`a b` is now its own node because two requests share it. Splitting **moves**
blocks between nodes; it never copies or frees KV. That is the whole trick, and
it is why the split has to update ref counts: the requests using the node keep
using the head.

### The prefix ends at the first token that differs, and that is a budget

The book's two figures, at block granularity:

| | A | B | cached |
|---|---|---|---|
| Figure 5.7 | `Weather in Paris ?` | `Weather in Rome ?` | 2 of 4 tokens |
| Figure 5.8 | `Weather in Paris ?` | `Forecast for Paris ?` | **0** of 4 tokens |

The second pair shares two tokens (`Paris ?`) and saves nothing, because the
first token already differs. Turning that into money, with a 1,600-token system
prompt and 20 requests:

| prompt order | token hit rate | prefilled |
|---|---|---|
| `[timestamp] + system + question` | **0.0%** | 33,220 |
| `system + question + [timestamp]` | **91.5%** | 2,820 |

Same content, same model, 12x less prefill. A per-request timestamp on line 1
costs everything after it. This is context engineering, not engine tuning.

### Only whole blocks are reusable

10 tokens with a 16-token block size gives 2 whole blocks. Re-sending the same
prompt reuses **8** tokens and prefills the last 2 again: a partial block is
never cached. Change one token inside the second block and 4 tokens survive
(block one), not 8.

A block is the unit of allocation, so its KV belongs to one request unless every
token in it matches. Sharing part of a block needs copy-on-write, which costs a
copy per block per request — a real trade, and the reason the partial tail is
left out.

### Eviction removes leaves, oldest first, and skips held blocks

Only leaves can go, so what remains in the tree is always a **complete path**
from the root; a block in the middle of a prefix can never be dropped while its
children stay. In a flat dict of hashes, nothing enforces that.

A leaf **held by a live request** cannot go either, because a decoding request
still needs its KV. In the walk-through `a b` is the least recently used leaf
but a request holds it, so `c d` is evicted instead. When every leaf is held the
cache refuses to free anything and the engine has to preempt a request (day 25).

Eviction can also take just the **last few blocks** of a node, keeping the head.
Without that, one node longer than the whole pool could never be evicted at all
— a 10-block prompt in a 4-block pool would deadlock.

### The tree barely beats the flat dict on hit rate, and that is the interesting part

Same traffic, same pool, same policy:

```
485 requests, 452,249 prompt tokens, 16-token blocks, cache of 800 blocks (12,800 tokens)

  cache                  tok hit req hit   prefilled  TTFT p50  TTFT p99  evicted  stranded
  -----------------------------------------------------------------------------------------
  no cache                 0.0%    0.0%     452,249     47.00     87.80        0         0
  flat LRU                70.6%   98.1%     132,841     12.35     47.20     7327        24
  flat, no refresh        59.3%   84.9%     183,993     15.60     74.30     9752       146
  radix, 800 blocks       71.4%   98.6%     129,433     12.35     43.60     7132         0
  radix, unlimited        82.2%   98.8%      80,585      9.40     29.55        0         0
```

`radix, unlimited` is the ceiling: with every block kept, 82.2% of prompt tokens
are reusable. At 800 blocks we get 71.4%, so **capacity is the binding
constraint, not the index.** The tree is worth 0.8 points over the flat dict on
this traffic, which is honest and a bit deflating.

What the tree actually buys:

- **No stranded blocks.** 24 blocks (3% of the pool) sat in the flat cache at
  the end with a missing ancestor, so their KV could never be read again. The
  tree takes this to exactly zero, by construction, not by tuning.
- **Correctness does not depend on remembering to refresh.** Turn off
  `move_to_end` in the flat cache's lookup path — a one-line mistake — and it
  drops to 59.3% with 146 stranded blocks, while still reporting 84.9% of
  requests as hits. Silently wrong is the worst kind of cache. A tree has no
  such knob.
- **Ref counts are per node**, covering every descendant at once. The flat dict
  has to pin each block separately, because it does not know what is below it.

The reason production engines use radix trees is therefore structure — sharing,
pinning, copy-on-write, evicting a suffix exactly — not a larger hit rate.

## Connection to earlier days

- Day 13: a KV cache inside one request, one growing tensor.
- Day 25: blocks and a free list, which is the pool this cache shares blocks
  from. That allocator could not share: it had no ref counts.
- Day 26: a KV-aware router with a flat dict of chained block hashes. This day
  is that cache, done properly, and the measurement of what "properly" is worth.
- Day 31: speculation speeds up decode. This speeds up prefill. They fix
  different numbers: ITL and TTFT.

## Caveats

- It indexes and accounts for blocks; it stores no KV tensors, so the memory
  numbers are counts of 16-token blocks, not bytes.
- TTFT is computed as `2 ms + 0.05 ms` per prefilled token, with no queue. The
  comparison between caches is fair, the absolute numbers are not a benchmark.
- Blocks are released as soon as the prefill is counted, so the trace measures
  cache quality only. A real engine holds them for the whole request; the
  walk-through shows the effect of that, the trace does not.
- Eviction is least-recently-used on the node, not on the block, and a node's
  blocks all share one timestamp. Real engines keep a per-block last-access.
- Hash collisions would merge two different prefixes. At this scale they do not
  happen; production uses 64-bit content hashes for the same reason.

## Things to try

1. Shrink `CACHE_BLOCKS` to 200 and 100. Where does the radix tree stop being
   able to hold one conversation's prefix at all?
2. Set `BLOCK_SIZE = 1`. The partial-block waste disappears and the tree gets
   more nodes; measure both.
3. Make the traffic adversarial for the flat cache: many short branches off one
   long shared system prompt, cache just below the working set.
4. Add copy-on-write: let two requests share a partial block, and count the
   copies.
5. Give the tree an `O(1)` LRU leaf candidate list instead of the full scan in
   `_evict`, and measure the difference as the pool grows.
