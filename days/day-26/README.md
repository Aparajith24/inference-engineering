# Day 26: NVIDIA Dynamo — a KV-aware router simulator

Book chapter: **4.4 NVIDIA Dynamo**

Task: when several copies (replicas) of a model are running, send each request
to the replica that already has the longest matching prefix in its KV cache.

## What we built

`kv_aware_router.py` simulates 4 replicas of one model. Each replica has:

- a **prefix cache** of 1,000 KV blocks (16 tokens each), least recently used
  evicted first;
- a queue, measured as "busy until".

The traffic is **400 chat conversations** of 1–6 turns, each using one of 8
system prompts (300–1,500 tokens). Every turn resends the whole chat so far, as
stateless APIs do (day 22), so the prompt grows with every turn. That makes
1,385 requests averaging 1,564 prompt tokens each.

The same requests go through five routers:

| Router | Picks |
|---|---|
| round robin | the next replica in turn |
| random | any replica |
| least loaded | the shortest queue |
| best prefix | the longest cached prefix, even if that replica is swamped |
| **kv-aware** | the lowest **estimated TTFT** = queue wait + prefill of the uncached part |

```
python3 days/day-26/kv_aware_router.py
```

It takes under a second.

## What we learned

### Block hashes find shared prefixes fast

Each full block of 16 tokens gets a hash **that includes the previous block's
hash**. Two prompts share block N's hash only if they are equal from token 0 up
to the end of block N. The router can then compare prompts by walking a list of
hashes, without looking at the tokens.

```
prompt B: same 40-token system prompt, different question -> shares 2 blocks with A
prompt C: one extra token at the front -> shares 0 blocks with A (a prefix must match from token 0)
```

The lesson for anyone building apps: keep what is shared (the system prompt,
documents) at the **start**, and put things that change (dates, user names)
after it.

### Cache hits and load pull in opposite directions

- Chasing cache hits alone sends every user of a popular system prompt to one
  replica. That replica gets swamped while the others sit idle.
- Spreading load alone sends each conversation's turns to random replicas. The
  cached prefix sits on one replica while the next turn lands on another and
  prefills everything again.
- Dynamo's router (and ours) **scores both**: how much prefill a replica can
  skip, against how long its queue is.

## Results

```
1385 requests from 400 conversations, 8 system prompts, 4 replicas, 1564 prompt tokens on average

        router cache hit TTFT p50 ms TTFT p99 ms requests per replica
---------------------------------------------------------------------
   round robin       58%         243        1424              346-347
        random       58%         453        2247              332-363
  least loaded       57%         190         787              294-420
   best prefix       76%         254       28487              161-737
      kv-aware       74%         103         457              345-349
```

- **Cache-blind routers reuse 57–58% of prompt tokens.** These are mostly
  system prompts that happen to be cached everywhere. Their TTFT p99 is
  0.8–2.2 s.
- **Best prefix** has the highest hit rate (76%), but one replica got 737
  requests and another 161. Its **p99 TTFT is 28 s**: a hotspot.
- **kv-aware** gets nearly the same hit rate (74%) with evenly spread load
  (345–349 per replica). It has the **best p50 (103 ms) and p99 (457 ms)**,
  2.3× and 3× better than round robin.

## Connection to earlier days

- Day 13: the KV cache. Prefix caching is the same cache kept after the
  request ends.
- Day 22: stateless chat means every turn resends the history. That is exactly
  why prefix caching pays off.
- Day 23: vLLM's `--enable-prefix-caching` is the cache inside one replica.
  The router decides **which** replica's cache to use.
- Day 25: block hashes work at the same block size as paged memory, so a cached
  prefix is just a list of blocks.

## Caveats

- Replicas handle one request at a time here (queue → prefill → decode). Real
  replicas batch (day 24), so their "load" is fuzzier.
- The router knows every replica's cache exactly. Dynamo learns it from events
  the replicas publish, which arrive slightly late.
- The tokens are random numbers, not real text. Only equality matters for
  prefix matching.
- `LOAD_WEIGHT = 1` treats one ms of queue like one ms of prefill. Dynamo
  makes this weight a setting.

## Things to try

1. Raise `CACHE_BLOCKS` to 3,000. Cache-blind routers catch up, because
   everything fits everywhere.
2. Set `LOAD_WEIGHT` to 0 (best prefix) and to 10. Watch the p99.
3. Make one system prompt much more popular than the others.
