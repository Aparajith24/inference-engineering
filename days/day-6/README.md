# Day 6: Latency and throughput

Book chapter: **1.4 Measuring Latency and Throughput**

Task: run 50 generations; compute TTFT, inter-token latency, p50/p95/p99, throughput.

## What we built

`latency_metrics.py` runs 50 generations on `distilgpt2` with different prompts
and random output lengths (32-128 tokens). It timestamps every generated token
and reports:

- **TTFT**: time to first token.
- **ITL**: inter-token latency, the gap between consecutive tokens.
- **Total response time**: from request start to the last token.
- **Perceived TPS**: tokens per second per user, after the first token.
- **Total TPS**: tokens per second for the whole service (throughput).
- **Percentiles** (P50 / P95 / P99) next to the mean.
- A small **batching demo** showing total TPS versus per-user TPS.

Run it:

```
python3 latency_metrics.py
```

It takes about 35 seconds on CPU.

## What we learned

### TPS means two different things

| Term | Meaning | Type |
|---|---|---|
| **Perceived TPS** | Tokens/s one user sees after the first token | Latency |
| **Total TPS** | Tokens/s the whole service produces | Throughput |
| **ITL** | Time between two tokens | Latency |

ITL and perceived TPS are the same thing seen from two sides:
`perceived TPS = 1 / ITL`. An ITL of 10 ms is 100 tokens/s per user. The code
checks this: `1 / mean ITL` matches the mean perceived TPS.

When someone just says "TPS", they usually mean perceived TPS. Be specific when
it matters.

### Why percentiles instead of the mean

Response times are **right-skewed**: most requests cluster around a typical
time, but a few outliers take much longer. Those outliers pull the mean above
the median.

| Percentile | Meaning |
|---|---|
| P50 (median) | 1 in 2 requests is slower |
| P90 | 1 in 10 is slower |
| P95 | 1 in 20 is slower |
| P99 | 1 in 100 is slower |

A product where most requests feel fast but 1 in 10 takes seconds loses user
trust. So good performance work reduces P90/P99, not only the average.

For TPS, higher is better, so the bad tail is the **low** end. That is why the
script reports P5 and P1 for TPS instead of P95 and P99.

### Exact per-token timing

Day 1 used `TextIteratorStreamer`, which buffers text and can group partial
words, so chunks only approximated tokens. Here a custom `BaseStreamer`
records `time.perf_counter()` every time `generate` hands it a token.
`generate` calls `put()` once with the prompt and then once per new token, so we
get exact token timestamps. It runs in the main thread, so no `Thread` is
needed.

### Inference time versus end-to-end time

This script measures **inference time only**: the model's time to produce
tokens on this machine. It does not include network latency or queue time. In
production you want both:

- Inference time shows how well your model performance work is going.
- End-to-end time shows what users feel.
- If inference is fast but end-to-end is slow, look at infrastructure (network,
  queues), not the model.

### Latency versus throughput (batching demo)

| Batch size | Total TPS | Per-user TPS |
|---|---|---|
| 1 | about 160 | about 160 |
| 4 | about 355 | about 89 |
| 16 | about 1100 | about 69 |

Batching multiplies total throughput, while each user's own speed drops. Cost
per token falls, but latency rises. This is the core trade-off between the two
kinds of TPS.

## Results (Apple M4, CPU, distilgpt2)

| Metric | Mean | P50 | P95 | P99 |
|---|---|---|---|---|
| TTFT | 9.1 ms | 8.4 ms | 12.5 ms | 13.2 ms |
| ITL | 6.9 ms | 6.8 ms | 7.5 ms | 8.0 ms |
| Total response time | 0.6 s | 0.6 s | 0.9 s | 0.9 s |

Perceived TPS was about 146 (P5 about 141). Total TPS was about 145, close to
per-user TPS because requests ran one at a time.

## Caveats

- **The tail here is very narrow** (P99 barely above P50). One client, one
  request at a time, no queue, and a tiny model produce steady timings. The
  large right-skew the book describes shows up under real traffic, with
  queueing, mixed request sizes and shared hardware. The percentile code is the
  same; the data would just be more spread out.
- Total response time varies mainly because requests ask for random lengths.
- TTFT starts the clock right before `generate`, so tokenization is excluded.
- Numbers vary between runs and depend on what else the machine is doing.
- 50 samples is enough to see a shape, but P99 from 50 samples is really just
  "close to the slowest request". Use hundreds or thousands for stable P99s.
- The batching demo uses greedy decoding on identical-length outputs and short
  prompts, so it is illustrative, not a serving benchmark.

## Things to try

1. Change `N_GENERATIONS` to 200 and see how stable P95/P99 become.
2. In `run_one`, right after `start = time.perf_counter()`, add
   `if random.random() < 0.1: time.sleep(0.2)` to simulate 1 in 10 requests
   hitting a queue. Watch P95/P99 TTFT jump while P50 barely moves.
3. Add a `mean` versus `P50` comparison after inserting one deliberately slow
   request, to see how outliers pull the mean but not the median.
4. Move the model to MPS (`model.to("mps")`, inputs too) and compare TTFT and ITL.
