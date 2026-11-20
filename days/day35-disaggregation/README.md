# Day 35: prefill/decode disaggregation, measured in goodput

Book chapter: **5.5 Disaggregation**

Task: separate prefill from decode, then measure what it is worth. The metric is
**goodput**: requests per second that arrive with TTFT and every inter-token gap
inside their targets. Tokens/s that land too late do not count.

## What we built

`pd_disagg_sim.py` is a discrete-event simulator: a 70B dense model at fp8 on
B200-class GPUs, Poisson traffic, and two ways to deploy the same 4 or 8 GPUs.

- **One GPU iteration is two resources.** Prefill is compute bound
  (`tokens / 15.7k tok/s`) and a decode step is memory bound
  (`70 GB of weights + seats x seq x 320 kB of KV` at 5 TB/s). They overlap, so
  the iteration costs the larger of the two plus a share of the smaller:

  ```
  iteration = max(chunk_ms, step_ms) + OVERLAP_LOSS * min(chunk_ms, step_ms)
  ```

  `OVERLAP_LOSS = 0.3` by default: a mixed iteration co-schedules prefill and
  decode kernels, but each runs with fewer SMs and less parallelism than when it
  owns the GPU. `--overlap-loss 0` is the perfect-overlap roofline, `1.0` is
  full serialisation. Section 6 shows this one number decides the whole answer.
- **Colocated engine**: iterations of `[prefill chunk, one decode step]`, where
  the chunk is the knob: 8192 is vLLM's default token budget (the whole prompt),
  128-512 protects ITL.
- **Disaggregated engine**: `P` prefill servers, a shared KV link with bandwidth
  and latency, `D` decode servers. Arrival → prefill → KV transfer (queued on the
  link) → decode steps. Batches are admitted at step boundaries up to the KV
  budget (101 GB per GPU, 308k tokens).
- **Conditional disaggregation**: requests whose *uncached* prompt is ≤ 512
  tokens are prefilled by the decode engine in 128-token chunks and never touch
  the link.
- **Targets**: TTFT ≤ 1000 ms, every ITL ≤ 30 ms.

```
python3 days/day35-disaggregation/pd_disagg_sim.py
python3 days/day35-disaggregation/pd_disagg_sim.py --workload chat --rate 8
python3 days/day35-disaggregation/pd_disagg_sim.py --overlap-loss 0.6
```

Standard library only, ~2 seconds, deterministic seed, saves `goodput.png` when
matplotlib is present.

## What we learned

### Unchunked colocation is a goodput disaster, and it is a single number

4 GPUs, 6 req/s, code-editor traffic (ISL 2000-8000, OSL 100-300):

```
  config                 GPUs  TTFT p99  ITL p99  worst   good/s  good %
  colocated, no chunk       4      655m     374m    517m     0.1     1%
  colocated, chunk 512      4      853m      42m     42m     0.1     1%
  colocated, chunk 256      4     1132m      25m     25m     5.4    97%
  colocated, chunk 128      4     7509m      23m     23m     1.3    27%
```

TTFT and throughput are *fine* in the first row. But a 5,000-token prompt is
318 ms of compute dropped into a decode stream, so the worst gap between two
tokens is 517 ms — a chat UI that freezes for half a second. 1% goodput is not
a tuning problem; it is the wrong architecture for a 30 ms ITL target.

### The chunk window is exactly "chunk compute ≈ decode step"

| chunk | ITL worst | TTFT p99 | goodput |
|---|---|---|---|
| 8192 (whole prompt) | 517 ms | 655 ms | 1% |
| 2048 | 140 ms | 733 ms | 1% |
| 512 | 42 ms | 853 ms | 1% |
| **256** | **25 ms** | **1132 ms** | **97%** |
| 128 | 23 ms | 7509 ms | 27% |

256 tokens of prefill is 16 ms of compute against a 19 ms decode step: it mostly
hides, ITL stays at 25 ms, and goodput is 97%. 512 crosses the line and every
iteration pays 42 ms. 128 is small enough but needs twice the iterations, and
TTFT goes to 7.5 s. **The window is one chunk size wide, and its position moves
with the decode step time, which moves with batch and context.** That fragility
is the real argument for disaggregation.

### Splitting the GPUs is a resource-balance question, and both ends starve

8 GPUs, 12 req/s:

```
  config                     TTFT p99  ITL p99  good/s  good %
  colocated x8, chunk 256        1149m      25m     9.9    98%
  1P7D                          48143m      17m     0.1     4%   <- prefill queue
  2P6D                          12680m      18m     0.9    12%
  3P5D                           2244m      20m     6.0    62%
  4P4D                            782m      21m    10.1   100%
  6P2D                            561m      30m     8.4    87%   <- decode saturated
  7P1D                          10509m      36m     0.1     2%
```

Prefill capacity is `15.7k tok/s x P` against `ISL x rate`; decode capacity is
`batch / step x D` against `OSL x rate`. Those two lines cross at a P:D set by
the workload, not by the model — change ISL:OSL and the answer moves, which is
why Dynamo makes `xPyD` configurable at runtime. The right column is the point:
6P2D has more prefill than 4P4D and gets 87% goodput because every request blows
its ITL instead.

### The prefill queue is the bottleneck you cannot see in throughput

1P7D serves 772 tok/s and 4P4D serves 2,005 tok/s, but the useful number is TTFT
p99: **48 s versus 0.8 s**. Once prefill is short of capacity, requests queue
*before* they own a KV cache, so nothing about the decode side matters. This is
the new bottleneck the book names, and it is a queueing problem, not a GPU one:
Dynamo's answer is a bounded prefill queue plus a runtime `xPyD` change.

### The KV link is not the bottleneck at this size, and the arithmetic says why

8 GPUs, 4P4D, 12 req/s, 12.4 GB/s of KV crossing the link:

```
  fabric   mean transfer   TTFT p99   goodput
  nvlink          3 ms        782m     100%
  ib800          14 ms        801m     100%
  pcie           82 ms        913m     100%
```

A 5,000-token prompt is 1.6 GB of KV; prefilling the same prompt is 318 ms of
compute. On NVLink the transfer is 1% of that, on PCIe 19%, and even then it
hides in the queue. The link starts to matter when its bandwidth approaches
`prompt x 320 kB / prefill time`, about **25 GB/s per engine**, or when prompts
get long enough that 320 kB per token stops being small. For a 70B model at 5k
tokens, IB and NVLink are both irrelevant and PCIe is marginal; at 200k tokens
all three matter. That is the quantitative form of "you can parallelize across
nodes if you can avoid bottlenecks from lower-bandwidth interconnects".

### At equal GPU count, tuned colocation ties or wins — until it doesn't

Load sweep, 4 GPUs:

```
  offered   colocated (chunk 256)   2P2D
   2 req/s    100% goodput          100%
   4 req/s    100%                  100%
   6 req/s     97%                   98%
   8 req/s     82%                   53%
  10 req/s     52%                   20%
  12 req/s     25%                   12%
```

Disaggregation is *not* free performance: 2P2D leaves two GPUs' compute mostly
idle (prefill) and two GPUs' memory mostly idle (decode), while a colocated
engine uses both resources of all four. That is why the book says "ideally,
prefill uses more compute resources, while decode uses more memory, and the two
can co-exist efficiently". At 8 GPUs the fair fight is a near-tie: colocated x8
98% at 9.9 good/s versus 4P4D 100% at 10.1 good/s, though 4P4D has the better
tails (TTFT 782 ms vs 1149 ms, ITL 21 vs 25 ms).

### The entire verdict is one unmeasured constant

```
  overlap loss   colocated, chunk 256   2P2D   verdict
           0%                   100%     53%   colocation
          30%                    82%     53%   colocation
          60%                     1%     53%   disaggregation
         100%                     0%     53%   disaggregation
```

How much of the smaller phase actually hides behind the larger one is a property
of kernels and batch mix on your hardware. At 30% the mixed iteration costs
25 ms, which is 5 ms inside the 30 ms target; at 60% it costs 34 ms and every
request dies. That cliff — not a smooth curve — is why the chunk window is so
hard to tune and why people reach for architecture instead.

### On chat traffic there is nothing to fix

Chat, 8 req/s, ISL 300-1200:

```
  colocated, no chunk       1% goodput   (ITL 63/84 ms)
  colocated, chunk 512      1%            (ITL 41 ms)
  colocated, chunk 256    100%            (ITL 24 ms, TTFT 134 ms)
  colocated, chunk 128    100%            (ITL 20 ms, TTFT 244 ms)
  1P3D / 2P2D / 3P1D      100%
```

Short prompts make prefill a few percent of the work, so a chunked colocated
engine keeps ITL at 24 ms and TTFT at 134 ms, and disaggregation has nothing
left to buy. This is
the book's third precondition failing, and the honest recommendation is extra
replicas: more users, not a better split.

### So when is disaggregation actually worth it?

Taking the book's three preconditions seriously, and adding what the numbers
here say:

1. **Enough traffic to right-size both roles.** At 12 req/s x 5,200 tokens this
   workload is ~5.4B tokens/day, past the book's 100M-1B band, and 4P4D is still
   only tied with 8 tuned replicas.
2. **A model big enough that a GPU cannot hold both roles' working set.** A 70B
   leaves 100 GB of KV per GPU; a 500B MoE on 8 GPUs leaves almost none, so the
   decode batch is memory-starved and prefill bursts hurt more.
3. **Prefill-heavy, long-context traffic.** This is where chunked colocation
   pays the overlap tax on every iteration and disaggregated prefill runs at the
   full 15.7k tok/s with no chunking at all.
4. **SLO isolation.** In the model, disaggregation's most reliable win is not
   goodput but *predictability*: a prefill burst on a colocated engine is an ITL
   spike by construction, and no chunk size fixes a workload whose prompts
   change length by 4x.
5. **Different parallelism per role.** Prefill needs less TP than decode, so a
   split deployment can spend NVLink bandwidth where it pays (day 34's planner
   prices exactly that).

## Connection to earlier days

- Day 1: TTFT is prefill and ITL is decode. This day is the bill for putting
  them on one device.
- Day 12: decode is memory bound, so the step time is weights + KV bytes — the
  term that places the chunk window.
- Day 24: continuous batching, reused for admission at step boundaries.
- Day 25: the 101 GB KV budget (308k tokens) is a paged-cache capacity, and it
  limits the decode batch before the weights do.
- Day 27: the load sweep and p50/p99. Goodput is that report with an SLO filter.
- Day 34: the planner prices TP, PP and EP. This prices the time split between
  two roles, and shows why the split has to be re-tuned per workload.

## Caveats

- `max + loss * min` is a two-resource model with one tuned constant. It has no
  SM occupancy, no kernel launch cost beyond the fixed 2 ms decode / 3 ms
  prefill, and no CUDA-graph effects. `--overlap-loss` is the honest knob.
- Prefill engines are FCFS with an unbounded queue. A real dispatcher sheds or
  reroutes, which is why Dynamo exposes queue depth as a threshold.
- The KV transfer is bandwidth + latency, not a real NIXL transport, and it
  assumes both roles use the same TP layout. Different layouts need the block
  transpose the book mentions.
- `cached` prefixes are applied on the prefill side only; a real 2P2D setup also
  has a prefix cache on the decode side, which changes the traffic mix and makes
  conditional disaggregation much more attractive.
- No admission control, no request timeouts, no priority classes.
- Absolute numbers are one 70B model on B200-class GPUs. The shapes and the
  crossovers are the point.

## Things to try

1. `--overlap-loss 0` and `0.6` on the same workload. Find the loss at which
   disaggregation overtakes tuned colocation for *your* batch mix.
2. `--workload chat --rate 16`: where does colocation with chunk 256 finally
   lose, and does it lose on TTFT or on ITL?
3. Set the decode engines to TP4 and the prefill engines to TP1 (the book's
   "prefill requires a lower TP than the decode engine") and compare goodput
   per GPU.
4. Bound the prefill queue at 8 and shed or reroute the overflow. How much
   goodput does shedding buy at 12 req/s?
5. Give 50% of requests a 90% cached prefix and re-run the conditional row: at
   what prefix-hit rate does conditional disaggregation beat plain 2P2D?
