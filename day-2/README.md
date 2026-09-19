# Day 2: Runtime techniques, scale, abstraction, map

Book chapters: **0.3-0.6 Runtime techniques, scale, abstraction, map**

## What we built

`cost_estimator.py` tokenizes sample prompts, counts tokens, and estimates
the daily cost and GPU-hours for a workload of **1M requests/day**.

Run it:

```
python3 cost_estimator.py
```

With the default assumptions it reports about 22 GPU-hours/day, about $56/day
(about $1,680/month), and 0.9 GPUs at average load or 2.8 at 3x peak.

## What we learned

### Tokens are the unit of work

Cost scales with tokens, not requests. Different models tokenize the same
text differently, so always count with the model's own tokenizer.

### Scale: turning a workload into hardware

```
requests/day x tokens/request   -> tokens/day
tokens/day / tokens/sec per GPU -> GPU-seconds
GPU-seconds / 3600              -> GPU-hours
GPU-hours x $/hour              -> daily cost
GPU-hours / 24                  -> average GPUs busy
average x peak factor           -> GPUs to provision
```

Prefill and decode are costed separately because their speeds differ
(prefill is fast and parallel, decode is slow and one token at a time). For
short prompts with long replies, decode dominates the bill.

### Runtime techniques

Ways to push TTFT down, TPS up, or serve more users on the same hardware:

| Technique | Idea | Helps most with |
|---|---|---|
| Batching | Serve many requests in one pass; weight reads are shared | Throughput, cost |
| Quantization | Store weights in fewer bits (16 -> 8 or 4) | Memory, TPS |
| Parallelism | Split a model across GPUs | Models too big for one GPU |
| Caching | KV cache and prefix cache: do not recompute | TTFT, long contexts |
| Speculation | Small draft model guesses, big model verifies | TPS, same quality |
| Disaggregation | Run prefill and decode on separate machines | TTFT and TPS at scale |

### Abstraction and map

The estimator hides hardware details behind a few numbers (`PREFILL_TPS_PER_GPU`,
`DECODE_TPS_PER_GPU`, `GPU_COST_PER_HOUR`). That abstraction is what makes
"what if" questions cheap: change a constant and rerun. Each runtime
technique above maps onto one of those constants. For example, quantization
and speculation raise decode TPS, and batching raises throughput per GPU.

(Adjust this section to match the book's framing of abstraction and the map.)

## Caveats

- The hardware numbers are **assumptions**, not measurements. Benchmark your
  own setup and edit the constants at the top of the file.
- The estimate assumes perfect utilization, so real costs will be higher.
  A utilization factor (say 60%) is a sensible next step.
- Only 5 sample prompts are used, so the average input length is rough.
