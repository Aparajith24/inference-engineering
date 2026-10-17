# Day 7: Review of Phase 1 — a benchmarking CLI

Mini project: combine Days 1, 5 and 6 into a CLI that benchmarks any small
Hugging Face model and prints a report.

## What we built

`bench_v1.py` is a command-line tool. Give it any Hugging Face model id (or
nothing, for `distilgpt2`) and it prints:

1. **Parameter count and memory report** (day 5): how much memory the model
   needs at fp32/fp16/bf16/int8/int4, and which of 10 GPUs it fits on a
   single card. This never downloads the model's weights, so it works even
   for huge models.
2. **Timed benchmark** (days 1 and 6): loads the real model, runs N
   generations (default 20, random 16-64 output tokens each) with exact
   per-token timestamps, and reports TTFT, inter-token latency (ITL) and
   total response time as mean/P50/P95/P99, plus perceived and total TPS.
3. **Batching demo** (day 6): total TPS vs. per-user TPS at a couple of
   batch sizes.

Run it:

```
python3 bench_v1.py                          # distilgpt2, best device auto-picked
python3 bench_v1.py gpt2                      # any Hugging Face model id
python3 bench_v1.py gpt2 --n 30 --device mps --dtype fp16
python3 bench_v1.py meta-llama/Llama-3-70B --skip-run   # memory report only
```

Useful flags: `--n`, `--min-tokens`/`--max-tokens`, `--device`, `--dtype`,
`--batch-sizes`, `--skip-run`, `--force` (run anyway above the size guard).

## What we learned

### It is days 1, 5 and 6, glued together

```
day 5  ->  count parameters, estimate memory, check which GPUs fit
day 1  ->  load the model, stream tokens, time TTFT
day 6  ->  repeat many times, compute ITL, percentiles, throughput
```

The memory report and the timed benchmark are two separate stages. The first
is cheap (config only, no weights) and always runs. The second is expensive
(downloads and runs the real model) and is skippable.

### Config-only parameter counting scales to any model

```python
config = AutoConfig.from_pretrained(model_name)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
n_params = sum(p.numel() for p in model.parameters())
```

`AutoConfig` downloads a small JSON describing the architecture, not the
weights. The `meta` device builds tensors with shapes but no memory behind
them, so this works instantly even for a 70B+ model. This is what makes it
safe to type any model name without risking a multi-GB download by accident.

### A safety guard for the timed run

```python
DEFAULT_MAX_RUN_PARAMS = 2_000_000_000
```

Above this many parameters, the script prints the memory report and skips
the timed run unless `--force` is passed. The memory report alone is often
enough to answer "will this fit", without downloading anything.

### Device and dtype defaults encode earlier lessons

```python
def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def pick_dtype(device):
    return "fp32" if device == "cpu" else "bf16"
```

Best GPU available, and fp32 on CPU vs. bf16 on GPU by default, because day 4
showed CPU has no fast half-precision path.

### Exact per-token timing

A custom `TokenTimestamps(BaseStreamer)` records `time.perf_counter()` every
time `generate()` hands it a new token (it calls `put()` once with the
prompt, then once per generated token). This gives exact per-token gaps,
more precise than day 1's text-chunk streamer, which could group partial
words.

### Batching needs its own warmup

```python
batched_throughput(1, tokenizer, model, device, n_tokens=8)  # warm up batched shapes
```

A batch of size 4 is a different tensor shape than a single request, and the
first call at a new shape can carry its own setup cost (see results below).
So the batching demo warms up before timing it, on top of the single-request
warmup done earlier.

## Results (Apple M4, distilgpt2)

**On CPU**, results were clean and consistent with earlier days:

| Metric | Mean | P50 | P95 | P99 |
|---|---|---|---|---|
| TTFT | 8.5 ms | 8.2 ms | 10.9 ms | 11.3 ms |
| ITL | 6.6 ms | 6.5 ms | 7.0 ms | 7.5 ms |

Perceived TPS about 151, total TPS about 150.

**On MPS** (the default, since it is the best available device), results
were spiky: TTFT P50 was 11 ms but P99 jumped to 720 ms, and a batch of 4
was *slower* than a batch of 1. This is very likely MPS recompiling its
kernels each time the input shape changes, and the script deliberately
varies output length per request (16-64 tokens) and batch size, so it keeps
hitting new shapes. It is a real artifact of Apple's GPU backend surfaced by
the benchmark, not a bug in the script.

## Connection to days 1-6

- Days 1 and 6 supplied the timing method (streaming, warmup, percentiles).
- Day 5 supplied the memory math and GPU fit table.
- Day 4's device/dtype lessons became the script's defaults.
- Day 2's batching trade-off (throughput up, per-user latency down) shows up
  directly in the batching demo's two TPS columns.

## Caveats

- The 20% memory overhead and the GPU list are the same rough figures as
  day 5, not measurements.
- 20 generations is enough to see a shape, not enough for a stable P99. Raise
  `--n` for tighter percentiles.
- The MPS spikiness is hardware/backend-specific; CPU numbers were far more
  stable on this machine.
- `--skip-run` and the size guard only protect the timed run; the memory
  report still needs network access to fetch the model's config.

## Things to try

1. Run `python3 bench_v1.py gpt2` and compare its report to `distilgpt2`'s.
2. Run with `--device cpu` vs. the default and compare the percentile spread.
3. Try `--batch-sizes 1 4 16` and watch total TPS vs. per-user TPS diverge.
4. Point it at a model just over `DEFAULT_MAX_RUN_PARAMS` and see the guard
   trigger, then rerun with `--force`.
