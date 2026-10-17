# Day 27: Benchmarking and load testing

Book chapter: **4.5 Benchmarking and load testing**

Task: an async load tester for the vLLM server from day 23. Sweep concurrency
and report TTFT, ITL and throughput. Runs on Colab, and works against any
OpenAI-compatible `/v1/completions` server, including day 22's.

## What we built

`load_tester.py`:

1. Sends one **warm-up** request and stops if it fails.
2. For each **concurrency level** (default 1, 2, 4, 8, 16, 32, 64), starts that
   many simulated users. Each user sends a streamed request, waits for the
   whole answer and sends the next, until the level has done at least
   `max(16, 3 × users)` requests.
3. For every request, records:
   - **TTFT:** when the first text arrived;
   - **ITL:** the time between pieces of text;
   - **E2E:** the total time;
   - **tokens:** from the final `usage`.
4. Prints a table, saves the numbers to `load_test_results.json` and the
   graphs to `load_test_results.png`.

Each request sends `ignore_eos: true` (a vLLM extra), so every answer is
exactly `--max-tokens` long and the levels can be compared fairly.

## How to run it

On Colab, with the day 23 server running:

```
!python load_tester.py
!python load_tester.py --concurrency 1,8,64 --max-tokens 256
```

Against day 22's server on the Mac:

```
python3 day-22/server.py
python3 day-27/load_tester.py --url http://localhost:8080/v1 --model distilgpt2 --concurrency 1,4,8 --requests 8 --rounds 1 --max-tokens 32
```

It needs `httpx` (already on Colab and on this Mac) and, for the graphs,
`matplotlib`.

## What we learned

### Why async

Every simulated user spends almost all its time **waiting** for the server.
With `asyncio`, one Python thread can keep 64 users waiting at once, the same
"one waiter, many tables" idea as day 22's server, now on the client side. With
one thread per user, the client itself could become the bottleneck.

### Closed loop vs open loop

This tester is **closed loop**: a user sends the next request only when the
last one is done, so the load can never exceed the concurrency level. It
answers "what do N busy users experience?"

An **open-loop** tester sends requests at a fixed rate whether or not the server
keeps up, the way real traffic does (day 24's simulator). It shows how queues
explode past capacity.

### What each number tells you

| Metric | Shows | Goes up with users because |
|---|---|---|
| TTFT | Queueing + prefill | New requests wait for a seat and share the GPU with decodes |
| ITL | Decode step time | Every step does more work for the bigger batch |
| Total tok/s | The GPU's output | More users share each weight read (day 12), until compute or KV memory runs out |
| Per-user tok/s | What one user feels | 1 / ITL, roughly |

The **knee** is the concurrency where total tok/s stops growing but TTFT and
ITL keep rising. The best operating point is just before it.

### Measure from the client

The server's own numbers leave out the network, HTTP and JSON. The client sees
what users see.

## Results

### Mac smoke test against day 22's server (distilgpt2, 32 tokens)

```
users   ok err TTFT p50 TTFT p99 ITL p50 ITL p99 E2E p50 E2E p99 total tok/s per user  req/s
--------------------------------------------------------------------------------------------
    1    6   2       24      122    11.8    16.6    0.39    0.50          71     83.3   2.23
    4    8   0       46      160    27.2   119.8    1.02    1.08         147     40.7   4.58
    8    8   0       66      144    27.3   103.1    1.00    1.00         255     34.0   7.97
```

- 1 → 8 users: total throughput up **3.6×**, while each user's speed falls from
  83 to 34 tok/s. This is the batching trade-off again.
- The 2 errors are **empty answers**. Day 22's server doesn't know
  `ignore_eos`, so distilgpt2 sometimes ends on its first token.

### vLLM on Colab T4 (Qwen2.5-0.5B)

```
Paste your output here
```

## Connection to earlier days

- Day 6: TTFT, ITL, p50/p99, total vs per-user TPS: the same metrics, now
  measured from outside over HTTP.
- Day 22: the load test there was a throwaway script. This is a reusable tool.
- Day 23: the server under test.
- Day 24: continuous batching is why total tok/s keeps climbing with
  concurrency.

## Caveats

- Closed loop only (see above).
- ITL is measured between SSE messages. vLLM sends about one token per
  message, but a server that groups tokens would show fewer, bigger gaps.
- The client and server share one machine (Colab, or the Mac), so they compete
  for CPU.
- The prompts are short, so this mostly tests decode. Long prompts would test
  prefill and TTFT.

## Things to try

1. Run it against day 22 and vLLM with the same settings, and compare.
2. Add long prompts (1,000+ tokens) and watch TTFT.
3. Add an open-loop mode: start requests at a fixed rate with
   `asyncio.create_task`.
