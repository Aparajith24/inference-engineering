# Inference Engineering, from zero

My path from knowing almost no Python or ML to understanding how large
language models are served fast and cheaply. It follows a 45-day plan based on
the book [Inference Engineering by Philip Kiely](https://www.baseten.co/inference-engineering/). Each day is a
folder with working code and a README that explains what I built, what I
learned and what the numbers showed.

**Progress: day 29 of 45.** More to come.

## What this is about

When a model answers a prompt, two things happen:

| Phase | What happens | Limited by | Users feel it as |
|---|---|---|---|
| **Prefill** | Read the whole prompt in one parallel pass | Compute | Time to first token (TTFT) |
| **Decode** | Make one token at a time | Memory bandwidth | Tokens per second (TPS) |

Almost everything in inference engineering comes from that difference: why
batching works, why quantization speeds up decode, why the KV cache matters,
and which GPU to buy. These folders build that understanding one piece at a
time, mostly by measuring it.

## The days so far

### Part 1: foundations (days 1–7)

| Day | Topic | What I built |
|---|---|---|
| [1](days/day-1) | Prefill vs decode | Stream tokens from distilgpt2 and time TTFT and TPS |
| [2](days/day-2) | Scale and cost | Turn requests per day into GPU-hours and dollars |
| [3](days/day-3) | A neural network forward pass | A 2-layer MLP in pure NumPy |
| [4.1](days/day-4.1) | Tensors, dtype, device | Matmul benchmark: fp32, fp16, bf16 on CPU and Mac GPU |
| [4.2](days/day-4.2) | `torch.nn` | The day 3 MLP as a PyTorch module, benchmarked |
| [5](days/day-5) | Model selection | Parameter count → memory → which GPUs it fits on |
| [6](days/day-6) | Latency and throughput | TTFT, inter-token latency, p50/p95/p99, total vs per-user TPS |
| [7](days/day-7) | Review project | A benchmarking CLI for any Hugging Face model |

### Part 2: how an LLM works inside (days 8–11)

| Day | Topic | What I built |
|---|---|---|
| [8](days/day-8) | Self-attention | Single-head attention with a causal mask, in NumPy |
| [9](days/day-9) | Multi-head attention | A PyTorch `MultiHeadAttention` module, checked against SDPA |
| [10](days/day-10) | A whole GPT | My own GPT with distilgpt2's weights loaded, matching Hugging Face token for token |
| [11](days/day-11) | Sampling | Temperature, top-k and top-p, checked against Hugging Face |

### Part 3: bottlenecks and attention (days 12–15)

| Day | Topic | What I built |
|---|---|---|
| [12](days/day-12) | The roofline | Measured that 16 tokens cost the same as 1 on my Mac GPU: decode is memory-bound |
| [13](days/day-13) | KV cache | Added a cache to my GPT: 8.6x faster at 512 tokens, identical output |
| [14](days/day-14) | Faster attention | SDPA vs naive attention, MHA vs GQA vs MQA, and a hand-built PagedAttention (81.5% → 0.5% wasted memory) |
| [15](days/day-15) | Image generation | A toy diffusion loop: latency grows with steps, and images are compute-bound |

### Part 4: hardware (days 16–19)

| Day | Topic | What I built |
|---|---|---|
| [16](days/day-16) | GPU architecture | CPU vs GPU matmul and the cost of copying over PCIe, on a Colab T4 |
| [17](days/day-17) | GPU generations | Estimate decode speed and TTFT from a spec sheet, T4 to B200 |
| [18](days/day-18) | Instances and cost | A planner that picks the cheapest GPU setup per million tokens, including MIG slices |
| [19](days/day-19) | Local inference | fp16 vs q8_0 vs q4_K_M with Ollama on my Mac: decode speed follows file size |

### Part 5: software (days 20–22)

| Day | Topic | What I built |
|---|---|---|
| [20](days/day-20) | CUDA kernels | My first GPU kernels in Triton: vector add and a fused softmax (Colab) |
| [21](days/day-21) | Frameworks | Eager mode vs `torch.compile` on a transformer block |
| [22](days/day-22) | Review project | An OpenAI-style `/v1/completions` server with SSE streaming and **continuous batching** |

### Part 6: inference engines (days 23–27)

| Day | Topic | What I built |
|---|---|---|
| [23](days/day-23) | vLLM | Qwen2.5-0.5B with vLLM: offline batch generation and the OpenAI-compatible server (Colab) |
| [24](days/day-24) | Continuous batching, revisited | A scheduler simulator: continuous batching gives about 4x the throughput of static |
| [25](days/day-25) | PagedAttention | A block-based KV allocator: 89% of memory holds tokens vs 36% for contiguous reservations |
| [26](days/day-26) | NVIDIA Dynamo | A KV-aware router: prefix hits plus load balance, 3x better p99 TTFT than round robin |
| [27](days/day-27) | Benchmarking | An async load tester that sweeps concurrency and reports TTFT, ITL and throughput |

### Part 7: techniques (days 28–29)

| Day | Topic | What I built |
|---|---|---|
| [28](days/day-28) | Quantization: int8 | Per-tensor vs per-channel int8 on all 24 linear layers of distilgpt2: output error 5.68% → 0.91% |
| [29](days/day-29) | Quantization: int4 | 4-bit group quantization: perplexity for fp16, int8 and int4, and real int4 kernels 3.4–3.8x faster at batch 1 |

## Highlights

- **My own GPT, from scratch:** embeddings, attention, MLP blocks and the
  generation loop, with distilgpt2's trained weights loaded in. It produces
  exactly the same tokens as Hugging Face.
- **Decode is memory-bound, measured:** on the Mac GPU, pushing 16 tokens
  through the weights took the same 3.3 ms as 1 token. With Ollama, a 3x smaller
  quantized file decoded 2.5x faster.
- **The KV cache:** with it, speed stays flat as the text grows; without it,
  speed collapses (20 vs 168 tok/s at 512 tokens).
- **A streaming server with continuous batching (day 22):** users join and leave
  the batch at every step. It gave about 7x the total throughput at 30 users,
  and the README records every design decision I made.
- **How engines like vLLM win (days 23–27):** continuous batching gave 4x the
  throughput of static batching, paged KV memory fit a 2.5x bigger batch, and
  a KV-aware router cut p99 TTFT 3x by sending each chat back to the replica
  that already has its prefix cached.
- **Quantization (days 28–29):** per-channel int8 is lossless for distilgpt2.
  int4 needs small groups: one scale per row sent perplexity from 45 to 17,489,
  groups of 32 kept it within 0.3%. At batch 1, int4 kernels ran 3.4–3.8x
  faster than fp16, but slower at batch 16, because they save bytes, not compute.

## Running the code

Most days run on a Mac with the system Python:

```
pip3 install torch transformers matplotlib numpy fastapi uvicorn
python3 days/day-1/token_timer.py
```

Days 16, 20, 21, 23 and 27 are meant for a free **Google Colab T4 GPU**. Day 19 needs
[Ollama](https://ollama.com). Each day's README has its own instructions. Some
days import earlier ones (10 uses 9, 11 and 13 use 10), so keep all the day
folders together in `days/` with their names as they are.

## What's next

Days 30–45 continue through the rest of the book: more model optimization
(speculative decoding, caching, disaggregation), multi-GPU parallelism, and
running inference in production. Each day
will get its own folder here, like the ones above.
