# Day 19: Local inference — quantization levels on your laptop

Book chapters: **3.4 Other Datacenter Accelerator Options**, **3.5 Local Inference**

Task: run a GGUF model locally with llama.cpp or Ollama; benchmark
quantization levels on your laptop.

## What we built

`local_bench.py` uses **Ollama** to run the same model (Qwen 2.5 1.5B
Instruct) at three quantization levels on your Mac. For each one it:

1. downloads it if needed,
2. measures prefill speed, time to first token and decode speed, averaged over
   3 runs,
3. compares decode speed with day 17's rule (bandwidth / model size),
4. asks all three the same question, to compare answers.

## How to run it

```
brew install ollama          # once
ollama serve                 # in one terminal: starts the server, leave it running
python3 local_bench.py       # in another terminal
```

The first run downloads about 5.7 GB (the three models). After that it takes
about a minute. Options: `--quants q2_K q4_K_M q8_0 fp16` to choose levels,
`--family` for another model, `--runs` for more repeats.

Two programs work together, like a real inference service:

- **`ollama serve`** is the **server**. It loads models and answers requests at
  `localhost:11434`.
- **`local_bench.py`** is the **client**. It sends prompts to the server and
  reads the timings it sends back.

Stop the server with Ctrl+C when you are done, so it does not keep memory in
use.

## What we learned

### GGUF, Ollama and llama.cpp

- **llama.cpp** is a C++ inference engine built to run models on normal
  computers (Mac, PC, even phones).
- **GGUF** is its file format: one file holding the weights (often quantized)
  and the tokenizer.
- **Ollama** wraps llama.cpp with a model library and a simple API, so you can
  `ollama run <model>` and start chatting.

### What the quantization names mean

| Name | Bits per weight | Meaning |
|---|---|---|
| fp16 | 16 | normal half precision, no quantization |
| q8_0 | about 8 | 8-bit integers, plus one scale per block of weights |
| q4_K_M | about 4.5 | 4-bit "K-quant", Medium: more bits kept for the most sensitive layers |

Each block of weights is stored as small integers plus a **scale**, a number
to multiply them by to get back near the original values. Chapter 5.1 covers
this in detail.

### Results on your Mac (Apple M4, 16 GB)

Qwen 2.5 1.5B, 222-token prompt, up to 128 new tokens, 3 runs each:

| Quant | File size | Prefill tok/s | TTFT | Decode tok/s | Day 17 "ideal" | Reached |
|---|---|---|---|---|---|---|
| q4_K_M | 0.99 GB | 841 | 264 ms | **79.4** | 122 | 65% |
| q8_0 | 1.65 GB | 865 | 257 ms | **49.7** | 73 | 68% |
| fp16 | 3.09 GB | 925 | 240 ms | **31.7** | 39 | 82% |

"Ideal" = 120 GB/s (the M4's memory bandwidth) / file size.

1. **Decode speed follows file size.** From fp16 to q4, the file got about 3x
   smaller and decode got **2.5x faster**. Fewer bytes to read per token means
   more tokens per second: decode is memory-bound, measured on your own laptop.
2. **Prefill barely changed** (841 to 925 tok/s). Prefill is compute-bound, so
   fewer bytes do not help. q4 is even slightly slower, because it spends extra
   work unpacking the 4-bit numbers back into fp16 before the math.
3. **Day 17's rule landed within 65-82%** of reality. The smaller files reached
   a lower share, because fixed costs (unpacking, launching work on the GPU)
   matter more when the weights are quick to read.
4. **Quality:** all three answered the test question correctly ("6:15 pm"). One
   easy question cannot show quality loss. Quantization errors show up on
   harder tasks, which is why you run evals (chapter 1) before shipping a
   quantized model.

### Other datacenter accelerators (3.4)

NVIDIA leads, but every competitor bets on one edge:

| Edge | Companies | Idea |
|---|---|---|
| Memory bandwidth | Cerebras, Groq | keep weights in SRAM for very fast decode |
| Power efficiency | Furiosa, Qualcomm | cheaper to run |
| Platform integration | Google (TPU), AWS (Inferentia) | built into their clouds |
| Specialized chips | Etched | a chip built only for transformers |

They all share the same problems: **software** (no CUDA, so the whole stack
must be rebuilt), **manufacturing**, and **getting chips installed**.

### Local inference (3.5)

| Advantages | Weaknesses |
|---|---|
| No network delay | much weaker hardware than a datacenter |
| Works offline | heat limits speed |
| Private: data never leaves the device | every device is different |
| Free for the developer | drains the battery |

- **Desktop:** Apple leads, because of **unified memory**: lots of memory, but
  slower bandwidth. An RTX 5090 has 32 GB at 1,792 GB/s; an M3 Ultra has
  512 GB at 819 GB/s. The Mac fits huge models but runs them slower.
- **Mixture of Experts** models help locally: each token only reads some of the
  weights, so decode reads fewer bytes.
- **Phones** struggle above 1-2B parameters. They suit small jobs like
  speech-to-text and translation (Android: Gemini Nano; iOS: Foundation
  Models, Core ML).
- **The future is both:** small, quick requests on the device; heavy ones in
  the cloud.
- Think about your users: most have older, slower devices than yours.

## Connection to earlier days

- Day 1 and 6: TTFT and decode tokens/s again, now reported by Ollama.
- Day 5: file size = parameters x bytes per weight.
- Day 12: decode is memory-bound (decode followed file size), prefill is
  compute-bound (prefill stayed flat).
- Day 13: Ollama reuses the KV cache when prompts start the same way. Each run
  starts with a different `[note set i]` label, so the prompt is processed
  fresh every time and the timing is fair.
- Day 17: the bandwidth / size rule, checked against reality.

## Caveats

- Speeds change with other apps running, battery mode, and heat.
- The M4's 120 GB/s is the spec. Day 12 measured about 94 GB/s with PyTorch.
- TTFT here is only the prefill time, after the model is already loaded. The
  first request after starting also has to load the model from disk.
- One question is not an eval.

## Things to try

1. `python3 local_bench.py --quants q2_K q4_K_M q8_0 fp16`: is 2-bit faster
   still, and does the answer stay right?
2. Chat with a model: `ollama run qwen2.5:1.5b-instruct-q4_K_M`.
3. Try a bigger model: `--family qwen2.5:3b-instruct --quants q4_K_M q8_0`.
   Is decode about half the speed of the 1.5B?
4. Run `ollama ps` while a model is loaded to see how much memory it uses.
