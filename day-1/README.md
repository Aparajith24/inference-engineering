# Day 1: Two phases, three layers

Book chapters: **0.1 Two phases**, **0.2 Three layers**

## What we built

`token_timer.py` loads `distilgpt2` with Hugging Face Transformers, streams
tokens as they are generated, and measures:

- **TTFT** (time to first token): how long before the first token appears.
- **TPS** (tokens per second): how fast tokens arrive after the first one.

Run it:

```
python3 token_timer.py
```

Sample result on a MacBook Air (CPU): TTFT about 9-12 ms, TPS about 150-170.

## What we learned

### 0.1 Two phases

Generating text has two distinct phases:

| Phase | What happens | Bound by | Shows up as |
|---|---|---|---|
| Prefill | Read the whole prompt in one parallel pass | Compute | TTFT |
| Decode | Produce one token at a time, each needing the previous one | Memory bandwidth (reading weights) | TPS |

That is why the timer separates the two numbers. TTFT includes prefill plus
the first token. TPS is measured only over the decode phase.

### 0.2 Three layers

Inference work can be seen in layers. Here is where this day's code sits:

- **Model:** the trained weights (`distilgpt2`) that predict the next token.
- **Runtime:** the code that runs the model efficiently (PyTorch and
  `model.generate`, which includes the KV cache and the token loop).
- **Application:** our script, which sends a prompt and measures the result.

(Adjust the layer names to match the book's wording if they differ.)

## Key concepts

- **Token:** a word or word piece. The tokenizer converts text to integers and back.
- **Inference:** using a trained model, as opposed to training it.
- **Greedy vs. sampling:** `do_sample=False` always picks the top token and
  small models get stuck repeating newlines. `do_sample=True, top_k=50`
  gives varied text. Timing is the same either way.
- **Streaming:** `TextIteratorStreamer` plus a background `Thread` lets us
  receive tokens as they are produced instead of waiting for the end.
- **Warmup:** the first run has one-time setup cost, so we run a throwaway
  generation before measuring.

## Notes

- The `NotOpenSSLWarning` from urllib3 is harmless. It comes from macOS's
  system Python using LibreSSL.
- Streamer chunks are approximately tokens, so TPS is approximate.
