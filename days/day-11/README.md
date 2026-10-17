# Day 11: Sampling — temperature, top-k and top-p

Book chapter: **2.2 LLM inference mechanics (sampling)**

Task: temperature, top-k and top-p sampling; compare outputs.

## What we built

`sampling.py` changes **only how the next token is picked**. The model is day
10's `TinyGPT` with `distilgpt2`'s trained weights. Instead of always taking
the top-scoring token (greedy), it picks at random, weighted by probability,
with three knobs to control the randomness:

```
tokens -> model (day 10) -> 50,257 logits
                               |
          temperature -> top-k -> top-p -> softmax -> weighted random pick
                               |
                          next token -> append -> repeat
```

Run it:

```
python3 sampling.py
```

It has three parts:

1. Show how each knob reshapes **one** next-token distribution.
2. **Check** our knobs against Hugging Face's own implementations.
3. **Generate text** with six settings, 3 samples each, and compare.

This file imports `load_gpt2_weights` from `day-10`, so keep the folder names
`day-9` and `day-10`.

## What we learned

### Greedy versus sampling

- **Greedy** (`argmax`) always takes the top token. It is deterministic, and
  small models get stuck in loops (day 1's and day 10's repeated newlines).
- **Sampling** (`torch.multinomial`) spins a weighted wheel: a token with
  probability 0.64 is picked about 64% of the time.

### The three knobs, on a toy example

Four candidate words with logits 2.0, 1.0, 0.0, -1.0 give probabilities
0.64, 0.24, 0.09, 0.03 (blue, green, red, banana).

**Temperature** divides the logits before softmax. It removes nothing, it only
reshapes the wheel:

| Word | T = 0.5 | T = 1 | T = 2 |
|---|---|---|---|
| blue | 0.86 | 0.64 | 0.46 |
| green | 0.12 | 0.24 | 0.28 |
| red | 0.02 | 0.09 | 0.17 |
| banana | 0.00 | 0.03 | 0.10 |

Low temperature is predictable, high temperature is creative (and lets
"banana" in). Temperature 0 means greedy.

**Top-k** keeps exactly the k best tokens. With k = 2, blue and green remain
and are renormalized to 0.73 and 0.27.

**Top-p** keeps the smallest group of top tokens whose probabilities add up to
at least p. With p = 0.8: blue (0.64) is not enough, blue + green (0.88) is,
so those two remain.

**Top-k is fixed-size; top-p adapts to confidence.** If the model is sure, top-p
keeps very few tokens. If it is unsure, top-p keeps many.

### How a token is removed

Set its logit to `-inf`. Softmax gives `exp(-inf) = 0`, so it can never be
picked, and the remaining tokens automatically renormalize to sum to 1.

### The top-p calculation

```python
sorted_logits, order = torch.sort(logits, descending=True)
probs = sorted_logits.softmax(dim=-1)
mass_before = probs.cumsum(dim=-1) - probs   # probability already covered above each token
remove = mass_before >= p                    # the set reached p without this token
```

```
probs         [0.64, 0.24, 0.09, 0.03]
mass_before   [0.00, 0.64, 0.88, 0.97]
remove (0.8)  [keep, keep, remove, remove]
```

The scores are then scattered back to their original vocabulary positions, so
token IDs are not scrambled.

## Results (distilgpt2, prompt "The future of AI inference is")

### One next-token distribution

| Setting | Tokens kept | Top 3 |
|---|---|---|
| temperature 0.5 | 50,257 | ' not' 0.27, ' a' 0.23, ' in' 0.11 |
| temperature 1.0 | 50,257 | ' not' 0.07, ' a' 0.07, ' in' 0.05 |
| temperature 1.5 | 50,257 | ' not' 0.02, ' a' 0.02, ' in' 0.01 |
| top-k 5 | 5 | ' not' 0.28, ' a' 0.26, ' in' 0.18 |
| top-k 50 | 50 | ' not' 0.11, ' a' 0.11, ' in' 0.07 |
| top-p 0.5 | 26 | ' not' 0.14, ' a' 0.13, ' in' 0.09 |
| top-p 0.9 | 443 | ' not' 0.08, ' a' 0.07, ' in' 0.05 |

- Temperature never removes tokens.
- Top-k keeps exactly k, so each kept token's chance rises.
- Top-p 0.9 needed 443 tokens because this small model is unsure.
- The ranking never changes. The knobs only change how often lower-ranked
  tokens get picked.

### Checks

- Our top-k and top-p filters remove **exactly the same tokens** as Hugging
  Face's `TopKLogitsWarper` and `TopPLogitsWarper`, and temperature matches
  `TemperatureLogitsWarper`.
- Temperature 0 and top-k 1 both produce text **identical to greedy**.

### Generated text (25 tokens, seeds 0-2)

| Setting | Example | Different outputs | Unique tokens |
|---|---|---|---|
| Greedy | " not yet clear.\n\n\n\n..." | 1 of 3 | 20% (stuck) |
| Temperature 0.7 | " up in the air, and you may not even know it..." | 3 of 3 | 80% |
| Temperature 1.5 | " up yet shape difficult news ones UNDER CHARLES QUIEED..." | 3 of 3 | 97% (word salad) |
| Top-k 5 | " a complex problem. We need to find ways to solve it..." | 3 of 3 | 75% |
| Top-p 0.9 | " a fundamental mystery. AI analysts and programmers..." | 3 of 3 | 91% |
| temp 0.8 + top-k 50 + top-p 0.95 | " uncertain. But the future of AI is uncertain. But..." | 3 of 3 | 69% |

- Greedy is the same every time, and stuck.
- Temperature 0.7 and top-p 0.9 are the sweet spot: readable and varied.
- Temperature 1.5 is too random. More variety is not always better.
- Even the combined setting produced one repetitive sample. Sampling makes
  loops less likely, not impossible.

## Why this matters for inference

- These are the `temperature`, `top_k` and `top_p` settings on every LLM API.
- They are **cheap**: sorting 50,257 numbers is tiny next to a forward pass, so
  they barely affect TTFT or TPS.
- Getting them **exactly right** matters for quality. A subtly wrong top-p
  gives worse text without anyone noticing, which is why we checked ours
  against Hugging Face's. Chapter 2.2 also mentions logit biasing for
  structured output (JSON, tool calls), which works on the same logits.

## Connection to earlier days

- Day 1: the greedy newline loop, and the `do_sample=True, top_k=50` fix.
- Day 8: softmax, and `-inf` to remove options (like the causal mask).
- Day 10: the same model and loop, with `argmax` replaced by `sample_next`.
- Chapter 2.2: temperature, top-k, top-p, and temperature 0 / top-k 1 as
  deterministic.

## Caveats

- Three samples per setting is enough to see patterns, not to measure them.
- `distilgpt2` is small, so even good settings produce weak text.
- No KV cache (day 13), so generation is slow.
- The order matters: temperature, then top-k, then top-p. Other engines may
  order them differently, which can change results slightly.

## Things to try

1. Change the seeds and see which patterns hold.
2. Try temperature 0.3 and 3.0 and compare against the table above.
3. Print how many tokens top-p 0.9 keeps at each step of a generation, and
   see it change with the model's confidence.
4. Use a prompt with an obvious answer, like "1, 2, 3, 4,", and compare how
   many tokens top-p 0.9 keeps.
