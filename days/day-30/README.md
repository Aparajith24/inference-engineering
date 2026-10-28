# Day 30: Speculative decoding — gpt2 drafts, gpt2-medium checks

Book chapter: **5.2 Speculative decoding**

Task: draft-target speculative decoding with `gpt2` (draft, 124M) and
`gpt2-medium` (target, 355M), including accept/reject.

## What we built

`spec_decode.py` builds speculative decoding by hand on top of Hugging Face
models, with a KV cache for each model.

1. **Toy accept/reject check:** a 4-word vocabulary where the draft and target
   disagree. It runs the accept/reject rule 20,000 times and shows that the
   output follows the **target's** probabilities, not the draft's.
2. **Walk-through:** "The cat sat on the", greedy, 4 drafts per pass. It prints
   every draft and whether the target accepted it.
3. **Costs:** one forward pass of each model, and the target checking 1, 3, 5
   and 9 tokens at once.
4. **Acceptance by prompt:** history, code, advice and a counting sequence.
5. **Benchmark:** normal decoding vs speculative decoding with 1–8 drafts per
   pass, at temperature 0 and 1. Each setting is timed 3 times and the fastest
   run is kept. At temperature 0 it also checks that the output is **identical**
   to normal decoding.

```
python3 days/day-30/spec_decode.py           # everything, about 5 minutes
python3 days/day-30/spec_decode.py --quick   # skip the benchmark
```

The first run downloads gpt2-medium (1.5 GB). Both models together use about
2 GB on the Mac GPU.

## How one speculative step works

The `Model` class wraps a model with its own KV cache. `forward` feeds only the
tokens the cache hasn't seen yet, and `rewind` cuts the cache back after
rejected drafts.

1. **Draft:** the small model makes `k` tokens one at a time, as normal decoding
   would, and remembers its probabilities `q` for each.
2. **Verify:** the big model runs **one** forward pass over the last real token
   plus all `k` drafts. Like a small prefill, this gives `k + 1` predictions at
   once: one to check each draft, plus one for the token after the last draft.
3. **Accept or reject, left to right:**
   - keep draft `d` with probability `min(1, p(d) / q(d))`;
   - at the first rejection, sample a replacement from what `p` has to spare,
     `max(0, p − q)` normalized, and **throw away every draft after it**;
   - if all `k` are accepted, take a **bonus** token from the last prediction.
4. **Rewind** both caches to the tokens that were kept.

Each pass produces **accepted + 1** tokens.

At temperature 0 the probabilities become "100% on the top token", and the
rule becomes simple: accept if the draft equals the target's top choice.

## What we learned

### The accept/reject rule keeps the output exact

```
Toy: "The cat sat on the ..." with 4 possible words
    word  draft q  target p   result
     mat     0.70      0.50    0.499
    sofa     0.10      0.30    0.297
   floor     0.10      0.15    0.154
    roof     0.10      0.05    0.050
  accepted 75% of drafts (theory: sum of min(p, q) = 0.75)
```

- The draft over-guesses "mat" (70% vs 50%), so some of its "mat" drafts are
  rejected.
- The draft under-guesses "sofa" (10% vs 30%). Rejected drafts are resampled
  from what the target has to spare, which is mostly "sofa".
- The result matches `p`. This is why speculative decoding is **lossless**: the
  big model's answers are unchanged, only faster.
- Acceptance = how much the two distributions overlap, `Σ min(p, q)`.

### A walk-through

```
Walk-through: 'The cat sat on the', greedy, 4 drafts per pass
  pass: ' floor' ok  ',' ok  ' and' NO -> ' staring'
  pass: ' at' ok  ' the' ok  ' ceiling' ok  '.' ok  + bonus '\n'
  pass: '\n' ok  '"' ok  'I' ok  "'m" ok  + bonus ' sorry'
  pass: ',' ok  ' I' ok  "'m" ok  ' sorry' ok  + bonus ',"'
  result: 'The cat sat on the floor, staring at the ceiling.\n\n"I\'m sorry, I\'m'
```

16 tokens from **4** passes of the big model instead of 16. In pass 1, the
draft guessed " and", the target wanted " staring", and the target's word was
used. Passes 2–4 accepted all 4 drafts **plus** a bonus: 5 tokens per pass.

### Checking is cheap: day 12 again

```
Costs: one forward pass with a cached prompt (ms)
          gpt2, 1 new token:   9.13
   gpt2-medium, 1 new token:  21.21
   gpt2-medium, 1 tokens checked at once:  21.65
   gpt2-medium, 3 tokens checked at once:  25.48
   gpt2-medium, 5 tokens checked at once:  26.82
   gpt2-medium, 9 tokens checked at once:  29.23
```

Checking **9** tokens costs only 35% more than generating 1. The weights are
read once either way (decode is memory-bound). That's the book's sudoku: solving
is hard, checking is easy.

But the draft is **not cheap**: gpt2 costs 9 ms, **43%** of a gpt2-medium pass.
It has only 3× fewer parameters, and at this size fixed overheads (Python,
kernel launches) dominate. The book's rule of thumb is a draft **at least 10×
smaller**.

### Subject matter changes acceptance

```
Acceptance by prompt (greedy, k = 4)
   81%  'The history of the Roman Empire'
   89%  'def fibonacci(n):\n    """Return the nth Fibon'
   91%  'The best way to learn a new language is'
  100%  '1, 2, 3, 4, 5, 6, 7,'
```

Predictable text (counting, code, set phrases) is easy to draft. Open-ended
text (history) is harder.

## Results (Apple M4, Mac GPU)

```
Benchmark: 4 prompts x 64 new tokens, draft gpt2, target gpt2-medium, mps, best of 3

temp       method accepted tokens/pass  tok/s speedup same as normal
--------------------------------------------------------------------
 0.0       normal                 1.00   40.8   1.00x
 0.0        k = 1      97%        1.95   45.1   1.10x    4/4 prompts
 0.0        k = 2      96%        2.88   52.0   1.27x    4/4 prompts
 0.0        k = 3      92%        3.71   55.2   1.35x    4/4 prompts
 0.0        k = 4      90%        4.49   56.4   1.38x    4/4 prompts
 0.0        k = 6      88%        5.82   56.1   1.37x    4/4 prompts
 0.0        k = 8      83%        7.31   56.8   1.39x    4/4 prompts

 1.0       normal                 1.00   40.5   1.00x
 1.0        k = 1      71%        1.71   40.4   1.00x  n/a (sampled)
 1.0        k = 2      61%        2.21   40.1   1.01x  n/a (sampled)
 1.0        k = 3      54%        2.61   39.7   0.98x  n/a (sampled)
 1.0        k = 4      47%        2.88   37.6   0.93x  n/a (sampled)
 1.0        k = 6      35%        3.01   30.3   0.75x  n/a (sampled)
 1.0        k = 8      30%        3.24   26.7   0.66x  n/a (sampled)
```

- **Lossless:** at temperature 0, every speculative run produced **exactly** the
  same text as normal decoding (4/4 prompts for every k).
- **Greedy: up to 1.39× faster**, flattening out from k = 3. With k = 4 the big
  model runs 4.49× fewer passes, but each pass now also includes 4 draft steps
  (~36 ms): `(36 + 27) / 4.49 ≈ 14 ms` per token instead of 24.5 ms. Python
  overhead takes the rest.
- **Acceptance falls with depth:** 97% at k = 1 down to 83% at k = 8. Later
  drafts depend on all the earlier ones being right.
- **Temperature 1 kills it:**
  - acceptance drops to 71% at k = 1 and 30% at k = 8;
  - there is no speedup at all;
  - long drafts make it **slower** (0.66× at k = 8), because the work spent on
    drafts that get thrown away costs more than the passes saved.
  - This is the book's warning: higher temperature makes tokens harder to
    predict.
- **Short, high-acceptance drafts win.** The best k here is 3–4.

## Connection to earlier days

- Day 11: temperature, the main thing that controls acceptance.
- Day 12: checking 9 tokens costs about the same as making 1. That spare
  compute is what speculation spends.
- Day 13: the KV cache, here with `crop` to undo rejected drafts.
- Day 22 and 24: at big batch sizes the spare compute is gone, which is why
  engines turn speculation off under load.

## Caveats

- **Batch size 1 only.** That's where speculation shines. With a busy batch,
  the GPU has no spare compute left.
- The draft is only 3× smaller than the target, and costs 43% of a target pass.
  A better pair (e.g. a 1B draft for a 70B target) would gain far more.
- Python drives every draft step. Real engines (vLLM, TensorRT-LLM) run the
  drafting loop on the GPU, and EAGLE folds the speculator into the target's own
  forward pass.
- At temperature 1, "same as normal" can't be checked token by token: the
  output is random. The toy example checks the distribution instead.

## Things to try

1. Use `distilgpt2` as the draft (`DRAFT_NAME`). It's cheaper, but does it agree
   less often?
2. Use `gpt2-large` (774M, 3 GB) as the target. The draft becomes 6× smaller.
   Does the speedup grow?
3. Try temperature 0.7: where is the best k now?
4. **Adaptive k:** draft more when recent acceptance is high, fewer when it's
   low.
