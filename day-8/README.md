# Day 8: LLM inference mechanics — single-head self-attention

Book chapter: **2.2 LLM inference mechanics**

Task: single-head self-attention from scratch in NumPy.

## What we built

`attention_numpy.py` implements single-head self-attention in pure NumPy:
softmax, the Q/K/V projections, the score matrix, an optional causal mask,
and the weighted sum of values. It also includes a slow loop version to check
the matrix version against, and a small demo of how cost grows with sequence
length.

Run it:

```
python3 attention_numpy.py
```

## What we learned

### The idea

Each token looks at every token (including itself), decides how much
attention to pay to each one, and blends their information into a new vector.
This is the core operation inside every transformer, including `distilgpt2`.

### The five steps

```python
Q = x @ Wq                         # what each token is looking for
K = x @ Wk                         # what each token offers to be matched against
V = x @ Wv                         # the information handed over if attended to
scores = Q @ K.T / np.sqrt(d_k)    # match every token against every token
weights = softmax(scores)          # turn scores into percentages (rows sum to 1)
out = weights @ V                  # blend the V rows using those percentages
```

- **Query, Key, Value** work like a library lookup: the query is the question,
  the key is the label on each shelf, the value is the content you get back.
- `Wq`, `Wk`, `Wv` are learned weight matrices, the same kind as `W1` and `W2`
  from day 3. Each projection is just a matmul. In our demo they are random,
  so the attention pattern means nothing. The mechanics are the point.

### Shapes

```
x        (T, d_model)   one row per token
Wq/Wk/Wv (d_model, d_k)
Q, K, V  (T, d_k)
scores   (T, T)         how well token i matches token j
weights  (T, T)         after softmax, each row sums to 1
output   (T, d_k)       one blended vector per token
```

The `(T, T)` matrix is the heart of it: row *i* says how much token *i*
attends to every token.

### Details worth remembering

- **Softmax** turns any scores into positive numbers that sum to 1. We
  subtract the max first so `exp()` cannot overflow. This does not change the
  result.
- **Dividing by `sqrt(d_k)`** stops the scores from growing with the vector
  size. Without it, softmax saturates (one weight near 1, the rest near 0)
  and learning stalls.
- **The causal mask** sets scores for later tokens to `-inf`, so their weight
  becomes exactly 0. The weight matrix becomes a lower triangle, and token 0
  attends only to itself (weight 1.0). This is how a language model avoids
  peeking at the future when predicting the next token.

### How we checked it

- The loop version and the matrix version give identical numbers.
- Every row of the weights sums to 1.
- With the causal mask, changing the last token changes only the last output,
  never the earlier ones.

## Why this matters for inference

The score matrix is `T x T`, so cost grows with the **square** of the
sequence length:

| T (tokens) | Scores (T*T) | Time |
|---|---|---|
| 256 | 65,536 | 0.4 ms |
| 1,024 | 1,048,576 | 5.8 ms |
| 4,096 | 16,777,216 | 168 ms |

Quadrupling the length makes the work about 16x bigger. This is why long
contexts are expensive, and why **prefill** (processing the whole prompt at
once, day 1) is compute-heavy.

It also explains the **KV cache** from day 2. During decode, the K and V of
earlier tokens never change, so we save them instead of recomputing. Each new
token computes only its own Q, K and V, and attends to the cached K and V.
That turns each decode step from a full `T x T` computation into a single
row. The cost is memory: the cache grows with context length, which is part
of the overhead we allowed for in day 5.

## Connection to days 1-7

- Day 1: prefill versus decode. Attention is where the two phases differ most.
- Day 2: the KV cache and long-context cost, from the toolbox.
- Day 3: `Wq`, `Wk`, `Wv` are the same kind of weight matrices as the MLP's.
- Day 5: the KV cache is part of the memory overhead.

## Caveats

- Weights are random and untrained, so the attention patterns carry no
  meaning.
- This is single-head. Real models use many heads in parallel (day 9) and
  project the result back with an output matrix, which we skip here.
- The timing table is rough: one run per size, float32 on CPU, and the
  4,096 row is affected by memory traffic as well as arithmetic.
- NumPy 2.0 with Apple's Accelerate BLAS prints spurious floating-point
  warnings (`divide by zero`, `overflow`) on larger matmuls. We checked the
  values are finite and match an independent computation to about 1e-16, so
  the warnings are false alarms. The script silences them only in the timing
  demo and asserts the outputs are finite.

## Things to try

1. Set `Wq` to all zeros and check that the weights become uniform (`1/T`
   everywhere). With no preference, every token gets equal attention.
2. Remove the `/ np.sqrt(d_k)` and use a large `d_k` (say 512). Watch the
   weights collapse toward one-hot.
3. Print `weights_c[3]` from the masked demo and explain each number in words.
4. Time the loop version against the matrix version for T=200.
