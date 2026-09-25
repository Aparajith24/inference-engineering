# Day 9: Multi-head attention with a causal mask

Book chapter: **2.2 LLM inference mechanics (continued)**

Task: multi-head attention with a causal mask in PyTorch.

## What we built

`mha_causal.py` defines a `MultiHeadAttention` PyTorch module. It runs day 8's
attention formula several times side by side (one per **head**), each head
working on a slice of every token's vector, with a causal mask so no token can
see the future. It then glues the heads back together and mixes them with an
output projection.

Run it:

```
python3 mha_causal.py
```

It prints the shapes at each stage, one head's attention grid, how each head
splits the last token's attention, and then runs four checks.

Day 10 imports this file to build a full GPT, so keep the folder name `day-9`.

## What we learned

### The big picture

Before attention, the text has already become vectors:

```
word  --tokenizer-->  ID (a label)  --embedding-->  vector (a point in space)
```

Embeddings are points in a vector space where similar words sit close
together. Attention lets each word look at the other words and **update its
vector** with what it finds. Stacking many rounds of attention and MLP refines
the vectors, and the last token's final vector is used to predict the next
token (day 10).

On day 9, `x` is random numbers standing in for those embedding vectors. There
is no tokenizer or text yet.

### Why several heads

One head learns one way of relating words. Several heads can track several
relationships at once, for example one linking a verb to its subject and
another linking a word to the one right before it. `distilgpt2` has 12 heads
per layer.

**Heads split the numbers inside each token, not the tokens.** Every head sees
every token, just a different slice of each token's vector:

```
one token's 32 numbers:  [ 8 | 8 | 8 | 8 ]
                           h0  h1  h2  h3
```

### The five steps

1. One linear layer computes Q, K and V together (`nn.Linear(d_model, 3 * d_model)`).
2. Cut each of Q, K, V into `n_heads` slices of `d_head = d_model / n_heads`.
3. Each head runs day 8's attention on its slice, with the causal mask.
4. Glue the heads' outputs back side by side.
5. Mix them with an output projection (`out_proj`), so heads can share what they found.

### B, T, C

`B, T, C = x.shape` names the three sizes of the input:

| Letter | Meaning | Demo value |
|---|---|---|
| B | Batch: sentences processed at once | 2 |
| T | Time: tokens per sentence | 6 |
| C | Channels: numbers per token (`d_model`) | 32 |

### Shapes through the module

```
x                      (2, 6, 32)      batch, tokens, d_model
q after qkv + split    (2, 6, 32)
q after view           (2, 6, 4, 8)    cut 32 into 4 heads x 8
q after transpose      (2, 4, 6, 8)    heads moved in front of tokens
scores  q @ k^T        (2, 4, 6, 6)    one 6x6 grid per head
out     weights @ v    (2, 4, 6, 8)
after transpose        (2, 6, 4, 8)
after reshape          (2, 6, 32)      heads glued back together
after out_proj         (2, 6, 32)      same shape as the input
```

**Why transpose before `@`:** `@` multiplies only the last two dimensions and
repeats for everything in front. Moving heads forward makes the last two
dimensions `(tokens, d_head)`, so `@` compares tokens with tokens, for every
sentence and every head at once, in one line.

**Same shape in and out** is what lets day 10 stack many of these blocks.

### Parameters do not grow with heads

The four matrices (Wq, Wk, Wv, Wo) are always `d_model x d_model`, so the
count is `4 * d_model^2` whatever the number of heads. More heads just cut the
same numbers into thinner slices:

| Heads | Numbers per head | Parameters (d_model 32) |
|---|---|---|
| 1 | 32 | 4,096 |
| 4 | 8 | 4,096 |
| 8 | 4 | 4,096 |

## Checks

1. Every row of every attention grid sums to 1, and everything above the
   diagonal is 0 (no head looks at the future).
2. Our attention matches PyTorch's built-in `F.scaled_dot_product_attention`.
3. **Causality:** changing the last token changes only the last output.
4. With one head, no biases and `out_proj` set to the identity, the output
   matches **day 8's NumPy code** exactly.
5. The parameter count is exactly `4 * d_model^2`.

## Connection to earlier and later days

- Day 3: `nn.Linear` layers are the same `x @ W + b` as the MLP.
- Day 4.1: `nn.Module`, `eval()` and `no_grad()`.
- Day 8: each head runs exactly that formula.
- Day 10: this module goes inside a transformer block, with real embeddings.
- Day 13: each head's K and V are what the KV cache stores.
- Day 14: GQA and MQA make heads share K and V to shrink that cache, and
  `scaled_dot_product_attention` is the fast version we benchmark against.

## Caveats

- Weights and inputs are random, so the attention patterns carry no meaning.
  The differences between heads are small here; in a trained model they are
  distinct.
- The mask is rebuilt on every call. Real implementations cache it or let a
  fused kernel handle it.

## Things to try

1. Change `n_heads` to 8 and check the shapes and the parameter count.
2. Set `n_heads = 5` and read the error from the `assert`.
3. Remove the `.transpose(1, 2)` calls and see what breaks.
4. For `distilgpt2` (`d_model = 768`, 12 heads), work out `d_head` and the
   attention parameters per layer, then check against `4 * 768^2`.
