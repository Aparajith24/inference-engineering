# Day 10: Tiny GPT with a greedy generation loop

Book chapter: **2.2 LLM inference mechanics (continued)**

Task: tiny GPT (embedding, blocks, LM head) with a greedy generation loop.

## What we built

`tiny_gpt_generate.py` builds a complete GPT-style language model around day
9's attention, and a loop that makes it write text:

```
"The future of AI inference is"
   | tokenizer
   v
[464, 2003, 286, 9552, 32278, 318]       IDs (labels)
   | tok_emb + pos_emb
   v
6 points in vector space
   | Block x N: attention + MLP           (day 9 + day 3)
   v
6 context-aware vectors
   | take the LAST one, LM head
   v
50,257 scores -> pick the highest -> append -> repeat
```

Run it:

```
python3 tiny_gpt_generate.py
```

It runs two experiments:

1. **A tiny random model** (64 numbers per token, 4 heads, 2 blocks), printing
   the shape at every stage. It is untrained, so it writes gibberish.
2. **The same `TinyGPT` class with `distilgpt2`'s trained weights copied in.**
   It matches Hugging Face's model exactly and writes real text.

This file imports `MultiHeadAttention` from `day-9`, and day 11 imports this
file, so keep the folder names `day-9` and `day-10`.

## What we learned

### The pieces of a GPT

| Piece | Code | What it does |
|---|---|---|
| Word table | `tok_emb = nn.Embedding(vocab_size, d_model)` | ID -> a point in space (a row lookup) |
| Position table | `pos_emb = nn.Embedding(max_len, d_model)` | Adds "where in the sentence", so word order matters |
| Block | `Block` | Attention (words talk) + MLP (each word thinks) |
| Stack | `nn.ModuleList` of blocks | Each block refines the vectors |
| Final norm | `ln_f` | One last tidy-up |
| LM head | `lm_head` | Final vector -> one score per word in the vocabulary |

`x = tok_emb(idx) + pos_emb(pos)` is the real version of the `x` that day 9
filled with random numbers.

### Inside a block: two new ideas

```python
x = x + self.attn(self.ln1(x))   # words exchange information
x = x + self.mlp(self.ln2(x))    # each word processes what it now knows
```

- **LayerNorm** rescales each token's numbers to a standard range, so they do
  not blow up or vanish as they pass through many layers.
- **The residual add** (`x = x + ...`) keeps the original vector and adds what
  the layer found, instead of replacing it. "cat" stays recognizably "cat" but
  picks up context like "big" and "running". This also keeps deep models
  stable.

The MLP is day 3's, expanding 4x in the middle (768 -> 3,072 -> 768) with a
GELU activation, a smooth version of ReLU. Blocks keep the shape
`(B, T, d_model)`, so they stack like Lego.

### Weight tying: one table used both ways

```python
self.lm_head.weight = self.tok_emb.weight
```

The LM head reuses the word table. It scores each word by how close the final
vector is to that word's point, so the same table maps ID -> point at the
start and point -> "which word is nearest?" at the end.

### The generation loop

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -model.max_len:]           # stay inside the context window
    logits = model(idx_cond)[:, -1, :]           # only the LAST position predicts the next token
    next_id = logits.argmax(dim=-1, keepdim=True)  # greedy: the top-scoring token
    idx = torch.cat([idx, next_id], dim=1)       # append and repeat
```

This is the autoregressive loop from chapter 2.2: predict, append, repeat.
The model produces scores for every position, but only the last one is about
what comes next.

### Loading trained weights

`load_gpt2_weights` reads `distilgpt2`'s `config.json` to build a `TinyGPT` of
the same shape (50,257 words, 1,024 positions, 768 numbers, 12 heads, 6
layers), then copies every weight across, translating names:

| Ours | Hugging Face GPT-2 |
|---|---|
| `tok_emb`, `pos_emb` | `wte`, `wpe` |
| `ln1`, `ln2`, `ln_f` | `ln_1`, `ln_2`, `ln_f` |
| `attn.qkv`, `attn.out_proj` | `attn.c_attn`, `attn.c_proj` |
| `mlp.fc`, `mlp.proj` | `mlp.c_fc`, `mlp.c_proj` |

GPT-2 stores its matrices flipped compared with `nn.Linear`, so linear
weights are transposed (`.T`) when copied.

## Results

**Tiny random model:**

```
embeddings    (1, 6, 64)
after block 0 (1, 6, 64)
after block 1 (1, 6, 64)
logits        (1, 6, 50257)
generated: ' is is is is is is ispainpainpain'
```

97% of its 3.3M parameters are the word table (50,257 x 64). In big models
the balance flips and the blocks dominate.

**With `distilgpt2`'s weights:**

| Check | Result |
|---|---|
| Parameter count | 81,912,576, identical to Hugging Face |
| Largest logit difference | 0.00008 (rounding only) |
| 30 greedy tokens | Identical to Hugging Face's `generate()`, token for token |

```
The future of AI inference is not yet clear.
(followed by repeated newlines)
```

That is exactly day 1's output, including greedy decoding's newline loop.

## Key takeaways

### The code is the shape; the weights are the knowledge

The same `TinyGPT` code wrote gibberish with random weights and real text with
trained ones. A model is an architecture plus trained numbers. This is why:

- One architecture can serve many models (sizes, base/instruct, fine-tunes),
  and an engine built for it runs all of them (chapter 2.2.1).
- What you download and store is mostly the weights, which is what day 5's
  memory calculator measures.
- Inference engineering runs already-trained weights; creating them is
  training, a separate job.

### No KV cache, so it is slower

Our loop reruns the whole sentence every step. Over 30 steps from a 6-token
prompt it pushes about 615 tokens through the model when only 30 are new.
Hugging Face's `generate()` keeps earlier tokens' K and V and runs only the new
token each step.

| | Tokens/s |
|---|---|
| Ours, no KV cache | about 87 |
| Hugging Face (day 1), with KV cache | about 150 |

The gap grows with length (quadratic versus linear total work). The cache
changes only speed, never the output. Day 13 adds it to this model.

## Connection to earlier days

- Day 1: the same output, now from a model we built ourselves.
- Day 3: the MLP inside every block.
- Day 5: the same 81,912,576 parameters we counted from the config.
- Day 8 and 9: the attention inside every block.
- Chapter 2.2: embedding layer, transformer blocks, LM head, autoregressive loop.

## Caveats

- Greedy decoding only; day 11 adds temperature, top-k and top-p.
- The loop assumes a batch of one when stopping on an end token.
- No KV cache, no fused attention, CPU only: this is for understanding, not
  speed.

## Things to try

1. Change the prompt and compare our output with Hugging Face's.
2. Load `gpt2` instead of `distilgpt2` (12 layers) and check the parameter count.
3. Remove the `.T` in `copy_linear` and read the error.
4. Time 100 tokens instead of 30 and see how tokens/s drops as the sentence grows.
5. Replace one block's residual (`x = x + ...`) with `x = ...` after loading the
   trained weights, and see how the output breaks.
