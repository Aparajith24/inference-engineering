# Day 28: Quantization — int8 per-tensor vs per-channel

Book chapter: **5.1 Quantization**

Task: int8 quantization of a linear layer, per-tensor vs per-channel, with the
error measured.

## What we built

`quantize_int8.py` has four parts:

1. **Toy examples:** four numbers with one outlier, and a 2×4 matrix with one
   small row and one big row, quantized per-tensor and per-channel.
2. **Every linear layer of distilgpt2** (6 blocks × 4 layers = 24), quantized
   both ways. For each layer, it measures the error in the **weights** and in
   the layer's **output**, using the real inputs that layer sees when the model
   reads a paragraph of text.
3. **The whole model:** all 24 layers quantized at once (weight-only int8). It
   compares the model's predictions with the original fp32 model: logit
   error, how often the top-1 next token is the same, and a greedy
   continuation.
4. **Memory:** the bytes for the linear weights in each format.

```
python3 days/day-28/quantize_int8.py
```

It runs on the CPU in about 10 s and uses well under 1 GB.

## What we learned

### Symmetric int8 in three lines

```python
scale = w.abs().max() / 127
q = round(w / scale).clamp(-127, 127)
w_back = q * scale
```

The biggest absolute value maps to 127, and everything else is scaled by the
same factor and rounded. `scale` is the **scale factor** from the book: the
multiplier that takes int8 back to float. **All the error is rounding.**

### Per-tensor vs per-channel

- **Per-tensor:** **one** scale for the whole matrix. One outlier anywhere
  stretches the ruler for everybody.
- **Per-channel:** one scale per **output channel**, i.e. per column of the
  weight (GPT-2 stores weights as `(in, out)`). Each channel gets a ruler that
  fits its own values.

```
Four numbers with one outlier: [0.1, 0.5, -2.0, 30.0]
  scale = 30 / 127 = 0.2362
  int8 values:   [0, 2, -8, 127]
  back to float: [0.0, 0.472, -1.89, 30.0]  <- 0.1 became 0

A 2x4 weight matrix: row 0 is small, row 1 has big values
  per-tensor : row 0 int8 = [1, -3, 1, 2], back = [0.071, -0.213, 0.071, 0.142], row 0 error 14.2%
  per-channel: row 0 int8 = [64, -127, 32, 95], back = [0.101, -0.2, 0.05, 0.15], row 0 error 0.4%
```

With one shared scale, the small row uses only 4 of the 255 int8 levels. With
its own scale, it uses all of them.

### The cost of granularity

Per-channel stores one more number per channel. For distilgpt2 that's 41,472
scales, **162 KB** on top of 40.5 MB: almost nothing. This is the book's
"more granularity = more scale factors to store and apply", at its cheapest.

### Why measure the output, not only the weights

What matters is what the layer **produces**. Weight error and output error
don't always match:

- In `h1.mlp.c_proj`, per-tensor has 36% weight error but only 6% output
  error. The badly rounded weights happen to meet small inputs.
- In `h2.mlp.c_proj`, 12% weight error becomes 18% output error.

## Results (Apple M4, CPU)

### Every linear layer

```
           layer       shape outlier ratio            weight err            output err
                                               tensor    channel     tensor    channel
--------------------------------------------------------------------------------------
  h0.attn.c_attn (768, 2304)           3.5       2.69       0.88       1.60       0.53
  h0.attn.c_proj  (768, 768)          23.3       4.75       0.95       1.97       0.39
     h0.mlp.c_fc (768, 3072)          10.1       8.45       0.96       3.72       0.42
   h0.mlp.c_proj (3072, 768)          20.0      19.33       1.99      18.56       3.12
  h1.attn.c_attn (768, 2304)           3.6       2.68       0.87       1.81       0.60
  h1.attn.c_proj  (768, 768)           7.5       6.46       1.00       6.89       1.06
     h1.mlp.c_fc (768, 3072)          25.2      18.71       0.89       9.68       0.49
   h1.mlp.c_proj (3072, 768)          35.9      36.10       1.94       5.96       0.34
  h2.attn.c_attn (768, 2304)           7.2       5.39       0.94       3.39       0.60
  h2.attn.c_proj  (768, 768)           5.5       4.64       0.90       6.51       1.28
     h2.mlp.c_fc (768, 3072)           5.2       3.90       0.80       2.26       0.46
   h2.mlp.c_proj (3072, 768)          12.2      11.98       1.35      17.64       1.98
  h3.attn.c_attn (768, 2304)           3.9       3.00       0.84       2.60       0.71
  h3.attn.c_proj  (768, 768)           5.7       4.68       0.89       4.60       0.85
     h3.mlp.c_fc (768, 3072)           2.9       2.16       0.77       1.20       0.43
   h3.mlp.c_proj (3072, 768)           8.5       8.72       1.17      12.17       1.62
  h4.attn.c_attn (768, 2304)           4.5       3.51       0.85       3.10       0.75
  h4.attn.c_proj  (768, 768)           4.2       3.26       0.81       3.03       0.75
     h4.mlp.c_fc (768, 3072)           5.7       4.27       0.79       2.37       0.44
   h4.mlp.c_proj (3072, 768)           7.9       8.18       1.25       7.96       1.26
  h5.attn.c_attn (768, 2304)           5.0       3.82       0.91       3.51       0.83
  h5.attn.c_proj  (768, 768)          15.0      10.46       0.95       1.98       0.19
     h5.mlp.c_fc (768, 3072)           4.6       3.48       0.82       1.83       0.43
   h5.mlp.c_proj (3072, 768)           8.8       9.59       1.78      12.02       2.25

average output error: per-tensor 5.68%, per-channel 0.91%
outlier ratio = biggest channel's max / the median channel's max
```

- **Per-channel wins in every one of the 24 layers.** The average output error
  falls from **5.68% to 0.91%**, about 6× less.
- **The outlier ratio predicts the damage.** In layers where one channel's
  biggest value is 20–36× the median channel's, per-tensor weight error reaches
  19–36%. Per-channel stays at 1–2% no matter what, because each channel gets
  its own scale.
- **The `mlp.c_proj` layers are the hardest** (the MLP's output layer, 3,072
  inputs). They have the biggest errors both ways. This is the book's point
  that some layers are more sensitive and may be left in higher precision.

### The whole model

```
     weights logit err same top-1   greedy continuation
------------------------------------------------------------------------------------------
        fp32     0.00%     100.0%   ' is a fascinating one. It is a fascinating one. It'
  per-tensor     5.17%      78.7%   ' is a great example of how to make a computer programmable'
 per-channel     0.53%      98.9%   ' is a fascinating one. It is a fascinating one. It'
```

- **Per-tensor changed the model.** It picks a different next token at 21% of
  positions and writes a different sentence. That's a quality change users
  would notice.
- **Per-channel** agrees with fp32 at 98.9% of positions and gives the same
  greedy answer.

### Memory

```
Linear weights: 42.5M parameters
  fp32:  162.0 MB
  fp16:   81.0 MB
  int8 per-tensor:    40.5 MB (one 4-byte scale per layer)
  int8 per-channel:   40.7 MB (41472 scales, 162 KB extra)
```

Half of fp16. By day 17's rule (decode TPS ≈ bandwidth / bytes), that means up
to 2× decode speed. Day 29 measures it.

## Connection to earlier days

- Day 4.1: fp32, fp16 and bf16, the native formats. int8 is the next step down.
- Day 10: the 24 linear layers quantized here are the ones we loaded into our
  own GPT.
- Day 17 and 19: halving the bytes is where quantization's decode speedup comes
  from.

## Caveats

- This is **"fake" quantization**: we quantize, then turn the values straight
  back into float32 and run the normal model. The math (and so the error) is
  exactly what an int8 kernel would compute, but nothing gets faster. Day 29
  times real int8 kernels.
- **Weight-only:** activations stay in float. W8A8 (int8 activations too) is
  harder, because activations have even bigger outliers.
- Embeddings and the output layer (tied in GPT-2) are not quantized, as the
  book suggests for first and last layers.
- One paragraph of text is a small test. Day 29 uses perplexity.

## Things to try

1. Quantize the activations (the layer inputs) per-tensor and per-token, and
   look at the error.
2. Leave only the six `mlp.c_proj` layers in fp32 with per-tensor everywhere
   else. How much does the whole-model error drop?
3. Change `QMAX` to 7 (that's 4-bit) and rerun. Then see day 29.
