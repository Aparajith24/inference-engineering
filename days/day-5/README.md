# Day 5: Model selection (parameter count + precision -> memory -> GPUs)

Book chapters: **1.1 Scale**, **1.2 About your app**, **1.3 Model selection**

## What we built

`model_memory_picker.py` is a small command-line calculator. Given a model
(a Hugging Face id like `distilgpt2`, or a size like `7B`), it prints:

1. The parameter count.
2. The memory needed at fp32, fp16, bf16, int8 and int4.
3. A table of which GPUs can hold it on a single card.
4. For models that fit nowhere, how many of the biggest GPU you would need.

Run it:

```
python3 model_memory_picker.py distilgpt2   # any Hugging Face model id
python3 model_memory_picker.py 7B           # or a raw size: 125M, 1.5B, 70B, ...
python3 model_memory_picker.py              # table for common sizes
```

## What we learned

### The core formula

```
memory (GB) = parameters x bytes per parameter / 1e9
```

| Precision | Bytes per parameter | 7B model |
|---|---|---|
| fp32 | 4 | 28 GB |
| fp16 / bf16 | 2 | 14 GB |
| int8 | 1 | 7 GB |
| int4 | 0.5 | 3.5 GB |

A handy rule: **at fp16, a model needs about 2 GB per billion parameters.**

This is why quantization (day 2) and dtypes (day 4) matter for choosing
hardware. Halving the bytes halves the memory, and can change which GPU is
enough. For example a 7B model does not fit on a 16 GB T4 at fp16 once you add
overhead, but it does at int8.

### Weights are only part of the memory

Serving also needs room for:

- **KV cache:** grows with batch size and context length (day 2 caching).
- **Activations and runtime buffers.**

We add a flat 20% on top of the weights as a rough allowance. It is a floor,
not a guarantee. At long context or large batch the KV cache alone can exceed
the weights.

### Counting parameters without downloading weights

```python
config = AutoConfig.from_pretrained(model_name)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
sum(p.numel() for p in model.parameters())
```

- `AutoConfig` downloads only a small JSON file describing the architecture.
- The `meta` device creates tensors that have shapes but no memory behind them,
  so even a 70B model can be "built" instantly.
- `numel()` is the number of elements in a tensor, so summing it over
  `parameters()` gives the parameter count. This is the same idea as the
  parameter count exercise from day 3.

For `distilgpt2` this gives 81,912,576 parameters (about 82M), which is about
0.33 GB at fp32.

### Example results

| Model | fp16 weights | Fits on one GPU at fp16? |
|---|---|---|
| distilgpt2 (82M) | 0.16 GB | Every GPU in the table |
| 7B | 14 GB (16.8 with overhead) | L4, A10G, RTX 4090 and up, but not a T4 |
| 70B | 140 GB (168 with overhead) | Only B200 |

A 70B model at int4 is about 42 GB with overhead, so it fits on a single
80 GB card. This is the trade-off between quality and cost that model selection
is about.

### When one GPU is not enough

The calculator reports how many of the biggest GPU you would need. Splitting a
model across GPUs is **parallelism** (day 2). It works, but the GPUs must talk
to each other constantly, so it adds cost and complexity.

## Connection to days 1-4

- Day 1: decode is memory-bound, so a smaller model in memory also means
  faster tokens, not just a cheaper GPU.
- Day 2: quantization, caching and parallelism all show up here as memory
  numbers.
- Day 4: dtype sizes (fp32, fp16, bf16) are the "bytes per parameter" column.

## Caveats

- The 20% overhead is a rule of thumb. Measure a real workload before buying.
- GPU memory sizes are the advertised capacities. Some memory is reserved by
  the driver and runtime, so usable memory is a little lower.
- Real int4 formats also store scaling values, so they use slightly more than
  0.5 bytes per parameter.
- Fitting in memory is necessary but not sufficient. Speed also depends on
  memory bandwidth (day 1) and compute.
- This is inference only. Training needs several times more memory for
  gradients and optimizer state.
- Apple Silicon uses unified memory shared with the OS, so it is not in the
  GPU table.

## Things to try

1. Run `python3 model_memory_picker.py gpt2-large` and compare with `1B`.
2. Change `OVERHEAD` to 1.5 and see which GPUs drop out for a 7B model.
3. Add a GPU to the `GPUS` list and check how the table changes.
4. Work out by hand how much memory a 13B model needs at int8, then check the
   calculator.
