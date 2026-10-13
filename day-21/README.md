# Day 21: Frameworks — eager mode vs torch.compile

Book chapter: **4.2 Deep Learning Frameworks and Libraries** (PyTorch,
torch.compile, model file formats, ONNX Runtime and TensorRT, transformers and
diffusers)

Task: eager mode vs `torch.compile` on a transformer block. Runs on Colab.

## What we built

`torch_compile_bench.py` builds 4 transformer blocks (like day 10's `Block`,
GPT-2 sized: d_model 768, 12 heads, 28M parameters) and runs them three ways:

| Mode | What it does |
|---|---|
| **eager** | normal PyTorch: one operation at a time, one kernel each |
| **default** (`torch.compile(model)`) | reads the whole model first, fuses small operations, picks kernels |
| **reduce-overhead** (`torch.compile(model, mode="reduce-overhead")`) | the same, plus a **CUDA graph**: records all the kernels once and replays them with one launch |

For each mode it measures:

- **first call:** how long the first run takes (this is when compiling happens),
- **per call:** average milliseconds per forward pass afterwards,
- **speedup** vs eager,
- **kernels:** how many GPU kernels one forward pass launches (fusion lowers this),
- **max diff:** the biggest difference from eager's output (should be tiny).

It does this for two input sizes:

- **small:** 8 sequences x 1 token, like a decode step: very little work per kernel,
- **large:** 8 sequences x 512 tokens, like prefill: lots of work per kernel.

## How to run it on Colab

1. colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. Upload `torch_compile_bench.py` (folder icon on the left).
4. Run `!python torch_compile_bench.py`. Expect 1-3 minutes; most of it is
   compiling.

It also runs on a Mac CPU (fp32, no kernel counts), which is how it was tested.
There, compile gave only about 1.1x on the small input and nothing on the large
one: the interesting numbers need an NVIDIA GPU.

## What we learned

### The software stack

```
4. Dynamo            run inference engines across many GPUs
3. Inference engine  vLLM, SGLang, TensorRT-LLM: serve models to users
2. Framework         PyTorch: write x @ W + b in Python          ← this day
1. CUDA              kernels: code that runs on the GPU           ← day 20
   GPU hardware
```

Most inference engineering happens at layers 3 and 4, but you need to
understand the layer below the one you work at.

### Frameworks

- **PyTorch** is the clear leader (made at Meta, now in the Linux Foundation),
  for both training and inference.
- **TensorFlow** (Google) was big in the 2010s and is fading.
- **JAX** (Google research) is cleaner but has "sharp edges".

**Autograd** records every operation so PyTorch can compute gradients for
training. Inference does not train, so we switch it off with
`torch.no_grad()` and `model.eval()`, as on days 4.2 and 10.

### Eager mode vs torch.compile

**Eager** runs your Python line by line. Each operation (LayerNorm, matmul,
add, GELU, ...) launches its own kernel, and each kernel reads its input from
VRAM and writes its output back.

**`torch.compile`** first reads the whole model, like a chef reading the whole
recipe before cooking. Then it:

1. **fuses** chains of small operations into one kernel (often generating
   Triton kernels like day 20's), so data makes fewer trips to memory,
2. **picks kernels** suited to your GPU,
3. with `mode="reduce-overhead"`, records everything into a **CUDA graph**, so
   one launch replays the whole forward pass.

```
eager:     LayerNorm | matmul | add bias | GELU | matmul | add | ...   many kernels, many memory trips
compiled:  LayerNorm+... | matmul | bias+GELU fused | ...               fewer kernels, fewer memory trips
```

The cost: the **first call is slow** (seconds), because that is when compiling
happens. Compiled code is cached, so a later compile of the same model can be
quicker.

### Why the input size changes the result

- **Small input (like decode):** each kernel has very little work, so the time
  goes mostly into **launching kernels** (a few microseconds each). Fewer
  kernels (fusion) and one launch for everything (CUDA graph) help a lot.
- **Large input (like prefill):** the big matmuls take most of the time, and
  they are cuBLAS kernels that compile cannot make faster. So the speedup is
  small.

### What compile cannot fuse

**Plugin kernels** are hand-written kernels from outside PyTorch's compiler:
cuBLAS matmuls, FlashAttention (here, inside `scaled_dot_product_attention`),
DeepGEMM. They are like **sealed jars**: compile can call them but cannot open
them and merge them with their neighbours. It can only fuse the small steps
between them (LayerNorm, adds, GELU). LLMs spend most of their time inside
those jars, which is why the book says compile is less useful for LLMs and
more useful for unusual architectures and long chains of small operations.

Compile can also **break** on Python tricks (an `if` on a tensor's value,
printing inside the model). Then you rewrite that part to be simpler.

### Model file formats

| Format | What is inside | Notes |
|---|---|---|
| **safetensors** | only the weights (numbers + names, shapes, dtypes) | safe and fast to load; the standard |
| pickle (`.bin`) | weights, but can also hold code that runs when loaded | unsafe: a bad file can run anything |
| **ONNX** | weights **plus** the model's graph | portable; used by ONNX Runtime and TensorRT |
| GGUF | weights (often quantized) + tokenizer | llama.cpp and Ollama (day 19) |

- safetensors is **safe** because it can only hold numbers: loading can never
  run code.
- It uses **memory mapping**: a file can be used without first copying all of
  it into RAM, so big models load faster.
- safetensors has no recipe. The structure comes from **code** (like day 10's
  `TinyGPT`) plus **`config.json`** (layers, heads, sizes). Day 10's
  `load_gpt2_weights` did exactly this: numbers from the file, shape from our
  code.

### ONNX Runtime and TensorRT

| | ONNX Runtime | TensorRT |
|---|---|---|
| Made by | open (Linux Foundation) | NVIDIA |
| Runs on | many kinds of hardware | NVIDIA only |
| Strength | portable | very fast; strong for image and video models |

Both run a model **exported** from PyTorch, but exporting does not support
everything (DeepSeek's MLA attention is hard to export). For popular LLMs,
people now skip exporting and serve safetensors weights with an inference
engine (vLLM, SGLang, TensorRT-LLM) directly.

### transformers and diffusers

Hugging Face's libraries are **reference implementations**: great for learning,
checking outputs and reading a model's exact inputs and outputs (we used
`transformers` on days 1, 6 and 10), but not built for production. They lack
cross-user batching, paged KV cache and the fastest kernels. That is the job of
inference engines (chapter 4.3).

## Results (Colab T4)

Paste your output here after running it. What to expect:

- **Small input:** eager launches dozens of kernels per pass and is held back
  by launch overhead. `default` should launch fewer kernels and be faster;
  `reduce-overhead` (CUDA graph) should be the fastest, often several times
  faster than eager.
- **Large input:** small speedups, because cuBLAS matmuls (sealed jars) take
  most of the time.
- **First call:** tens of seconds for compiled modes, versus well under a
  second for eager.
- **max diff:** small (fp16 rounding), never large.

## Connection to earlier days

- Day 4.2 and 10: `nn.Module`, `eval()`, `no_grad()`; the same Block design.
- Day 12: small inputs do little work per byte; here they are even below that,
  limited by kernel launches.
- Day 14: `scaled_dot_product_attention` is a plugin kernel compile cannot open.
- Day 19: the smallest models reached the lowest share of the ideal speed,
  partly because of per-kernel overhead.
- Day 20: compile generates fused Triton kernels like the ones we wrote.

## Caveats

- Random weights: speed does not depend on the weight values.
- No KV cache, so the "decode-like" input is only decode-sized work, not real
  decoding.
- `reduce-overhead` gets no kernel count, because a CUDA graph launches its
  kernels as one replay.
- Compile results depend on the PyTorch version.

## Things to try

1. Change `N_LAYERS` to 12. Does the compile time grow? Does the speedup stay?
2. Add a `("tiny", 1, 1)` shape. How big is the CUDA graph speedup now?
3. Add `mode="max-autotune"` to `MODES`: it tries many kernels and keeps the
   fastest. How long does it compile, and is it faster?
4. Replace `scaled_dot_product_attention` with day 9's step-by-step attention.
   Now compile can see inside attention: does the large-input speedup grow?
