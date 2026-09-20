# Day 4.1: The day 3 MLP as a PyTorch nn.Module

Follow-up to day 3 and day 4: rewrite the NumPy MLP with `torch.nn` and benchmark it
across sizes and dtypes (fp32/fp16/bf16) on CPU and MPS.

## What we built

`mlp_bench.py` builds the day 3 two-layer MLP as a PyTorch `nn.Module` and
times its forward pass for layer widths N from 128 to 4096, in fp32, fp16 and
bf16, on CPU and on Apple's GPU (MPS). It reports time per forward pass and
TFLOPS, and saves a plot to `mlp_bench.png`.

Run it:

```
pip3 install matplotlib   # once, for the plot
python3 mlp_bench.py
```

It takes about 15 seconds. CPU is capped at N=2048 because CPU half-precision
is slow.

## What we learned

### Tensors

A tensor is a grid of numbers, the basic PyTorch data type. It is like a NumPy
array that can also live on a GPU and track gradients. Inputs, weights and
outputs of a model are all tensors.

### dtype

The dtype is how each number is stored.

| dtype | Bits | Exponent / mantissa bits | Trade-off |
|---|---|---|---|
| fp32 | 32 | 8 / 23 | Accurate, the default |
| fp16 | 16 | 5 / 10 | Half the memory, small range (can overflow) |
| bf16 | 16 | 8 / 7 | Half the memory, fp32's range, coarser precision |

Fewer bits means fewer bytes to read from memory, which matters because decode
is memory-bound (day 1). This is the simplest form of quantization (day 2).

### Device

The device is where a tensor lives and where math on it runs: `cpu`, `cuda`
(NVIDIA GPU) or `mps` (Apple GPU). Tensors must be on the same device to be
combined.

### torch.nn: the same MLP as a module

Day 3 wrote the MLP by hand with NumPy arrays. `torch.nn` packages the same
ideas:

| Day 3 (NumPy) | Day 4 (`torch.nn`) |
|---|---|
| `x @ W + b` written by hand | `nn.Linear(in, out)`, which owns `W` and `b` |
| `np.maximum(0, x)` | `nn.ReLU()` |
| `mlp_forward(x, W1, b1, W2, b2)` | `TwoLayerMLP.forward(x)` |
| `init_params(...)` with random numbers | `nn.Linear` initializes its own weights |
| Passing weights around as arguments | Weights live inside the module |

```python
class TwoLayerMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

- `__init__` declares the layers. Assigning an `nn.Module` to `self` registers
  its parameters automatically.
- `forward` says how data flows through them. Call the model like a function,
  `model(x)`, and PyTorch runs `forward`.
- `model.to(device=..., dtype=...)` moves and converts **every parameter at
  once**. That is how one line switches the whole model to MPS or fp16.
- `model.eval()` selects inference mode, and `torch.no_grad()` skips gradient
  tracking, since we only run forward.

### Benchmarking a GPU correctly

- **Synchronize.** GPU calls are asynchronous: `model(x)` only submits the
  job. Call `torch.mps.synchronize()` before reading the clock, or you time
  the submission instead of the math.
- **Warm up.** The first calls include one-time setup cost, so run a few
  untimed passes first.
- **Average.** Time several runs and divide, to smooth out noise.
- **TFLOPS.** Each layer is a matmul of about `2 * N^3` operations, and there
  are two layers, so `4 * N^3 / seconds / 1e12` gives throughput independent
  of size. Bias and ReLU are cheap and ignored.

## Results (Apple M4)

| Device | Size | fp32 | fp16 | bf16 |
|---|---|---|---|---|
| CPU | 2048 | 21 ms | 172 ms | 242 ms |
| MPS | 4096 | 134 ms | 110 ms | 110 ms |

1. **On CPU, fp16/bf16 are 8-12x slower than fp32.** The CPU has fast fp32
   paths but no fast half-precision path. Lower precision is not automatically
   faster; it depends on hardware support.
2. **On MPS, fp16/bf16 are about 20% faster than fp32** at large sizes. Many
   NVIDIA GPUs gain 2-8x from dedicated half-precision hardware.
3. **Small workloads underuse the GPU.** MPS reaches about 0.05 TFLOPS at
   N=128 and about 2.5 at N=4096. Overhead dominates small jobs, which is why
   batching helps.
4. **Time grows about 8x each time N doubles**, because the work is `N^3`.
5. **fp16 and bf16 run at about the same speed on MPS.** Both are 16 bits.
   They differ in range and precision, not speed.
6. **CPU beats the GPU at small sizes**, and the GPU wins only when the work
   is big enough.

Note: `day-4/matmul_bench.py` times a bare `a @ b` and sees CPU
fp16 at about 500x slower than fp32. Going through `nn.Linear` gave a much
smaller gap (about 10x), because it calls a different PyTorch kernel. The
same math can take different code paths, so benchmark the code you will
actually run.

## Connection to days 1-3

- Day 3 did the same forward pass in NumPy (CPU, float64). Here it is an
  `nn.Module` timed across device and dtype.
- Matmul is where nearly all of a model's compute goes, so its speed is
  model speed. `distilgpt2` from day 1 is built from `nn.Linear` layers like
  these.
- Device, dtype and matrix size are the three levers that the day 2
  techniques (batching, quantization) pull.

## Caveats

- Numbers vary a little between runs, and other apps on the machine affect
  them.
- This measures speed only, not accuracy. fp16 and bf16 lose precision, so
  results differ slightly from fp32.
- The CPU x-axis shows sizes up to 4096 but the CPU has no data past 2048.
- The weights are random and untrained. Speed does not depend on their values.

## Things to try

1. Set `ITERS = 1` and see how much noisier the numbers get.
2. Comment out `sync(device)` in `bench_mlp` on MPS. The times look
   impossibly fast, which shows the async behavior.
3. Print the parameter count: `sum(p.numel() for p in model.parameters())`.
   For N=1024 it is 2 * (1024 * 1024 + 1024).
4. Measure the accuracy cost:
   ```python
   model = TwoLayerMLP(256); x = torch.randn(8, 256)
   exact = model(x)
   approx = model.bfloat16()(x.bfloat16()).float()
   print((exact - approx).abs().max())
   ```
   Repeat with `.half()` and compare.
