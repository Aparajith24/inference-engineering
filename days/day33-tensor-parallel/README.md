# Day 33: a tensor-parallel MLP on two CPU processes

Book chapter: **5.4 Model parallelism**

Task: split one linear layer across two processes with `torch.distributed`
(backend `gloo`), check that the answer is the same as on one process, and
measure what the split costs.

## What we built

`tp_gloo.py` runs 2 processes with `torch.multiprocessing.spawn`, a
`file://` rendezvous and the gloo backend, then splits one MLP
(`2048 -> 8192 -> 2048`, GELU) the way Megatron-LM does.

- **`tp_mlp`**, the real split. The up-projection `W1 [inter, hidden]` is cut
  along its output rows, so each rank computes `[batch, inter/2]` on a full
  copy of the input: **no communication**. The down-projection
  `W2 [hidden, inter]` is cut along its input columns, so each rank computes a
  **partial sum** `[batch, hidden]` and one `all_reduce` adds the two halves.
  One collective per layer.
- **`naive_mlp`**, the split that looks symmetric and is not. Both weights are
  cut along their input dim, which needs an `all_reduce` on the wide
  intermediate (`inter`) *and* an `all_gather` on the output.
- **`benchmark_collectives`**, a measured `t = latency + bytes / bandwidth`
  fit for gloo on this machine, so the projections are anchored to something.

Five sections: the split, correctness against a single process, collective
cost, communication bytes against weight bytes, and end-to-end timing.

```
python3 days/day33-tensor-parallel/tp_gloo.py
```

It takes about 7 seconds, uses 2 processes and 1 thread each, and needs only
PyTorch. `--hidden`, `--inter`, `--batches` change the shape.

## What we learned

### The split direction decides what travels, and the two directions are not symmetric

Only one of them is free:

| | W1 | W2 | per forward |
|---|---|---|---|
| fused (Megatron) | column: `[inter/2, hidden]` | row: `[hidden, inter/2]` | 1 all-reduce of `hidden` |
| naive | row: `[inter, hidden/2]` | column: `[hidden/2, inter]` | 1 all-reduce of `inter` + 1 all-gather |

```
bytes moved per forward (batch 3, fp32): fused 24 kB, naive 108 kB (4.5x)
```

The ratio is `inter/hidden = 4`. The intermediate is wide — 4x the hidden size
in a SwiGLU MLP — so communicating it instead of the residual stream is exactly
the mistake the column-then-row order avoids. The fused order only works
because the loss of `W1`'s output layout is compensated by `W2` accepting a
sharded input: **the shard boundary is invisible to the next layer**, which is
why TP can be applied per block without a single all-gather in the whole model.

### It is numerically the same, and the ranks really do different work

```
single process vs TP:   max |diff| = 6.79e-06  (2.90e-06 relative)
rank 0's partial vs rank 1's partial, before the all-reduce: max |diff| = 2.51
```

The residual difference is fp32 summation order, nothing else: each rank adds
`inter/2` products where the single process adds `inter`. The 2.51 before the
all-reduce is the proof the work was split — if both ranks had computed the
full layer, that number would be 0 and the all-reduce would be a no-op.

### Per-rank memory halves, and nothing is duplicated

```
weights: 134.2 MB on one process, 67.1 MB per rank, 134.2 MB total
```

`W1` loses half its rows, `W2` half its columns, and the total across ranks
equals the original. That is the entire reason TP exists: 2 GPUs now hold a
model that does not fit on 1, and each reads half the weight bytes per step.
The cost is that the *activations* are replicated: every rank needs the full
residual stream `[batch, hidden]` in and out.

### What an all-reduce costs: a fixed 88 us, then 6.4 GB/s

```
    payload  ring bytes/rank   all-reduce   effective
        4 kB              4 kB       99.1 us      0.04 GB/s
       64 kB             64 kB      101.0 us      0.65 GB/s
     1024 kB           1024 kB      221.1 us      4.74 GB/s
    16384 kB          16384 kB     2706.0 us      6.20 GB/s

  fit: 88.0 us fixed + 6.41 GB/s per rank
```

Two regimes, and the first is the one that matters for decode:

- **Latency-bound.** 4 kB and 64 kB cost the same 100 us. Nothing about the
  bandwidth matters below ~256 kB; it is a round trip through TCP.
- **Bandwidth-bound.** From 1 MB on, time is linear in bytes at ~6.4 GB/s,
  which is a loopback socket, not a fabric.

The same fit for NVLink 5 (900 GB/s per GPU, 5 us) puts a 64 kB all-reduce at
**5.1 us**. That is a 20x difference in the constant, and it is why TP is a
single-node technique: the collective is per layer, so the constant is paid
`2 x layers` times per token.

### The decision rule is "is the layer bigger than the latency?"

For a decode step, TP replaces `params_per_rank x bytes` of weight reading with
one all-reduce of `batch x hidden x bytes`. At batch 1 that message is tiny:

```
   batch  ring kB/rank   NVLink us  weight read us  comm/read
       1             4        5.0             4.2       1.19
     256          1024        6.2             4.2       1.47
```

So at batch 1 on a B200 the collective costs **more** than the read it
replaces, for this 33M-param MLP. The break-even is a size, not a batch:

```
5 us of read at 8 TB/s = 20M params per rank per layer
DeepSeek-V3.1 at TP8: 1.38B per rank per layer, read 344 us, 69x the latency
```

That is the quantitative version of "TP is for big models". Half a billion
params per layer per rank and the collective is 1.5% overhead; 16M and it is
100%. Nothing here is specific to GPUs — the ratio is
`params_per_rank x dtype / HBM_bw` against `latency + batch x hidden x dtype / link_bw`.

### On this machine the split wins, 1.8–2.1x, and that is not the interesting number

```
   batch   1 process   TP fused   TP naive   TP/1 proc
       1      3.20 ms     1.75 ms     1.98 ms        0.55x
      16      6.43 ms     3.55 ms     4.20 ms        0.55x
     256     29.06 ms    13.94 ms    20.67 ms        0.48x
```

Half the arithmetic per rank, so ~2x, less the collective. Two processes on a
10-core machine are not two GPUs: the compute here is so slow (ms) relative to
the loopback collective (100 us) that the ratio TCP:HBM (1:1000) is nowhere
near what a GPU sees. On a B200 the same layer computes in ~4 us and the
collective is 5 us, so the accounting flips. **The CPU run proves the
mechanism and the byte counts; the GPU numbers in section 4 are the ones that
decide deployments.**

## Connection to earlier days

- Day 12 (roofline): decode is memory-bound, so `params_per_rank / HBM_bw` is
  the term TP is trying to divide. Every number in section 4 is that day's
  arithmetic with a link budget added.
- Day 17 (spec sheets): HBM bandwidth and NVLink bandwidth are the only two
  hardware numbers TP cares about. Their ratio, ~9:1 on a B200, is why one
  collective can be worth 300 us of saved reading and not 4 us.
- Day 14 (GQA/MQA): TP splits attention heads, so `num_kv_heads` is the hard
  cap on TP inside one node unless you replicate KV. An 8-KV-head model cannot
  use TP16 without duplication.
- Day 25 (paged KV): TP also splits the KV cache by head, so the cache per GPU
  drops with TP exactly as the weights do. Day 34's planner adds that term.

## Caveats

- gloo over loopback TCP, not NCCL over NVLink. The measured 88 us constant and
  6.4 GB/s are this laptop; the NVLink column is a model
  (`5 us + bytes / 900 GB/s x 0.8`), labelled as such in the code.
- Rings, not trees. Real NCCL picks a topology-aware algorithm, so the byte
  count here (`2(N-1)/N x payload`) is the upper bound for a ring.
- MLP only: no attention, no LayerNorm, no autograd. An attention block adds
  one more all-reduce for the output projection, which is why a transformer
  layer has 2 per layer, and why TP8 on 80 layers is 160 collectives per token.
- Both ranks build the full weights from a seed and then slice, so the script
  can prove the shards are exact. Production shards at load time and never
  materialises the full tensor on a rank.
- 1 thread per rank (see the code) so the comparison is 2 processes against 1,
  not 2 processes against 10 cores.

## Things to try

1. `--world-size 4`. Where does the 100 us constant start to dominate, and at
   what batch does the fused path stop beating 4 single-process ranks?
2. `--inter 32768` (a wide MLP). The naive split's penalty should grow with
   `inter/hidden`; check the 4.5x becomes ~17x.
3. `--hidden 512 --inter 2048`. Compute the break-even params/rank and confirm
   the fused path loses on this machine.
4. Replace the all-reduce with `all_gather` + local add. Same bytes on 2 ranks,
   different constant; at 8 ranks all-gather moves `(N-1)/N x payload` against
   all-reduce's `2(N-1)/N`, which is why sequence parallelism exists.
5. Time `dist.all_reduce` with `async_op=True` overlapped with the next
   layer's column-parallel matmul, and see how much of the 88 us hides.
