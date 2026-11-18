# Day 34: a planner for TP, PP and EP

Book chapter: **5.4 Model parallelism**

Task: given a model and a GPU count, work out the memory and the communication
for each way of splitting it, and pick a layout.

## What we built

`parallelism_planner.py` enumerates rank layouts and prices them.

- **The model** comes from its config, not its card: attention, FFN, routers,
  embeddings and per-expert size are computed from `hidden`, `layers`, `heads`,
  `kv_heads`, `experts`. For DeepSeek-V3.1 that gives **682B** against the card's
  671B, a 1.6% check that the structure is right.
- **The layouts**: `PP | GPUs`, `TP | stage`, `EP | stage`, and whatever is left
  over becomes whole replicas. Illegal cuts are dropped: TP may not cross a node
  (`tp <= 8`), attention heads must divide, `tp * ep` must fit in the stage.
- **Memory per GPU**: `dense/(pp*tp) + experts/(pp*ep)` for weights,
  `batch * seq * kv_per_token * layers/pp / kv_split` for KV, activations and a
  2 GB runtime on top.
- **Time per step**, which is where the whole thing lives:

  ```
  read   = dense/tp  +  hot(batch)/ep * per_expert * moe_layers
  tp     = 2 all-reduces per layer   x ring bytes 2(tp-1)/tp x batch x hidden
  ep     = 2 all-to-alls per MoE layer, dispatch + combine
  pp     = (pp-1) hops between stages, over InfiniBand
  step   = max(read / HBM, flops / peak) + tp + ep + pp
  ```

  `hot(batch)` is the number of **distinct** experts a step touches:
  `experts * (1 - (1 - topk/experts)^batch)`. At batch 1 that is 8, and it
  saturates at all 256 around batch 94.

```
python3 days/day34-parallelism-planner/parallelism_planner.py --model deepseek-v3.1 --gpus 8
python3 days/day34-parallelism-planner/parallelism_planner.py --sweep
python3 days/day34-parallelism-planner/parallelism_planner.py --check
```

Standard library only. Also `--model llama-3.3-70b | gpt-oss-120b |
qwen3-235b-a22b`, `--gpu h200`, `--dtype fp8|bf16`, `--batch`, `--ctx`.

## What we learned

### Four B200s hold the weights and still cannot serve the model

The book's figure 5.11, reproduced to the gigabyte:

```
  4 GPUs: weights 170.5 GB of a 172.8 GB budget, best total 177.3 GB
          weights fit: True, layouts that also hold KV: 0   PASS
```

671B at fp8 is 171 GB, so on 4×B200 every GPU is at 96% of a 90%-of-VRAM
budget **before the KV cache exists**. Adding a 8192-token context for 32
requests costs another 6.8 GB per GPU and it OOMs. The model is not "too big";
the model plus a working batch is too big, which is the point of the figure.

### TP and PP divide memory identically, and only TP divides time

Llama-3.3-70B on 8×B200, same bytes per GPU for four different layouts:

```
  plan               weights       KV   activ    total      of
  TP8                   8.8G    10.7G    2.0G    21.6G  172.8G
  TP4 PP2               8.8G    10.7G    2.0G    21.6G  172.8G
  TP2 PP4               8.8G    10.7G    2.0G    21.6G  172.8G
  PP8                   8.8G    10.7G    2.0G    21.6G  172.8G
```

and then the timings separate them by 5x:

```
  plan                read     TP     PP    step   tok/s/user   maxB   tok/s total
  TP8                 1.6m   0.8m   0.0m    2.4m          420    482        63,010
  TP4 PP2             3.1m   0.8m   0.0m    4.0m          252    482        43,412
  PP8                12.6m   0.0m   0.1m   12.7m           79    482        15,536
  1 GPU              12.6m   0.0m   0.0m   12.6m           79    298        23,653
```

Memory cannot tell TP from PP. Time can, because a token still walks every
layer: PP divides the storage, not the path. TP8 reads 1/8 of the weights per
rank *at the same time*; PP8 reads 1/8 per rank *in sequence*, so the step is
the same 12.6 ms as one GPU. That is the numerical version of "PP: not
recommended due to poor latency and utilization from step-by-step pipeline".

### EP inside a node is TP with extra steps

```
  TP8                 1.1m   0.6m   0.0m    1.7m    590 tok/s/user   85.2G weights
  EP8                 5.5m   0.0m   0.6m    6.1m    165 tok/s/user  109.8G weights
```

With 8 GPUs in one node, `TP8` already splits each expert 8 ways, so `EP8`
splits the *same* weights the *same* number of ways while adding 0.6 ms of
all-to-all and losing TP's split of the 28B-parameter dense part (it is
replicated on every rank instead: 109.8 GB vs 85.2 GB of weights).
**EP is not a faster TP. It is a way to spread experts over more GPUs than one
node**, which is where the book uses it: EP16 at 16 GPUs, EP32 at 32.

### Replicas beat every form of model parallelism, once the model fits

The sweep, same model, same workload:

```
   GPUs lowest latency    latency  tok/s/user   maxB  tok/s total  vs smallest
      8 TP8                  1.7m         590    148         9064  1.0x
     16 TP8                  1.7m         590    297        18190  2.0x
     32 TP8                  1.7m         590    594        36379  4.0x
     64 TP8                  1.7m         590   1188        72758  8.0x
```

Eight GPUs is the smallest that serves DeepSeek at this context and batch, and
from there the planner just adds replicas: latency flat, throughput linear.
At 16 GPUs that is also the *best* option the planner can find — `TP8` twice
(18,190 tok/s at 1.7 ms) beats `TP8PP2` (12,251 tok/s at 10.7 ms) and `EP16`
(4,246 tok/s at 16.4 ms) on both axes. It is exactly the book's advice: once
the model fits, extra nodes are for copies, not for cleverer splits.

### The MoE read is not constant, so throughput has a knee

For a dense model, weight bytes per step are fixed and batching is free until
compute or KV runs out. For an MoE, the *distinct* experts touched grows with
the batch:

| batch | distinct experts of 256 | expert bytes per rank (TP8) |
|---|---|---|
| 1 | 8 | 2.5 GB |
| 32 | 163 | 52 GB |
| 94 | 256 | 82 GB |

So DeepSeek's step time goes 1.7 ms → 16.3 ms between batch 1 and batch 148,
with the same weights and the same GPUs. Per-user speed collapses from 590 to
61 tok/s while aggregate throughput rises from 590 to 9,064 tok/s. This is the
real content of "EP improves throughput": spreading 256 experts over more GPUs
is the only way to stop every rank from reading all of them once the batch is
big enough to activate them all.

### Where my model disagrees with the book

The check asserts what it can and prints what it cannot:

```
  16 GPUs at batch 32: TP8PP2 10.7 ms/step, batch 438 max, 12251 tok/s
                    EP16    16.4 ms/step, batch 176 max, 4246 tok/s
          TP8PP2 lower latency than EP16: True
          EP16 throughput above TP8PP2: False
```

The book says EP16 yields higher overall system throughput than TP8PP2. In this
model it does not, for two model-specific reasons: EP16 leaves the 28B dense
part at TP1, so every rank reads all of it every step, and PP already splits
the layers so TP8PP2's per-stage read is half of EP16's. The book's claim rests
on PP behaving worse than the ideal pipeline this planner prices (stage
imbalance, pipeline fill latency, and PP being hoppier as you add nodes), plus
EP16 serving a much larger batch once attention is data-parallel across ranks.
I chose to print the disagreement instead of tuning a fudge factor until it
went away.

## Connection to earlier days

- Day 5 and day 17: params → memory → GPUs. This is the same arithmetic with a
  `pp`, `tp` and `ep` in the denominator, plus a KV term that day 17 ignored.
- Day 13 and day 25: the KV cache is what makes 4 GPUs fail after the weights
  pass. The KV term here is the day 25 block count, per layer.
- Day 33: the measured gloo all-reduce became the `tp` term: `2 x layers`
  collectives of `batch x hidden`, and the same latency-versus-bytes split.
- Day 24: continuous batching is why `maxB` is a throughput lever at all.

## Caveats

- Everything is a roofline: `0.7` of peak HBM bandwidth, `0.8` of ring bandwidth,
  `0.5` of peak flops. Real engines land 2–4x below these numbers on MoE decode.
  Compare plans, not absolute tok/s.
- PP is priced as one batch walking all stages per step. A pipeline kept full
  with `pp` independent microbatches recovers up to `pp`x throughput; that is
  what PP is for in practice, and it is deliberately not modelled here because
  the book's warning is about the step-by-step case.
- No stage imbalance, no pipeline fill/drain, no overlapping communication with
  compute. All three flatter the plans with more stages.
- `kv_split = 1` for MLA, so DeepSeek's KV is replicated across the TP group.
  That is what makes 8 GPUs hold only ~148 requests at 8k context; a real MLA
  deployment shards the latent differently and will beat this number.
- Expert routing is treated as uniform, so `hot()` is an expectation, not a
  guarantee. Skew makes the hot set larger, sooner.
- `tp * ep` ranks form one replica, `stage/(tp*ep)` replicas the rest. Real
  deployments sometimes overlap the TP and EP groups differently (attention DP
  plus EP), which changes which weights are replicated.

## Things to try

1. `--ctx 32768`. Which layouts survive, and does the answer become EP because
   TP's KV replication is what runs out first?
2. `--model qwen3-235b-a22b --gpus 8` and `--gpus 16`. Where does EP overtake TP
   for a model whose dense part is 4B instead of 28B?
3. `--dtype bf16`. Halve the bytes, double the flops ceiling: does TP8 still fit
   DeepSeek on 8 GPUs? (No.)
4. Add stage imbalance: give a PP stage 2 attention layers and the next 8 MoE
   layers, and watch PP's throughput gain from pipelining evaporate.
5. Model pipelining properly: `rate = read/pp` for throughput with `latency =
   read`. Then re-run the 16-GPU comparison and see how much of the book's
   EP claim it explains.
