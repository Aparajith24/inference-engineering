# Day 18: Instances — how many GPUs, and what it costs

Book chapter: **3.3 Instances** (3.3.1 Multi-GPU Instances, 3.3.2 Multi-Instance GPUs)

Task: planner: model + traffic → instance count and cost per 1M tokens.

## What we built

`instance_planner.py` takes a model, your traffic and a speed target, tries
every instance shape (1, 2, 4 or 8 GPUs of L4, A100, H100, H200, B200, plus two
MIG slices of an H100), and prints them from cheapest to most expensive per
1 million tokens.

Run it:

```
python3 instance_planner.py
python3 instance_planner.py --model qwen2.5-1.5b --rps 2
python3 instance_planner.py --model llama-3.3-70b --dtype fp8
```

Options: `--model`, `--dtype` (fp16, fp8), `--rps` (requests per second at
peak), `--input` and `--output` (tokens per request), `--min-tps` (the slowest
tokens/s any user may get). Pure arithmetic, no libraries.

## What we learned

### An instance is more than a GPU

An **instance** is the virtual machine you rent:

| Part | What it is |
|---|---|
| GPUs (device) | where inference runs |
| CPUs (host) | everything else: tokenizing, the web server |
| Host memory | normal RAM |
| Storage | disk, for loading model files |
| Networking | the connection to the internet |
| Interconnect | GPU-to-GPU and node-to-node links |

Any of these can be the bottleneck. Even the same GPU comes in versions (an
A100 SXM has 5% more bandwidth than an A100 PCIe).

### Model too big for one GPU: multi-GPU

```
inside a node (8 GPUs):  NVLink      900 GB/s (Hopper), 1,800 GB/s (Blackwell)
between nodes:           InfiniBand  400 Gb/s per NIC = 50 GB/s   (Ethernet: 100 Gb/s)
```

- A **node** is 8 GPUs. **NVLink** connects pairs of GPUs, and **NVSwitch** lets
  all 8 talk to each other at once.
- Note **Gb/s (bits) vs GB/s (bytes)**: 8x apart. InfiniBand is about 20-40x
  slower than NVLink, so keep one model inside one node when you can.
- **NVL72** is a whole rack: 72 Blackwell GPUs all on NVLink, for the biggest
  models.

### Model too small for one GPU: MIG

**Multi-Instance GPU** cuts an A100, H100, H200 or B200 into up to **7 slices**.
Each slice gets part of the compute, part of the memory, and part of the CPU,
RAM and network.

```
H100: 7 compute slices, 8 memory slices (10 GB each)
1g.10gb: 1/7 of the compute, 10 GB
3g.40gb: 3/7 of the compute, 40 GB (half the memory)
```

Seven, not eight, because the H100's 132 SMs do not split evenly. Seven equal
slices are made and the leftover SMs sit idle.

### How the planner works

For each instance shape:

1. **Does the model fit?** Weights + 50% headroom ≤ VRAM.
2. **How many users at once?** The largest batch where every user still gets at
   least `--min-tps` tokens/s, limited by KV cache memory and capped at 256
   (real engines cap it too). Step time comes from day 17's rule.
3. **Requests per second per instance** = `1 / (prefill time + output tokens x
   step time / batch)`.
4. **How many instances** to handle the traffic at 80% busy (room for spikes).
5. **Cost per 1M tokens** = cost per hour / millions of tokens served per hour.

### Results

**Llama 3.1 8B, fp16, 50 requests/s, 500 in + 200 out tokens, at least 30 tok/s
per user** (top 6 of 17):

| Instance | $/hr each | Users at once | User tok/s | TTFT | Count | $/hr total | $ per 1M tokens |
|---|---|---|---|---|---|---|---|
| 1x H100 | 2.50 | 256 | 59 | 16 ms | 2 | 5.00 | **0.040** |
| 2x H100 | 5.00 | 256 | 119 | 8 ms | 1 | 5.00 | 0.040 |
| H100 MIG 3g.40gb | 1.07 | 250 | 30 | 38 ms | 5 | 5.36 | 0.043 |
| 1x B200 | 5.50 | 256 | 142 | 6 ms | 1 | 5.50 | 0.044 |
| 1x H200 | 3.50 | 256 | 85 | 16 ms | 2 | 7.00 | 0.056 |
| 1x A100 | 1.80 | 256 | 36 | 51 ms | 5 | 9.00 | 0.071 |

Cheapest: 2 instances of 1x H100, $5.00/hour, $120/day.

- The **cheapest per hour is not the cheapest per token**. The A100 costs least
  per hour but you need 5 of them.
- Bigger instances give users more speed for a similar cost per token. Faster
  service is a choice you can buy.

**A small model (Qwen 2.5 1.5B) with light traffic (2 requests/s):** the
cheapest is a **1/7 MIG slice of an H100**, at $0.36/hour ($0.071 per 1M
tokens), less than half the price of a whole L4. This is the book's MIG point:
a small model wastes a whole H100.

**Llama 3.3 70B at fp8, 50 requests/s:** the cheapest is **3 x 1x B200** at
$16.50/hour ($0.131 per 1M tokens), even though the B200 costs the most per
hour. A100s are the worst at $72/hour: they have no fp8 compute, so prefill is
slow and they need 2 GPUs just to fit the model.

## Connection to earlier days

- Day 2: the first cost estimate used fixed tokens/s per GPU. This one works
  out the tokens/s from the hardware.
- Day 5: does it fit = parameters x bytes x 1.5.
- Day 6: batching raises total throughput but lowers each user's speed; the
  speed target decides how far you can push it.
- Day 13 and 14: the KV cache decides how many users fit at once.
- Day 17: step time = max(memory time, compute time).

## Caveats

- **Prices are examples** (edit them in `GPUS`). Real prices vary by provider,
  region, and whether you reserve GPUs in advance.
- Efficiency is fixed (70% of bandwidth, 50% of compute). Real engines vary.
- Splitting across GPUs is treated as perfect, with no communication cost.
- MIG bandwidth is assumed to follow the share of memory slices.
- Traffic is assumed steady at `--rps`. Real traffic goes up and down, and you
  either keep spare GPUs or scale up and down (autoscaling, later chapters).

## Things to try

1. `--min-tps 100`: what happens to cost when every user must get 100 tok/s?
2. `--input 8000`: long prompts (like coding assistants). Which GPU wins now?
3. `--rps 500`: does the cheapest option change at high traffic?
4. Change the B200 price and see when it becomes the best choice for 8B.
