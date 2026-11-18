"""Day 33 — tensor parallelism on two CPU processes with torch.distributed (gloo).

Splits one MLP across 2 ranks the way Megatron-LM does: the first linear is
column-parallel (no communication), the second is row-parallel (one all-reduce
returns the full activation). So a layer costs one all-reduce, not two, and
what travels is the *activation*, not the weight shard.

Run: python3 days/day33-tensor-parallel/tp_gloo.py
"""

import argparse
import math
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Projections for a real GPU, so the CPU run can be read as "what this costs on
# hardware where the interconnect is ~1/9 of HBM instead of slower than it".
PROJECT_HBM_GB_S = 8000.0  # B200 HBM3e per GPU
PROJECT_NV_BW_GB_S = 900.0  # NVLink 5, per GPU, unidirectional
PROJECT_NV_LAT_US = 5.0  # fixed cost of one collective
PROJECT_EFF = 0.8  # ring collectives do not hit peak bandwidth


def section(title):
    print(f"\n{title}\n{'=' * len(title)}")


def shard(w, dim, rank, world):
    """Hand each rank one contiguous slice of the tensor along `dim`."""
    return w.chunk(world, dim=dim)[rank].contiguous()


def ring_bytes(size, world):
    """Bytes each rank moves in a ring all-reduce of `size` bytes."""
    return 2 * (world - 1) / world * size


def comm_us(size_bytes, bw_gb_s, lat_us):
    return lat_us + size_bytes / (bw_gb_s * 1e9) * 1e6


def tp_mlp(x, w1_local, w2_local):
    """Megatron split: column-parallel up-projection, row-parallel down-projection.

    The shards are state, not work: they are loaded once, like any weight. Only
    the activation crosses the wire, once. Returns the full activation on every
    rank and this rank's contribution before the all-reduce.
    """
    h = torch.nn.functional.gelu(x @ w1_local.T)  # [batch, inter/world], no comm
    y_partial = h @ w2_local.T  # [batch, hidden], a partial sum
    y = y_partial.clone()
    dist.all_reduce(y)
    return y, y_partial


def naive_mlp(x_local, w1_local, w2_local, world):
    """Split both linears along their input instead: one all-reduce *and* one all-gather."""
    h = x_local @ w1_local.T  # [batch, inter], partial over the input dim
    dist.all_reduce(h)
    h = torch.nn.functional.gelu(h)
    y_local = h @ w2_local.T  # [batch, hidden/world]
    parts = [torch.empty_like(y_local) for _ in range(world)]
    dist.all_gather(parts, y_local)
    return torch.cat(parts, dim=1)


def timed(fn, iters, reps=3):
    """Fastest average seconds per call over a few repetitions; both ranks in lock step."""
    best = float("inf")
    for _ in range(reps):
        fn()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dist.barrier()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best


def benchmark_collectives(world, rank):
    """Latency and bandwidth of gloo all-reduce, measured, not assumed."""
    sizes_kb = [4, 16, 64, 256, 1024, 4096, 16384]
    rows = []
    for kb in sizes_kb:
        buf = torch.ones(kb * 1024 // 4)
        iters = max(3, min(200, int(64_000 / kb)))
        us = timed(lambda: dist.all_reduce(buf), iters) * 1e6
        rows.append((kb, us, ring_bytes(buf.numel() * 4, world) / (us * 1e-6) / 1e9))
    # Fit t = latency + bytes / bw from the middle and the largest message.
    (s0, t0, _), (s1, t1, _) = rows[3], rows[-1]
    b0, b1 = ring_bytes(s0 * 1024, world), ring_bytes(s1 * 1024, world)
    bw = (b1 - b0) / ((t1 - t0) * 1e-6) / 1e9
    lat = t0 - b0 / (bw * 1e9) * 1e6
    if rank == 0:
        print(f"  {'payload':>9} {'ring bytes/rank':>16} {'all-reduce':>12} {'effective':>11}")
        print(f"  {'':>9} {'':>16} {'us':>12} {'GB/s':>11}")
        for kb, us, gb_s in rows:
            print(f"  {kb:>7} kB {ring_bytes(kb * 1024, world) / 1024:>14.0f} kB {us:>10.1f} us {gb_s:>9.2f}")
        print(f"\n  fit: {lat:.1f} us fixed + {bw:.2f} GB/s per rank")
        print(f"  on NVLink ({PROJECT_NV_BW_GB_S:.0f} GB/s, {PROJECT_NV_LAT_US:.0f} us) the same")
        print(f"  64 kB all-reduce costs {comm_us(ring_bytes(64 * 1024, world), PROJECT_NV_BW_GB_S, PROJECT_NV_LAT_US):.1f} us "
              f"and gloo costs {rows[1][1]:.1f} us")
    return rows, lat, bw


def worker(rank, world, init_file, cfg):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    hidden, inter, batches, seed = cfg["hidden"], cfg["inter"], cfg["batches"], cfg["seed"]
    # One thread per rank: a rank stands in for one GPU, so the fused path must
    # win on parallelism, not on having more cores than the baseline.
    torch.set_num_threads(1)

    # Same weights on every rank, built from one seed. In production they are
    # loaded already sharded; here we build full and slice, so both paths see
    # bit-identical parameters.
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(inter, hidden, generator=g) / math.sqrt(hidden)  # up
    w2 = torch.randn(hidden, inter, generator=g) / math.sqrt(inter)  # down
    w1_col = shard(w1, 0, rank, world)  # [inter/world, hidden]: output split
    w2_row = shard(w2, 1, rank, world)  # [hidden, inter/world]: input split
    w1_in = shard(w1, 1, rank, world)  # [inter, hidden/world]: the naive split
    w2_out = shard(w2, 0, rank, world)  # [hidden/world, inter]

    def sync_print(*args):
        if rank == 0:
            print(*args)

    if rank == 0:
        section("1. the split")
    sync_print(f"  MLP: {hidden} -> {inter} -> {hidden}, GELU, {2 * hidden * inter / 1e6:.1f}M params")
    sync_print(f"  world_size = {dist.get_world_size()}, backend = gloo, every rank has the full input")
    sync_print(f"  W1 {tuple(w1.shape)} -> per rank {tuple(shard(w1, 0, rank, world).shape)}  (column split: outputs)")
    sync_print(f"  W2 {tuple(w2.shape)} -> per rank {tuple(shard(w2, 1, rank, world).shape)}  (row split: inputs)")
    full_mb = (w1.numel() + w2.numel()) * 4 / 1e6
    shard_mb = (w1.numel() + w2.numel()) / world * 4 / 1e6
    sync_print(f"  weights: {full_mb:.1f} MB on one process, {shard_mb:.1f} MB per rank, "
               f"{shard_mb * world:.1f} MB total (no duplication)")

    if rank == 0:
        section("2. one MLP, two ways")
    x = torch.randn(3, hidden, generator=torch.Generator().manual_seed(7))
    ref = torch.nn.functional.gelu(x @ w1.T) @ w2.T
    y_tp, y_partial = tp_mlp(x, w1_col, w2_row)
    y_naive = naive_mlp(shard(x, 1, rank, world), w1_in, w2_out, world)
    parts = [torch.empty_like(y_partial) for _ in range(world)]
    dist.all_gather(parts, y_partial)
    if rank == 0:
        print(f"  single process vs TP:   max |diff| = {(ref - y_tp).abs().max():.2e}"
              f"  ({((ref - y_tp).abs().max() / ref.abs().max()).item():.2e} relative)")
        print(f"  single process vs naive: max |diff| = {(ref - y_naive).abs().max():.2e}")
        print(f"  rank 0's partial vs rank 1's partial, before the all-reduce: "
              f"max |diff| = {(parts[0] - parts[1]).abs().max():.2e}")
        print("  -> the all-reduce is not cosmetic: each rank computes half of the sum.")
        b = 3 * hidden * 4
        fused = ring_bytes(b, world)
        naive = ring_bytes(3 * inter * 4, world) + (3 * hidden * 4 / world) * 1.0
        print(f"\n  bytes moved per forward (batch 3, fp32): fused {fused / 1024:.0f} kB, "
              f"naive {naive / 1024:.0f} kB ({naive / fused:.1f}x)")
        print("  the split direction decides *what* travels: hidden vs inter = 1:4 here.")

    if rank == 0:
        section("3. what an all-reduce costs here (gloo, 2 processes, loopback)")
    rows, gloo_lat_us, gloo_bw = benchmark_collectives(world, rank)

    if rank == 0:
        section("4. a decode step: communication bytes against weight bytes")
    dtype_b = 2  # fp16 weights, as served
    params_per_rank = 2 * hidden * inter / world
    read_us = params_per_rank * dtype_b / (PROJECT_HBM_GB_S * 1e9) * 1e6
    if rank == 0:
        print(f"  one layer, batch B, hidden {hidden}, fp16: the all-reduce moves "
              f"{hidden * dtype_b / 1024:.0f} kB per token")
        print(f"  {'batch':>6} {'ring kB/rank':>13} {'gloo us':>9} {'NVLink us':>10} {'weight read us':>15} {'comm/read':>10}")
        for b in batches:
            size = b * hidden * dtype_b
            rb = ring_bytes(size, world)
            print(f"  {b:>6} {rb / 1024:>13.0f} {comm_us(rb, gloo_bw, gloo_lat_us):>9.1f} "
                  f"{comm_us(rb, PROJECT_NV_BW_GB_S, PROJECT_NV_LAT_US):>10.1f} {read_us:>15.1f} "
                  f"{comm_us(rb, PROJECT_NV_BW_GB_S, PROJECT_NV_LAT_US) / read_us:>10.2f}")
        worth = PROJECT_NV_LAT_US * 1e-6 * PROJECT_HBM_GB_S * 1e9 / dtype_b
        print(f"\n  At batch 1 the all-reduce is {hidden * dtype_b / (PROJECT_NV_BW_GB_S * 1e9) * 1e9:.1f} ns of bytes")
        print(f"  inside {PROJECT_NV_LAT_US:.0f} us of fixed cost, so latency decides, not bandwidth.")
        print(f"  A layer earns the {PROJECT_NV_LAT_US:.0f} us back once its read per rank exceeds it:")
        print(f"  {worth / 1e6:.0f}M params/rank. This MLP has {params_per_rank / 1e6:.1f}M per rank, "
              f"read {read_us:.1f} us.")
        big = 671e9 / 61 / 8
        big_read = big * dtype_b / (PROJECT_HBM_GB_S * 1e9) * 1e6
        print(f"  DeepSeek-V3.1 at TP8: {big / 1e9:.2f}B per rank per layer, read {big_read:.0f} us, "
              f"{big_read / PROJECT_NV_LAT_US:.0f}x the latency.")
        if read_us > gloo_lat_us:
            be = (read_us - PROJECT_NV_LAT_US) * 1e-6 * PROJECT_NV_BW_GB_S * 1e9 / (hidden * dtype_b)
            print(f"  Bytes overtake the read at batch ~{be:,.0f}. Below that: latency-bound, "
                  f"so raise the layer size, not the batch.")
        else:
            print("  Here the latency alone exceeds the read: at this layer size TP cannot win.")

    if rank == 0:
        section("5. end to end on CPU: what the split costs when the interconnect is TCP")
        print(f"  {'batch':>6} {'1 process':>11} {'TP fused':>10} {'TP naive':>10} {'TP/1 proc':>11}")
    for b in batches:
        xb = torch.randn(b, hidden, generator=torch.Generator().manual_seed(11))
        iters = max(3, min(50, 200 // b))
        one = timed(lambda: torch.nn.functional.gelu(xb @ w1.T) @ w2.T, iters) * 1e3
        xb_local = shard(xb, 1, rank, world)
        fused = timed(lambda: tp_mlp(xb, w1_col, w2_row), iters) * 1e3
        naive = timed(lambda: naive_mlp(xb_local, w1_in, w2_out, world), iters) * 1e3
        if rank == 0:
            print(f"  {b:>6} {one:>9.2f} ms {fused:>8.2f} ms {naive:>8.2f} ms {fused / one:>11.2f}x")
    if rank == 0:
        print("\n  TP/1 proc below 1.0 means the split won: half the arithmetic per rank,")
        print(f"  one all-reduce of the activation. Here that bill is {rows[2][1]:.0f} us of gloo latency")
        print("  against milliseconds of arithmetic, so even loopback TCP wins.")
        print("  On NVLink the same collective is 5 us but the arithmetic is ~50x faster,")
        print("  and then the layer has to be big enough to pay for it (section 4).")

    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=8192)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.inter % args.world_size or args.hidden % args.world_size:
        raise SystemExit("TP size must divide both hidden and inter")

    cfg = {"hidden": args.hidden, "inter": args.inter, "batches": args.batches, "seed": args.seed}
    tmp = tempfile.mkdtemp(prefix="tp-gloo-")
    try:
        mp.spawn(worker, args=(args.world_size, os.path.join(tmp, "init"), cfg),
                 nprocs=args.world_size, join=True)
    finally:
        for name in os.listdir(tmp):
            os.unlink(os.path.join(tmp, name))
        os.rmdir(tmp)


if __name__ == "__main__":
    main()
