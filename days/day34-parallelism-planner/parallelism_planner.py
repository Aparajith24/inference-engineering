"""Day 34 — plan TP, PP and EP: what each shape costs in memory and communication.

Given a model and a GPU count, this enumerates the rank layouts
(tensor parallel x pipeline parallel x expert parallel), keeps the ones that
fit in memory, and estimates the two numbers that decide a deployment:

  tokens/s for one user at batch 1   -> latency, what TP buys
  total tokens/s at max batch        -> throughput, what EP buys

Cost model for one decode step, per rank (all terms in `analyse`):

  step = max(read_bytes / (HBM x eff), flops / peak) + tp + ep + pp

  read = dense/(pp*tp) + active_experts/ep * expert * moe_layers/pp   [weight bytes]
  tp   = 2 all-reduces per layer over the tp group, ring: 2(tp-1)/tp x batch x hidden
  ep   = 2 all-to-alls per MoE layer over the ep group, dispatch + combine
  pp   = (pp-1) serial hops between stages, over the inter-node fabric

Run: python3 days/day34-parallelism-planner/parallelism_planner.py --model deepseek-v3.1 --gpus 8
     python3 days/day34-parallelism-planner/parallelism_planner.py --sweep
     python3 days/day34-parallelism-planner/parallelism_planner.py --check
"""

import argparse
import math
from dataclasses import dataclass

GB = 1e9

# How much of the peak a well-tuned engine actually gets. The only fudge
# factors, identical for every plan, so the comparison holds.
MEM_EFF = 0.7  # bytes read per step against peak HBM bandwidth
COMM_EFF = 0.8  # ring collective against peak link bandwidth
FLOPS_EFF = 0.5  # matmul against peak dense flops
HEADROOM = 0.9  # fraction of VRAM a plan may use
RUNTIME_GB = 2.0  # context, allocator, workspace
BUFFERS = 6  # activation copies of [batch, hidden] kept live


@dataclass(frozen=True)
class Model:
    name: str
    total: float  # params, from the model card
    active: float  # params touched per token
    layers: int
    hidden: int
    heads: int
    head_dim: int
    kv_heads: int
    inter: int  # dense FFN width
    experts: int  # 0 for a dense model
    active_experts: int
    moe_inter: int
    dense_layers: int
    shared_experts: int
    vocab: int
    attn_per_layer: float = 0.0  # override for MLA, where q/kv/o are not plain matmuls
    kv_bytes_per_token_layer: float = 0.0  # override, bytes of KV per layer per token
    kv_shardable: bool = True  # False for MLA: the latent is not split by head


MODELS = {
    "deepseek-v3.1": Model("DeepSeek-V3.1", 671e9, 37e9, 61, 7168, 128, 128, 128, 18432,
                           256, 8, 2048, 3, 1, 129280,
                           attn_per_layer=366e6, kv_bytes_per_token_layer=576 * 2,
                           kv_shardable=False),
    "llama-3.3-70b": Model("Llama-3.3-70B", 70.6e9, 70.6e9, 80, 8192, 64, 128, 8, 28672,
                           0, 0, 0, 80, 0, 128256),
    "gpt-oss-120b": Model("gpt-oss-120B", 116.8e9, 5.1e9, 36, 2880, 64, 64, 8, 2880,
                          128, 4, 2880, 0, 0, 201088),
    "qwen3-235b-a22b": Model("Qwen3-235B-A22B", 235e9, 22e9, 94, 4096, 64, 128, 4, 4096,
                             128, 8, 1536, 0, 0, 151936),
}

HARDWARE = {
    "b200": dict(mem=192, hbm=8000, flops=4500, node=8, nv_bw=900, nv_lat=5.0,
                 ib_bw=100, ib_lat=15.0),
    "h200": dict(mem=141, hbm=4800, flops=2000, node=8, nv_bw=450, nv_lat=5.0,
                 ib_bw=50, ib_lat=15.0),
}

FABRIC_IB = {"nvlink": None, "ib400": 50.0, "ib800": 100.0}

DTYPE_BYTES = {"fp8": 1, "bf16": 2, "fp16": 2}
PEAK = {"fp8": 1.0, "bf16": 0.5, "fp16": 0.5}  # x the card's fp8 number


@dataclass(frozen=True)
class Plan:
    gpus: int
    tp: int = 1
    pp: int = 1
    ep: int = 1

    @property
    def label(self):
        bits = [f"TP{self.tp}"] if self.tp > 1 else []
        if self.pp > 1:
            bits.append(f"PP{self.pp}")
        if self.ep > 1:
            bits.append(f"EP{self.ep}")
        return " ".join(bits) or "1 GPU"


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def structure(m):
    """Parameter breakdown from the config, checked against the model card."""
    if m.attn_per_layer:
        attn = m.attn_per_layer
    else:
        q = m.hidden * m.heads * m.head_dim
        kv = 2 * m.hidden * m.kv_heads * m.head_dim
        o = m.heads * m.head_dim * m.hidden
        attn = q + kv + o
    expert = 3 * m.hidden * m.moe_inter  # SwiGLU: gate, up, down
    moe_layers = m.layers - m.dense_layers
    routed = m.experts * expert * moe_layers
    dense = (attn * m.layers + 3 * m.hidden * m.inter * m.dense_layers
             + m.shared_experts * expert * moe_layers
             + m.hidden * m.experts * moe_layers  # routers
             + 2 * m.vocab * m.hidden)  # embeddings + untied lm_head
    return {"attn": attn, "per_expert": expert, "routed": routed, "dense": dense,
            "total": dense + routed, "moe_layers": moe_layers}


def kv_per_token_layer(m, kv_bytes):
    """Bytes of KV cache for one token in one layer, in the KV dtype."""
    if m.kv_bytes_per_token_layer:
        return m.kv_bytes_per_token_layer * (kv_bytes / 2)
    return 2 * m.kv_heads * m.head_dim * kv_bytes


def link(group, hw):
    """(latency us, bandwidth GB/s) for a collective over `group` ranks."""
    if group <= hw["node"]:
        return hw["nv_lat"], hw["nv_bw"]
    return hw["ib_lat"], hw["ib_bw"]


def legal(m, hw, plan):
    """A layout is illegal when a tensor cannot be cut that way or it crosses a node."""
    stage = plan.gpus // plan.pp
    if plan.gpus % plan.pp or stage % plan.tp or stage % plan.ep:
        return False
    if stage % (plan.tp * plan.ep):  # the rest of the stage is whole replicas
        return False
    if m.heads % plan.tp:  # attention heads must divide
        return False
    if plan.tp > hw["node"]:  # TP is a single-node technique
        return False
    if plan.pp > m.layers:
        return False
    if plan.ep > 1 and (not m.experts or plan.ep > m.experts or m.experts % plan.ep):
        return False
    return True


def hot_experts(m, tokens):
    """How many distinct experts one step activates. MoE decode fills up as the batch grows."""
    if not m.experts:
        return 0
    hit = 1 - (1 - m.active_experts / m.experts) ** tokens
    return m.experts * hit


def analyse(m, part, hw, plan, batch, seq, dtype, kv_dtype):
    """Memory and time for one layout, in GB and ms.

    One step is one batch moving through the whole model, so PP never shortens
    the path: it divides memory and then adds hops. That is the book's "poor
    latency and utilization from step-by-step pipeline", made numerical.

    Bytes read per step, per rank, and why each term is what it is:
      dense   = dense/tp, once per step whatever the batch (that is batching)
      experts = hot(batch)/ep * per_expert * moe_layers, where hot() saturates:
                at batch 1 only topk experts are touched, at batch 1000 all 256
      kv      = batch * seq * kv_per_token_layer * layers/pp / kv_split
    """
    w_bytes, act_bytes = DTYPE_BYTES[dtype], kv_dtype
    stage = plan.gpus // plan.pp
    layers_stage = -(-m.layers // plan.pp)
    moe_stage = part["moe_layers"] * layers_stage / m.layers
    expert_split = plan.ep if plan.ep > 1 else plan.tp
    copies = stage // (plan.tp * plan.ep)  # independent replicas of the whole model

    weights = (part["dense"] / (plan.pp * plan.tp)
               + part["routed"] / (plan.pp * expert_split)) * w_bytes
    kv_split = plan.tp if (m.kv_shardable and m.kv_heads % plan.tp == 0) else 1
    kv_rank = batch * seq * kv_per_token_layer(m, kv_dtype) * layers_stage / kv_split
    act_rank = batch * m.hidden * act_bytes * BUFFERS
    other = act_rank + RUNTIME_GB * GB
    total = weights + kv_rank + other
    budget = HEADROOM * hw["mem"] * GB

    hot = hot_experts(m, batch)
    read_rank = (part["dense"] / plan.tp
                 + hot / expert_split * part["per_expert"] * part["moe_layers"]) * w_bytes
    read_s = read_rank / (hw["hbm"] * GB * MEM_EFF)
    flops = hw["flops"] * PEAK[dtype] * 1e12
    compute_s = 2 * (part["dense"] / plan.tp
                     + hot / expert_split * part["per_expert"] * part["moe_layers"]) * batch \
        / (flops * FLOPS_EFF)

    tp_us = ep_us = pp_us = 0.0
    if plan.tp > 1:
        lat, bw = link(plan.tp, hw)
        nbytes = 2 * (plan.tp - 1) / plan.tp * batch * m.hidden * act_bytes
        tp_us = 2 * m.layers * (lat + nbytes / (bw * GB * COMM_EFF) * 1e6)
    if plan.ep > 1:
        lat, bw = link(plan.ep, hw)
        nbytes = (plan.ep - 1) / plan.ep * batch * m.active_experts * m.hidden * act_bytes
        ep_us = 2 * part["moe_layers"] * (lat + nbytes / (bw * GB * COMM_EFF) * 1e6)
    if plan.pp > 1:
        nbytes = batch * m.hidden * act_bytes
        pp_us = (plan.pp - 1) * (hw["ib_lat"] + nbytes / (hw["ib_bw"] * GB * COMM_EFF) * 1e6)

    comm_us = tp_us + ep_us + pp_us
    step_s = max(read_s, compute_s) + comm_us * 1e-6

    per_request = (seq * kv_per_token_layer(m, kv_dtype) * layers_stage / kv_split
                   + m.hidden * act_bytes * BUFFERS)
    max_batch = max(0.0, (budget - weights - RUNTIME_GB * GB) / per_request)
    return {
        "plan": plan,
        "weights": weights, "kv": kv_rank, "other": other, "total": total, "budget": budget,
        "fits": total <= budget,
        "read_ms": read_s * 1e3, "compute_ms": compute_s * 1e3, "comm_ms": comm_us / 1e3,
        "tp_ms": tp_us / 1e3, "ep_ms": ep_us / 1e3, "pp_ms": pp_us / 1e3,
        "step_ms": step_s * 1e3,
        "tps_user": 1 / step_s,
        "max_batch": max_batch * copies,  # the whole deployment, every replica
        "copies": copies,
        "hot": hot, "expert_split": expert_split, "kv_split": kv_split,
        "weights_budget": budget,
    }


def candidates(m, hw, gpus):
    out = []
    for pp in divisors(gpus):
        stage = gpus // pp
        for tp in divisors(stage):
            eps = [1] if not m.experts else [e for e in divisors(stage) if e <= m.experts]
            for ep in eps:
                if m.experts and m.experts % ep:
                    continue
                plan = Plan(gpus, tp, pp, ep)
                if legal(m, hw, plan):
                    out.append(plan)
    return out


def throughput(r, m, hw, seq, dtype, kv_dtype):
    """Total tokens/s at the largest batch this plan can hold."""
    b = int(r["max_batch"])
    if b < 1:
        return 0.0, 0.0, 0.0
    per_replica = b // r["copies"]  # each replica holds its own batch
    s = analyse(m, part_of(m), hw, r["plan"], max(1, per_replica), seq, dtype, kv_dtype)
    return b, s["step_ms"], b / (s["step_ms"] / 1e3)


_PART = {}


def part_of(m):
    if m.name not in _PART:
        _PART[m.name] = structure(m)
    return _PART[m.name]


def plan_rows(m, hw, gpus, batch, seq, dtype, kv_dtype):
    """One row per layout: memory at the requested batch, latency at batch 1."""
    rows = []
    for plan in candidates(m, hw, gpus):
        r = analyse(m, part_of(m), hw, plan, batch, seq, dtype, kv_dtype)
        r1 = analyse(m, part_of(m), hw, plan, 1, seq, dtype, kv_dtype)
        b, ms, tps = throughput(r, m, hw, seq, dtype, kv_dtype)
        r.update(batch_max=b, batch_step_ms=ms, tps_total=tps,
                 lat_step_ms=r1["step_ms"], lat_read_ms=r1["read_ms"],
                 lat_comm_ms=r1["comm_ms"], lat_tp_ms=r1["tp_ms"],
                 lat_ep_ms=r1["ep_ms"], lat_pp_ms=r1["pp_ms"],
                 lat_tps_user=r1["tps_user"])
        rows.append(r)
    return rows


def show_table(rows, model, batch, seq, dtype, limit=12):
    fits = [r for r in rows if r["fits"]]
    best = sorted(fits, key=lambda r: r["step_ms"])[:limit]
    print(f"\n  memory per GPU, batch {batch}, {seq}-token context, {dtype} weights")
    print(f"  {'plan':<16} {'weights':>9} {'KV':>8} {'activ':>7} {'total':>8} {'of':>7} {'use':>6}")
    print("  " + "-" * 68)
    for r in sorted(rows, key=lambda r: r["total"])[:limit]:
        flag = "ok" if r["fits"] else "OOM"
        print(f"  {r['plan'].label:<16} {r['weights'] / GB:>8.1f}G {r['kv'] / GB:>7.1f}G "
              f"{r['other'] / GB:>6.1f}G {r['total'] / GB:>7.1f}G {r['budget'] / GB:>6.1f}G "
              f"{r['total'] / r['budget']:>5.0%} {flag}")

    print(f"\n  one decode step at batch 1, and at the largest batch this layout can hold")
    print(f"  {'plan':<16} {'read':>7} {'TP':>6} {'EP':>6} {'PP':>6} {'step':>7} {'tok/s':>7} "
          f"{'maxB':>5} {'step':>7} {'tok/s':>8}")
    print("  " + "-" * 82)
    for r in best:
        print(f"  {r['plan'].label:<16} {r['lat_read_ms']:>6.1f}m {r['lat_tp_ms']:>5.1f}m "
              f"{r['lat_ep_ms']:>5.1f}m {r['lat_pp_ms']:>5.1f}m {r['lat_step_ms']:>6.1f}m "
              f"{r['lat_tps_user']:>7.0f} {r['batch_max']:>5.0f} {r['batch_step_ms']:>6.1f}m "
              f"{r['tps_total']:>8.0f}")

    if not fits:
        cheapest = min(rows, key=lambda r: r["total"])
        print(f"\n  nothing fits: the smallest layout, {cheapest['plan'].label}, needs "
              f"{cheapest['total'] / GB:.1f} GB per GPU of {cheapest['budget'] / GB:.1f} GB.")
        print(f"  weights alone are {cheapest['weights'] / GB:.1f} GB per GPU.")
        return []
    fastest = min(fits, key=lambda r: r["lat_step_ms"])
    fattest = max(fits, key=lambda r: r["tps_total"])
    print(f"\n  lowest latency: {fastest['plan'].label} at {fastest['lat_tps_user']:.0f} tok/s for one user")
    print(f"  highest throughput: {fattest['plan'].label} at batch {fattest['batch_max']:.0f}, "
          f"{fattest['tps_total']:.0f} tok/s total")
    if fattest["plan"] != fastest["plan"]:
        print(f"  -> {fattest['plan'].label} runs {fattest['copies']} replica(s) with "
              f"{fattest['batch_max']:.0f} requests in flight and gives up latency for users")
    unique = []
    for r in (fastest, fattest):
        if all(r["plan"] != u["plan"] for u in unique):
            unique.append(r)
    for r in unique:
        if model.experts and r["expert_split"] > 1 and r["plan"].ep <= 1:
            print(f"  note: {r['plan'].label} splits each expert {r['expert_split']}-way with TP, "
                  f"so the expert FFN pays the same all-reduce as attention")
        if model.experts and r["hot"] < model.experts:
            full = math.log(0.05) / math.log(1 - model.active_experts / model.experts)
            print(f"  note: at batch {batch} a step touches {r['hot']:.0f} of {model.experts} experts "
                  f"(topk {model.active_experts}); by batch ~{full:.0f} it touches all of them")
    return fits


def sweep(m, hw, batch, seq, dtype, kv_dtype):
    print(f"\n  {m.name}: {m.total / 1e9:.0f}B params ({m.active / 1e9:.1f}B active), "
          f"{hw['mem']} GB per GPU, {dtype} weights, {kvd(kv_dtype)} KV, "
          f"{batch} requests x {seq} tokens")
    print(f"\n  {'GPUs':>5} {'lowest latency':<16} {'latency':>8} {'tok/s/user':>11} "
          f"{'maxB':>6} {'tok/s total':>12} {'vs smallest':>12}")
    print("  " + "-" * 76)
    base = None
    label = ""
    for gpus in (1, 2, 4, 8, 16, 32, 64):
        rows = plan_rows(m, hw, gpus, batch, seq, dtype, kv_dtype)
        fits = [r for r in rows if r["fits"]]
        if not fits:
            print(f"  {gpus:>5} {'-':<16} {'OOM':>9}")
            continue
        fast = min(fits, key=lambda r: r["lat_step_ms"])
        fat = max(fits, key=lambda r: r["tps_total"])
        if base is None:
            base, label = fat["tps_total"], f"vs {gpus} GPUs"
        rel = f"{fat['tps_total'] / base:.1f}x {label}"
        print(f"  {gpus:>5} {fast['plan'].label:<16} {fast['lat_step_ms']:>7.1f}m {fast['lat_tps_user']:>11.0f} "
              f"{fat['batch_max']:>6.0f} {fat['tps_total']:>12.0f} {rel:>12}")
    print("\n  lowest latency = fastest single-user step at batch 1;")
    print("  maxB / tok/s total = the same GPU count serving the largest batch that fits.")


def kvd(b):
    return f"{int(b * 8)}-bit"


def self_check(args):
    """The claims in section 5.4, run against the planner. Prints what holds and what does not."""
    m = MODELS["deepseek-v3.1"]
    hw = hardware(args)
    batch, seq = 32, 8192
    print(f"\n  {m.name} on {hw['mem']} GB GPUs, batch {batch}, {seq}-token context, fp8")
    print("  " + "-" * 70)
    failures = []

    four = plan_rows(m, hw, 4, batch, seq, "fp8", 2)
    cheap = min(four, key=lambda r: r["weights"])
    weights_fit = cheap["weights"] <= cheap["budget"]
    none_fit = not any(r["fits"] for r in four)
    print(f"  4 GPUs: weights {cheap['weights'] / GB:.1f} GB of a {cheap['budget'] / GB:.1f} GB budget, "
          f"best total {min(r['total'] for r in four) / GB:.1f} GB")
    print(f"          weights fit: {weights_fit}, layouts that also hold KV: "
          f"{sum(r['fits'] for r in four)}   "
          f"{'PASS' if weights_fit and none_fit else 'FAIL'}")
    if not (weights_fit and none_fit):
        failures.append("4 GPUs should hold weights but not weights + KV")

    eight = plan_rows(m, hw, 8, batch, seq, "fp8", 2)
    ok8 = [r for r in eight if r["fits"]]
    best8 = min(ok8, key=lambda r: r["lat_step_ms"]) if ok8 else None
    print(f"  8 GPUs: {len(ok8)} layouts fit, fastest {best8['plan'].label if best8 else '-'} at "
          f"{best8['lat_step_ms']:.1f} ms/step and {best8['total'] / GB:.1f} GB per GPU   "
          f"{'PASS' if ok8 else 'FAIL'}")
    if not ok8:
        failures.append("8 GPUs should serve the model")

    sixteen = plan_rows(m, hw, 16, batch, seq, "fp8", 2)
    tp8pp2 = next((r for r in sixteen if r["plan"] == Plan(16, 8, 2, 1) and r["fits"]), None)
    ep16 = next((r for r in sixteen if r["plan"] == Plan(16, 1, 1, 16) and r["fits"]), None)
    tp8 = next((r for r in sixteen if r["plan"] == Plan(16, 8, 1, 1) and r["fits"]), None)
    if tp8pp2 and ep16:
        print(f"  16 GPUs at batch {batch}: TP8PP2 {tp8pp2['step_ms']:.1f} ms/step, "
              f"batch {tp8pp2['batch_max']:.0f} max, {tp8pp2['tps_total']:.0f} tok/s")
        print(f"                    EP16   {ep16['step_ms']:.1f} ms/step, "
              f"batch {ep16['batch_max']:.0f} max, {ep16['tps_total']:.0f} tok/s")
        if not tp8pp2["step_ms"] < ep16["step_ms"]:
            failures.append("TP8PP2 should be lower latency than EP16")
        print(f"          TP8PP2 lower latency than EP16: {tp8pp2['step_ms'] < ep16['step_ms']}")
        print(f"          EP16 throughput above TP8PP2: {ep16['tps_total'] > tp8pp2['tps_total']}"
              f"  (the book says it should be, this model says no: EP reads the")
        print(f"          whole dense part on every rank, and PP already splits the layers)")
        if tp8:
            print(f"          2 replicas of {tp8['plan'].label}: same latency, "
                  f"{tp8['tps_total']:.0f} tok/s, on the same 16 GPUs")
    else:
        failures.append("TP8PP2 and EP16 should both fit on 16 GPUs")
    if failures:
        raise SystemExit("self-check failed: " + "; ".join(failures))


def hardware(args):
    hw = dict(HARDWARE[args.gpu])
    ib = FABRIC_IB[args.fabric]
    if ib is not None:
        hw["ib_bw"] = ib
    return hw


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="deepseek-v3.1", choices=sorted(MODELS))
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--gpu", default="b200", choices=sorted(HARDWARE))
    p.add_argument("--fabric", default="ib800", choices=sorted(FABRIC_IB))
    p.add_argument("--dtype", default="fp8", choices=sorted(DTYPE_BYTES))
    p.add_argument("--kv-dtype", type=int, default=2, help="bytes per KV element")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--sweep", action="store_true", help="best plans from 1 to 64 GPUs")
    p.add_argument("--check", action="store_true", help="assert the claims in section 5.4")
    args = p.parse_args()

    if args.check:
        return self_check(args)
    m = MODELS[args.model]
    hw = hardware(args)
    part = part_of(m)
    print(f"\n  {m.name}: card says {m.total / 1e9:.0f}B total, {m.active / 1e9:.1f}B active; "
          f"this config gives {part['total'] / 1e9:.0f}B "
          f"({part['dense'] / 1e9:.0f}B dense + {part['routed'] / 1e9:.0f}B in experts)")
    print(f"  {m.layers} layers, hidden {m.hidden}, {m.heads} heads ({m.kv_heads} KV), "
          f"{m.experts or 'no'} experts, {m.active_experts or '-'} active")
    print(f"  one token costs {kv_per_token_layer(m, args.kv_dtype) * m.layers / 1e3:.0f} kB of KV per layer-set")
    print(f"  hardware: {args.gpus}x {args.gpu.upper()} ({hw['mem']} GB, {hw['hbm']} GB/s HBM, "
          f"{hw['flops']} TFLOPS fp8), {hw['node']}/node, NVLink {hw['nv_bw']} GB/s, "
          f"IB {hw['ib_bw']} GB/s")
    if args.sweep:
        return sweep(m, hw, args.batch, args.ctx, args.dtype, args.kv_dtype)
    rows = plan_rows(m, hw, args.gpus, args.batch, args.ctx, args.dtype, args.kv_dtype)
    show_table(rows, m, args.batch, args.ctx, args.dtype)


if __name__ == "__main__":
    main()
