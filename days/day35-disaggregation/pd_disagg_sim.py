"""Day 35 — prefill/decode disaggregation: goodput under latency targets.

Simulates the same workload through colocated engines (one GPU does prefill and
decode) and through xPyD deployments (P prefill engines, D decode engines, KV
cache moved over a link), and counts goodput: requests/s that arrive with
TTFT and every inter-token gap inside their targets.

Run: python3 days/day35-disaggregation/pd_disagg_sim.py
     python3 days/day35-disaggregation/pd_disagg_sim.py --workload chat --rate 6
"""

import argparse
import heapq
import os
import random
import statistics
from dataclasses import dataclass, field

# One B200-class GPU per engine, a 70B dense model at fp8. Every number below
# is derived from these, so the trade-offs are physics, not opinions.
PARAMS_B = 70e9
WEIGHT_GB = 70.0  # fp8
KV_BYTES_PER_TOKEN = 80 * 8 * 128 * 2 * 2  # 80 layers, 8 KV heads, 128 dim, K+V, 2 bytes
GPU_GB = 192.0
HBM_GB_S = 5000.0  # 8 TB/s peak, ~62% realised
FP8_TFLOPS = 2200.0  # 4500 TFLOPS dense fp8, ~50% realised
PREFILL_TPS = FP8_TFLOPS * 1e12 / (2 * PARAMS_B)  # compute bound: 2 flops per param per token
DECODE_OVERHEAD_MS = 2.0
PREFILL_OVERHEAD_MS = 3.0
KV_BUDGET_GB = 0.9 * GPU_GB - WEIGHT_GB - 2.0  # 100.8 GB of KV per decode GPU

# (GB/s per engine, one-way latency ms)
FABRICS = {"nvlink": (450.0, 0.005), "ib800": (100.0, 0.02), "pcie": (25.0, 0.05)}

WORKLOADS = {
    "code": dict(isl=(2000, 8000), osl=(100, 300), prefix_share=0.35),
    "chat": dict(isl=(300, 1200), osl=(200, 600), prefix_share=0.15),
}

# A mixed iteration runs prefill and decode kernels on one device. Perfect
# overlap (0.0) is the roofline; fully serialised (1.0) is the naive model.
# Real engines land in between: smaller, less parallel kernels per phase.
OVERLAP_LOSS = 0.3
TTFT_SLO_MS = 1000.0
ITL_SLO_MS = 30.0
LOCAL_PREFILL_MAX = 512  # conditional disaggregation: shorter than this stays on the decode engine
LOCAL_CHUNK = 128  # ... and is chunked so it does not wreck that engine's ITL
MAX_BATCH = 256


@dataclass
class Req:
    """One request and the times it reached each stage, in ms."""

    id: int
    arrival: float
    isl: int
    osl: int
    cached: int = 0
    remaining_prefill: int = 0
    prefill_done: float = 0.0
    first_token: float = 0.0
    finish: float = 0.0
    tokens: int = 0
    last_token: float = 0.0
    xfer_ms: float = 0.0
    itls: list = field(default_factory=list)

    def __post_init__(self):
        self.remaining_prefill = self.isl - self.cached


def prefill_ms(tokens):
    """Prefill is compute bound: linear in tokens, one fixed cost per request."""
    return tokens / PREFILL_TPS * 1000 + PREFILL_OVERHEAD_MS


def decode_ms(batch, kv_tokens):
    """Decode is memory bound: weights read once per step, plus every seat's KV."""
    nbytes = WEIGHT_GB * 1e9 + kv_tokens * KV_BYTES_PER_TOKEN
    return nbytes / (HBM_GB_S * 1e9) * 1000 + DECODE_OVERHEAD_MS


def iteration_ms(compute_ms, memory_ms, loss=None):
    """One GPU iteration: prefill work is compute bound, the decode step is memory bound.

    They overlap, so the ideal cost is the larger of the two, not their sum; the
    smaller phase is hidden behind the other. `loss` is the share of that hidden
    phase that does not hide in practice. With loss=0 colocation is at its best,
    with loss=1 the two phases serialise.
    """
    loss = OVERLAP_LOSS if loss is None else loss
    return max(compute_ms, memory_ms) + loss * min(compute_ms, memory_ms)


def kv_gb(tokens):
    return tokens * KV_BYTES_PER_TOKEN / 1e9


def make_requests(rate, workload, n, seed):
    """Poisson arrivals; `rate` requests per second, ms timestamps from the first arrival."""
    wl, rng, t, out = WORKLOADS[workload], random.Random(seed), 0.0, []
    for i in range(n):
        t += rng.expovariate(rate) * 1000
        isl = rng.randint(*wl["isl"])
        cached = 0
        if rng.random() < wl["prefix_share"]:
            cached = min(isl - 1, int(isl * rng.uniform(0.5, 0.9) / 128) * 128)
        out.append(Req(i, t, isl, rng.randint(*wl["osl"]), cached))
    return out


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def metrics(requests, label, extra=None):
    """Latency percentiles, throughput, and goodput against the two targets."""
    made = [r for r in requests if r.tokens]
    if not made:
        out = {k: 0.0 for k in ("ttft_p50", "ttft_p99", "itl_p50", "itl_p99", "itl_worst",
                                "tok_s", "good_s", "good_pct", "span_s", "xfer_ms_mean")}
        out.update(label=label, **(extra or {}))
        return out
    ttft = [r.first_token - r.arrival for r in made]
    itl = [gap for r in made for gap in r.itls]
    worst = [max(r.itls) if r.itls else 0.0 for r in made]
    good = [r for r in made
            if r.first_token - r.arrival <= TTFT_SLO_MS and (not r.itls or max(r.itls) <= ITL_SLO_MS)]
    span = max(r.finish for r in made) - min(r.arrival for r in requests)
    out = {
        "label": label,
        "ttft_p50": percentile(ttft, 50),
        "ttft_p99": percentile(ttft, 99),
        "itl_p50": percentile(itl, 50),
        "itl_p99": percentile(itl, 99),
        "itl_worst": percentile(worst, 99),
        "tok_s": sum(r.tokens for r in made) / (span / 1000),
        "good_s": len(good) / (span / 1000),
        "good_pct": len(good) / max(1, len(made)),
        "span_s": span / 1000,
        "xfer_ms_mean": statistics.mean(r.xfer_ms for r in made),
    }
    out.update(extra or {})
    return out


# --------------------------------------------------------------------------- #
# Disaggregated: P engines prefill, the KV cache crosses a link, D engines decode
# --------------------------------------------------------------------------- #


def simulate_disagg(requests, p_engines, d_engines, fabric="nvlink", local_max=0, name=""):
    """Event-driven: arrival -> prefill engine -> KV transfer -> decode engine.

    `local_max` > 0 turns on conditional disaggregation: requests whose uncached
    prompt fits in `local_max` tokens are prefilled by the decode engine itself,
    in `LOCAL_CHUNK` chunks, and never touch the link.
    """
    bw, lat = FABRICS[fabric]
    for r in requests:  # fresh copy of the mutable fields
        r.remaining_prefill, r.tokens, r.itls, r.last_token = r.isl - r.cached, 0, [], 0.0
        r.first_token = r.finish = 0.0
        r.xfer_ms = 0.0

    p_free = [0.0] * p_engines
    d_active = [[] for _ in range(d_engines)]
    d_waiting = [[] for _ in range(d_engines)]
    d_local = [[] for _ in range(d_engines)]  # conditional: local prefills, chunk by chunk
    d_busy = [None] * d_engines  # time of the step in flight
    link_free = 0.0
    transferred_gb = 0.0

    events = [(r.arrival, 0, r) for r in requests]
    heapq.heapify(events)

    def admit(eng, now):
        """Top up the batch from the queue, as far as KV memory allows."""
        used = sum(kv_gb(r.isl + r.tokens) for r in d_active[eng])
        while d_waiting[eng] and len(d_active[eng]) < MAX_BATCH:
            r = d_waiting[eng][0]
            need = kv_gb(r.isl + r.tokens)
            if used + need > KV_BUDGET_GB:
                break
            d_waiting[eng].pop(0)
            d_active[eng].append(r)
            used += need

    def start_step(eng, now):
        """Run one decode step, with at most one local prefill chunk in front of it."""
        extra = 0.0
        if d_local[eng]:
            r = d_local[eng][0]
            take = min(r.remaining_prefill, LOCAL_CHUNK)
            r.remaining_prefill -= take
            extra = prefill_ms(take)
            if r.remaining_prefill <= 0:
                d_local[eng].pop(0)
                r.prefill_done = now + extra
                d_waiting[eng].append(r)
        admit(eng, now)
        if not d_active[eng] and not d_local[eng]:
            d_busy[eng] = None
            return
        kv = sum(r.isl + r.tokens for r in d_active[eng])
        step = decode_ms(len(d_active[eng]), kv) if d_active[eng] else 0.0
        dur = iteration_ms(extra, step)
        d_busy[eng] = now + dur
        heapq.heappush(events, (d_busy[eng], 3, eng))

    def pick_decode(now):
        """Least loaded decode engine: fewest requests already holding a seat or a queue slot."""
        return min(range(d_engines), key=lambda e: len(d_active[e]) + len(d_waiting[e]) + len(d_local[e]))

    while events:
        now, kind, payload = heapq.heappop(events)
        if kind == 0:  # a request arrives
            r = payload
            if local_max and r.remaining_prefill <= local_max:
                eng = pick_decode(now)
                d_local[eng].append(r)
                if d_busy[eng] is None:
                    start_step(eng, now)
                continue
            pe = min(range(p_engines), key=lambda e: p_free[e])
            start = max(now, p_free[pe])
            p_free[pe] = start + prefill_ms(r.remaining_prefill)
            r.prefill_done = p_free[pe]
            heapq.heappush(events, (p_free[pe], 1, r))
        elif kind == 1:  # prefill finished: hand the KV cache to the decode side
            r = payload
            start = max(now, link_free)
            dur = kv_gb(r.isl - r.cached) / bw * 1000 + lat
            r.xfer_ms = dur + (start - now)  # transfer plus any queueing on the link
            link_free = start + dur
            transferred_gb += kv_gb(r.isl - r.cached)
            heapq.heappush(events, (link_free, 2, r))
        elif kind == 2:  # KV arrived: wait for a decode seat
            eng = pick_decode(now)
            d_waiting[eng].append(payload)
            if d_busy[eng] is None:
                start_step(eng, now)
        else:  # a decode step finished
            eng = payload
            for r in d_active[eng]:
                if r.tokens == 0:
                    r.first_token = now
                else:
                    r.itls.append(now - r.last_token)
                r.last_token = now
                r.tokens += 1
                if r.tokens >= r.osl:
                    r.finish = now
            d_active[eng] = [r for r in d_active[eng] if r.tokens < r.osl]
            d_busy[eng] = None
            start_step(eng, now)

    m = metrics(requests, name or f"{p_engines}P{d_engines}D",
                {"xfers_gb": transferred_gb, "gpus": p_engines + d_engines})
    m["xfer_gb_s"] = transferred_gb / m["span_s"]
    return m


# --------------------------------------------------------------------------- #
# Colocated: one GPU per engine, prefill and decode share it
# --------------------------------------------------------------------------- #


def simulate_colocated(requests, engines, prefill_budget=8192, name=""):
    """Each engine runs iterations of [prefill chunks, one decode step].

    `prefill_budget` is how many prompt tokens one iteration may take: the whole
    prompt (8192, vLLM's default batch budget) or a fixed chunk. A smaller chunk
    protects ITL and costs TTFT, which is the trade disaggregation is meant to
    escape.
    """
    for r in requests:
        r.remaining_prefill, r.tokens, r.itls, r.last_token = r.isl - r.cached, 0, [], 0.0
        r.first_token = r.finish = 0.0
    # Least-work-first dispatch, so the engines see the same offered load.
    per_engine = [[] for _ in range(engines)]
    load = [0.0] * engines
    for r in requests:
        e = min(range(engines), key=lambda i: load[i])
        per_engine[e].append(r)
        load[e] += prefill_ms(r.remaining_prefill) + r.osl * 15.0

    for e in range(engines):
        run_colocated_engine(per_engine[e], prefill_budget)
    return metrics(requests, name or f"colocated x{engines}", {"gpus": engines})


def run_colocated_engine(reqs, prefill_budget):
    """One GPU: every iteration does up to `prefill_budget` prefill tokens, then one decode step."""
    reqs = sorted(reqs, key=lambda r: r.arrival)
    now, i, active, waiting, blocked = 0.0, 0, [], [], []
    while i < len(reqs) or active or waiting or blocked:
        if not active and not waiting and not blocked:
            now = max(now, reqs[i].arrival)
        while i < len(reqs) and reqs[i].arrival <= now:
            waiting.append(reqs[i])
            i += 1

        budget, taken = prefill_budget, 0
        ready = []
        while waiting and budget > 0:
            r = waiting[0]
            take = min(r.remaining_prefill, budget)
            r.remaining_prefill -= take
            budget -= take
            taken += take
            if r.remaining_prefill <= 0:
                waiting.pop(0)
                r.prefill_done = now
                ready.append(r)

        # A request that just finished prefill gets its first token in this step.
        candidates = blocked + ready
        blocked = []
        used = sum(kv_gb(r.isl + r.tokens) for r in active)
        for r in candidates:
            need = kv_gb(r.isl + r.tokens)
            if len(active) < MAX_BATCH and used + need <= KV_BUDGET_GB:
                active.append(r)
                used += need
            else:
                blocked.append(r)

        chunk = prefill_ms(taken) if taken else 0.0
        step = decode_ms(len(active), sum(r.isl + r.tokens for r in active)) if active else 0.0
        now += iteration_ms(chunk, step)

        for r in active:
            if r.tokens == 0:
                r.first_token = now
            else:
                r.itls.append(now - r.last_token)
            r.last_token = now
            r.tokens += 1
            if r.tokens >= r.osl:
                r.finish = now
        active = [r for r in active if r.tokens < r.osl]
        if not active and not waiting and not blocked and i == len(reqs):
            break
    return now


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def header(ctx):
    print(f"  model: {PARAMS_B / 1e9:.0f}B dense, {WEIGHT_GB:.0f} GB fp8 weights, "
          f"{KV_BYTES_PER_TOKEN / 1024:.0f} kB of KV per token")
    print(f"  engine: prefill {PREFILL_TPS / 1000:.1f}k tok/s (compute bound), "
          f"decode {decode_ms(1, 0):.1f} ms/step at batch 1 (memory bound)")
    print(f"  decode KV budget per GPU: {KV_BUDGET_GB:.0f} GB "
          f"({KV_BUDGET_GB * 1e9 / KV_BYTES_PER_TOKEN / 1000:.0f}k tokens)")
    print(f"  targets: TTFT <= {TTFT_SLO_MS:.0f} ms, every ITL <= {ITL_SLO_MS:.0f} ms; "
          f"prefill/decode overlap loss {OVERLAP_LOSS:.0%}")
    print(f"  workload: {ctx['workload']} (ISL {ctx['isl'][0]}-{ctx['isl'][1]}, "
          f"OSL {ctx['osl'][0]}-{ctx['osl'][1]}, {ctx['prefix_share']:.0%} prefix hits), "
          f"{ctx['rate']:g} req/s offered, {ctx['n']} requests")


def table(rows, title, note=""):
    print(f"\n{title}")
    link = any("xfer_gb_s" in m for m in rows)
    head = (f"  {'config':<22} {'GPUs':>4} {'TTFT p50':>9} {'TTFT p99':>9} {'ITL p50':>8} "
            f"{'ITL p99':>8} {'worst':>7} {'tok/s':>7} {'good/s':>7} {'good %':>7}"
            + (f" {'KV GB/s':>6} {'xfer':>8}" if link else ""))
    print(head)
    print("  " + "-" * (len(head) - 2))
    for m in rows:
        tail = (f" {m['xfer_gb_s']:>6.1f} {m['xfer_ms_mean']:>7.0f}m"
                if link and "xfer_gb_s" in m else (" " * 15 if link else ""))
        print(f"  {m['label']:<22} {m['gpus']:>4} {m['ttft_p50']:>8.0f}m {m['ttft_p99']:>8.0f}m "
              f"{m['itl_p50']:>7.0f}m {m['itl_p99']:>7.0f}m {m['itl_worst']:>6.0f}m "
              f"{m['tok_s']:>7.0f} {m['good_s']:>7.1f} {m['good_pct']:>6.0%}{tail}")
    if note:
        print(f"  {note}")


def plot(load_rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in dict.fromkeys(r["label"] for r in load_rows):
        mine = [r for r in load_rows if r["label"] == label]
        ax.plot([r["rate"] for r in mine], [r["good_s"] for r in mine], marker="o", label=label)
    ax.set(xlabel="offered load (req/s)", ylabel="goodput (req/s inside both targets)",
           title="Goodput: colocated vs disaggregated")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved {os.path.basename(path)}")


def main():
    global OVERLAP_LOSS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workload", default="code", choices=sorted(WORKLOADS))
    p.add_argument("--rate", type=float, default=6.0, help="offered requests per second")
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fabric", default="nvlink", choices=sorted(FABRICS))
    p.add_argument("--overlap-loss", type=float, default=OVERLAP_LOSS,
                   help="share of the smaller phase that does not hide behind the larger")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    OVERLAP_LOSS = args.overlap_loss

    wl = WORKLOADS[args.workload]
    ctx = dict(workload=args.workload, rate=args.rate, n=args.requests, **wl)
    print()
    header(ctx)
    base = make_requests(args.rate, args.workload, args.requests, args.seed)
    gpus = 4

    # 1. The same four GPUs, five ways.
    rows = [
        simulate_colocated(list(base), gpus, 8192, "colocated, no chunk"),
        simulate_colocated(list(base), gpus, 512, "colocated, chunk 512"),
        simulate_colocated(list(base), gpus, 256, "colocated, chunk 256"),
        simulate_colocated(list(base), gpus, 128, "colocated, chunk 128"),
        simulate_disagg(list(base), 1, 3, args.fabric, name="1P3D"),
        simulate_disagg(list(base), 2, 2, args.fabric, name="2P2D"),
        simulate_disagg(list(base), 3, 1, args.fabric, name="3P1D"),
        simulate_disagg(list(base), 2, 2, args.fabric, LOCAL_PREFILL_MAX, "2P2D conditional"),
    ]
    table(rows, f"1. same {gpus} GPUs, {args.rate:g} req/s offered, {args.fabric} KV transfer",
          "good % = requests that met both targets; worst = p99 of each request's worst ITL gap")

    # 2. Where the GPUs should go: prefill or decode.
    rows, total = [], 8
    heavy_split = make_requests(args.rate * 2, args.workload, args.requests, args.seed)
    rows.append(simulate_colocated(list(heavy_split), total, 256, f"colocated x{total}, chunk 256"))
    for pe in (1, 2, 3, 4, 6, 7):
        rows.append(simulate_disagg(list(heavy_split), pe, total - pe, args.fabric))
    table(rows, f"2. xPyD split of {total} GPUs, {args.rate * 2:g} req/s, {args.fabric}",
          "prefill scales with prompt tokens, decode with concurrent seats: the split follows ISL:OSL")

    # 3. The chunk-size frontier inside one colocated engine.
    rows = [simulate_colocated(list(base), gpus, b, f"colocated, chunk {b}") for b in (8192, 2048, 512, 256, 128)]
    table(rows, f"3. colocated chunk size, {gpus} GPUs",
          "smaller chunk protects ITL and spends TTFT; that is the trade disaggregation removes")

    # 4. The link, at 3x the load so the transfer is a real share of the work.
    heavy = make_requests(args.rate * 2, args.workload, args.requests, args.seed)
    rows = [simulate_disagg(list(heavy), 4, 4, f, name=f"4P4D, {f}") for f in ("nvlink", "ib800", "pcie")]
    table(rows, f"4. KV transfer fabric, 4P4D, {args.rate * 2:g} req/s",
          "xfer = mean transfer time per request, queueing included; a 5k-token prompt is 1.6 GB of KV")

    # 5. Goodput against load, the whole point.
    load_rows = []
    for rate in (2, 4, 6, 8, 10, 12, 16):
        reqs = make_requests(rate, args.workload, args.requests, args.seed)
        for row in (simulate_colocated(list(reqs), gpus, 256, f"colocated @ {rate:g}/s"),
                    simulate_disagg(list(reqs), 2, 2, args.fabric, name=f"2P2D @ {rate:g}/s")):
            row["rate"] = rate
            load_rows.append(row)
    table(load_rows, f"5. goodput against offered load, {gpus} GPUs", "")

    # 6. The one assumption that decides the answer: how much of the smaller phase
    #    hides behind the larger one inside a colocated iteration.
    print(f"\n  6. sensitivity: how much of the hidden phase is actually lost, "
          f"{args.rate * 4 / 3:g} req/s")
    print(f"  {'overlap loss':>13} {'colocated, chunk 256':>22} {'2P2D':>8}   verdict")
    print("  " + "-" * 62)
    tuned = make_requests(args.rate * 4 / 3, args.workload, args.requests, args.seed)
    fixed = simulate_disagg(list(tuned), 2, 2, args.fabric)["good_pct"]
    for loss in (0.0, 0.3, 0.6, 1.0):
        OVERLAP_LOSS = loss
        fused = simulate_colocated(list(tuned), gpus, 256)["good_pct"]
        verdict = "colocation" if fused > fixed else "disaggregation"
        print(f"  {loss:>12.0%} {fused:>21.0%} {fixed:>7.0%}   {verdict}")
    OVERLAP_LOSS = args.overlap_loss
    print("  colocation only wins while the smaller phase mostly hides; that is a property")
    print("  of the kernels and the batch mix on your hardware, not of the math above")
    if not args.no_plot:
        plot(load_rows, os.path.join(os.path.dirname(os.path.abspath(__file__)), "goodput.png"))


if __name__ == "__main__":
    main()
