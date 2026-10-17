import os
import random
import statistics
from dataclasses import dataclass

N_REQUESTS = 500
MAX_BATCH = 32
STEP_BASE_MS, STEP_PER_SEQ_MS = 6.0, 0.35
PREFILL_MS_PER_TOKEN = 0.05
PROMPT_TOKENS = (20, 400)
SHORT_ANSWER, LONG_ANSWER, LONG_SHARE = (10, 60), (200, 500), 0.2
ARRIVAL_RATES = [2, 5, 10, 20, 40, 80]
SEED = 0


@dataclass
class Request:
    """One user request and the times it reached each stage."""

    arrival: float
    prompt_len: int
    output_len: int
    generated: int = 0
    first_token: float = 0.0
    finished: float = 0.0


def make_requests(rate, seed=SEED):
    """Make N_REQUESTS requests arriving at random (Poisson) at `rate` per second."""
    rng = random.Random(seed)
    t, requests = 0.0, []
    for _ in range(N_REQUESTS):
        t += rng.expovariate(rate) * 1000
        answer = LONG_ANSWER if rng.random() < LONG_SHARE else SHORT_ANSWER
        requests.append(Request(t, rng.randint(*PROMPT_TOKENS), rng.randint(*answer)))
    return requests


def step_ms(batch):
    """How long one decode step takes for `batch` users: weights read once, a little extra per user."""
    return STEP_BASE_MS + STEP_PER_SEQ_MS * batch


def prefill(requests, now):
    """Prefill new requests one after another; each gets its first token when its prefill ends."""
    for r in requests:
        now += PREFILL_MS_PER_TOKEN * r.prompt_len
        r.generated, r.first_token = 1, now
        if r.output_len == 1:
            r.finished = now
    return now


def static_batching(requests):
    """Take whoever is waiting, run the whole group until its longest answer ends, then repeat."""
    now, i, steps, used_steps = 0.0, 0, 0, 0
    while i < len(requests):
        now = max(now, requests[i].arrival)
        batch = []
        while i < len(requests) and requests[i].arrival <= now and len(batch) < MAX_BATCH:
            batch.append(requests[i])
            i += 1
        now = prefill(batch, now)
        while any(r.generated < r.output_len for r in batch):
            now += step_ms(len(batch))
            steps += 1
            for r in batch:
                if r.generated < r.output_len:
                    used_steps += 1
                    r.generated += 1
                    if r.generated == r.output_len:
                        r.finished = now
    return now, used_steps / max(steps, 1)


def continuous_batching(requests):
    """Every step, let finished users leave and waiting users take the free seats."""
    now, i, active, steps, used_steps = 0.0, 0, [], 0, 0
    while i < len(requests) or active:
        if not active:
            now = max(now, requests[i].arrival)
        joining = []
        while i < len(requests) and requests[i].arrival <= now and len(active) + len(joining) < MAX_BATCH:
            joining.append(requests[i])
            i += 1
        now = prefill(joining, now)
        active = [r for r in active + joining if r.generated < r.output_len]
        if not active:
            continue
        now += step_ms(len(active))
        steps += 1
        used_steps += len(active)
        for r in active:
            r.generated += 1
            if r.generated == r.output_len:
                r.finished = now
        active = [r for r in active if r.generated < r.output_len]
    return now, used_steps / max(steps, 1)


def percentile(values, p):
    """The value below which p percent of `values` fall."""
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def report(requests, end_ms, avg_batch):
    """Throughput and latency numbers for one finished simulation."""
    tokens = sum(r.output_len for r in requests)
    ttft = [r.first_token - r.arrival for r in requests]
    e2e = [r.finished - r.arrival for r in requests]
    return {
        "tok/s": tokens / (end_ms / 1000),
        "ttft_p50": percentile(ttft, 50),
        "ttft_p99": percentile(ttft, 99),
        "e2e_p50": percentile(e2e, 50),
        "e2e_p99": percentile(e2e, 99),
        "short_e2e": statistics.mean(r.finished - r.arrival for r in requests if r.output_len <= SHORT_ANSWER[1]),
        "batch": avg_batch,
    }


def tiny_example():
    """Three users with answers of 2, 5 and 3 tokens: who holds a seat at each step."""
    print("Tiny example: answers of 2, 5 and 3 tokens, seats A B C (x = making a token, . = idle seat)")
    lengths = {"A": 2, "B": 5, "C": 3}
    print("  static:     ", "  ".join(f"{u}:" + "".join("x" if s < n else "." for s in range(5)) for u, n in lengths.items()))
    print("  continuous: ", "  ".join(f"{u}:" + "x" * n + " " * (5 - n) for u, n in lengths.items()),
          " (A's seat is free after step 2 for a new user)")
    print()


def main():
    """Run both schedulers at several arrival rates and print a comparison table."""
    tiny_example()
    print(f"{N_REQUESTS} requests, max batch {MAX_BATCH}, {LONG_SHARE:.0%} long answers. Times in ms.\n")
    header = f"{'req/s':>5} {'scheduler':>10} {'tok/s':>7} {'TTFT p50':>9} {'TTFT p99':>9} {'E2E p50':>8} {'E2E p99':>8} {'short E2E':>9} {'avg batch':>9}"
    print(header)
    print("-" * len(header))
    rows = []
    for rate in ARRIVAL_RATES:
        for name, scheduler in (("static", static_batching), ("continuous", continuous_batching)):
            requests = make_requests(rate)
            end_ms, avg_batch = scheduler(requests)
            m = report(requests, end_ms, avg_batch)
            rows.append((rate, name, m))
            print(f"{rate:>5} {name:>10} {m['tok/s']:>7.0f} {m['ttft_p50']:>9.0f} {m['ttft_p99']:>9.0f} "
                  f"{m['e2e_p50']:>8.0f} {m['e2e_p99']:>8.0f} {m['short_e2e']:>9.0f} {m['batch']:>9.1f}")
    plot(rows)


def plot(rows):
    """Save throughput and p99 TTFT against arrival rate for both schedulers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("static", "continuous"):
        mine = [(rate, m) for rate, n, m in rows if n == name]
        left.plot([r for r, _ in mine], [m["tok/s"] for _, m in mine], marker="o", label=name)
        right.plot([r for r, _ in mine], [m["ttft_p99"] / 1000 for _, m in mine], marker="o", label=name)
    left.set(xlabel="arrival rate (req/s)", ylabel="total tokens/s", title="Throughput")
    right.set(xlabel="arrival rate (req/s)", ylabel="TTFT p99 (s)", title="Tail time to first token", yscale="log")
    for ax in (left, right):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "batching_sim.png"), dpi=120)
    print("\nSaved batching_sim.png")


if __name__ == "__main__":
    main()
