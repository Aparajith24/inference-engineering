import argparse
import asyncio
import json
import statistics
import time

import httpx

PROMPTS = [
    "Explain how a GPU runs a large language model, step by step.",
    "Write a short story about a robot who wants to learn how to paint.",
    "What are the pros and cons of continuous batching?",
    "Describe the city of Paris to someone who has never been there.",
    "List ten ideas for a weekend project with a Raspberry Pi and explain each one.",
    "Why is the sky blue? Give a detailed answer.",
    "Summarize the history of the internet in a few paragraphs.",
    "How does a KV cache make text generation faster?",
]


def percentile(values, p):
    """The value below which p percent of `values` fall (0 if there are none)."""
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


async def one_request(client, args, prompt):
    """Stream one completion and record when it started, when each piece of text arrived, and the token count."""
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        body["ignore_eos"] = True
    start = time.perf_counter()
    arrivals, tokens = [], 0
    try:
        async with client.stream("POST", f"{args.url}/completions", json=body) as response:
            if response.status_code != 200:
                await response.aread()
                return {"error": f"{response.status_code} {response.text[:100]}"}
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    arrivals.append(time.perf_counter())
                if chunk.get("usage"):
                    tokens = chunk["usage"]["completion_tokens"]
    except httpx.HTTPError as error:
        return {"error": repr(error)}
    end = time.perf_counter()
    if not arrivals:
        return {"error": "empty answer (the model stopped at once)"}
    return {
        "ttft": arrivals[0] - start,
        "itl": [b - a for a, b in zip(arrivals, arrivals[1:])],
        "e2e": end - start,
        "tokens": tokens or len(arrivals),
        "decode_time": arrivals[-1] - arrivals[0],
    }


async def run_level(client, args, concurrency):
    """Keep `concurrency` requests in flight until the level's requests are all done."""
    n_requests = max(args.requests, concurrency * args.rounds)
    counter = iter(range(n_requests))
    results = []

    async def user():
        """One simulated user: send a request, wait for the whole answer, send the next."""
        for i in counter:
            results.append(await one_request(client, args, PROMPTS[i % len(PROMPTS)]))

    start = time.perf_counter()
    await asyncio.gather(*(user() for _ in range(concurrency)))
    wall = time.perf_counter() - start
    return summarize(concurrency, results, wall)


def summarize(concurrency, results, wall):
    """Turn one level's raw timings into TTFT, ITL, end-to-end and throughput numbers."""
    good = [r for r in results if "error" not in r]
    errors = [r["error"] for r in results if "error" in r]
    ttft = [r["ttft"] * 1000 for r in good]
    itl = [x * 1000 for r in good for x in r["itl"]]
    e2e = [r["e2e"] for r in good]
    tokens = sum(r["tokens"] for r in good)
    per_user = [(r["tokens"] - 1) / r["decode_time"] for r in good if r["decode_time"] > 0]
    return {
        "concurrency": concurrency,
        "requests": len(good),
        "errors": len(errors),
        "first_error": errors[0] if errors else None,
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p99_ms": percentile(ttft, 99),
        "itl_p50_ms": percentile(itl, 50),
        "itl_p99_ms": percentile(itl, 99),
        "e2e_p50_s": percentile(e2e, 50),
        "e2e_p99_s": percentile(e2e, 99),
        "output_tok_s": tokens / wall,
        "per_user_tok_s": statistics.mean(per_user) if per_user else 0.0,
        "requests_per_s": len(good) / wall,
    }


def print_table(rows):
    """Print one line per concurrency level."""
    header = (f"{'users':>5} {'ok':>4} {'err':>3} {'TTFT p50':>8} {'TTFT p99':>8} {'ITL p50':>7} {'ITL p99':>7} "
              f"{'E2E p50':>7} {'E2E p99':>7} {'total tok/s':>11} {'per user':>8} {'req/s':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['concurrency']:>5} {r['requests']:>4} {r['errors']:>3} {r['ttft_p50_ms']:>8.0f} {r['ttft_p99_ms']:>8.0f} "
              f"{r['itl_p50_ms']:>7.1f} {r['itl_p99_ms']:>7.1f} {r['e2e_p50_s']:>7.2f} {r['e2e_p99_s']:>7.2f} "
              f"{r['output_tok_s']:>11.0f} {r['per_user_tok_s']:>8.1f} {r['requests_per_s']:>6.2f}")
    print("\nTTFT and ITL in ms, E2E in s. per user = tokens per second each user sees while their answer streams.")


def plot(rows, path):
    """Save throughput, TTFT and ITL against concurrency."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    users = [r["concurrency"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(users, [r["output_tok_s"] for r in rows], marker="o", label="total")
    axes[0].plot(users, [r["per_user_tok_s"] for r in rows], marker="o", label="per user")
    axes[0].set(title="Throughput", ylabel="tokens/s")
    axes[1].plot(users, [r["ttft_p50_ms"] for r in rows], marker="o", label="p50")
    axes[1].plot(users, [r["ttft_p99_ms"] for r in rows], marker="o", label="p99")
    axes[1].set(title="Time to first token", ylabel="ms")
    axes[2].plot(users, [r["itl_p50_ms"] for r in rows], marker="o", label="p50")
    axes[2].plot(users, [r["itl_p99_ms"] for r in rows], marker="o", label="p99")
    axes[2].set(title="Inter-token latency", ylabel="ms")
    for ax in axes:
        ax.set(xlabel="concurrent users", xscale="log")
        ax.set_xticks(users)
        ax.set_xticklabels(users)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


async def main(args):
    """Warm up, sweep the concurrency levels, then print, save and plot the results."""
    levels = [int(x) for x in args.concurrency.split(",")]
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=max(levels) + 1, max_keepalive_connections=max(levels) + 1)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        warmup = await one_request(client, args, PROMPTS[0])
        if "error" in warmup:
            raise SystemExit(f"Warm-up failed: {warmup['error']}")
        rows = []
        for concurrency in levels:
            print(f"concurrency {concurrency} ...", flush=True)
            rows.append(await run_level(client, args, concurrency))
    print()
    print_table(rows)
    with open(args.out, "w") as f:
        json.dump({"settings": vars(args), "results": rows}, f, indent=2)
    print(f"Saved {args.out}")
    plot(rows, args.out.rsplit(".", 1)[0] + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async load tester for an OpenAI-compatible /v1/completions server")
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    parser.add_argument("--requests", type=int, default=16, help="at least this many requests per level")
    parser.add_argument("--rounds", type=int, default=3, help="each user sends at least this many requests")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--out", default="load_test_results.json")
    asyncio.run(main(parser.parse_args()))
