import random
from collections import OrderedDict

BLOCK_SIZE = 16
N_REPLICAS = 4
CACHE_BLOCKS = 1000
PREFILL_MS_PER_TOKEN = 0.2
DECODE_MS_PER_TOKEN = 0.5
N_SYSTEM_PROMPTS = 8
SYSTEM_TOKENS = (300, 1500)
N_CONVERSATIONS = 400
TURNS = (1, 6)
USER_TOKENS = (20, 200)
ANSWER_TOKENS = (30, 200)
THINK_SECONDS = (2, 15)
CONVERSATIONS_PER_SECOND = 6
LOAD_WEIGHT = 1.0
SEED = 0


def block_hashes(tokens):
    """One hash per full block; each hash includes the one before it, so equal hashes mean equal prefixes."""
    hashes, parent = [], None
    for i in range(0, len(tokens) - BLOCK_SIZE + 1, BLOCK_SIZE):
        parent = hash((parent, tuple(tokens[i:i + BLOCK_SIZE])))
        hashes.append(parent)
    return hashes


class Replica:
    """One model server: a prefix cache of KV blocks (least recently used are evicted) and a queue."""

    def __init__(self, name):
        self.name = name
        self.cache = OrderedDict()
        self.busy_until = 0.0
        self.served = 0

    def cached_prefix(self, hashes):
        """How many blocks at the start of this prompt are already in the cache."""
        n = 0
        for h in hashes:
            if h not in self.cache:
                break
            n += 1
        return n

    def queue_ms(self, now):
        """How long a new request would wait before this replica starts on it."""
        return max(0.0, self.busy_until - now)

    def remember(self, hashes):
        """Store these blocks in the cache, marking them as just used, and evict the oldest if full."""
        for h in hashes:
            self.cache[h] = True
            self.cache.move_to_end(h)
        while len(self.cache) > CACHE_BLOCKS:
            self.cache.popitem(last=False)


def make_turns(seed=SEED):
    """Multi-turn conversations: each turn's prompt is the system prompt plus the whole chat so far."""
    rng = random.Random(seed)
    systems = [[rng.randrange(50000) for _ in range(rng.randint(*SYSTEM_TOKENS))] for _ in range(N_SYSTEM_PROMPTS)]
    turns, start = [], 0.0
    for c in range(N_CONVERSATIONS):
        start += rng.expovariate(CONVERSATIONS_PER_SECOND)
        history, t = list(rng.choice(systems)), start
        for _ in range(rng.randint(*TURNS)):
            history += [rng.randrange(50000) for _ in range(rng.randint(*USER_TOKENS))]
            answer = [rng.randrange(50000) for _ in range(rng.randint(*ANSWER_TOKENS))]
            turns.append((t * 1000, c, list(history), answer))
            history += answer
            t += rng.uniform(*THINK_SECONDS) + len(answer) * DECODE_MS_PER_TOKEN / 1000
    return sorted(turns, key=lambda turn: turn[0])


def round_robin(replicas, hashes, now, state):
    """Next replica in turn, ignoring caches and load."""
    state["next"] = (state.get("next", -1) + 1) % len(replicas)
    return replicas[state["next"]]


def random_choice(replicas, hashes, now, state):
    """Any replica at random."""
    return state.setdefault("rng", random.Random(SEED)).choice(replicas)


def least_loaded(replicas, hashes, now, state):
    """The replica with the shortest queue."""
    return min(replicas, key=lambda r: r.queue_ms(now))


def best_prefix(replicas, hashes, now, state):
    """The replica holding the longest prefix of this prompt, even if it is busy."""
    return max(replicas, key=lambda r: (r.cached_prefix(hashes), -r.queue_ms(now)))


def kv_aware(replicas, hashes, now, state):
    """Lowest estimated time to first token: queue wait plus prefill of the part that is not cached."""
    def cost(r):
        uncached = (len(hashes) - r.cached_prefix(hashes)) * BLOCK_SIZE
        return LOAD_WEIGHT * r.queue_ms(now) + uncached * PREFILL_MS_PER_TOKEN, r.served
    return min(replicas, key=cost)


def percentile(values, p):
    """The value below which p percent of `values` fall."""
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def simulate(router, turns):
    """Send every turn through one router and measure cache hits and time to first token."""
    replicas = [Replica(f"r{i}") for i in range(N_REPLICAS)]
    state, ttfts, prompt_tokens, cached_tokens = {}, [], 0, 0
    for now, _, prompt, answer in turns:
        hashes = block_hashes(prompt)
        replica = router(replicas, hashes, now, state)
        hit = replica.cached_prefix(hashes) * BLOCK_SIZE
        start = now + replica.queue_ms(now)
        prefill_ms = (len(prompt) - hit) * PREFILL_MS_PER_TOKEN
        ttfts.append(start + prefill_ms - now)
        replica.busy_until = start + prefill_ms + len(answer) * DECODE_MS_PER_TOKEN
        replica.remember(block_hashes(prompt + answer))
        replica.served += 1
        prompt_tokens += len(prompt)
        cached_tokens += hit
    served = [r.served for r in replicas]
    return {
        "hit": cached_tokens / prompt_tokens,
        "p50": percentile(ttfts, 50),
        "p99": percentile(ttfts, 99),
        "spread": f"{min(served)}-{max(served)}",
    }


def demo():
    """Show how block hashes find a shared prefix."""
    system = list(range(40))
    a, b, c = system + [100, 101, 102] * 4, system + [200] * 12, [7] + system
    print(f"Block hashes (blocks of {BLOCK_SIZE} tokens):")
    print(f"  prompt A: {len(a)} tokens -> {len(block_hashes(a))} full blocks")
    shared = sum(x == y for x, y in zip(block_hashes(a), block_hashes(b)))
    print(f"  prompt B: same 40-token system prompt, different question -> shares {shared} blocks with A")
    shared = sum(x == y for x, y in zip(block_hashes(a), block_hashes(c)))
    print(f"  prompt C: one extra token at the front -> shares {shared} blocks with A (a prefix must match from token 0)")
    print()


def main():
    """Compare five routers on the same conversations."""
    demo()
    turns = make_turns()
    tokens = sum(len(p) for _, _, p, _ in turns)
    print(f"{len(turns)} requests from {N_CONVERSATIONS} conversations, {N_SYSTEM_PROMPTS} system prompts, "
          f"{N_REPLICAS} replicas, {tokens / len(turns):.0f} prompt tokens on average\n")
    header = f"{'router':>14} {'cache hit':>9} {'TTFT p50 ms':>11} {'TTFT p99 ms':>11} {'requests per replica':>20}"
    print(header)
    print("-" * len(header))
    for name, router in (("round robin", round_robin), ("random", random_choice), ("least loaded", least_loaded),
                         ("best prefix", best_prefix), ("kv-aware", kv_aware)):
        m = simulate(router, turns)
        print(f"{name:>14} {m['hit']:>9.0%} {m['p50']:>11.0f} {m['p99']:>11.0f} {m['spread']:>20}")


if __name__ == "__main__":
    main()
