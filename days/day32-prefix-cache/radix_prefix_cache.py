"""Day 32 - a prefix cache on a radix tree of token blocks.

Book chapter: 5.3 Caching (prefix caching and KV cache re-use)

The cache stores the KV of *whole blocks* of tokens. Each node of the radix
tree owns a run of blocks and remembers which tokens those blocks hold. A
request walks the tree block by block to find the longest prefix whose KV is
already in the pool, so prefill can skip those tokens.

    root
      "Weather in" | "Paris ?"     <- the first request inserted this
      "Weather in" | "Rome ?"      <- the second shares the first block

Two implementations are compared:

  RadixCache      the tree. Prefixes are nodes, so only leaves can be evicted
                  and what is left in the tree is always a set of valid
                  prefixes.
  FlatBlockCache  day 26's design: a flat dict of chained block hashes, LRU
                  per block. Simpler, but it can evict a block in the middle
                  of a prefix and leave the blocks after it stranded.

Run it with: python3 days/day32-prefix-cache/radix_prefix_cache.py
"""

from __future__ import annotations

import random
import time
from collections import OrderedDict
from dataclasses import dataclass

BLOCK_SIZE = 16                 # tokens of KV per block (vLLM's default page size)
PREFILL_MS_PER_TOKEN = 0.05     # 20k tokens/s of prefill, a small model on one GPU
PREFILL_BASE_MS = 2.0           # one kernel launch and the sampling of the first token
DECODE_MS_PER_TOKEN = 0.4
EVICT_HEADROOM = 0.15           # free this share of the pool at once, not one block at a time
SEED = 0

N_CONVERSATIONS = 200
TURNS = (1, 4)
N_SYSTEM_PROMPTS = 6
SYSTEM_TOKENS = (400, 1200)
USER_TOKENS = (20, 150)
ANSWER_TOKENS = (30, 200)
THINK_SECONDS = (2, 15)
CONVERSATIONS_PER_SECOND = 5
CACHE_BLOCKS = 800


# --------------------------------------------------------------------------
# the pool of blocks
# --------------------------------------------------------------------------


class BlockPool:
    """A fixed pool of KV blocks. One block holds BLOCK_SIZE tokens of K and V, for every layer."""

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.free = list(range(num_blocks - 1, -1, -1))   # pop() hands out 0, 1, 2, ...
        self.used = 0
        self.peak = 0

    def alloc(self, n):
        """Take n blocks off the free list."""
        if n > len(self.free):
            raise MemoryError(f"wanted {n} blocks, {len(self.free)} free")
        ids = [self.free.pop() for _ in range(n)]
        self.used += n
        self.peak = max(self.peak, self.used)
        return ids

    def release(self, ids):
        """Give blocks back."""
        self.free.extend(ids)
        self.used -= len(ids)


# --------------------------------------------------------------------------
# the radix tree
# --------------------------------------------------------------------------


def block_keys(tokens, block_size=BLOCK_SIZE):
    """Split tokens into whole blocks. A tail shorter than one block is never cached.

    Only whole blocks are shareable: a block is the unit of allocation, and two
    requests with different tokens in the same block cannot point at one block
    without copying its KV. So the last, partial block is always recomputed.
    """
    n = len(tokens) // block_size * block_size
    return tuple(tuple(tokens[i:i + block_size]) for i in range(0, n, block_size))


class Node:
    """One node: a run of whole blocks plus its children, keyed by the first block of theirs."""

    __slots__ = ("key", "blocks", "children", "parent", "ref_count", "last_used")

    def __init__(self, key, parent=None):
        self.key = key                # tuple of blocks, each a tuple of tokens
        self.blocks = []              # pool ids, one per block in key
        self.children = {}            # block -> Node
        self.parent = parent
        self.ref_count = 0            # live requests whose matched prefix ends at or below here
        self.last_used = 0


class RadixCache:
    """Prefix cache as a radix tree over blocks.

    Invariants:
      * every node owns exactly len(key) blocks from the pool;
      * a node's children start with different blocks, so a prefix match is a
        single walk down the tree;
      * only leaves are evicted, so every block still in the tree is reachable
        from the root and therefore usable by some future request.
    """

    def __init__(self, num_blocks, block_size=BLOCK_SIZE):
        self.block_size = block_size
        self.pool = BlockPool(num_blocks)
        self.root = Node(())
        self.clock = 0
        self.evicted_blocks = 0
        self.requests = 0
        self.hit_requests = 0
        self.blocks_requested = 0
        self.blocks_hit = 0
        self.tokens_requested = 0
        self.tokens_hit = 0

    # -- lookup ------------------------------------------------------------

    def match(self, keys):
        """Walk the tree for the longest cached prefix of `keys`.

        Returns (node, blocks_matched). `node` is the last node reached; holding
        it pins the blocks of the whole path until release(). A query that stops
        in the middle of an edge splits that edge, so the shared part becomes a
        node of its own.
        """
        node = self.root
        matched = 0
        while matched < len(keys):
            child = node.children.get(keys[matched])
            if child is None:
                break
            n = 0
            while (n < len(child.key)
                   and matched + n < len(keys)
                   and child.key[n] == keys[matched + n]):
                n += 1
            if n < len(child.key):
                child = self._split(child, n)
            matched += n
            node = child
        return node, matched

    def _split(self, node, n):
        """Cut an edge after n blocks. The head takes the node's place under its parent.

        No block is copied or freed: they stay allocated, split between the two
        nodes. Live requests are unaffected because ref_count moves to the head
        along with the blocks it now owns.
        """
        parent = node.parent
        head = Node(node.key[:n], parent)
        head.blocks = node.blocks[:n]
        head.ref_count = node.ref_count
        head.last_used = node.last_used
        node.key = node.key[n:]
        node.blocks = node.blocks[n:]
        node.parent = head
        head.children[node.key[0]] = node
        parent.children[head.key[0]] = head
        return head

    def lookup(self, tokens):
        """How much of `tokens` is already cached. Returns (node to release later, tokens hit)."""
        keys = block_keys(tokens, self.block_size)
        node, matched = self.match(keys)
        self.clock += 1
        self._touch(node)
        self._acquire(node)
        self.requests += 1
        self.tokens_requested += len(tokens)
        self.blocks_requested += len(keys)
        self.blocks_hit += matched
        self.tokens_hit += matched * self.block_size
        self.hit_requests += 1 if matched else 0
        return node, matched * self.block_size

    # -- insert ------------------------------------------------------------

    def insert(self, keys):
        """Record that these blocks now hold valid KV. Returns the number added."""
        node, matched = self.match(keys)
        rest = keys[matched:]
        if not rest:
            self.clock += 1
            self._touch(node)
            return 0
        self._make_room(len(rest))
        new = Node(rest, node)
        new.blocks = self.pool.alloc(len(rest))
        self.clock += 1
        new.last_used = self.clock
        node.children[rest[0]] = new
        return len(rest)

    # -- eviction ----------------------------------------------------------

    def _leaves(self):
        """Every node with no children, the only nodes that can be evicted."""
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
            else:
                yield node

    def _make_room(self, n):
        """Free pool blocks until n fit, with headroom so this is not a scan per insert."""
        if n > self.pool.num_blocks:
            raise MemoryError(
                f"a {n}-block prefix cannot fit in a pool of {self.pool.num_blocks} blocks"
            )
        if n <= len(self.pool.free):
            return
        target = max(n, int(self.pool.num_blocks * EVICT_HEADROOM))
        self._evict(target)
        if n > len(self.pool.free):
            raise MemoryError(
                f"no room for {n} block(s), {len(self.pool.free)} free: every leaf is held by a live request"
            )

    def _evict(self, target_free):
        """Remove least recently used leaves until the pool has target_free blocks free.

        A leaf held by a live request cannot go. A leaf can also give up only
        its last few blocks, keeping the head of its run, so a node longer than
        the whole pool is still evictable. Removing a leaf can make its parent a
        leaf, which the next round may remove in turn, oldest first.
        """
        while len(self.pool.free) < target_free:
            candidates = [n for n in self._leaves() if n.ref_count == 0]
            if not candidates:
                return
            candidates.sort(key=lambda n: n.last_used)
            for leaf in candidates:
                need = target_free - len(self.pool.free)
                if need <= 0:
                    return
                take = min(len(leaf.blocks), need)
                first = leaf.key[0]
                self.pool.release(leaf.blocks[-take:])
                leaf.blocks = leaf.blocks[:-take]
                leaf.key = leaf.key[:-take]
                self.evicted_blocks += take
                if not leaf.blocks:
                    del leaf.parent.children[first]

    # -- references --------------------------------------------------------

    def _path(self, node):
        while node is not self.root:
            yield node
            node = node.parent

    def _touch(self, node):
        for n in self._path(node):
            n.last_used = self.clock

    def _acquire(self, node):
        for n in self._path(node):
            n.ref_count += 1

    def release(self, node):
        """A request is done with these blocks; they may now be evicted."""
        for n in self._path(node):
            n.ref_count -= 1

    # -- stats -------------------------------------------------------------

    def count_nodes(self):
        stack, n = list(self.root.children.values()), 0
        while stack:
            node = stack.pop()
            n += 1
            stack.extend(node.children.values())
        return n

    def stats(self):
        return {
            "requests": self.requests,
            "hit_rate": self.hit_requests / max(1, self.requests),
            "token_hit_rate": self.tokens_hit / max(1, self.tokens_requested),
            "block_hit_rate": self.blocks_hit / max(1, self.blocks_requested),
            "reused_tokens": self.tokens_hit,
            "prefilled_tokens": self.tokens_requested - self.tokens_hit,
            "blocks_held": self.pool.used,
            "peak_blocks": self.pool.peak,
            "blocks_evicted": self.evicted_blocks,
            "nodes": self.count_nodes(),
            "stranded_blocks": 0,
        }


# --------------------------------------------------------------------------
# the baseline: a flat dict of chained block hashes (day 26)
# --------------------------------------------------------------------------


class FlatBlockCache:
    """One dict of chained block hashes, LRU per block. No tree.

    A block's hash includes every block before it, so equal hashes mean equal
    prefixes. That is enough to *find* a prefix, but eviction still happens one
    block at a time: a block in the middle of a prefix can go while the blocks
    after it stay. Those survivors are stranded - no request can use them,
    because a prefix has to match from the first token.

    Same pool size, same least-recently-used eviction and the same headroom as
    RadixCache, so the only difference is the data structure. Set
    refresh=False to see what happens when the lookup path forgets to mark a
    hit as recently used: long-lived shared prefixes sink and are evicted first.
    """

    def __init__(self, num_blocks, block_size=BLOCK_SIZE, refresh=True):
        self.capacity = num_blocks
        self.block_size = block_size
        self.refresh = refresh
        self.cache = OrderedDict()      # hash -> True, oldest first
        self.parent = {}                # hash -> hash of the block before it
        self.evicted_blocks = 0
        self.requests = 0
        self.hit_requests = 0
        self.blocks_requested = 0
        self.blocks_hit = 0
        self.tokens_requested = 0
        self.tokens_hit = 0

    @staticmethod
    def chain(keys):
        """One chained hash per block."""
        out, parent = [], None
        for block in keys:
            parent = hash((parent, block))
            out.append(parent)
        return out

    def lookup(self, tokens):
        keys = block_keys(tokens, self.block_size)
        hashes = self.chain(keys)
        matched = 0
        for h in hashes:
            if h not in self.cache:
                break
            if self.refresh:
                self.cache.move_to_end(h)
            matched += 1
        self.requests += 1
        self.tokens_requested += len(tokens)
        self.blocks_requested += len(keys)
        self.blocks_hit += matched
        self.tokens_hit += matched * self.block_size
        self.hit_requests += 1 if matched else 0
        return None, matched * self.block_size

    def insert(self, keys):
        """Add blocks whose KV was just computed. Already-cached blocks are left where they are.

        Only new blocks go to the end of the order, the way a real engine only
        writes the blocks it actually prefilled.
        """
        parent = None
        for block in keys:
            h = hash((parent, block))
            self.parent[h] = parent
            if h not in self.cache:
                self.cache[h] = True
                self.cache.move_to_end(h)
            parent = h
        target = max(0, int(self.capacity * EVICT_HEADROOM))
        if len(self.cache) > self.capacity:
            while len(self.cache) > self.capacity - target:
                self.cache.popitem(last=False)
                self.evicted_blocks += 1
        return len(keys)

    def release(self, node):
        """Nothing to release: this cache does not track live requests."""

    def unreachable(self):
        """Cached blocks whose ancestor chain is broken; their KV can never be read."""
        n = 0
        for h in self.cache:
            p = self.parent[h]
            while p is not None:
                if p not in self.cache:
                    n += 1
                    break
                p = self.parent[p]
        return n

    def stats(self):
        return {
            "requests": self.requests,
            "hit_rate": self.hit_requests / max(1, self.requests),
            "token_hit_rate": self.tokens_hit / max(1, self.tokens_requested),
            "block_hit_rate": self.blocks_hit / max(1, self.blocks_requested),
            "reused_tokens": self.tokens_hit,
            "prefilled_tokens": self.tokens_requested - self.tokens_hit,
            "blocks_held": len(self.cache),
            "peak_blocks": self.capacity,
            "blocks_evicted": self.evicted_blocks,
            "nodes": len(self.cache),
            "stranded_blocks": self.unreachable(),
        }


# --------------------------------------------------------------------------
# printing the tree
# --------------------------------------------------------------------------


def label(key, max_blocks=4):
    """A readable edge: the tokens of its blocks, joined."""
    parts = [" ".join(map(str, block)) for block in key[:max_blocks]]
    if len(key) > max_blocks:
        parts.append("…")
    return " | ".join(parts)


def tree_lines(cache):
    """The tree as indented text."""
    out = []

    def walk(node, depth):
        for child in sorted(node.children.values(), key=lambda n: n.key[0]):
            held = f"  [held by {child.ref_count}]" if child.ref_count else ""
            out.append("    " * depth + f"{len(child.key)} block(s): {label(child.key)}{held}"
                       + f"   blocks {child.blocks[:3]}{'…' if len(child.blocks) > 3 else ''}")
            walk(child, depth + 1)

    walk(cache.root, 1)
    return "\n".join(out) if out else "    (empty)"


# --------------------------------------------------------------------------
# demo 1: building the tree, and the split
# --------------------------------------------------------------------------


def demo_tree():
    print("=" * 78)
    print("1. The tree: shared prefixes are shared nodes")
    print("=" * 78)
    print(f"block_size = 2 tokens, so every line below is one or two blocks\n")

    cache = RadixCache(num_blocks=16, block_size=2)
    steps = [
        ["a", "b", "c", "d"],
        ["a", "b", "c", "e"],      # shares two tokens, then diverges
        ["a", "b"],                # stops on a node boundary
        ["a", "x", "y", "z"],      # first block already differs
    ]
    for tokens in steps:
        node, hit = cache.lookup(tokens)
        added = cache.insert(block_keys(tokens, 2))
        cache.release(node)
        print(f'insert {" ".join(tokens):12s}  matched {hit} token(s), added {added} block(s)')
        print(tree_lines(cache))
        print()

    print("The second insert split the first node in two: 'a b' is now its own")
    print("node, because two requests share it. Splitting moves blocks between")
    print("nodes; it never copies or frees KV.\n")


# --------------------------------------------------------------------------
# demo 2: the book's two figures
# --------------------------------------------------------------------------


def demo_book_figures():
    print("=" * 78)
    print("2. The prefix ends at the first token that differs")
    print("=" * 78)
    block_size = 2

    def run(first, second):
        cache = RadixCache(num_blocks=16, block_size=block_size)
        cache.lookup(first)
        cache.insert(block_keys(first, block_size))
        node, hit = cache.lookup(second)
        cache.release(node)
        print(f'  A: {" ".join(first)}')
        print(f'  B: {" ".join(second)}')
        print(f'  cached: {hit} of {len(second)} tokens, prefill {len(second) - hit}\n')
        return hit

    print("Figure 5.7, shared prefix:")
    run(["Weather", "in", "Paris", "?"],
        ["Weather", "in", "Rome", "?"])

    print("Figure 5.8, first token differs:")
    run(["Weather", "in", "Paris", "?"],
        ["Forecast", "for", "Paris", "?"])

    print("The second pair still shares two tokens ('Paris ?') and saves nothing:")
    print("a prefix is a promise about everything")
    print("before the token, not about the token itself.\n")


# --------------------------------------------------------------------------
# demo 3: whole blocks only
# --------------------------------------------------------------------------


def demo_partial_blocks():
    print("=" * 78)
    print("3. Only whole blocks are reusable")
    print("=" * 78)
    cache = RadixCache(num_blocks=16, block_size=4)
    prompt = [f"t{i}" for i in range(1, 11)]          # 10 tokens -> 2 whole blocks
    cache.insert(block_keys(prompt, 4))
    node, hit = cache.lookup(prompt)
    cache.release(node)
    print(f"  {len(prompt)} tokens, block_size 4 -> {len(block_keys(prompt, 4))} whole blocks")
    print(f"  the same prompt again: {hit} of {len(prompt)} tokens reusable,")
    print(f"  the last {len(prompt) - hit} are prefilled again: a partial block is not cached\n")

    changed = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "X", "t9", "t10"]
    node, hit = cache.lookup(changed)
    cache.release(node)
    print("  same prompt with one token changed inside the second block:")
    print(f"  {hit} tokens reusable: block one survives, block two is invalid\n")

    print("A block is the unit of allocation, so its KV belongs to one request")
    print("unless every token in it matches. Copy-on-write would let the shared")
    print("part be reused, at the cost of a copy per block per request.\n")


# --------------------------------------------------------------------------
# demo 4: eviction, LRU order, and pinning
# --------------------------------------------------------------------------


def demo_eviction():
    print("=" * 78)
    print("4. Eviction: leaves only, oldest first, never a held block")
    print("=" * 78)
    cache = RadixCache(num_blocks=3, block_size=2)
    print("A pool of 3 blocks, block_size 2.\n")

    for tokens in (["a", "b"], ["c", "d"], ["e", "f"]):
        node, _ = cache.lookup(tokens)
        cache.insert(block_keys(tokens, 2))
        cache.release(node)
    print("three requests, three blocks:")
    print(tree_lines(cache))

    held = [cache.lookup(["a", "b"])[0]]     # a decoding request pins 'a b'
    for tokens in (["c", "d"], ["e", "f"]):   # ... and the other two get used after it
        node, _ = cache.lookup(tokens)
        cache.release(node)

    node, _ = cache.lookup(["g", "h"])
    cache.insert(block_keys(["g", "h"], 2))
    cache.release(node)
    print("\n'a b' is now the least recently used leaf, but a request holds it.")
    print("Insert 'g h':")
    print(tree_lines(cache))
    print("  'c d' went instead: eviction skips held leaves\n")

    held.append(cache.lookup(["g", "h"])[0])
    node, _ = cache.lookup(["i", "j"])
    cache.insert(block_keys(["i", "j"], 2))
    cache.release(node)
    print("insert 'i j' after pinning 'g h' too:")
    print(tree_lines(cache))

    held.append(cache.lookup(["i", "j"])[0])
    try:
        node, _ = cache.lookup(["k", "l"])
        cache.insert(block_keys(["k", "l"], 2))
        cache.release(node)
    except MemoryError as error:
        print(f"\ninsert 'k l' with every leaf held:\n  {error}")
    for node in held:
        cache.release(node)

    print("\nThe engine cannot evict its way out, so it preempts a request (day 25)")
    print("instead. In a tree that question is one ref_count per node covering all")
    print("of its descendants; a flat block dict has to pin blocks one by one,")
    print("because it knows nothing about ancestry.\n")


# --------------------------------------------------------------------------
# demo 5: context engineering decides the hit rate
# --------------------------------------------------------------------------


def demo_context_engineering():
    print("=" * 78)
    print("5. Where you put the unique tokens decides the hit rate")
    print("=" * 78)
    system = [f"s{i}" for i in range(1600)]

    def run(name, build):
        cache = RadixCache(num_blocks=512, block_size=BLOCK_SIZE)
        for turn in range(20):
            prompt = build(turn)
            node, hit = cache.lookup(prompt)
            cache.insert(block_keys(prompt, BLOCK_SIZE))
            cache.release(node)
        stats = cache.stats()
        print(f"  {name:22s} token hit rate {stats['token_hit_rate']:6.1%}   "
              f"prefilled {stats['prefilled_tokens']:6d} tokens")

    # A per-request timestamp at position 0 changes every token after it.
    run("timestamp first", lambda t: [f"ts{t}"] + system + [f"q{t}"] * 60)
    # The same prompt with the novel tokens at the end.
    run("timestamp last", lambda t: system + [f"q{t}"] * 60 + [f"ts{t}"])
    print()


# --------------------------------------------------------------------------
# the trace: chat traffic with shared system prompts
# --------------------------------------------------------------------------


@dataclass
class Turn:
    """One request: the whole prompt, and how many tokens the answer will be."""

    at: float
    prompt: list
    answer: int


def make_trace(seed=SEED):
    """Conversations over a handful of long system prompts, resending all history every turn."""
    rng = random.Random(seed)
    systems = [[rng.randrange(50_000) for _ in range(rng.randint(*SYSTEM_TOKENS))]
               for _ in range(N_SYSTEM_PROMPTS)]
    turns, start = [], 0.0
    for _ in range(N_CONVERSATIONS):
        start += rng.expovariate(CONVERSATIONS_PER_SECOND)
        history = list(rng.choice(systems))
        at = start
        for _ in range(rng.randint(*TURNS)):
            history += [rng.randrange(50_000) for _ in range(rng.randint(*USER_TOKENS))]
            answer = rng.randint(*ANSWER_TOKENS)
            turns.append(Turn(at, list(history), answer))
            history += [rng.randrange(50_000) for _ in range(answer)]
            at += rng.uniform(*THINK_SECONDS) + answer * DECODE_MS_PER_TOKEN / 1000
    turns.sort(key=lambda turn: turn.at)
    return turns


def percentile(values, p):
    """The value below which p percent of `values` fall."""
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def serve(cache, turns):
    """Send every turn through one cache and measure the prefill it saved."""
    ttfts = []
    for turn in turns:
        node, hit = cache.lookup(turn.prompt)
        uncached = len(turn.prompt) - hit
        ttfts.append(PREFILL_BASE_MS + uncached * PREFILL_MS_PER_TOKEN)
        cache.insert(block_keys(turn.prompt, cache.block_size))
        cache.release(node)
    return ttfts


def run_trace():
    print("=" * 78)
    print("6. The same traffic through three caches")
    print("=" * 78)
    turns = make_trace()
    prompt_tokens = sum(len(t.prompt) for t in turns)
    print(f"{len(turns)} requests, {prompt_tokens:,} prompt tokens, "
          f"{BLOCK_SIZE}-token blocks, cache of {CACHE_BLOCKS} blocks "
          f"({CACHE_BLOCKS * BLOCK_SIZE:,} tokens)\n")

    class NoCache:
        block_size = BLOCK_SIZE

        def lookup(self, tokens):
            return None, 0

        def insert(self, keys):
            return len(keys)

        def release(self, node):
            pass

        def stats(self):
            return {"requests": len(turns), "hit_rate": 0.0, "token_hit_rate": 0.0,
                    "block_hit_rate": 0.0, "reused_tokens": 0, "prefilled_tokens": prompt_tokens,
                    "blocks_held": 0, "peak_blocks": 0, "blocks_evicted": 0, "nodes": 0,
                    "stranded_blocks": 0}

    runs = [
        ("no cache", NoCache()),
        ("flat LRU", FlatBlockCache(CACHE_BLOCKS)),
        ("flat, no refresh", FlatBlockCache(CACHE_BLOCKS, refresh=False)),
        (f"radix, {CACHE_BLOCKS} blocks", RadixCache(CACHE_BLOCKS)),
        ("radix, unlimited", RadixCache(4 * CACHE_BLOCKS + 20_000)),
    ]

    header = (f"  {'cache':22s}{'tok hit':>8s}{'req hit':>8s}{'prefilled':>12s}"
              f"{'TTFT p50':>10s}{'TTFT p99':>10s}{'evicted':>9s}{'stranded':>10s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, cache in runs:
        ttfts = serve(cache, turns)
        s = cache.stats()
        print(f"  {name:22s}{s['token_hit_rate']:7.1%}{s['hit_rate']:8.1%}"
              f"{s['prefilled_tokens']:12,d}{percentile(ttfts, 50):10.2f}"
              f"{percentile(ttfts, 99):10.2f}{s['blocks_evicted']:9d}"
              f"{s['stranded_blocks']:10d}")

    print("\n  TTFT is computed, not measured: base 2 ms + 0.05 ms per prefilled token.")
    print()


def main():
    started = time.perf_counter()
    demo_tree()
    demo_book_figures()
    demo_partial_blocks()
    demo_eviction()
    demo_context_engineering()
    run_trace()
    print(f"ran in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
