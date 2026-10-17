import random

BLOCK_SIZE = 16
TOTAL_SLOTS = 8192
MAX_MODEL_LEN = 1024
N_REQUESTS = 400
PROMPT_TOKENS = (20, 400)
ANSWER_TOKENS = (10, 500)
KV_BYTES_PER_TOKEN = 36 * 1024
SEED = 0


class PagedAllocator:
    """A KV cache split into fixed-size blocks; each request gets blocks from anywhere, listed in a block table."""

    def __init__(self, total_slots, block_size=BLOCK_SIZE):
        self.block_size = block_size
        self.num_blocks = total_slots // block_size
        self.free_blocks = list(range(self.num_blocks))
        self.block_tables = {}
        self.lengths = {}

    def blocks_needed(self, n_tokens):
        """How many blocks hold n_tokens."""
        return -(-n_tokens // self.block_size)

    def can_alloc(self, n_tokens):
        """Is there room for a new request with n_tokens of prompt?"""
        return self.blocks_needed(n_tokens) <= len(self.free_blocks)

    def alloc(self, seq, n_tokens):
        """Give a new request enough blocks for its prompt."""
        if not self.can_alloc(n_tokens):
            raise MemoryError(f"no room for {seq}")
        self.block_tables[seq] = [self.free_blocks.pop() for _ in range(self.blocks_needed(n_tokens))]
        self.lengths[seq] = n_tokens

    def append(self, seq):
        """Add one generated token; take a new block only when the last one is full."""
        if self.lengths[seq] % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError(f"no block for {seq}")
            self.block_tables[seq].append(self.free_blocks.pop())
        self.lengths[seq] += 1

    def free(self, seq):
        """A request finished: give all its blocks back."""
        self.free_blocks.extend(self.block_tables.pop(seq))
        self.lengths.pop(seq)

    def slot_of(self, seq, position):
        """Where token `position` of a request lives in memory: (block id, offset in the block)."""
        return self.block_tables[seq][position // self.block_size], position % self.block_size

    def stats(self):
        """Memory used by tokens, reserved but empty, and free."""
        used_blocks = sum(len(t) for t in self.block_tables.values())
        tokens = sum(self.lengths.values())
        return {
            "tokens": tokens,
            "reserved_empty": used_blocks * self.block_size - tokens,
            "free": len(self.free_blocks) * self.block_size,
            "largest_free_run": len(self.free_blocks) * self.block_size,
        }


class ContiguousAllocator:
    """The old way: each request reserves one unbroken range for its prompt plus max_tokens, first fit."""

    def __init__(self, total_slots):
        self.total_slots = total_slots
        self.ranges = {}
        self.lengths = {}

    def holes(self):
        """The free gaps between reserved ranges, as (start, size)."""
        gaps, cursor = [], 0
        for start, size in sorted(self.ranges.values()):
            if start > cursor:
                gaps.append((cursor, start - cursor))
            cursor = start + size
        if cursor < self.total_slots:
            gaps.append((cursor, self.total_slots - cursor))
        return gaps

    def can_alloc(self, n_reserve):
        """Is there one gap big enough for n_reserve slots?"""
        return any(size >= n_reserve for _, size in self.holes())

    def alloc(self, seq, n_tokens, n_reserve):
        """Reserve n_reserve slots in the first gap that fits; n_tokens of them are filled now."""
        for start, size in self.holes():
            if size >= n_reserve:
                self.ranges[seq] = (start, n_reserve)
                self.lengths[seq] = n_tokens
                return
        raise MemoryError(f"no gap for {seq}")

    def append(self, seq):
        """Add one generated token inside the request's own range."""
        self.lengths[seq] += 1

    def free(self, seq):
        """A request finished: its range becomes a gap."""
        self.ranges.pop(seq)
        self.lengths.pop(seq)

    def stats(self):
        """Memory used by tokens, reserved but empty, and free (plus the biggest single gap)."""
        tokens = sum(self.lengths.values())
        reserved = sum(size for _, size in self.ranges.values())
        holes = self.holes()
        return {
            "tokens": tokens,
            "reserved_empty": reserved - tokens,
            "free": self.total_slots - reserved,
            "largest_free_run": max((size for _, size in holes), default=0),
        }


def make_requests(seed=SEED):
    """Random requests: (id, prompt tokens, answer tokens)."""
    rng = random.Random(seed)
    return [(f"r{i}", rng.randint(*PROMPT_TOKENS), rng.randint(*ANSWER_TOKENS)) for i in range(N_REQUESTS)]


def simulate(kind):
    """Serve every request with one allocator: admit while memory allows, one token per step for everyone."""
    allocator = PagedAllocator(TOTAL_SLOTS) if kind == "paged" else ContiguousAllocator(TOTAL_SLOTS)
    waiting = make_requests()
    running = {}
    steps, batch_sum, samples = 0, 0, []
    blocked_by_holes, preempted = 0, 0
    while waiting or running:
        while waiting:
            seq, prompt, answer = waiting[0]
            if kind == "paged":
                if allocator.blocks_needed(prompt) + len(running) > len(allocator.free_blocks):
                    break
                allocator.alloc(seq, prompt)
            else:
                reserve = min(prompt + answer if kind == "exact" else MAX_MODEL_LEN, MAX_MODEL_LEN)
                if not allocator.can_alloc(reserve):
                    if allocator.stats()["free"] >= reserve:
                        blocked_by_holes += 1
                    break
                allocator.alloc(seq, prompt, reserve)
            running[seq] = answer
            waiting.pop(0)
        if kind == "paged":
            preempted += make_room(allocator, running, waiting)
        steps += 1
        batch_sum += len(running)
        for seq in list(running):
            allocator.append(seq)
            running[seq] -= 1
            if running[seq] == 0:
                allocator.free(seq)
                del running[seq]
        if steps % 25 == 0 and running:
            samples.append(allocator.stats())
    return steps, batch_sum / steps, samples, blocked_by_holes, preempted


def make_room(allocator, running, waiting):
    """Before a step, make sure every running request can store its next token; preempt the newest if not."""
    preempted = 0
    while sum(allocator.lengths[seq] % allocator.block_size == 0 for seq in running) > len(allocator.free_blocks):
        seq = list(running)[-1]
        waiting.insert(0, (seq, allocator.lengths[seq], running.pop(seq)))
        allocator.free(seq)
        preempted += 1
    return preempted


def average(samples, key):
    """Average share of all memory that `key` took across the samples."""
    return sum(s[key] for s in samples) / len(samples) / TOTAL_SLOTS


def demo():
    """A small walk-through: three requests, alloc, grow, free, reuse."""
    print(f"Walk-through with blocks of {BLOCK_SIZE} tokens and a pool of 8 blocks")
    paged = PagedAllocator(8 * BLOCK_SIZE)
    paged.alloc("A", 20)
    paged.alloc("B", 5)
    print("  A has 20 tokens -> blocks", paged.block_tables["A"], "| B has 5 tokens -> blocks", paged.block_tables["B"])
    for _ in range(12):
        paged.append("A")
    print("  A grew to 32 tokens -> still blocks", paged.block_tables["A"], "(only the last block is ever part-empty)")
    paged.append("A")
    print("  A's token 33 -> new block, table", paged.block_tables["A"], "| token 33 lives at", paged.slot_of("A", 32))
    paged.free("B")
    paged.alloc("C", 40)
    print("  B finished, C (40 tokens) -> blocks", paged.block_tables["C"], "(B's old block reused)")
    print("  stats:", paged.stats())
    print()


def main():
    """Compare a paged allocator with contiguous ones on the same requests."""
    demo()
    mb = TOTAL_SLOTS * KV_BYTES_PER_TOKEN / 2**20
    print(f"Pool: {TOTAL_SLOTS} token slots ({mb:.0f} MB of distilgpt2 KV cache), {N_REQUESTS} requests\n")
    header = f"{'allocator':>31} {'steps':>6} {'avg batch':>9} {'tokens':>7} {'reserved, empty':>15} {'free':>6} {'blocked by holes':>16} {'preempted':>9}"
    print(header)
    print("-" * len(header))
    for kind, label in (("max", f"contiguous, reserve {MAX_MODEL_LEN}"),
                        ("exact", "contiguous, knows answer length"),
                        ("paged", f"paged, blocks of {BLOCK_SIZE}")):
        steps, avg_batch, samples, blocked, preempted = simulate(kind)
        print(f"{label:>31} {steps:>6} {avg_batch:>9.1f} {average(samples, 'tokens'):>7.0%} "
              f"{average(samples, 'reserved_empty'):>15.1%} {average(samples, 'free'):>6.0%} {blocked:>16} {preempted:>9}")
    print("\ntokens / reserved, empty / free = average share of the pool while requests were running")
    print("blocked by holes = steps where the next request waited although enough memory was free, just not in one piece")
    print("preempted = times a running request was paused and its blocks freed, to be recomputed later")


if __name__ == "__main__":
    main()
