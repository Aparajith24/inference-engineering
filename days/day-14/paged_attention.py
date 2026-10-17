import math
import random

import torch

from gqa_sdpa_bench import fmt_bytes, kv_cache_bytes  # same folder: the day 14 calculator

BLOCK_SIZE = 16           # tokens per block (page)
N_HEADS, HEAD_DIM = 4, 8  # small, so the demo is easy to read


class PagedKVCache:
    """A KV cache stored in fixed-size blocks taken from one shared pool.

    Instead of one long, growing tensor per conversation (day 13's torch.cat),
    every conversation gets blocks from wherever they are free, and a block
    table remembers which blocks are its, in order.
    """

    def __init__(self, num_blocks, n_heads, head_dim, block_size=BLOCK_SIZE):
        self.block_size = block_size
        # The pool: every block in the GPU's KV cache memory, allocated once, up front
        self.k_pool = torch.zeros(num_blocks, n_heads, block_size, head_dim)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.free_blocks = list(range(num_blocks))  # block ids nobody is using
        self.block_tables = {}                      # conversation -> its block ids, in order
        self.lengths = {}                           # conversation -> tokens stored

    def append(self, seq, k, v):
        """Store one new token's K and V, each (n_heads, head_dim)."""
        n = self.lengths.get(seq, 0)
        table = self.block_tables.setdefault(seq, [])
        if n % self.block_size == 0:  # first token, or the last block is full: take a free block
            if not self.free_blocks:
                raise MemoryError("out of KV cache blocks")
            table.append(self.free_blocks.pop(0))
        block, offset = table[n // self.block_size], n % self.block_size
        self.k_pool[block, :, offset] = k
        self.v_pool[block, :, offset] = v
        self.lengths[seq] = n + 1

    def gather(self, seq):
        """Read one conversation's K and V back in order: each (n_heads, tokens, head_dim)."""
        table = torch.tensor(self.block_tables[seq])
        n, heads = self.lengths[seq], self.k_pool.shape[1]

        def read(pool):
            blocks = pool[table]  # (n_blocks, heads, block_size, head_dim): the lookup
            return blocks.transpose(0, 1).reshape(heads, -1, pool.shape[-1])[:, :n]

        return read(self.k_pool), read(self.v_pool)

    def free(self, seq):
        """The conversation finished: give its blocks back to the pool."""
        self.free_blocks = sorted(self.free_blocks + self.block_tables.pop(seq))
        self.lengths.pop(seq)


def decode_attention(q, k, v):
    """One new token attending to its whole cache. q: (heads, d); k, v: (heads, tokens, d)."""
    scores = (q.unsqueeze(1) @ k.transpose(-2, -1)).squeeze(1) / math.sqrt(q.shape[-1])
    return (scores.softmax(dim=-1).unsqueeze(1) @ v).squeeze(1)  # (heads, d)


def run(cache, reference, seq, n_tokens):
    """Add n_tokens to a conversation in both the paged cache and a plain torch.cat cache."""
    for _ in range(n_tokens):
        k, v = torch.randn(N_HEADS, HEAD_DIM), torch.randn(N_HEADS, HEAD_DIM)
        cache.append(seq, k, v)
        old_k, old_v = reference.get(seq, (torch.empty(N_HEADS, 0, HEAD_DIM),) * 2)
        reference[seq] = (torch.cat([old_k, k[:, None]], dim=1), torch.cat([old_v, v[:, None]], dim=1))


def check(cache, reference, seq):
    """Paged cache must hold exactly what the plain cache holds, and give the same attention."""
    k, v = cache.gather(seq)
    ref_k, ref_v = reference[seq]
    assert torch.equal(k, ref_k) and torch.equal(v, ref_v)
    q = torch.randn(N_HEADS, HEAD_DIM)
    assert torch.allclose(decode_attention(q, k, v), decode_attention(q, ref_k, ref_v))


def show(cache):
    for seq, table in cache.block_tables.items():
        print(f"  {seq}: {cache.lengths[seq]:>3} tokens -> blocks {table}")
    print(f"  free blocks: {cache.free_blocks}")


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- Part 1: three conversations sharing one pool, growing at the same time ----
    print(f"PART 1: a pool of 12 blocks x {BLOCK_SIZE} tokens, shared by every conversation\n")
    cache = PagedKVCache(num_blocks=12, n_heads=N_HEADS, head_dim=HEAD_DIM)
    reference = {}  # day 13's way: one torch.cat tensor per conversation

    targets = {"A": 40, "B": 20, "C": 33}
    for step in range(max(targets.values())):  # decode steps: every active conversation adds a token
        for seq, n in targets.items():
            if step < n:
                run(cache, reference, seq, 1)
    print("A, B and C generated tokens at the same time, so their blocks are interleaved:")
    show(cache)
    for seq in targets:
        check(cache, reference, seq)
    print("Check passed: each conversation's K, V and attention match a plain torch.cat cache.\n")

    cache.free("B")
    print("B finished, so its blocks go back to the pool:")
    show(cache)

    run(cache, reference, "D", 45)
    print("\nD starts and reuses B's old blocks first, wherever they are:")
    show(cache)
    check(cache, reference, "D")
    print("Check passed: D's K, V and attention are correct even though its blocks are scattered.")

    # ---- Part 2: why this matters, at a real model's size ----
    per_token = kv_cache_bytes(layers=32, kv_heads=8, head_dim=128, tokens=1)  # 8B GQA-8 shape, fp16
    budget = 64e9          # an 80 GB GPU after 16 GB of 8B-model weights (day 14, Part 3)
    max_context = 8192     # the longest a conversation may grow
    rng = random.Random(0)
    lengths = [min(max_context, max(32, int(rng.expovariate(1 / 1500)))) for _ in range(2000)]

    def fits(reserve):
        """How many conversations fit, if each one reserves reserve(length) tokens of cache."""
        used = count = 0
        for n in lengths:
            if used + reserve(n) * per_token > budget:
                break
            used += reserve(n) * per_token
            count += 1
        return count

    def waste(reserve):
        return 1 - sum(lengths) / sum(reserve(n) for n in lengths)

    contiguous = lambda n: max_context                                   # noqa: E731  must reserve the max up front
    paged = lambda n: math.ceil(n / BLOCK_SIZE) * BLOCK_SIZE            # noqa: E731  only whole blocks actually used

    print(f"\nPART 2: {len(lengths)} conversations, 8B model (GQA-8, fp16: {fmt_bytes(per_token)} per token), "
          f"{fmt_bytes(budget)} of KV cache memory")
    print(f"Conversation lengths vary from 32 to {max_context} tokens; "
          f"the average is {sum(lengths) / len(lengths):.0f}.\n")
    print(f"{'approach':<38}{'memory wasted':>15}{'conversations that fit':>25}")
    for name, reserve in [
        (f"one contiguous block of {max_context} tokens", contiguous),
        (f"paged, {BLOCK_SIZE}-token blocks", paged),
    ]:
        print(f"{name:<38}{waste(reserve):>14.1%}{fits(reserve):>25}")
    print("\nContiguous must reserve the maximum up front, because nobody knows how long a reply will be.")
    print(f"Paged only wastes the unused part of each conversation's last block (at most {BLOCK_SIZE - 1} tokens).")
