# Day 25: PagedAttention — a block-based KV cache allocator

Book chapter: **PagedAttention idea (engines)**

Task: a KV cache allocator built from blocks, with alloc, free and
fragmentation stats.

## What we built

`paged_kv_allocator.py` manages KV cache **memory slots** (one slot holds one
token's K and V for every layer), not the tensors themselves. That makes the
memory question easy to see. There are two allocators:

- **`PagedAllocator`:** memory is cut into **blocks of 16 tokens**. A request
  gets blocks from anywhere in memory, and its **block table** lists them in
  order. It has these operations:
  - `alloc`: give a new request blocks for its prompt.
  - `append`: add one token, taking a new block only when the last one is full.
  - `free`: give all of a request's blocks back.
  - `slot_of`: return where token N lives.
  - `stats`
- **`ContiguousAllocator`:** the old way. Each request reserves **one
  unbroken range** (first fit), sized either for the model's maximum length or
  for exactly prompt + answer. The exact size is a cheat, since a real server
  never knows the answer length in advance.

The script runs a walk-through first, then serves the same 400 requests with
all three setups on an 8,192-slot pool. That is 288 MB of KV cache for
distilgpt2, at 36 KB per token.

```
python3 day-25/paged_kv_allocator.py
```

## What we learned

### Three kinds of waste

| Waste | What it is | Contiguous | Paged |
|---|---|---|---|
| **Over-reservation** | Room saved for tokens that may never come | Huge: reserves 1024 when most answers are short | None: blocks are taken as the answer grows |
| **Internal fragmentation** | The empty end of a request's last block | n/a | At most 15 slots per request |
| **External fragmentation** | Free memory cut into holes too small to use | Yes: the next request waits although enough memory is free in total | None: any free block works |

### The block table is a page table

The same idea as virtual memory in an operating system:

- token 33 of request A → `block_tables["A"][33 // 16]`, offset `33 % 16`;
- logical order: A's tokens 0, 1, 2, … ;
- physical order: whatever blocks were free.

The attention kernel follows the block table to read K and V. Day 14 did this
gather in PyTorch; vLLM does it inside the kernel.

### Preemption

Paged memory admits more users, so it can run out **in the middle** of
answers. Then the scheduler **preempts**: it pauses the newest request, frees
its blocks, and later recomputes its prefill (prompt plus tokens so far). vLLM
does the same.

## Results

```
Walk-through with blocks of 16 tokens and a pool of 8 blocks
  A has 20 tokens -> blocks [7, 6] | B has 5 tokens -> blocks [5]
  A grew to 32 tokens -> still blocks [7, 6] (only the last block is ever part-empty)
  A's token 33 -> new block, table [7, 6, 4] | token 33 lives at (4, 0)
  B finished, C (40 tokens) -> blocks [5, 3, 2] (B's old block reused)

Pool: 8192 token slots (288 MB of distilgpt2 KV cache), 400 requests

                      allocator  steps avg batch  tokens reserved, empty   free blocked by holes preempted
----------------------------------------------------------------------------------------------------------
       contiguous, reserve 1024  13302       7.8     36%           61.9%     2%                0         0
contiguous, knows answer length   8614      12.1     55%           25.2%    20%             7547         0
            paged, blocks of 16   5344      19.5     89%            1.8%     9%                0       112
```

- **Reserve 1024:** only 36% of memory holds real tokens, and 62% is reserved
  for tokens that never come. About 8 users fit at once.
- **Knowing the answer length** (impossible in practice) still wastes 45%:
  25% sits empty in reservations that fill up over time, and 20% is free but
  cut into holes. For 7,547 steps, a request waited although enough memory was
  free in total.
- **Paged:** **89%** of memory holds tokens, and only **1.8%** is lost in
  half-full last blocks. The average batch is **19.5**, 2.5× the first setup,
  so all 400 requests finish in **2.5× fewer steps**. The price: 112
  preemptions.

This is vLLM's main claim: the same GPU memory holds more users, so batches are
bigger and throughput is higher.

## Connection to earlier days

- Day 13: the KV cache, which grows by one token per step.
- Day 14: the first PagedAttention, with real K/V tensors and a gather.
  81.5% → 0.5% waste there, 62% → 2% here.
- Day 22: our server used one growing tensor per user, re-padded every step.
  This allocator is what it was missing.
- Day 24: bigger batches mean more throughput. Paging is how you fit them.

## Caveats

- It counts slots and steps, not real time. A preempted request's
  recomputation is not charged as extra work here.
- Admission is simple: admit while there is room for the prompt plus one
  spare block per running request.
- Freed blocks go back to the end of the free list. Real allocators also track
  **reference counts**, so that several requests can share one block (day 26).

## Things to try

1. Change `BLOCK_SIZE` to 1, 64 and 256. Smaller blocks waste less but make a
   longer block table. Where does waste become noticeable?
2. Make the pool smaller (`TOTAL_SLOTS = 4096`) and count the preemptions.
3. Add **block sharing**: two requests with the same system prompt point at the
   same blocks, each with a reference count.
