import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-10"))
from tiny_gpt_generate import generate_greedy, load_gpt2_weights  # noqa: E402  (day 10's model)

LENGTHS = [16, 64, 256, 512]  # new tokens to generate in the speed comparison

# blue and orange: the first two slots of the reference palette
NO_CACHE_COLOR, CACHE_COLOR = "#eb6834", "#2a78d6"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def attention_with_cache(attn, x, past):
    """Day 9's attention, but it can reuse the K and V of earlier tokens.

    x:    (B, T_new, C) only the NEW tokens
    past: None, or (K, V) from earlier tokens, each (B, n_heads, T_past, d_head)
    returns (output for the new tokens, updated (K, V) including the new tokens)
    """
    B, T, C = x.shape
    q, k, v = attn.qkv(x).split(C, dim=-1)  # Q, K, V for the new tokens only
    q, k, v = (t.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2) for t in (q, k, v))

    if past is not None:
        k = torch.cat([past[0], k], dim=2)  # append the new K after the cached ones
        v = torch.cat([past[1], v], dim=2)
    T_total = k.shape[2]
    T_past = T_total - T

    scores = q @ k.transpose(-2, -1) / math.sqrt(attn.d_head)  # (B, n_heads, T, T_total)
    # new token i sits at position T_past + i, so it may see keys 0 .. T_past + i
    mask = torch.triu(torch.ones(T, T_total, dtype=torch.bool, device=x.device), diagonal=T_past + 1)
    scores = scores.masked_fill(mask, float("-inf"))
    out = scores.softmax(dim=-1) @ v

    out = out.transpose(1, 2).reshape(B, T, C)
    return attn.out_proj(out), (k, v)


def forward_with_cache(model, idx_new, cache):
    """Run only the new tokens through the model, using and extending the cache.

    cache: a list with one (K, V) per layer, or None on the first call (prefill).
    """
    T_past = 0 if cache is None else cache[0][0].shape[2]
    T = idx_new.shape[1]
    pos = torch.arange(T_past, T_past + T, device=idx_new.device)  # positions continue on
    x = model.tok_emb(idx_new) + model.pos_emb(pos)

    new_cache = []
    for i, block in enumerate(model.blocks):
        a, kv = attention_with_cache(block.attn, block.ln1(x), None if cache is None else cache[i])
        x = x + a
        x = x + block.mlp(block.ln2(x))
        new_cache.append(kv)
    return model.lm_head(model.ln_f(x)), new_cache


@torch.no_grad()
def generate_greedy_cached(model, idx, max_new_tokens, step_times=None):
    """Prefill the whole prompt once, then feed ONE token per step."""
    assert idx.shape[1] + max_new_tokens <= model.max_len, "stay inside the context window"
    logits, cache = forward_with_cache(model, idx, None)  # prefill: builds the cache
    for _ in range(max_new_tokens):
        start = time.perf_counter()
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        logits, cache = forward_with_cache(model, next_id, cache)  # decode: just the new token
        if step_times is not None:
            step_times.append(time.perf_counter() - start)
    return idx, cache


@torch.no_grad()
def generate_greedy_timed(model, idx, max_new_tokens, step_times):
    """Day 10's loop (no cache), with each step timed."""
    for _ in range(max_new_tokens):
        start = time.perf_counter()
        logits = model(idx[:, -model.max_len:])[:, -1, :]
        idx = torch.cat([idx, logits.argmax(dim=-1, keepdim=True)], dim=1)
        step_times.append(time.perf_counter() - start)
    return idx


def cache_bytes(cache):
    return sum(t.numel() * t.element_size() for kv in cache for t in kv)


def plot_steps(no_cache, cached, prompt_len, path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    steps = range(1, len(no_cache) + 1)
    ax.plot(steps, [t * 1e3 for t in no_cache], color=NO_CACHE_COLOR, linewidth=2)
    ax.plot(steps, [t * 1e3 for t in cached], color=CACHE_COLOR, linewidth=2)
    ax.annotate("no cache: reruns the whole sentence", (len(no_cache), no_cache[-1] * 1e3),
                xytext=(-8, 8), textcoords="offset points", ha="right", color=INK, fontsize=10)
    ax.annotate("KV cache: runs only the new token", (len(cached), cached[-1] * 1e3),
                xytext=(-8, 10), textcoords="offset points", ha="right", color=INK, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Decode step (prompt was {prompt_len} tokens)", color=MUTED)
    ax.set_ylabel("Time for this step (ms)", color=MUTED)
    fig.suptitle("Time per generated token, with and without a KV cache", x=0.01, ha="left",
                 color=INK, fontsize=13, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot to {path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model, hf = load_gpt2_weights("distilgpt2")
    prompt = "The future of AI inference is"
    idx = tok(prompt, return_tensors="pt")["input_ids"]
    n_layers, d_model = len(model.blocks), model.tok_emb.weight.shape[1]

    # ---- 1. Same answer: the cache changes speed, never the output ----
    with torch.no_grad():
        full_logits = model(idx)
        pre_logits, cache = forward_with_cache(model, idx[:, :-1], None)
        step_logits, _ = forward_with_cache(model, idx[:, -1:], cache)
    assert torch.allclose(full_logits[:, -1], step_logits[:, -1], atol=1e-4)

    for n in [30, 100]:
        plain = generate_greedy(model, idx, n)
        cached, _ = generate_greedy_cached(model, idx, n)
        assert torch.equal(plain, cached)
    print("Check passed: with and without the cache, the tokens are identical.")

    # ---- 2. What the cache holds ----
    _, cache = generate_greedy_cached(model, idx, 100)
    n_tok = cache[0][0].shape[2]
    per_token = 2 * n_layers * d_model * cache[0][0].element_size()
    print(f"\nAfter {n_tok} tokens the cache holds {n_layers} layers x (K, V), "
          f"each {tuple(cache[0][0].shape)} = (batch, heads, tokens, d_head)")
    print(f"cache size: {cache_bytes(cache) / 1e6:.2f} MB = {n_tok} tokens x {per_token / 1024:.0f} KB/token")
    print(f"formula:    2 (K,V) x {n_layers} layers x {d_model} numbers x "
          f"{cache[0][0].element_size()} bytes = {per_token:,} bytes per token")
    assert cache_bytes(cache) == n_tok * per_token

    # ---- 3. Speed: total time for N new tokens ----
    print(f"\n{'new tokens':>10}{'no cache (s)':>14}{'cache (s)':>11}{'no cache tok/s':>16}"
          f"{'cache tok/s':>13}{'speedup':>9}")
    for n in LENGTHS:
        start = time.perf_counter()
        a = generate_greedy(model, idx, n)
        t_plain = time.perf_counter() - start
        start = time.perf_counter()
        b, _ = generate_greedy_cached(model, idx, n)
        t_cache = time.perf_counter() - start
        assert torch.equal(a, b)
        print(f"{n:>10}{t_plain:>14.2f}{t_cache:>11.2f}{n / t_plain:>16.0f}{n / t_cache:>13.0f}"
              f"{t_plain / t_cache:>8.1f}x")

    # Hugging Face's generate() also uses a KV cache: our cached version should be close
    n = LENGTHS[-1]
    with torch.no_grad():
        start = time.perf_counter()
        hf.generate(idx, attention_mask=torch.ones_like(idx), max_new_tokens=n,
                    min_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        t_hf = time.perf_counter() - start
    print(f"{'(Hugging Face generate, with its own cache)':<44}{n / t_hf:>13.0f} tok/s for {n} tokens")

    # ---- 4. Time per step: no cache grows, cache stays flat ----
    no_cache_steps, cache_steps = [], []
    generate_greedy_timed(model, idx, LENGTHS[-1], no_cache_steps)
    generate_greedy_cached(model, idx, LENGTHS[-1], cache_steps)
    first, last = slice(0, 10), slice(-10, None)
    avg = lambda xs: sum(xs) / len(xs) * 1e3  # noqa: E731
    print(f"\nms per step    first 10 steps   last 10 steps")
    print(f"no cache      {avg(no_cache_steps[first]):>14.1f}{avg(no_cache_steps[last]):>16.1f}")
    print(f"KV cache      {avg(cache_steps[first]):>14.1f}{avg(cache_steps[last]):>16.1f}")
    plot_steps(no_cache_steps, cache_steps, idx.shape[1], Path(__file__).parent / "kv_cache.png")
