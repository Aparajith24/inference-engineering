"""Day 31: prompt-lookup (n-gram) decoding, and acceptance vs draft length k.

There is no draft model here. The context itself is the drafter: find the most
recent earlier occurrence of the last ``n`` tokens and propose the tokens that
followed it. The target model checks all proposals in one forward pass.

Only greedy decoding is used, so "accepted" means the proposal equals the
target's argmax. The emitted text is identical to normal greedy decoding.

Run:
    python3 days/day31-prompt-lookup/prompt_lookup_decoding.py
    python3 days/day31-prompt-lookup/prompt_lookup_decoding.py --quick
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_NAME = "gpt2-medium"
# Three prompts with different amounts of repetition. Prompt lookup wins when
# the model copies or repeats; it wins less on varied, novel text.
PROMPTS = {
    "copy": (
        "Repeat the following text exactly, word for word:\n\n"
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog again. "
        "A slow green turtle walks under the busy road. "
        "The quick brown fox jumps over the lazy dog. "
        "\n\nRepeat:\n\n"
    ),
    "code": (
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def sub(a, b):\n"
        "    return a - b\n\n"
        "def mul(a, b):\n"
        "    return a * b\n\n"
        "def div(a, b):"
    ),
    "story": "Write a short story about a detective in Paris. Detective Moreau",
}
DRAFT_LENGTHS = [1, 2, 3, 4, 5, 6, 7, 8]
NGRAM = 2
NEW_TOKENS = 96
REPEATS = 2
PLOT_PATH = Path(__file__).with_name("acceptance_vs_k.png")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


def sync():
    """GPU work is asynchronous; wait before reading the clock."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


class Model:
    """A causal LM plus its KV cache: each forward pass only sees new tokens."""

    def __init__(self, name):
        self.lm = AutoModelForCausalLM.from_pretrained(name).to(DEVICE).eval()
        self.cache = DynamicCache()

    def reset(self):
        self.cache = DynamicCache()

    def seen(self):
        return self.cache.get_seq_length()

    def forward(self, tokens):
        new = torch.tensor([tokens[self.seen() :]], device=DEVICE)
        with torch.no_grad():
            out = self.lm(new, past_key_values=self.cache, use_cache=True)
        return out.logits[0]

    def rewind(self, length):
        """Drop cached tokens past ``length`` after rejected proposals."""
        self.cache.crop(length)


def lookup_draft(tokens, k, n):
    """Propose up to k tokens by continuing the previous match of the last n.

    Search backwards for the most recent occurrence of the suffix before the
    suffix itself. The continuation may overlap the current suffix; that is
    what makes periodic text (lists, repeated phrases) draftable.
    """
    if len(tokens) < n + 1:
        return []
    suffix = tokens[-n:]
    for i in range(len(tokens) - n - 1, -1, -1):
        if tokens[i : i + n] == suffix:
            return tokens[i + n : i + n + k]
    return []


def normal_decode(model, prompt_ids, n_new):
    """Baseline: one token per target forward pass."""
    model.reset()
    tokens = list(prompt_ids)
    passes = 0
    while len(tokens) - len(prompt_ids) < n_new:
        logits = model.forward(tokens)
        tokens.append(int(logits[-1].argmax()))
        passes += 1
    return tokens[len(prompt_ids) :][:n_new], {"passes": passes}


def prompt_lookup_decode(model, prompt_ids, n_new, k, n):
    """Draft from the context, verify k proposals in one target pass."""
    model.reset()
    tokens = list(prompt_ids)
    stats = {
        "passes": 0,
        "draft_passes": 0,
        "fallbacks": 0,
        "drafted": 0,
        "accepted": 0,
    }

    while len(tokens) - len(prompt_ids) < n_new:
        drafts = lookup_draft(tokens, k, n)

        if not drafts:
            # No n-gram match: an ordinary target step.
            logits = model.forward(tokens)
            tokens.append(int(logits[-1].argmax()))
            stats["passes"] += 1
            stats["fallbacks"] += 1
            continue

        logits = model.forward(tokens + drafts)
        # The logits for the last prompt token predict draft 0. The final
        # position is the bonus token if every draft survives.
        offset = len(logits) - len(drafts) - 1
        accepted = 0
        for j, draft in enumerate(drafts):
            target = int(logits[offset + j].argmax())
            if draft != target:
                tokens.append(target)  # correction from the target
                break
            tokens.append(draft)
            accepted += 1
        else:
            tokens.append(int(logits[offset + len(drafts)].argmax()))

        stats["passes"] += 1
        stats["draft_passes"] += 1
        stats["drafted"] += len(drafts)
        stats["accepted"] += accepted
        # Rejected proposals are in the cache; keep only the emitted tokens
        # except the newest one, whose logits are needed on the next pass.
        model.rewind(len(tokens) - 1)

    return tokens[len(prompt_ids) :][:n_new], stats


def timed(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return result, time.perf_counter() - start


def best_of(fn, repeats):
    """Acceptance is deterministic; repeats only reduce timing noise."""
    best = None
    for _ in range(repeats):
        result, dt = timed(fn)
        if best is None or dt < best[0]:
            best = (dt, result)
    dt, result = best
    return result, dt


def benchmark(model, tokenizer, prompts, draft_lengths, n_new, n, repeats):
    results = {}
    print(
        f"\n{'prompt':>6} {'k':>3} {'accept':>7} {'drafted':>8} "
        f"{'acc/pass':>9} {'passes':>7} {'tok/s':>7} {'speedup':>8}"
    )
    print("-" * 66)

    for name, text in prompts.items():
        prompt_ids = tokenizer(text)["input_ids"]
        (base_ids, base_stats), base_dt = best_of(
            lambda: normal_decode(model, prompt_ids, n_new), repeats
        )
        base_tps = n_new / base_dt
        base_text = tokenizer.decode(base_ids)

        results[name] = {"acceptance": {}, "speedup": {}, "tokens_per_pass": {}}
        print(
            f"{name:>6} {'-':>3} {'-':>7} {'-':>8} {'-':>9} "
            f"{base_stats['passes']:>7} {base_tps:>7.1f} {1.0:>7.2f}x"
        )

        for k in draft_lengths:
            (out_ids, stats), dt = best_of(
                lambda: prompt_lookup_decode(model, prompt_ids, n_new, k, n),
                repeats,
            )
            if tokenizer.decode(out_ids) != base_text:
                raise AssertionError(f"{name}, k={k}: output differs from greedy baseline")

            drafted = stats["drafted"]
            acceptance = stats["accepted"] / drafted if drafted else 0.0
            tps = n_new / dt
            tokens_per_pass = n_new / stats["passes"]

            results[name]["acceptance"][k] = acceptance
            results[name]["speedup"][k] = tps / base_tps
            results[name]["tokens_per_pass"][k] = tokens_per_pass

            print(
                f"{name:>6} {k:>3} {acceptance:>6.1%} {drafted:>8} "
                f"{stats['accepted'] / max(stats['draft_passes'], 1):>9.2f} "
                f"{stats['passes']:>7} {tps:>7.1f} {tps / base_tps:>7.2f}x"
            )
    return results


def plot_acceptance(results, draft_lengths):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, data in results.items():
        ys = [100 * data["acceptance"][k] for k in draft_lengths]
        ax.plot(draft_lengths, ys, marker="o", label=name)

    mean = [
        100
        * sum(data["acceptance"][k] for data in results.values())
        / len(results)
        for k in draft_lengths
    ]
    ax.plot(draft_lengths, mean, color="black", linestyle="--", label="mean")

    ax.set_xlabel("draft length k")
    ax.set_ylabel("acceptance rate (%)")
    ax.set_title(f"Prompt-lookup acceptance vs k (n={NGRAM}, greedy)")
    ax.set_xticks(draft_lengths)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"\nSaved {PLOT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--ngram", type=int, default=NGRAM)
    parser.add_argument("--tokens", type=int, default=NEW_TOKENS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--quick", action="store_true", help="fewer tokens, one repeat"
    )
    args = parser.parse_args()

    n_new = 32 if args.quick else args.tokens
    repeats = 1 if args.quick else args.repeats

    print(f"device: {DEVICE}, model: {args.model}, n-gram: {args.ngram}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Model(args.model)

    # Warm up the first shape and the graph.
    normal_decode(model, tokenizer("warm up")["input_ids"], 2)

    results = benchmark(
        model, tokenizer, PROMPTS, DRAFT_LENGTHS, n_new, args.ngram, repeats
    )
    plot_acceptance(results, DRAFT_LENGTHS)

    print("\nacceptance by k:")
    print(f"{'prompt':>6} " + " ".join(f"{k:>6}" for k in DRAFT_LENGTHS))
    for name, data in results.items():
        print(
            f"{name:>6} "
            + " ".join(f"{100 * data['acceptance'][k]:>5.1f}%" for k in DRAFT_LENGTHS)
        )


if __name__ == "__main__":
    main()
