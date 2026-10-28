import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DRAFT_NAME = "gpt2"
TARGET_NAME = "gpt2-medium"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_NEW_TOKENS = 64
DRAFT_LENGTHS = [1, 2, 3, 4, 6, 8]
TEMPERATURES = [0.0, 1.0]
SEED = 0
TIMING_REPEATS = 3
PROMPTS = [
    "The history of the Roman Empire",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "The best way to learn a new language is",
    "1, 2, 3, 4, 5, 6, 7,",
]


class Model:
    """A causal LM plus its KV cache, so it only ever processes tokens it has not seen yet."""

    def __init__(self, name):
        self.name = name
        self.lm = AutoModelForCausalLM.from_pretrained(name).to(DEVICE).eval()
        self.cache = DynamicCache()

    def reset(self):
        """Forget everything: start a new sequence."""
        self.cache = DynamicCache()

    def seen(self):
        """How many tokens are already in the KV cache."""
        return self.cache.get_seq_length()

    def forward(self, tokens):
        """Feed the tokens the cache has not seen yet and return the logits for each of them."""
        new = torch.tensor([tokens[self.seen():]], device=DEVICE)
        with torch.no_grad():
            out = self.lm(new, past_key_values=self.cache, use_cache=True)
        return out.logits[0]

    def rewind(self, length):
        """Drop cached tokens past `length` (used after rejected drafts)."""
        self.cache.crop(length)


def probabilities(logits, temperature):
    """Turn logits into a probability distribution; temperature 0 puts all probability on the top token."""
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample(probs):
    """Pick one token id from a probability distribution."""
    return torch.multinomial(probs, 1).item()


def accept_or_reject(draft_token, p, q):
    """Speculative sampling: keep the draft with probability min(1, p/q), else resample from what p has spare.

    Returns (accepted, token). This rule makes the output follow the target's distribution p exactly.
    """
    if torch.rand(1).item() < min(1.0, (p[draft_token] / q[draft_token]).item()):
        return True, draft_token
    leftover = torch.clamp(p - q, min=0)
    return False, sample(leftover / leftover.sum())


def normal_decode(target, prompt_ids, n_new, temperature):
    """The baseline: the target model alone, one token per forward pass."""
    target.reset()
    tokens = list(prompt_ids)
    passes = 0
    while len(tokens) - len(prompt_ids) < n_new:
        logits = target.forward(tokens)
        tokens.append(sample(probabilities(logits[-1], temperature)))
        passes += 1
    return tokens[len(prompt_ids):], {"target_passes": passes}


def speculative_decode(draft, target, prompt_ids, n_new, temperature, k, trace=None):
    """Draft k tokens with the small model, check them all in one pass of the big model, keep the good ones."""
    draft.reset()
    target.reset()
    tokens = list(prompt_ids)
    stats = {"target_passes": 0, "drafted": 0, "accepted": 0}
    while len(tokens) - len(prompt_ids) < n_new:
        start = len(tokens)
        drafts, draft_probs = [], []
        for _ in range(k):
            q = probabilities(draft.forward(tokens + drafts)[-1], temperature)
            drafts.append(sample(q))
            draft_probs.append(q)

        logits = target.forward(tokens + drafts)
        offset = len(logits) - k - 1
        target_probs = [probabilities(logits[offset + j], temperature) for j in range(k + 1)]
        stats["target_passes"] += 1
        stats["drafted"] += k

        for j, d in enumerate(drafts):
            accepted, token = accept_or_reject(d, target_probs[j], draft_probs[j])
            tokens.append(token)
            if trace is not None:
                trace.append((d, target_probs[j].argmax().item(), accepted, token))
            if not accepted:
                break
            stats["accepted"] += 1
        else:
            bonus = sample(target_probs[k])
            tokens.append(bonus)
            if trace is not None:
                trace.append((None, bonus, True, bonus))

        target.rewind(len(tokens) - 1)
        draft.rewind(min(draft.seen(), len(tokens) - 1))
        if trace is not None:
            trace.append("end of pass")
    return tokens[len(prompt_ids):][:n_new], stats


def toy_rejection_sampling():
    """Show that accept/reject turns the draft's guesses into exactly the target's distribution."""
    words = ["mat", "sofa", "floor", "roof"]
    p = torch.tensor([0.50, 0.30, 0.15, 0.05])
    q = torch.tensor([0.70, 0.10, 0.10, 0.10])
    counts = torch.zeros(4)
    accepted = 0
    n = 20000
    for _ in range(n):
        d = sample(q)
        ok, token = accept_or_reject(d, p, q)
        accepted += ok
        counts[token] += 1
    print('Toy: "The cat sat on the ..." with 4 possible words')
    print(f"  {'word':>6} {'draft q':>8} {'target p':>9} {'result':>8}")
    for i, w in enumerate(words):
        print(f"  {w:>6} {q[i]:>8.2f} {p[i]:>9.2f} {counts[i] / n:>8.3f}")
    print(f"  accepted {accepted / n:.0%} of drafts (theory: sum of min(p, q) = {torch.minimum(p, q).sum():.2f})")
    print("  'result' matches the target p, not the draft q: speculation does not change the output distribution\n")


def walk_through(draft, target, tokenizer, k=4):
    """One prompt, greedy, printing every draft and the target's verdict on it."""
    prompt = "The cat sat on the"
    ids = tokenizer(prompt).input_ids
    trace = []
    out, _ = speculative_decode(draft, target, ids, 16, 0.0, k, trace)
    print(f"Walk-through: {prompt!r}, greedy, {k} drafts per pass")
    line = []
    for item in trace:
        if item == "end of pass":
            print("  pass: " + "  ".join(line))
            line = []
            continue
        d, wanted, ok, token = item
        if d is None:
            line.append(f"+ bonus {tokenizer.decode([token])!r}")
        elif ok:
            line.append(f"{tokenizer.decode([d])!r} ok")
        else:
            line.append(f"{tokenizer.decode([d])!r} NO -> {tokenizer.decode([token])!r}")
    print(f"  result: {prompt + tokenizer.decode(out)!r}\n")


def timed(fn):
    """Run fn and return its result and the seconds it took, waiting for the GPU to finish."""
    if DEVICE == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    result = fn()
    if DEVICE == "mps":
        torch.mps.synchronize()
    return result, time.perf_counter() - start


def best_of(fn):
    """Run fn TIMING_REPEATS times with the same seed and keep the fastest time (less noise from the rest of the Mac)."""
    best = None
    for _ in range(TIMING_REPEATS):
        torch.manual_seed(SEED)
        result, seconds = timed(fn)
        best = seconds if best is None else min(best, seconds)
    return result, best


def costs(draft, target, tokenizer):
    """Time one forward pass of each model, and the target checking 1 to 9 tokens at once."""
    ids = tokenizer(PROMPTS[0]).input_ids * 4
    print("Costs: one forward pass with a cached prompt (ms)")
    for model in (draft, target):
        model.reset()
        model.forward(ids)
        times = []
        for _ in range(20):
            _, s = timed(lambda: model.forward(ids + [11]))
            model.rewind(len(ids))
            times.append(s * 1000)
        print(f"  {model.name:>12}, 1 new token: {sorted(times)[10]:6.2f}")
    for n in (1, 3, 5, 9):
        times = []
        for _ in range(20):
            _, s = timed(lambda: target.forward(ids + [11] * n))
            target.rewind(len(ids))
            times.append(s * 1000)
        print(f"  {target.name:>12}, {n} tokens checked at once: {sorted(times)[10]:6.2f}")
    print()


def benchmark(draft, target, tokenizer):
    """Normal decoding vs speculative decoding for several draft lengths and temperatures."""
    prompts = [tokenizer(p).input_ids for p in PROMPTS]
    print(f"Benchmark: {len(prompts)} prompts x {MAX_NEW_TOKENS} new tokens, draft {DRAFT_NAME}, target {TARGET_NAME}, {DEVICE}, "
          f"best of {TIMING_REPEATS}\n")
    header = f"{'temp':>4} {'method':>12} {'accepted':>8} {'tokens/pass':>11} {'tok/s':>6} {'speedup':>7} {'same as normal':>14}"
    print(header)
    print("-" * len(header))
    for temperature in TEMPERATURES:
        normal, seconds = best_of(lambda: [normal_decode(target, p, MAX_NEW_TOKENS, temperature) for p in prompts])
        base = len(prompts) * MAX_NEW_TOKENS / seconds
        print(f"{temperature:>4} {'normal':>12} {'':>8} {1.0:>11.2f} {base:>6.1f} {1.0:>6.2f}x {'':>14}")
        for k in DRAFT_LENGTHS:
            runs, seconds = best_of(lambda: [speculative_decode(draft, target, p, MAX_NEW_TOKENS, temperature, k) for p in prompts])
            drafted = sum(s["drafted"] for _, s in runs)
            accepted = sum(s["accepted"] for _, s in runs)
            passes = sum(s["target_passes"] for _, s in runs)
            tok_s = len(prompts) * MAX_NEW_TOKENS / seconds
            same = "n/a (sampled)" if temperature else f"{sum(a == b[0] for (a, _), b in zip(runs, normal))}/{len(prompts)} prompts"
            print(f"{temperature:>4} {f'k = {k}':>12} {accepted / drafted:>8.0%} {len(prompts) * MAX_NEW_TOKENS / passes:>11.2f} "
                  f"{tok_s:>6.1f} {tok_s / base:>6.2f}x {same:>14}")
        print()


def per_prompt(draft, target, tokenizer, k=4):
    """Acceptance rate for each prompt at greedy: the subject matter matters."""
    print(f"Acceptance by prompt (greedy, k = {k})")
    for text in PROMPTS:
        _, s = speculative_decode(draft, target, tokenizer(text).input_ids, MAX_NEW_TOKENS, 0.0, k)
        print(f"  {s['accepted'] / s['drafted']:>4.0%}  {text[:45]!r}")
    print()


def main():
    """Toy check, a walk-through, costs, then the benchmark."""
    parser = argparse.ArgumentParser(description="Speculative decoding with a small draft model and a big target model")
    parser.add_argument("--quick", action="store_true", help="skip the benchmark")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    toy_rejection_sampling()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_NAME)
    draft, target = Model(DRAFT_NAME), Model(TARGET_NAME)
    walk_through(draft, target, tokenizer)
    costs(draft, target, tokenizer)
    per_prompt(draft, target, tokenizer)
    if not args.quick:
        benchmark(draft, target, tokenizer)


if __name__ == "__main__":
    main()
