import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-10"))
from tiny_gpt_generate import load_gpt2_weights  # noqa: E402  (day 10's model, trained weights)


def top_k_filter(logits, k):
    """Keep the k highest-scoring tokens; every other token gets -inf (probability 0)."""
    kth_best = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth_best, float("-inf"))


def top_p_filter(logits, p):
    """Keep the smallest set of top tokens whose probabilities add up to at least p."""
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs  # probability of all the tokens ranked above
    remove = mass_before >= p                   # the set already reached p without this token
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter(-1, order, sorted_logits)  # undo the sort


def next_token_probs(logits, temperature=1.0, top_k=None, top_p=None):
    """The distribution we actually sample from, after temperature, top-k and top-p."""
    logits = logits / temperature  # < 1 sharpens, > 1 flattens
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    return logits.softmax(dim=-1)  # filtered tokens had -inf, so they get exactly 0


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0:  # temperature 0 means greedy: always the top token
        return logits.argmax(dim=-1, keepdim=True)
    probs = next_token_probs(logits, temperature, top_k, top_p)
    return torch.multinomial(probs, num_samples=1, generator=generator)  # weighted random pick


@torch.no_grad()
def generate(model, idx, max_new_tokens, generator=None, **sampling):
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.max_len:])[:, -1, :]
        idx = torch.cat([idx, sample_next(logits, generator=generator, **sampling)], dim=1)
    return idx


def describe(name, probs, tok):
    kept = int((probs > 0).sum())
    top = torch.topk(probs, 3)
    tops = ", ".join(f"{tok.decode([i])!r} {v:.2f}" for v, i in zip(top.values.tolist(), top.indices.tolist()))
    print(f"{name:<22}{kept:>8}   {tops}")


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model, _ = load_gpt2_weights("distilgpt2")
    prompt = "The future of AI inference is"
    idx = tok(prompt, return_tensors="pt")["input_ids"]

    with torch.no_grad():
        logits = model(idx)[0, -1]  # scores for the token after the prompt

    # ---- 1. How each setting reshapes ONE next-token distribution ----
    print(f"Next-token distribution after {prompt!r} (vocabulary: {logits.numel():,} tokens)\n")
    print(f"{'setting':<22}{'# kept':>8}   top 3 tokens")
    for name, kw in [
        ("temperature 0.5", dict(temperature=0.5)),
        ("temperature 1.0", dict(temperature=1.0)),
        ("temperature 1.5", dict(temperature=1.5)),
        ("top-k 5", dict(top_k=5)),
        ("top-k 50", dict(top_k=50)),
        ("top-p 0.5", dict(top_p=0.5)),
        ("top-p 0.9", dict(top_p=0.9)),
    ]:
        describe(name, next_token_probs(logits, **kw), tok)

    # ---- 2. Checks ----
    # Our filters keep exactly the same tokens as Hugging Face's own implementations
    scores = logits.unsqueeze(0)
    for ours, theirs in [
        (top_k_filter(scores, 50), TopKLogitsWarper(top_k=50)(idx, scores.clone())),
        (top_p_filter(scores, 0.9), TopPLogitsWarper(top_p=0.9)(idx, scores.clone())),
        (top_p_filter(scores, 0.5), TopPLogitsWarper(top_p=0.5)(idx, scores.clone())),
        (scores / 0.7, TemperatureLogitsWarper(0.7)(idx, scores.clone())),
    ]:
        assert torch.equal(torch.isfinite(ours), torch.isfinite(theirs))
        assert torch.allclose(ours[torch.isfinite(ours)], theirs[torch.isfinite(theirs)])
    # Temperature 0 and top-k 1 are both deterministic and equal to greedy
    greedy = generate(model, idx, 15, temperature=0)
    g = torch.Generator().manual_seed(123)
    assert torch.equal(generate(model, idx, 15, top_k=1, generator=g), greedy)
    print("\nChecks passed: filters match Hugging Face's, temperature 0 == top-k 1 == greedy.")

    # ---- 3. Compare generated text: 3 samples per setting ----
    n_new, n_samples = 25, 3
    settings = [
        ("greedy (temperature 0)", dict(temperature=0)),
        ("temperature 0.7", dict(temperature=0.7)),
        ("temperature 1.5", dict(temperature=1.5)),
        ("top-k 5", dict(top_k=5)),
        ("top-p 0.9", dict(top_p=0.9)),
        ("temp 0.8 + top-k 50 + top-p 0.95", dict(temperature=0.8, top_k=50, top_p=0.95)),
    ]
    print(f"\nGenerated continuations ({n_samples} samples each, {n_new} tokens, same seeds):")
    for name, kw in settings:
        print(f"\n== {name}")
        outputs = []
        for s in range(n_samples):
            g = torch.Generator().manual_seed(s)
            out = generate(model, idx, n_new, generator=g, **kw)[0, idx.shape[1]:]
            outputs.append(out.tolist())
            text = tok.decode(out).replace("\n", "\\n")
            print(f"  [{s}] {text}")
        distinct = len({tuple(o) for o in outputs})
        variety = sum(len(set(o)) / len(o) for o in outputs) / n_samples
        print(f"  -> {distinct}/{n_samples} different outputs, "
              f"{variety:.0%} of tokens within a sample are unique (low = repetitive)")
