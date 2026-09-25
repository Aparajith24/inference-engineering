import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-9"))
from mha_causal import MultiHeadAttention  # noqa: E402  (day 9's attention, reused as-is)


class MLP(nn.Module):
    """The day 3 MLP: linear -> activation -> linear, 4x wider in the middle."""

    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))  # GPT-2 uses the tanh GELU


class Block(nn.Module):
    """One transformer block: attention then MLP, each with LayerNorm and a residual add."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # tokens exchange information
        x = x + self.mlp(self.ln2(x))   # each token is processed on its own
        return x


class TinyGPT(nn.Module):
    """token ids (B, T) -> logits (B, T, vocab_size)."""

    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)  # token id -> vector
        self.pos_emb = nn.Embedding(max_len, d_model)     # position -> vector
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # vector -> one score per token
        self.lm_head.weight = self.tok_emb.weight  # weight tying, as in GPT-2

    def forward(self, idx, trace=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        if trace:
            print(f"  embeddings    {tuple(x.shape)}")
        for i, block in enumerate(self.blocks):
            x = block(x)
            if trace:
                print(f"  after block {i} {tuple(x.shape)}")
        logits = self.lm_head(self.ln_f(x))
        if trace:
            print(f"  logits        {tuple(logits.shape)}")
        return logits


@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens, eos_id=None):
    """Autoregressive loop: predict, append, repeat. No KV cache yet (day 13)."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_len:]      # never feed more than the context window
        logits = model(idx_cond)[:, -1, :]      # only the LAST position predicts the next token
        next_id = logits.argmax(dim=-1, keepdim=True)  # greedy: take the top token
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and (next_id == eos_id).all():
            break
    return idx


def load_gpt2_weights(model_name="distilgpt2"):
    """Build a TinyGPT with the same shape as a GPT-2 model and copy its trained weights in."""
    hf = AutoModelForCausalLM.from_pretrained(model_name)
    cfg = hf.config
    model = TinyGPT(cfg.vocab_size, cfg.n_positions, cfg.n_embd, cfg.n_head, cfg.n_layer)
    sd = hf.state_dict()

    def copy_linear(ours, prefix):
        # GPT-2 stores these as (in, out) for x @ W; nn.Linear stores (out, in), so transpose
        ours.weight.copy_(sd[prefix + ".weight"].T)
        ours.bias.copy_(sd[prefix + ".bias"])

    def copy_ln(ours, prefix):
        ours.weight.copy_(sd[prefix + ".weight"])
        ours.bias.copy_(sd[prefix + ".bias"])

    with torch.no_grad():
        model.tok_emb.weight.copy_(sd["transformer.wte.weight"])  # also the lm_head (tied)
        model.pos_emb.weight.copy_(sd["transformer.wpe.weight"])
        for i, block in enumerate(model.blocks):
            p = f"transformer.h.{i}"
            copy_ln(block.ln1, f"{p}.ln_1")
            copy_linear(block.attn.qkv, f"{p}.attn.c_attn")
            copy_linear(block.attn.out_proj, f"{p}.attn.c_proj")
            copy_ln(block.ln2, f"{p}.ln_2")
            copy_linear(block.mlp.fc, f"{p}.mlp.c_fc")
            copy_linear(block.mlp.proj, f"{p}.mlp.c_proj")
        copy_ln(model.ln_f, "transformer.ln_f")
    return model.eval(), hf.eval()


def count(module):
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    prompt = "The future of AI inference is"
    idx = tok(prompt, return_tensors="pt")["input_ids"]

    # ---- Part 1: a tiny random model, to see the pipeline and the shapes ----
    print("PART 1: tiny random GPT (untrained, so the text is gibberish)\n")
    tiny = TinyGPT(vocab_size=tok.vocab_size, max_len=128, d_model=64, n_heads=4, n_layers=2).eval()
    print(f"prompt {prompt!r} -> token ids {tuple(idx.shape)}: {idx[0].tolist()}")
    with torch.no_grad():
        tiny(idx, trace=True)
    out = generate_greedy(tiny, idx, max_new_tokens=10)
    print(f"generated: {tok.decode(out[0, idx.shape[1]:])!r}")

    emb = tiny.tok_emb.weight.numel() + tiny.pos_emb.weight.numel()
    blocks = sum(count(b) for b in tiny.blocks)
    print(f"\nparameters: {count(tiny):,} total = {emb:,} embeddings + {blocks:,} blocks + "
          f"{count(tiny.ln_f):,} final norm  (lm_head shares the embedding)")

    # ---- Part 2: the same architecture with distilgpt2's trained weights ----
    print("\nPART 2: same TinyGPT class, loaded with distilgpt2's trained weights\n")
    ours, hf = load_gpt2_weights("distilgpt2")
    print(f"our model: {count(ours):,} parameters | Hugging Face distilgpt2: {count(hf):,}")

    with torch.no_grad():
        ours_logits = ours(idx)
        hf_logits = hf(idx).logits
    diff = (ours_logits - hf_logits).abs().max().item()
    print(f"max logit difference vs Hugging Face: {diff:.2e}")
    assert diff < 1e-3

    n_new = 30
    start = time.perf_counter()
    out = generate_greedy(ours, idx, max_new_tokens=n_new)
    secs = time.perf_counter() - start
    with torch.no_grad():
        ref = hf.generate(idx, attention_mask=torch.ones_like(idx), max_new_tokens=n_new,
                          do_sample=False, pad_token_id=tok.eos_token_id)
    assert torch.equal(out, ref), "our greedy loop should match Hugging Face token for token"

    print(f"\ngreedy output ({n_new} tokens):\n{prompt}{tok.decode(out[0, idx.shape[1]:])}")
    print(f"\nIdentical to Hugging Face's generate() token for token. "
          f"({n_new} tokens in {secs:.2f}s, {n_new / secs:.0f} tokens/s, no KV cache)")
