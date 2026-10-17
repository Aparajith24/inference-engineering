import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    Input and output are (B, T, d_model): B sequences, T tokens each.
    """

    def __init__(self, d_model, n_heads, bias=True, use_sdpa=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_sdpa = use_sdpa
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)  # Wq, Wk, Wv in one matmul
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)  # mixes the heads back together

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)  # each (B, T, C)
        # split C into n_heads slices of d_head: (B, T, C) -> (B, n_heads, T, d_head)
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v))

        if self.use_sdpa:  # PyTorch's built-in fused version (day 14), for comparison
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            weights = None
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, n_heads, T, T)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))  # block every future position
            weights = scores.softmax(dim=-1)
            out = weights @ v  # (B, n_heads, T, d_head)

        out = out.transpose(1, 2).reshape(B, T, C)  # glue the heads back side by side
        out = self.out_proj(out)
        return (out, weights) if return_weights else out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d_model, n_heads = 2, 6, 32, 4
    x = torch.randn(B, T, d_model)
    mha = MultiHeadAttention(d_model, n_heads)
    mha.eval()

    with torch.no_grad():
        out, weights = mha(x, return_weights=True)
    print(f"input  x       : {tuple(x.shape)}   (batch, tokens, d_model)")
    print(f"output         : {tuple(out.shape)}   same shape as the input")
    print(f"weights        : {tuple(weights.shape)}   (batch, heads, T, T): one (T, T) grid PER HEAD")
    print(f"d_head         : {mha.d_head}   (d_model {d_model} / {n_heads} heads)")

    print("\nHead 0 attention weights for sequence 0 (lower triangle = causal):")
    print(np.round(weights[0, 0].numpy(), 2))
    print("\nLast token's attention, head by head (each head looks at different tokens):")
    for h in range(n_heads):
        print(f"  head {h}: {np.round(weights[0, h, -1].numpy(), 2)}")

    # ---- Checks ----
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, n_heads, T), atol=1e-6)
    assert torch.allclose(torch.triu(weights, diagonal=1), torch.zeros_like(weights))

    # 1. The fused PyTorch kernel gives the same answer
    mha_sdpa = MultiHeadAttention(d_model, n_heads, use_sdpa=True)
    mha_sdpa.load_state_dict(mha.state_dict())
    with torch.no_grad():
        assert torch.allclose(out, mha_sdpa(x), atol=1e-5)

    # 2. Causality: changing the LAST token must not change any earlier output
    x2 = x.clone()
    x2[:, -1] = torch.randn(B, d_model)
    with torch.no_grad():
        out2 = mha(x2)
    assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-6)
    assert not torch.allclose(out[:, -1], out2[:, -1])

    # 3. One head, no biases == the NumPy attention we wrote on day 8
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-8"))
    from attention_numpy import causal_mask, self_attention

    one = MultiHeadAttention(d_model, n_heads=1, bias=False)
    with torch.no_grad():
        one.out_proj.weight.copy_(torch.eye(d_model))  # make out_proj a no-op
        W = one.qkv.weight.detach().numpy()  # (3*d_model, d_model); nn.Linear computes x @ W.T
        Wq, Wk, Wv = (W[i * d_model:(i + 1) * d_model].T for i in range(3))
        ref, _ = self_attention(x[0].numpy(), Wq, Wk, Wv, mask=causal_mask(T))
        mine = one(x)[0].numpy()
    assert np.allclose(mine, ref, atol=1e-5)

    # 4. Parameter count: 4 * d_model^2 without biases (Wq, Wk, Wv, Wo)
    n_params = sum(p.numel() for p in MultiHeadAttention(d_model, n_heads, bias=False).parameters())
    assert n_params == 4 * d_model**2
    print(f"\nChecks passed: PyTorch fused kernel matches, causality holds, "
          f"1 head matches day 8's NumPy, params = 4*d_model^2 = {n_params}.")
