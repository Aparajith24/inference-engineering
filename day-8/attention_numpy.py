import time

import numpy as np


def softmax(x, axis=-1):
    """Turn scores into probabilities that sum to 1 along `axis`."""
    x = x - x.max(axis=axis, keepdims=True)  # subtract the max so exp() cannot overflow
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(T):
    """True where a token is NOT allowed to look: every position after itself."""
    return np.triu(np.ones((T, T), dtype=bool), k=1)


def self_attention(x, Wq, Wk, Wv, mask=None):
    """Single-head self-attention.

    x:  (T, d_model)   one row per token
    Wq, Wk, Wv: (d_model, d_k)
    returns (output (T, d_k), attention weights (T, T))
    """
    Q = x @ Wq  # what each token is looking for
    K = x @ Wk  # what each token offers to be matched against
    V = x @ Wv  # the information each token hands over if attended to

    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # (T, T): how well token i matches token j
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)  # exp(-inf) = 0, so masked tokens get weight 0
    weights = softmax(scores)  # each row sums to 1
    return weights @ V, weights  # each output is a weighted average of the V rows


def self_attention_loop(x, Wq, Wk, Wv, mask=None):
    """Same thing with explicit loops, one token at a time. Slow, but easy to read."""
    T = x.shape[0]
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    d_k = Q.shape[-1]
    out = np.zeros_like(V)
    for i in range(T):
        scores = np.array([Q[i] @ K[j] / np.sqrt(d_k) for j in range(T)])
        if mask is not None:
            scores[mask[i]] = -np.inf
        w = softmax(scores)
        out[i] = sum(w[j] * V[j] for j in range(T))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d_model, d_k = 5, 8, 4

    x = rng.normal(size=(T, d_model))
    Wq, Wk, Wv = (rng.normal(0, 0.5, size=(d_model, d_k)) for _ in range(3))

    out, weights = self_attention(x, Wq, Wk, Wv)
    print("x:", x.shape, " Wq/Wk/Wv:", Wq.shape)
    print("scores/weights:", weights.shape, " output:", out.shape)
    print("\nAttention weights (row i = how much token i attends to each token):")
    print(np.round(weights, 2))
    print("Row sums:", np.round(weights.sum(axis=1), 6))

    # Causal mask: a token may only look at itself and earlier tokens
    out_c, weights_c = self_attention(x, Wq, Wk, Wv, mask=causal_mask(T))
    print("\nWith causal mask:")
    print(np.round(weights_c, 2))

    # Checks
    assert np.allclose(weights.sum(axis=1), 1)
    assert np.allclose(out, self_attention_loop(x, Wq, Wk, Wv))
    assert np.allclose(out_c, self_attention_loop(x, Wq, Wk, Wv, mask=causal_mask(T)))
    assert np.allclose(np.triu(weights_c, k=1), 0)

    # Under a causal mask, changing the LAST token must not change any earlier output
    x2 = x.copy()
    x2[-1] = rng.normal(size=d_model)
    out_c2, _ = self_attention(x2, Wq, Wk, Wv, mask=causal_mask(T))
    assert np.allclose(out_c[:-1], out_c2[:-1])
    assert not np.allclose(out_c[-1], out_c2[-1])
    print("\nChecks passed (loop == matrix version, rows sum to 1, causality holds).")

    # The (T, T) score matrix grows with the SQUARE of the sequence length
    print("\nCost vs. sequence length (d_k = 64):")
    print(f"{'T':>6}{'scores (T*T)':>16}{'time (ms)':>12}")
    d = 64
    self_attention(rng.normal(size=(64, d)), *(rng.normal(size=(d, d)) for _ in range(3)))  # warmup
    for T in [256, 512, 1024, 2048, 4096]:
        xs = rng.normal(size=(T, d)).astype(np.float32)
        W = [rng.normal(0, 0.1, size=(d, d)).astype(np.float32) for _ in range(3)]
        # NumPy 2.0 with Apple's Accelerate BLAS raises spurious floating-point warnings
        # on larger matmuls, so silence them here and check the values ourselves instead.
        with np.errstate(all="ignore"):
            start = time.perf_counter()
            out, _ = self_attention(xs, *W)
            ms = (time.perf_counter() - start) * 1000
        assert np.isfinite(out).all()
        print(f"{T:>6}{T * T:>16,}{ms:>12.1f}")
