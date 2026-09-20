import numpy as np


def relu(x):
    """Activation: keep positives, zero out negatives."""
    return np.maximum(0, x)


def linear(x, W, b):
    """One dense layer: matrix multiply, then add bias."""
    return x @ W + b


def mlp_forward(x, W1, b1, W2, b2):
    """2-layer MLP: input -> linear -> ReLU -> linear -> output."""
    hidden = relu(linear(x, W1, b1))   # (batch, hidden_dim)
    out = linear(hidden, W2, b2)       # (batch, output_dim)
    return out


def init_params(input_dim, hidden_dim, output_dim, seed=0):
    """Random weights (small values), zero biases."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.1, size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    W2 = rng.normal(0, 0.1, size=(hidden_dim, output_dim))
    b2 = np.zeros(output_dim)
    return W1, b1, W2, b2


if __name__ == "__main__":
    batch, input_dim, hidden_dim, output_dim = 4, 3, 5, 2

    rng = np.random.default_rng(42)
    x = rng.normal(size=(batch, input_dim))
    W1, b1, W2, b2 = init_params(input_dim, hidden_dim, output_dim)

    # Step by step, so we can see every shape
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2

    print("x  :", x.shape)
    print("W1 :", W1.shape, " b1:", b1.shape)
    print("z1 :", z1.shape, "(before activation)")
    print("a1 :", a1.shape, "(after ReLU)")
    print("W2 :", W2.shape, " b2:", b2.shape)
    print("out:", z2.shape)

    print("\nz1 (negatives present):\n", np.round(z1, 3))
    print("a1 (negatives zeroed): \n", np.round(a1, 3))

    out = mlp_forward(x, W1, b1, W2, b2)
    assert np.allclose(out, z2)
    print("\nOutput:\n", np.round(out, 3))

    # Sanity check: one input alone gives the same answer as inside the batch
    single = mlp_forward(x[0:1], W1, b1, W2, b2)
    assert np.allclose(single, out[0:1])
    print("\nChecks passed.")
