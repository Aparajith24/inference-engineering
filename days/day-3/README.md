# Day 3: A 2-layer MLP forward pass in NumPy

## What we built

`mlp_numpy.py` implements a 2-layer MLP (multi-layer perceptron) forward
pass in pure NumPy: matmul, bias, and an activation function. There is no
training and no deep learning framework.

Run it:

```
python3 mlp_numpy.py
```

It prints the shape at every step, the hidden layer before and after ReLU,
and the final output, then runs two sanity checks.

## What we learned

### Forward pass = inference

A forward pass runs an input through the network to get an output. That is
exactly what inference means. The `distilgpt2` model from day 1 does the same
thing with much larger layers, once per generated token.

```
input (3) -> linear -> ReLU -> linear -> output (2)
```

### The three ingredients

| Piece | Code | What it does |
|---|---|---|
| Matmul | `x @ W` | Each output is a weighted sum of the inputs. `W` holds the learned weights. |
| Bias | `+ b` | Shifts each output up or down, like the intercept in `y = mx + c`. |
| Activation | `np.maximum(0, x)` (ReLU) | Zeroes negatives. Adds non-linearity. |

`x @ W + b` together is a **linear layer**.

### Why the activation matters

Without it, two linear layers collapse into one, because `(x @ W1) @ W2` equals
`x @ (W1 @ W2)`. Depth would add nothing. A non-linear step between layers is
what lets a network model complicated relationships.

### Shapes

```
x   (4, 3)   batch of 4 examples, 3 features each
W1  (3, 5)   maps 3 -> 5
z1  (4, 5)   hidden layer, before ReLU (has negatives)
a1  (4, 5)   hidden layer, after ReLU
W2  (5, 2)   maps 5 -> 2
out (4, 2)   final output
```

- In a matmul the inner dimensions must match and disappear: `(4,3) @ (3,5) -> (4,5)`.
- The batch dimension (4) never changes. Every row is processed independently,
  but all at once in a single matmul. This is why batching is efficient.
- The bias has shape `(5,)` and is added to every row through NumPy
  **broadcasting**.

## Connection to inference engineering

- Every transformer layer, including in `distilgpt2`, contains an MLP block
  much like this one (larger, with a smoother activation called GELU).
- Most of a model's parameters are matrices like `W1` and `W2`. Reading them
  from memory for each token is what makes decode memory-bound (day 1).
- **Quantization** (day 2) shrinks these matrices, and **batching** (day 2)
  pushes many rows through one matmul.

## Caveats

- Weights are random and never trained, so the output numbers are meaningless.
  The point is the mechanics.
- Training (backpropagation) adjusts `W` and `b` to make outputs useful. That
  is not covered here.

## Things to try

1. Set `hidden_dim` to 100 and watch the shapes change.
2. Replace `relu` with `lambda x: x` and confirm the output equals
   `x @ (W1 @ W2)` (the layer collapse).
3. Print `W1.size + W2.size` to count parameters.
