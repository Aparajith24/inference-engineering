import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
QMAX = 127
LAYER_NAMES = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
TEXT = (
    "The history of computing is full of surprising turns. Early machines filled whole rooms and "
    "needed teams of engineers to keep them running, yet they could do less than a cheap calculator "
    "does today. Over the following decades, transistors replaced vacuum tubes, chips replaced "
    "circuit boards, and the price of a single calculation fell by a factor of billions. Today a "
    "phone in your pocket can run a language model that writes stories, answers questions and "
    "translates between languages."
)


def quantize_per_tensor(w):
    """One scale for the whole matrix: the biggest absolute value maps to 127."""
    scale = w.abs().max() / QMAX
    q = torch.clamp(torch.round(w / scale), -QMAX, QMAX).to(torch.int8)
    return q, scale


def quantize_per_channel(w, dim):
    """One scale per output channel; `dim` is the axis that holds the inputs of each channel."""
    scale = w.abs().amax(dim=dim, keepdim=True) / QMAX
    q = torch.clamp(torch.round(w / scale), -QMAX, QMAX).to(torch.int8)
    return q, scale


def dequantize(q, scale):
    """Turn int8 values back into floats."""
    return q.float() * scale


def relative_error(approx, exact):
    """Size of the error compared with the size of the true values, in percent."""
    return 100 * ((approx - exact).norm() / exact.norm()).item()


def toy_example():
    """The outlier problem with four numbers, then with a 2x4 matrix."""
    x = torch.tensor([0.1, 0.5, -2.0, 30.0])
    q, scale = quantize_per_tensor(x)
    print("Four numbers with one outlier:", [round(v, 3) for v in x.tolist()])
    print(f"  scale = 30 / 127 = {scale:.4f}")
    print("  int8 values:  ", q.tolist())
    print("  back to float:", [round(v, 3) for v in dequantize(q, scale).tolist()], " <- 0.1 became 0\n")

    w = torch.tensor([[0.10, -0.20, 0.05, 0.15],
                      [8.00, -6.00, 3.00, 9.00]])
    print("A 2x4 weight matrix: row 0 is small, row 1 has big values")
    for name, (q, scale) in (("per-tensor ", quantize_per_tensor(w)), ("per-channel", quantize_per_channel(w, dim=1))):
        back = dequantize(q, scale)
        print(f"  {name}: row 0 int8 = {q[0].tolist()}, back = {[round(v, 3) for v in back[0].tolist()]}, "
              f"row 0 error {relative_error(back[0], w[0]):.1f}%")
    print()


def layer_inputs(model, tokenizer):
    """Run the text through the model once and keep the input of every linear layer."""
    inputs, hooks = {}, []
    for i, block in enumerate(model.transformer.h):
        for name in LAYER_NAMES:
            layer = block.get_submodule(name)
            hooks.append(layer.register_forward_hook(
                lambda module, args, out, key=f"h{i}.{name}": inputs.__setitem__(key, args[0][0])))
    with torch.no_grad():
        model(**tokenizer(TEXT, return_tensors="pt"))
    for hook in hooks:
        hook.remove()
    return inputs


def layer_study(model, inputs):
    """For every linear layer, compare per-tensor and per-channel int8 on weights and on outputs."""
    print("Every linear layer of distilgpt2, int8 weights (errors in %, lower is better)\n")
    header = (f"{'layer':>16} {'shape':>11} {'outlier ratio':>13} {'weight err':>21} {'output err':>21}")
    print(header)
    print(f"{'':>16} {'':>11} {'':>13} {'tensor':>10} {'channel':>10} {'tensor':>10} {'channel':>10}")
    print("-" * len(header))
    totals = [0.0, 0.0]
    for i, block in enumerate(model.transformer.h):
        for name in LAYER_NAMES:
            layer = block.get_submodule(name)
            w, x = layer.weight.data, inputs[f"h{i}.{name}"]
            exact = x @ w + layer.bias
            channel_max = w.abs().amax(dim=0)
            ratio = (channel_max.max() / channel_max.median()).item()
            row = []
            for q, scale in (quantize_per_tensor(w), quantize_per_channel(w, dim=0)):
                back = dequantize(q, scale)
                row += [relative_error(back, w), relative_error(x @ back + layer.bias, exact)]
            totals[0] += row[1]
            totals[1] += row[3]
            print(f"{f'h{i}.{name}':>16} {str(tuple(w.shape)):>11} {ratio:>13.1f} "
                  f"{row[0]:>10.2f} {row[2]:>10.2f} {row[1]:>10.2f} {row[3]:>10.2f}")
    n = len(model.transformer.h) * len(LAYER_NAMES)
    print(f"\naverage output error: per-tensor {totals[0] / n:.2f}%, per-channel {totals[1] / n:.2f}%")
    print("outlier ratio = biggest channel's max / the median channel's max\n")


def quantize_model(model, quantizer):
    """Replace every linear layer's weights with their int8 round trip (weight-only quantization)."""
    for block in model.transformer.h:
        for name in LAYER_NAMES:
            layer = block.get_submodule(name)
            layer.weight.data = dequantize(*quantizer(layer.weight.data))
    return model


def whole_model(tokenizer):
    """Quantize all linear layers and compare the model's predictions with the original."""
    ids = tokenizer(TEXT, return_tensors="pt").input_ids
    original = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
    with torch.no_grad():
        exact = original(ids).logits[0]
    prompt = tokenizer("The history of computing", return_tensors="pt")
    print(f"Whole model, all 24 linear layers in int8 ({ids.shape[1]} tokens of text)\n")
    print(f"{'weights':>12} {'logit err':>9} {'same top-1':>10}   greedy continuation")
    print("-" * 90)
    variants = (("fp32", None), ("per-tensor", quantize_per_tensor),
                ("per-channel", lambda w: quantize_per_channel(w, dim=0)))
    for name, quantizer in variants:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
        if quantizer:
            quantize_model(model, quantizer)
        with torch.no_grad():
            logits = model(ids).logits[0]
            out = model.generate(**prompt, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        same = (logits.argmax(-1) == exact.argmax(-1)).float().mean().item()
        text = tokenizer.decode(out[0, prompt.input_ids.shape[1]:])
        print(f"{name:>12} {relative_error(logits, exact):>8.2f}% {same:>10.1%}   {text!r}")


def memory():
    """Bytes for distilgpt2's linear weights in each format."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    weights = [block.get_submodule(n).weight for block in model.transformer.h for n in LAYER_NAMES]
    params = sum(w.numel() for w in weights)
    channels = sum(w.shape[1] for w in weights)
    print(f"\nLinear weights: {params / 1e6:.1f}M parameters")
    print(f"  fp32: {params * 4 / 2**20:6.1f} MB")
    print(f"  fp16: {params * 2 / 2**20:6.1f} MB")
    print(f"  int8 per-tensor:  {(params + 4 * len(weights)) / 2**20:6.1f} MB (one 4-byte scale per layer)")
    print(f"  int8 per-channel: {(params + 4 * channels) / 2**20:6.1f} MB ({channels} scales, "
          f"{4 * channels / 1024:.0f} KB extra)")


def main():
    """Toy examples, a per-layer study, the whole model, and memory."""
    torch.manual_seed(0)
    toy_example()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
    layer_study(model, layer_inputs(model, tokenizer))
    whole_model(tokenizer)
    memory()


if __name__ == "__main__":
    main()
