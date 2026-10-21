import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LAYER_NAMES = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
GROUP_SIZES = [128, 64, 32]
SPEED_SIZES = [4096, 8192]
SPEED_BATCHES = [1, 16]
SPEED_GROUP = 128
SPEED_REPEATS = 200
TEXT = (
    "The history of computing is full of surprising turns. Early machines filled whole rooms and "
    "needed teams of engineers to keep them running, yet they could do less than a cheap calculator "
    "does today. Over the following decades, transistors replaced vacuum tubes, chips replaced "
    "circuit boards, and the price of a single calculation fell by a factor of billions.\n\n"
    "Cities grew in a similar way. A small market town on a river would attract traders, then "
    "workshops, then schools and hospitals. Roads and railways connected it to other towns, and "
    "within a few generations the town had become a city of a million people, with its own "
    "newspapers, sports teams and traffic problems.\n\n"
    "Cooking is another field where small changes add up. A pinch of salt, a few minutes more in "
    "the oven, or a different kind of flour can turn an ordinary loaf of bread into something "
    "people remember. Good cooks taste as they go, adjust the recipe, and write down what worked "
    "so they can do it again the next day.\n\n"
    "Scientists study the weather by collecting measurements from thousands of stations, balloons "
    "and satellites. They feed these numbers into computer models that divide the atmosphere into "
    "a grid of boxes and calculate how heat, water and wind move from one box to the next. The "
    "forecasts are not perfect, but they have improved steadily for fifty years."
)


def quantize_int8(w):
    """Symmetric int8 with one scale per output row (day 28's per-channel). w is (out, in)."""
    scale = w.abs().amax(dim=1, keepdim=True) / 127
    return torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8), scale


def quantize_int4(w, group_size):
    """Asymmetric 4-bit: each group of `group_size` inputs gets its own scale and minimum, values 0..15."""
    out_features, in_features = w.shape
    groups = w.reshape(out_features, in_features // group_size, group_size)
    low, high = groups.amin(dim=2, keepdim=True), groups.amax(dim=2, keepdim=True)
    scale = (high - low).clamp(min=1e-8) / 15
    q = torch.clamp(torch.round((groups - low) / scale), 0, 15).to(torch.uint8)
    return q.reshape(out_features, in_features), scale.squeeze(2), low.squeeze(2)


def dequantize_int4(q, scale, low, group_size):
    """Turn 4-bit values back into floats: value * scale + minimum, group by group."""
    out_features, in_features = q.shape
    groups = q.reshape(out_features, in_features // group_size, group_size).float()
    return (groups * scale[..., None] + low[..., None]).reshape(out_features, in_features)


def pack_int4(q):
    """Put two 4-bit values in one byte: the first in the high half, the second in the low half."""
    return (q[:, ::2] << 4) | q[:, 1::2]


def unpack_int4(packed):
    """Split each byte back into its two 4-bit values."""
    return torch.stack([packed >> 4, packed & 15], dim=2).reshape(packed.shape[0], -1)


def toy_example():
    """Eight weights, 4-bit, one group vs two groups of four."""
    w = torch.tensor([[0.02, -0.05, 0.01, 0.04, 1.60, -0.03, 0.06, -0.01]])
    print("Eight weights:", [round(v, 3) for v in w[0].tolist()])
    for group_size in (8, 4):
        q, scale, low = quantize_int4(w, group_size)
        back = dequantize_int4(q, scale, low, group_size)
        print(f"  groups of {group_size}: 4-bit values {q[0].tolist()}")
        print(f"               back = {[round(v, 3) for v in back[0].tolist()]}")
    packed = pack_int4(q)
    print(f"  packed: {q.shape[1]} values in {packed.shape[1]} bytes {packed[0].tolist()}, "
          f"unpacks correctly: {torch.equal(unpack_int4(packed), q)}\n")


def linear_weights(model):
    """distilgpt2's 24 linear layers. GPT-2 stores weights as (in, out), so each is transposed to (out, in)."""
    return [block.get_submodule(name) for block in model.transformer.h for name in LAYER_NAMES]


def apply_quantization(model, scheme):
    """Replace every linear layer's weights with their quantized round trip; embeddings stay in fp32."""
    for layer in linear_weights(model):
        w = layer.weight.data.T
        if scheme == "fp16":
            back = w.half().float()
        elif scheme == "int8":
            q, scale = quantize_int8(w)
            back = q.float() * scale
        elif scheme.startswith("int4"):
            group_size = w.shape[1] if scheme == "int4 per-channel" else int(scheme.split("g")[-1])
            back = dequantize_int4(*quantize_int4(w, group_size), group_size)
        else:
            continue
        layer.weight.data = back.T.contiguous()
    return model


def bits_per_weight(model, scheme):
    """Average storage per weight, counting the scales (and minimums) that have to be stored too."""
    layers = linear_weights(model)
    params = sum(l.weight.numel() for l in layers)
    if scheme == "fp32":
        return 32.0
    if scheme == "fp16":
        return 16.0
    if scheme == "int8":
        return (8 * params + 16 * sum(l.weight.shape[1] for l in layers)) / params
    groups = 0
    for l in layers:
        in_features, out_features = l.weight.shape
        group_size = in_features if scheme == "int4 per-channel" else int(scheme.split("g")[-1])
        groups += out_features * in_features // group_size
    return (4 * params + 2 * 16 * groups) / params


def perplexity(model, ids):
    """exp(average surprise) over every token of the text: lower means the model predicts it better."""
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return math.exp(loss.item())


def quality(tokenizer):
    """Perplexity of distilgpt2 with its linear layers in each format."""
    ids = tokenizer(TEXT, return_tensors="pt").input_ids
    schemes = ["fp32", "fp16", "int8", "int4 per-channel"] + [f"int4 g{g}" for g in GROUP_SIZES]
    print(f"Quality: perplexity on {ids.shape[1]} tokens of text, all 24 linear layers quantized\n")
    print(f"{'weights':>17} {'bits/weight':>11} {'linear MB':>9} {'perplexity':>10} {'change':>8}")
    print("-" * 59)
    base = None
    for scheme in schemes:
        model = apply_quantization(AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval(), scheme)
        ppl = perplexity(model, ids)
        base = base or ppl
        bits = bits_per_weight(model, scheme)
        params = sum(l.weight.numel() for l in linear_weights(model))
        print(f"{scheme:>17} {bits:>11.2f} {params * bits / 8 / 2**20:>9.1f} {ppl:>10.2f} {100 * (ppl / base - 1):>+7.1f}%")
    print()


def time_ms(fn):
    """Average milliseconds per call on DEVICE, after a warm-up."""
    for _ in range(10):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(SPEED_REPEATS):
        fn()
    sync()
    return (time.perf_counter() - start) / SPEED_REPEATS * 1000


def sync():
    """Wait for the GPU to finish its queued work, so the timing is honest."""
    if DEVICE == "mps":
        torch.mps.synchronize()


def int4_kernel_inputs(w, group_size):
    """Quantize w (out, in) and lay it out the way PyTorch's int4 kernel wants it."""
    q, scale, low = quantize_int4(w.float(), group_size)
    zero = low + 8 * scale
    scales_and_zeros = torch.stack([scale.T, zero.T], dim=2).to(w.dtype).contiguous()
    if DEVICE == "cpu":
        packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(q.to(torch.int32), 2)
    else:
        packed = torch.ops.aten._convert_weight_to_int4pack(pack_int4(q), 2)
    return packed, scales_and_zeros, dequantize_int4(q, scale, low, group_size).to(w.dtype)


def int4_matmul(x, packed, scales_and_zeros, group_size):
    """x @ W.T with W stored in 4 bits, using PyTorch's built-in int4 kernel."""
    if DEVICE == "cpu":
        return torch.ops.aten._weight_int4pack_mm_for_cpu(x, packed, group_size, scales_and_zeros)
    return torch.ops.aten._weight_int4pack_mm(x, packed, group_size, scales_and_zeros)


def speed():
    """Time one big linear layer in fp16, int8 and int4 with PyTorch's built-in quantized kernels."""
    dtype = torch.float16 if DEVICE == "mps" else torch.bfloat16
    print(f"Speed: one linear layer on {DEVICE}, x @ W.T, {SPEED_REPEATS} runs each "
          f"({'fp16' if DEVICE == 'mps' else 'bf16'} activations, int4 groups of {SPEED_GROUP})\n")
    print(f"{'size':>11} {'batch':>5} {'weights':>7} {'MB':>6} {'ms':>7} {'GB/s':>6} {'speedup':>7} {'max err':>8}")
    print("-" * 64)
    for size in SPEED_SIZES:
        torch.manual_seed(0)
        w = (torch.randn(size, size) * 0.02).to(DEVICE, dtype)
        q8, scale8 = quantize_int8(w.float())
        q8, scale8 = q8.contiguous(), scale8.squeeze(1).to(dtype)
        packed, scales_and_zeros, w4 = int4_kernel_inputs(w, SPEED_GROUP)
        for batch in SPEED_BATCHES:
            x = torch.randn(batch, size, device=DEVICE, dtype=dtype)
            runs = (
                ("fp16", w.numel() * 2, lambda: x @ w.T, x @ w.T),
                ("int8", w.numel() + size * 2, lambda: torch._weight_int8pack_mm(x, q8, scale8),
                 x @ (q8.to(dtype) * scale8[:, None]).T),
                ("int4", w.numel() // 2 + scales_and_zeros.numel() * 2,
                 lambda: int4_matmul(x, packed, scales_and_zeros, SPEED_GROUP), x @ w4.T),
            )
            base = None
            for name, nbytes, fn, reference in runs:
                err = (fn().float() - reference.float()).abs().max().item()
                ms = time_ms(fn)
                base = base or ms
                print(f"{f'{size}x{size}':>11} {batch:>5} {name:>7} {nbytes / 2**20:>6.0f} {ms:>7.3f} "
                      f"{nbytes / ms / 1e6:>6.0f} {base / ms:>6.2f}x {err:>8.4f}")
        del w, q8, packed, w4
    print("\nmax err = difference from doing the same math by hand (dequantize, then multiply): rounding only")


def main():
    """Toy example, quality (perplexity) and speed."""
    torch.manual_seed(0)
    toy_example()
    quality(AutoTokenizer.from_pretrained(MODEL_NAME))
    speed()


if __name__ == "__main__":
    main()
