import argparse
import math
import re

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Bytes needed to store one parameter at each precision
BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}

# Rule of thumb: extra memory for KV cache, activations and runtime buffers
OVERHEAD = 1.2

# (name, memory in GB)
GPUS = [
    ("T4", 16),
    ("L4", 24),
    ("A10G", 24),
    ("RTX 4090", 24),
    ("L40S", 48),
    ("A100 40GB", 40),
    ("A100 80GB", 80),
    ("H100", 80),
    ("H200", 141),
    ("B200", 192),
]

# Sizes shown when no model is given
EXAMPLES = ["125M", "1B", "7B", "13B", "70B", "405B"]


def parse_size(text):
    """'7B' -> 7e9, '125M' -> 125e6, '1.5B' -> 1.5e9, '350000' -> 350000."""
    m = re.fullmatch(r"([\d.]+)\s*([kKmMbBtT]?)", text.strip())
    if not m:
        return None
    scale = {"": 1, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
    return float(m.group(1)) * scale[m.group(2).lower()]


def count_params(model_name):
    """Count parameters of a Hugging Face model without downloading its weights."""
    config = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):  # tensors with shapes but no memory behind them
        model = AutoModelForCausalLM.from_config(config)
    return sum(p.numel() for p in model.parameters())


def memory_gb(n_params, precision):
    """Weights only: parameters x bytes per parameter."""
    return n_params * BYTES_PER_PARAM[precision] / 1e9


def print_report(label, n_params):
    print(f"\n{label}: {n_params:,.0f} parameters ({n_params / 1e9:.2f}B)\n")

    precisions = list(BYTES_PER_PARAM)
    weights = {p: memory_gb(n_params, p) for p in precisions}
    needed = {p: weights[p] * OVERHEAD for p in precisions}

    # Memory needed at each precision
    print(f"{'':<18}" + "".join(f"{p:>10}" for p in precisions))
    print(f"{'weights (GB)':<18}" + "".join(f"{weights[p]:>10.2f}" for p in precisions))
    print(f"{'+20% overhead (GB)':<18}" + "".join(f"{needed[p]:>10.2f}" for p in precisions))

    # Does it fit on a single GPU?
    print(f"\nFits on ONE GPU? (needs weights + {round((OVERHEAD - 1) * 100)}% overhead)\n")
    print(f"{'GPU':<18}" + "".join(f"{p:>10}" for p in precisions))
    for name, vram in GPUS:
        cells = "".join(f"{'yes' if needed[p] <= vram else '-':>10}" for p in precisions)
        print(f"{name + f' ({vram}GB)':<18}{cells}")

    # For models that fit nowhere, say how many big GPUs are needed
    biggest_name, biggest = max(GPUS, key=lambda g: g[1])
    print()
    for p in precisions:
        if needed[p] > biggest:
            n = math.ceil(needed[p] / biggest)
            print(f"{p}: needs at least {n} x {biggest_name} ({biggest}GB) to hold it")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate GPU memory for a model.")
    parser.add_argument(
        "model",
        nargs="?",
        help="Hugging Face model id (e.g. distilgpt2) or a size (e.g. 7B, 125M). "
        "Leave empty for a table of common sizes.",
    )
    args = parser.parse_args()

    if args.model is None:
        for size in EXAMPLES:
            print_report(size, parse_size(size))
    else:
        n_params = parse_size(args.model)
        if n_params is None:  # not a size like "7B", so treat it as a model id
            n_params = count_params(args.model)
        print_report(args.model, n_params)
