import time
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_NAME = "distilgpt2"


def load(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def token_timer(prompt, tokenizer, model, max_new_tokens=64):
    """Stream tokens for `prompt`; return (text, ttft_seconds, tps)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )

    start = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    first = None
    n_chunks = 0
    text = ""
    for chunk in streamer:
        now = time.perf_counter()
        if first is None:
            first = now
        n_chunks += 1
        text += chunk
        print(chunk, end="", flush=True)
    end = time.perf_counter()
    thread.join()
    print()

    ttft = first - start
    # TPS over the decode phase (after first token); streamer chunks ~= tokens
    decode_time = end - first
    tps = (n_chunks - 1) / decode_time if n_chunks > 1 and decode_time > 0 else float("nan")
    return text, ttft, tps


if __name__ == "__main__":
    torch.manual_seed(0)
    tokenizer, model = load()
    # warmup so first-call overhead doesn't pollute TTFT
    token_timer("Hello", tokenizer, model, max_new_tokens=4)
    print("-" * 40)
    _, ttft, tps = token_timer("The future of AI inference is", tokenizer, model)
    print("-" * 40)
    print(f"TTFT: {ttft * 1000:.1f} ms")
    print(f"TPS:  {tps:.1f} tokens/s")
