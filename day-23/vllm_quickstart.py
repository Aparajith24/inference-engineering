import argparse
import json
import time

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SERVER_URL = "http://localhost:8000/v1"
MAX_TOKENS = 64
TEMPERATURE = 0.7
N_BATCH_PROMPTS = 64

PROMPTS = [
    "The cat sat on the",
    "Explain what a KV cache is in one sentence.",
    "Write a haiku about GPUs.",
    "The capital of France is",
    "List three reasons why batching makes inference cheaper.",
    "Once upon a time",
    "What is the difference between prefill and decode?",
    "Water boils at",
]


def offline():
    """Generate answers for a batch of prompts with vLLM's LLM class and time it."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_NAME, dtype="half", gpu_memory_utilization=0.8, max_model_len=2048)
    params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    for output in llm.generate(PROMPTS, params):
        print(repr(output.prompt), "->", repr(output.outputs[0].text[:80]))

    for n in (1, 8, N_BATCH_PROMPTS):
        prompts = (PROMPTS * N_BATCH_PROMPTS)[:n]
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        seconds = time.perf_counter() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"{n:>3} prompts: {tokens:>5} tokens in {seconds:5.2f} s = {tokens / seconds:7.0f} tok/s")

    messages = [{"role": "user", "content": "In one sentence, what does vLLM do?"}]
    print(llm.chat(messages, params)[0].outputs[0].text)


def stream_completion(prompt):
    """Stream one completion from the vLLM server and return TTFT, token count and total time."""
    import requests

    body = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft, tokens, text = None, 0, ""
    with requests.post(f"{SERVER_URL}/completions", json=body, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk["choices"] and chunk["choices"][0]["text"]:
                if ttft is None:
                    ttft = time.perf_counter() - start
                text += chunk["choices"][0]["text"]
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
    return text, ttft, tokens, time.perf_counter() - start


def client():
    """Call the running vLLM server: list models, one normal answer, one chat answer, one stream."""
    import requests

    print(requests.get(f"{SERVER_URL}/models").json()["data"][0]["id"])

    body = {"model": MODEL_NAME, "prompt": PROMPTS[0], "max_tokens": MAX_TOKENS}
    answer = requests.post(f"{SERVER_URL}/completions", json=body).json()
    print(repr(answer["choices"][0]["text"]), answer["choices"][0]["finish_reason"], answer["usage"])

    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": PROMPTS[1]}], "max_tokens": MAX_TOKENS}
    answer = requests.post(f"{SERVER_URL}/chat/completions", json=body).json()
    print(repr(answer["choices"][0]["message"]["content"]))

    text, ttft, tokens, seconds = stream_completion(PROMPTS[2])
    print(repr(text))
    print(f"TTFT {ttft * 1000:.0f} ms, {tokens} tokens in {seconds:.2f} s = {tokens / seconds:.0f} tok/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM quickstart: offline batch or calls to the server")
    parser.add_argument("mode", choices=["offline", "client"])
    args = parser.parse_args()
    if args.mode == "offline":
        offline()
    else:
        client()
