import argparse
import json
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"
FAMILY = "qwen2.5:1.5b-instruct"
QUANTS = ["q4_K_M", "q8_0", "fp16"]
RUNS = 3
MAX_TOKENS = 128
MAC_BANDWIDTH_GBS = 120

PROMPT = (
    "You are helping a student learn about inference. Here are their notes. "
    "A GPU has many streaming multiprocessors, each with Tensor Cores for matrix math, CUDA cores for single "
    "numbers, and special function units for things like exponentials. VRAM is large but slow compared to the "
    "on-chip SRAM caches. During prefill, the whole prompt is processed at once, so the GPU does a lot of math "
    "for every weight it reads, and the limit is compute. During decode, one token is produced at a time, and "
    "every weight must be read from memory for each token, so the limit is memory bandwidth. Batching lets many "
    "users share one read of the weights. The KV cache stores keys and values so earlier tokens are not "
    "recomputed. Quantization stores weights in fewer bits, so there are fewer bytes to read per token. "
    "Using these notes, write a detailed explanation of why quantization makes decode faster."
)
QUALITY_QUESTION = "A train leaves at 3:45 pm and the trip takes 2 hours 30 minutes. When does it arrive? Answer briefly."


def call(path, payload=None):
    """Send a request to the local Ollama server and return its JSON reply."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(OLLAMA + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def pull(tag):
    """Download a model from the Ollama library, printing progress."""
    req = urllib.request.Request(OLLAMA + "/api/pull", data=json.dumps({"model": tag}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            msg = json.loads(line)
            if msg.get("total"):
                print(f"\r  downloading {tag}: {msg.get('completed', 0) / msg['total']:.0%}   ", end="", flush=True)
    print()


def installed_models():
    """Return a dict of installed model tag -> size on disk in GB."""
    return {m["name"]: m["size"] / 1e9 for m in call("/api/tags")["models"]}


def generate(tag, prompt, max_tokens=MAX_TOKENS):
    """Run one generation with greedy decoding and return Ollama's reply, including its timings."""
    options = {"temperature": 0, "seed": 0, "num_predict": max_tokens}
    return call("/api/generate", {"model": tag, "prompt": prompt, "stream": False, "options": options})


def bench(tag, runs):
    """Measure prefill tok/s, time to first token and decode tok/s, averaged over several runs."""
    generate(tag, "Hi", 8)
    prefill, ttft, decode = [], [], []
    for i in range(runs):
        r = generate(tag, f"[note set {i}] " + PROMPT)
        prefill.append(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9))
        ttft.append(r["prompt_eval_duration"] / 1e6)
        decode.append(r["eval_count"] / (r["eval_duration"] / 1e9))
    return sum(prefill) / runs, sum(ttft) / runs, sum(decode) / runs, r["prompt_eval_count"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark quantization levels of one model with Ollama.")
    parser.add_argument("--family", default=FAMILY)
    parser.add_argument("--quants", nargs="+", default=QUANTS)
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    try:
        call("/api/version")
    except urllib.error.URLError:
        raise SystemExit("Ollama is not running. Start it with: ollama serve")

    tags = [f"{args.family}-{q}" for q in args.quants]
    for tag in tags:
        if tag not in installed_models():
            pull(tag)
    sizes = installed_models()

    results = []
    for tag in tags:
        print(f"benchmarking {tag} ...")
        results.append((tag, sizes[tag], *bench(tag, args.runs)))

    prompt_tokens = results[0][5]
    print(f"\n{args.family}: {prompt_tokens}-token prompt, up to {MAX_TOKENS} new tokens, {args.runs} runs each\n")
    print(f"{'quant':<9}{'size GB':>9}{'prefill tok/s':>15}{'TTFT ms':>9}{'decode tok/s':>14}"
          f"{'ideal tok/s':>13}{'of ideal':>10}")
    for tag, size, prefill, ttft, decode, _ in results:
        ideal = MAC_BANDWIDTH_GBS / size
        print(f"{tag.split('-')[-1]:<9}{size:>9.2f}{prefill:>15.0f}{ttft:>9.0f}{decode:>14.1f}{ideal:>13.0f}"
              f"{decode / ideal:>10.0%}")
    print(f"\n'ideal' is day 17's rule: {MAC_BANDWIDTH_GBS} GB/s of memory bandwidth / model size.")

    print(f"\nSame question to each quantization level (greedy):\n  {QUALITY_QUESTION}\n")
    for tag in tags:
        answer = generate(tag, QUALITY_QUESTION, 60)["response"].strip().replace("\n", " ")
        print(f"  {tag.split('-')[-1]:<8} {answer[:160]}")
