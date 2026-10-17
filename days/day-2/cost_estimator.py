from transformers import AutoTokenizer

MODEL_NAME = "distilgpt2"

# --- Workload assumptions ---
REQUESTS_PER_DAY = 1_000_000
AVG_OUTPUT_TOKENS = 200        # tokens generated per request
PEAK_FACTOR = 3                # peak traffic vs. daily average

# --- Hardware assumptions ---
PREFILL_TPS_PER_GPU = 20_000   # prompt tokens processed per second per GPU
DECODE_TPS_PER_GPU = 2_500     # generated tokens/s per GPU, with batching
GPU_COST_PER_HOUR = 2.50       # USD

SAMPLE_PROMPTS = [
    "Summarize the following article in three sentences.",
    "Write a Python function that reverses a linked list.",
    "What is the capital of France?",
    "Explain the difference between TCP and UDP to a beginner, "
    "including when you would choose one over the other.",
    "Translate 'Where is the nearest train station?' into Spanish.",
]


def count_tokens(prompts, tokenizer):
    """Return the token count of each prompt."""
    counts = []
    for p in prompts:
        counts.append(len(tokenizer(p)["input_ids"]))
    return counts


def estimate(requests_per_day, avg_input, avg_output):
    input_tokens = requests_per_day * avg_input
    output_tokens = requests_per_day * avg_output

    # GPU-seconds needed for each phase
    prefill_s = input_tokens / PREFILL_TPS_PER_GPU
    decode_s = output_tokens / DECODE_TPS_PER_GPU
    gpu_hours = (prefill_s + decode_s) / 3600

    cost = gpu_hours * GPU_COST_PER_HOUR
    avg_gpus = gpu_hours / 24           # GPUs busy on average over a day
    peak_gpus = avg_gpus * PEAK_FACTOR  # GPUs to provision for peak traffic

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "gpu_hours": gpu_hours,
        "cost": cost,
        "avg_gpus": avg_gpus,
        "peak_gpus": peak_gpus,
    }


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    counts = count_tokens(SAMPLE_PROMPTS, tokenizer)
    for prompt, n in zip(SAMPLE_PROMPTS, counts):
        print(f"{n:>4} tokens | {prompt}")
    avg_input = sum(counts) / len(counts)
    print(f"\nAverage prompt length: {avg_input:.1f} tokens")

    r = estimate(REQUESTS_PER_DAY, avg_input, AVG_OUTPUT_TOKENS)

    print(f"\nWorkload: {REQUESTS_PER_DAY:,} requests/day")
    print(f"  Input tokens/day:  {r['input_tokens']:,.0f}")
    print(f"  Output tokens/day: {r['output_tokens']:,.0f}")
    print(f"\nGPU-hours/day:       {r['gpu_hours']:.1f}")
    print(f"Daily cost:          ${r['cost']:.2f}  (@ ${GPU_COST_PER_HOUR}/GPU-hr)")
    print(f"Monthly cost:        ${r['cost'] * 30:,.2f}")
    print(f"GPUs (average load): {r['avg_gpus']:.1f}")
    print(f"GPUs (peak, {PEAK_FACTOR}x):    {r['peak_gpus']:.1f}")
