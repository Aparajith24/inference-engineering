# Day 23: Inference engines — vLLM with Qwen2.5-0.5B

Book chapter: **4.3 Inference engines**

Task: run a real inference engine, vLLM, two ways: offline batch generation
from Python, and as an OpenAI-compatible server. Runs on Colab.

## What we built

`vllm_quickstart.py` has two modes:

- **`offline`**: loads `Qwen/Qwen2.5-0.5B-Instruct` with vLLM's `LLM` class.
  It answers 8 prompts in one call, then times 1, 8 and 64 prompts to show
  total tokens/s rising with batch size. It ends with one chat-format answer
  (`llm.chat`), which applies the model's chat template for us.
- **`client`**: talks to a running `vllm serve` server over HTTP.
  1. `GET /v1/models`
  2. a normal `/v1/completions` answer
  3. a `/v1/chat/completions` answer
  4. one **streamed** answer, timing TTFT and tokens/s from the client's side

## How to run it on Colab

vLLM needs an NVIDIA GPU, so this cannot run on a Mac.

1. colab.research.google.com → **New notebook** → **Runtime → Change runtime
   type → T4 GPU**.
2. `!pip install vllm` (a few minutes; restart the session if Colab asks).
3. Upload `vllm_quickstart.py` with the folder icon on the left.
4. Offline: `!python vllm_quickstart.py offline`
5. Server: start it in the background, wait until the log says
   `Application startup complete`, then run the client:

```
!nohup vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --gpu-memory-utilization 0.8 --max-model-len 2048 > vllm.log 2>&1 &
!tail -5 vllm.log
!python vllm_quickstart.py client
```

Run the offline mode first. It takes 80% of the GPU's memory and gives it back
when the script ends; the server then needs that memory for itself.

`--dtype half`: the T4 has no bf16 support, so we use fp16.

## What we learned

### What an inference engine adds

Day 22's server was an engine of our own: a scheduler, a KV cache per user, and
an HTTP layer. vLLM does the same jobs, at production quality:

| Job | Our day 22 server | vLLM |
|---|---|---|
| Batching | Continuous, 30 seats | Continuous, seats limited by KV memory |
| KV cache | One tensor per user, re-padded every step | **PagedAttention**: fixed blocks, never copied (day 25) |
| Memory | Whatever PyTorch allocates | Takes `gpu_memory_utilization` of the GPU up front, splits it into KV blocks |
| Kernels | PyTorch eager | Fused CUDA kernels (FlashAttention, etc.) and CUDA graphs |
| API | `/v1/completions` | `/v1/completions`, `/v1/chat/completions`, `/v1/models`, and more |
| Extras | none | Prefix caching, quantization, speculative decoding, tensor parallelism |

### The memory grab at startup

vLLM loads the weights, runs a test batch to measure how much memory the
activations need, and turns everything else (up to `gpu_memory_utilization`)
into KV cache blocks. The startup log prints how many blocks and tokens fit.
That number limits how many users can be seated at once.

### Offline vs server

- **Offline (`LLM.generate`)**: you hand over all prompts at once, and vLLM
  schedules them as tightly as it can. Best for batch jobs such as evaluations
  or labelling a dataset.
- **Server (`vllm serve`)**: requests arrive whenever users send them. The
  same scheduler runs, but the batch is only as full as the traffic.

### `SamplingParams`

The same knobs as day 11: `temperature`, `top_p`, `top_k`, `max_tokens`, plus
stop strings. They are set per request, unlike day 22, which fixed them on
the server.

### Completions vs chat completions

`/v1/completions` continues raw text. `/v1/chat/completions` takes a list of
messages, and the server wraps them in the model's chat template (special
tokens that mark who is speaking) before generating. Instruct models like
Qwen2.5-Instruct answer far better through chat.

## Results (Colab T4)

```
Paste your output here
```

## Connection to earlier days

- Day 5: a 0.5B model in fp16 is about 1 GB of weights, a small slice of the
  T4's 16 GB. Most of the rest becomes KV cache.
- Day 11: the same sampling settings, now as request fields.
- Day 14: PagedAttention, the idea behind vLLM's name, built for real.
- Day 22: our own engine. vLLM answers the same `curl` commands.

## Caveats

- Colab's T4 is an older GPU (no bf16, slow FP16 tensor cores next to an A100
  or H100), so absolute numbers are low.
- vLLM's first start compiles and captures CUDA graphs, so it is slow. Later
  requests are not.
- The `client` mode uses `requests`, which Colab already has.

## Things to try

1. Lower `gpu_memory_utilization` and watch the KV block count in the log fall.
2. Pass `--enable-prefix-caching` and send the same long system prompt twice:
   the second TTFT should drop.
3. Point day 22's `curl` commands at port 8000.
