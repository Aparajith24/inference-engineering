import asyncio
import json
import time
import uuid
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

HOST, PORT = "127.0.0.1", 8080
MODEL_NAME = "distilgpt2"
DEFAULT_MAX_TOKENS = 16
TEMPERATURE, TOP_K = 0.7, 10
WELCOME_TEXT = "hi! welcome"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
CONTEXT_LENGTH = model.config.n_positions
app = FastAPI()


class CompletionRequest(BaseModel):
    """What the client sends: a non-empty prompt, optionally max_tokens, and whether to stream (yes by default)."""

    prompt: str = Field(min_length=1)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1)
    stream: bool = True


class AsyncTokenStreamer(BaseStreamer):
    """Receives each new token id from generate() on its thread and passes it to the event loop through a queue."""

    def __init__(self, loop):
        self.loop, self.queue, self.seen_prompt = loop, asyncio.Queue(), False

    def put(self, value):
        """Called by generate() once with the prompt, then once per new token; only the new tokens go on the queue."""
        if not self.seen_prompt:
            self.seen_prompt = True
            return
        for token_id in value.view(-1).tolist():
            self.loop.call_soon_threadsafe(self.queue.put_nowait, token_id)

    def end(self):
        """Called when generation is over; None on the queue tells the reader to stop."""
        self.loop.call_soon_threadsafe(self.queue.put_nowait, None)


async def model_tokens(prompt_ids, max_tokens):
    """Run generate() on a background thread and yield its token ids as they arrive, leaving the event loop free."""
    streamer = AsyncTokenStreamer(asyncio.get_running_loop())
    input_ids = torch.tensor([prompt_ids])
    kwargs = dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=max_tokens,
                  do_sample=True, temperature=TEMPERATURE, top_k=TOP_K,
                  pad_token_id=tokenizer.eos_token_id, streamer=streamer)

    def run():
        """Generate on this thread; if it crashes, still end the stream so the reader is not stuck waiting."""
        try:
            model.generate(**kwargs)
        except Exception:
            streamer.end()
            raise

    Thread(target=run, daemon=True).start()
    while (token_id := await streamer.queue.get()) is not None:
        yield token_id


async def welcome_tokens(max_tokens):
    """Yield the welcome message's token ids and then the end token, as if the model had said it; cut at max_tokens."""
    for token_id in (tokenizer.encode(WELCOME_TEXT) + [tokenizer.eos_token_id])[:max_tokens]:
        yield token_id


async def text_pieces(token_source):
    """For each token, yield the new text it adds and all ids so far; decoding the whole answer keeps split characters whole."""
    ids, sent = [], ""
    async for token_id in token_source:
        ids.append(token_id)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if text.endswith("�"):
            yield "", ids
            continue
        yield text[len(sent):], ids
        sent = text


def finish_reason(ids):
    """'stop' if the answer ended with the end token, otherwise 'length' (it hit max_tokens)."""
    return "stop" if ids and ids[-1] == tokenizer.eos_token_id else "length"


def completion_object(request_id, created, text, finish, prompt_tokens, completion_tokens):
    """One OpenAI text_completion object; used for a whole answer and for each streamed piece."""
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


async def sse_events(prompt_ids, token_source):
    """Send one SSE message per token, then one with finish_reason and the token usage, then [DONE]."""
    request_id, created, ids = f"cmpl-{uuid.uuid4().hex}", int(time.time()), []
    async for piece, ids in text_pieces(token_source):
        chunk = completion_object(request_id, created, piece, None, len(prompt_ids), len(ids))
        chunk["usage"] = None
        yield f"data: {json.dumps(chunk)}\n\n"
    chunk = completion_object(request_id, created, "", finish_reason(ids), len(prompt_ids), len(ids))
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def full_answer(prompt_ids, token_source):
    """Collect every token, then return the whole answer as one object."""
    text, ids = "", []
    async for piece, ids in text_pieces(token_source):
        text += piece
    return completion_object(f"cmpl-{uuid.uuid4().hex}", int(time.time()), text, finish_reason(ids),
                             len(prompt_ids), len(ids))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Check the request, pick the model or the welcome message, then stream the answer or send it whole."""
    prompt_ids = tokenizer.encode(request.prompt)
    if len(prompt_ids) + request.max_tokens > CONTEXT_LENGTH:
        raise HTTPException(status_code=400, detail=(
            f"prompt ({len(prompt_ids)} tokens) + max_tokens ({request.max_tokens}) is over "
            f"{MODEL_NAME}'s context length of {CONTEXT_LENGTH} tokens"))
    if request.prompt.strip():
        source = model_tokens(prompt_ids, request.max_tokens)
    else:
        source = welcome_tokens(request.max_tokens)
    if request.stream:
        return StreamingResponse(sse_events(prompt_ids, source), media_type="text/event-stream")
    return await full_answer(prompt_ids, source)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
