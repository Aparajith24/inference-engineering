import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """What the client sends: a non-empty prompt, and optionally how many tokens it wants back at most."""

    prompt: str = Field(min_length=1)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1)


def generate(prompt_ids, max_tokens):
    """Run the model on the prompt and return only the new token ids, sampling from the TOP_K likeliest tokens."""
    input_ids = torch.tensor([prompt_ids])
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=max_tokens,
                                do_sample=True, temperature=TEMPERATURE, top_k=TOP_K,
                                pad_token_id=tokenizer.eos_token_id)
    return output[0, len(prompt_ids):].tolist()


def completion_response(prompt_ids, answer_ids, finish_reason):
    """Wrap an answer in OpenAI's text_completion shape."""
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "text": tokenizer.decode(answer_ids, skip_special_tokens=True),
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(answer_ids),
                  "total_tokens": len(prompt_ids) + len(answer_ids)},
    }


@app.post("/v1/completions")
def completions(request: CompletionRequest):
    """Continue a prompt with the model; a prompt of only spaces gets a welcome message instead."""
    prompt_ids = tokenizer.encode(request.prompt)
    if len(prompt_ids) + request.max_tokens > CONTEXT_LENGTH:
        raise HTTPException(status_code=400, detail=(
            f"prompt ({len(prompt_ids)} tokens) + max_tokens ({request.max_tokens}) is over "
            f"{MODEL_NAME}'s context length of {CONTEXT_LENGTH} tokens"))
    if not request.prompt.strip():
        return completion_response(prompt_ids, tokenizer.encode(WELCOME_TEXT), "stop")
    answer_ids = generate(prompt_ids, request.max_tokens)
    finish_reason = "stop" if answer_ids[-1] == tokenizer.eos_token_id else "length"
    return completion_response(prompt_ids, answer_ids, finish_reason)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
