import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

HOST, PORT = "127.0.0.1", 8080
MODEL_NAME = "distilgpt2"
DEFAULT_MAX_TOKENS = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
CONTEXT_LENGTH = tokenizer.model_max_length
app = FastAPI()


class CompletionRequest(BaseModel):
    """What the client sends: a non-empty prompt, and optionally how many tokens it wants back at most."""

    prompt: str = Field(min_length=1)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1)


def placeholder_answer(prompt):
    """Stand-in for the model until we add one: the token ids of 'you said: <prompt>'."""
    return tokenizer.encode(f"you said: {prompt}")


@app.post("/v1/completions")
def completions(request: CompletionRequest):
    """Answer a prompt in OpenAI's text_completion shape, cut at max_tokens."""
    prompt_ids = tokenizer.encode(request.prompt)
    if len(prompt_ids) + request.max_tokens > CONTEXT_LENGTH:
        raise HTTPException(status_code=400, detail=(
            f"prompt ({len(prompt_ids)} tokens) + max_tokens ({request.max_tokens}) is over "
            f"{MODEL_NAME}'s context length of {CONTEXT_LENGTH} tokens"))
    answer_ids = placeholder_answer(request.prompt)
    finish_reason = "stop" if len(answer_ids) <= request.max_tokens else "length"
    answer_ids = answer_ids[:request.max_tokens]
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "text": tokenizer.decode(answer_ids), "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(answer_ids),
                  "total_tokens": len(prompt_ids) + len(answer_ids)},
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
