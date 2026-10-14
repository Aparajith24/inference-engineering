import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

HOST, PORT = "127.0.0.1", 8080
MODEL_NAME = "placeholder"

app = FastAPI()


class ChatRequest(BaseModel):
    """What the client must send: a prompt."""

    prompt: str


@app.post("/v1/chat/{id}")
def chat(id: str, request: ChatRequest):
    """Answer a prompt in OpenAI's response shape; the text is a placeholder until we add a model."""
    text = f"you said: {request.prompt}"
    prompt_words, answer_words = len(request.prompt.split()), len(text.split())
    return {
        "id": id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_words, "completion_tokens": answer_words,
                  "total_tokens": prompt_words + answer_words},
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
