
import json
from uuid import uuid4
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.rag_pipeline import ask_question

app = FastAPI(title="Haystack RAG API")

MODEL_ID = "haystack-rag"


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    team: Literal["hamta", "karbordi"] | None = None
    confidentiality: Literal["private", "public"] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


def make_chunk(completion_id, delta, finish_reason=None):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    user_messages = [
        message.content
        for message in request.messages
        if message.role == "user"
    ]

    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="A user message is required.",
        )

    question = user_messages[-1]

    if not isinstance(question, str) or not question.strip():
        raise HTTPException(
            status_code=400,
            detail="The user message must be text.",
        )

    try:
        result = ask_question(
            question.strip(),
            team=request.team,
            confidentiality=request.confidentiality,
        )
        answer = result["answer"]

        sources = [
            {
                "source": doc.meta.get("source", "نامشخص"),
                "content": (doc.content or "")[:500],
                "score": doc.score,
            }
            for doc in result.get("sources", [])
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Haystack pipeline failed: {exc}",
        ) from exc

    completion_id = f"chatcmpl-{uuid4().hex}"

    if request.stream:
        def generate():
            role_chunk = make_chunk(
                completion_id,
                {"role": "assistant"},
            )
            yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

            for start in range(0, len(answer), 30):
                chunk = make_chunk(
                    completion_id,
                    {"content": answer[start:start + 30]},
                )
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            final_chunk = make_chunk(
                completion_id,
                {},
                finish_reason="stop",
            )
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": 0,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "sources": sources,
    }
