from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.chatbot import chat, stream_chat

app = FastAPI()


class ChatRequest(BaseModel):
    thread_id: str
    query: str


@app.get("/")
def home():
    return {
        "message": "Server Running"
    }


@app.post("/chat")
async def chat_api(payload: ChatRequest):

    answer = chat(
        payload.thread_id,
        payload.query
    )

    return {
        "answer": answer
    }


@app.post("/chat-stream")
async def chat_stream_api(payload: ChatRequest):

    async def event_generator():

        for chunk in stream_chat(
            payload.thread_id,
            payload.query
        ):
            yield f"data: {chunk}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )