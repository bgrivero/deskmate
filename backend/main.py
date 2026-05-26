from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Annotated
from pathlib import Path as FilePath

from socratic import conversation, store

class ChatRequest(BaseModel):
    message: str

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    response = conversation.invoke(
            {"input": request.message},
            config={"configurable": {"session_id": "default"}}
    )
    return {"message": response.content}

@app.get("/history/{sessionID}")
async def fetch_history(sessionID: str):
    try:
        chat_history = store[sessionID]
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Session ID not found")
    messages = [message.content for message in chat_history.messages]
    return messages


