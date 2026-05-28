from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Annotated
from pathlib import Path as FilePath
from contextlib import asynccontextmanager
import datetime
import uuid

from socratic import conversation, store, get_embeddings
from db import get_db, create_tables, connection_pool

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield
    connection_pool.closeall()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(request: ChatRequest, 
               conn=Depends(get_db)):
    curs = conn.cursor()
    input_message = request.message
    session_id = request.session_id

    response = conversation.invoke(
            {"input": input_message},
            config={"configurable": {"session_id": session_id}}
    )
    output_message = response.content
    request_embedding = get_embeddings(request.message)
    response_embedding = get_embeddings(output_message)
    curs.execute("""
    INSERT INTO messages (role, content, embedding, created_at, session_id)
    VALUES 
    (%s, %s, %s, %s, %s),
    (%s, %s, %s, %s, %s)
    """, (
        'user', input_message, request_embedding, datetime.datetime.now(), session_id,
        'assistant', output_message, response_embedding, datetime.datetime.now(), session_id
    ))
    curs.close()
    
    return {"message": output_message, "session_id": session_id}

@app.get("/history/{session_id}")
async def fetch_history(session_id: str,
                        conn=Depends(get_db)):
    curs = conn.cursor()
    try:
        query = '''SELECT role, content, created_at
        FROM messages 
        WHERE session_id = (%s)'''
        curs.execute(query, (session_id, ))
        chat_history = curs.fetchall()
        curs.close()
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Session ID not found")
    return chat_history

@app.post("/session")
async def new_chat(title: Annotated[str, Query(max_length=50)], 
                   conn=Depends(get_db)):
    session_id = uuid.uuid4()
    curs = conn.cursor()
    curs.execute("""
    INSERT INTO sessions (id, title, created_at)
    VALUES 
    (%s, %s, %s)
    """, (
        session_id, title, datetime.datetime.now()
    ))
    curs.close()
    return {"session_id": str(session_id)}


