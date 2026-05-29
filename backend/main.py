from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Annotated
from pathlib import Path as FilePath
from contextlib import asynccontextmanager
import datetime
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

from socratic import chain, get_history, get_embeddings, store
from db import get_db, create_tables, connection_pool, find_similar_chunks

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    use_rag: bool = False

class UploadRequest(BaseModel):
    filepath: str
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
    query_message = request.message
    session_id = request.session_id
    request_embedding = get_embeddings(request.message)
    history = get_history(session_id, conn)

    if request.use_rag:
        similar_chunks = find_similar_chunks(conn, request_embedding, session_id)
        context = "\n\n".join(similar_chunks)
        query_message = f"Context from notes:\n{context}\n\nQuestion: {input_message}"

    response = chain.invoke({
        "input": query_message,
        "history": history
    })
    
    output_message = response.content
    store[session_id].append(HumanMessage(content=input_message))
    store[session_id].append(AIMessage(content=output_message))
    
    response_embedding = get_embeddings(output_message)
    curs.execute("""
    INSERT INTO messages (role, content, embedding, created_at, session_id)
    VALUES 
    (%s, %s, %s, %s, %s),
    (%s, %s, %s, %s, %s)
    """, (
        'user', input_message, str(request_embedding), datetime.datetime.now(), session_id,
        'assistant', output_message, str(response_embedding), datetime.datetime.now(), session_id
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

@app.post("/upload/{session_id}")
async def upload_document(upload_request: UploadRequest,
                          conn=Depends(get_db)):
    document = PyPDFLoader(upload_request.filepath)
    data = document.load()
    curs = conn.cursor()

    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 50
    SEPARATORS = ["\n\n", "\n", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS
    )

    document_chunked = text_splitter.split_documents(data)

    title = document_chunked[0].metadata["title"]
    for chunk in document_chunked:
        chunk_text = chunk.page_content
        embedding = str(get_embeddings(chunk_text))
        curs.execute("""
        INSERT INTO document_chunks (filename, chunk_text, embedding, session_id)
        VALUES (%s, %s, %s, %s)
        """, (title, chunk_text, embedding, upload_request.session_id))
    
    return {"uploaded_filename": title, "session_id":upload_request.session_id}


