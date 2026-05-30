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

from langchain import BaseChat, SocracticChat
from db import get_db, create_tables, connection_pool, find_similar_chunks
from models import ChatRequest, UploadRequest, SearchRequest

base_chat = BaseChat()
socratic_chat = SocracticChat()

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield
    connection_pool.closeall()

app = FastAPI(lifespan=lifespan)

###################################
# Base Chat Functionality
###################################
@app.post("/chat")
async def chat(request: ChatRequest, 
               conn=Depends(get_db)):
    curs = conn.cursor()
    input_message = request.message
    query_message = request.message
    session_id = request.session_id
    request_embedding = base_chat.get_embeddings(request.message)
    history = base_chat.get_history(session_id, conn)

    if request.use_rag:
        similar_chunks = find_similar_chunks(conn, request_embedding, session_id)
        context = "\n\n".join(similar_chunks)
        query_message = f"Context from notes:\n{context}\n\nQuestion: {input_message}"

    response = base_chat.chain.invoke({
        "input": query_message,
        "history": history
    })
    
    output_message = response.content
    base_chat.store[session_id].append(HumanMessage(content=input_message))
    base_chat.store[session_id].append(AIMessage(content=output_message))
    
    response_embedding = base_chat.get_embeddings(output_message)
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
        embedding = str(base_chat.get_embeddings(chunk_text))
        curs.execute("""
        INSERT INTO document_chunks (filename, chunk_text, embedding, session_id)
        VALUES (%s, %s, %s, %s)
        """, (title, chunk_text, embedding, upload_request.session_id))
    
    return {"uploaded_filename": title, "session_id":upload_request.session_id}


###################################
# Socratic Chat Functionality
###################################
@app.post("/socratic/chat")
async def socratic(request: ChatRequest, 
                        conn=Depends(get_db)):
    curs = conn.cursor()
    input_message = request.message
    query_message = request.message
    session_id = request.session_id
    request_embedding = socratic_chat.get_embeddings(request.message)
    history = socratic_chat.get_history(session_id, conn)

    if request.use_rag:
        similar_chunks = find_similar_chunks(conn, request_embedding, session_id)
        context = "\n\n".join(similar_chunks)
        query_message = f"Context from notes:\n{context}\n\nQuestion: {input_message}"

    response = socratic_chat.chain.invoke({
        "input": query_message,
        "history": history
    })
    
    output_message = response.content
    socratic_chat.store[session_id].append(HumanMessage(content=input_message))
    socratic_chat.store[session_id].append(AIMessage(content=output_message))
    
    response_embedding = socratic_chat.get_embeddings(output_message)
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


###################################
# Search Messages Functionality
###################################
@app.post("/search")
async def search_messages(request: SearchRequest,
                          conn=Depends(get_db)):
    curs = conn.cursor()
    search_query = request.search_query
    request_embedding = base_chat.get_embeddings(request.search_query)
    ## Temporarily limits to top 10 search results, will add pagination in the future
    if request.keyword_toggle:
        curs.execute("""
        SELECT sessions.id, sessions.title, sessions.created_at, messages.content, messages.created_at
        FROM messages
        JOIN sessions ON messages.session_id = sessions.id 
        WHERE messages.content LIKE %s
        ORDER BY messages.embedding <-> %s 
        LIMIT 10;
        """, (f'%%{search_query}%%', str(request_embedding)))
        results = curs.fetchall()
        curs.close()
    else:
        curs.execute("""
        SELECT sessions.id, sessions.title, sessions.created_at, messages.content, messages.created_at
        FROM messages
        JOIN sessions ON messages.session_id = sessions.id 
        ORDER BY messages.embedding <-> %s 
        LIMIT 10;
        """, (str(request_embedding), ))
        results = curs.fetchall()
        curs.close()

    return [{
        "session_id": row[0],
        "session_title": row[1],
        "session_created_at": row[2],
        "content": row[3],
        "message_created_at": row[4]
    }
    for row in results
    ]
