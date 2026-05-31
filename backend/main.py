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
import json
import re

from langchain_calls import BaseChat, SocracticChat, ScorerChat, InsightChat
from db import get_db, create_tables, connection_pool, find_similar_chunks, find_similar_notes
from models import ChatRequest, UploadRequest, SearchRequest, TopicBlockRequest, AnalyticsRequest

base_chat = BaseChat()
socratic_chat = SocracticChat()
scorer_chat = ScorerChat()
insight_chat = InsightChat()

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
        if request.use_notes:
            similar_chunks = find_similar_notes(conn, request_embedding)
        else:
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
    INSERT INTO messages (role, content, embedding, mode, created_at, session_id)
    VALUES 
    (%s, %s, %s, %s, %s, %s),
    (%s, %s, %s, %s, %s, %s)
    """, (
        'user', input_message, str(request_embedding), 'base', datetime.datetime.now(), session_id,
        'assistant', output_message, str(response_embedding), 'base', datetime.datetime.now(), session_id
    ))
    curs.close()
    
    return {"message": output_message, "session_id": session_id, "query": query_message}

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

@app.get("/session")
async def get_all_sessions(conn=Depends(get_db)):
    curs = conn.cursor()
    curs.execute("""
    SELECT id, title, created_at
    FROM sessions
    ORDER BY created_at DESC
    """)
    results = curs.fetchall()
    curs.close()
    return [{
        "session_id": row[0],
        "title": row[1],
        "created_at": row[2]
    } for row in results]

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

@app.get("/history/base/{session_id}")
async def fetch_history(session_id: str,
                        conn=Depends(get_db)):
    curs = conn.cursor()
    try:
        query = '''SELECT role, content, created_at
        FROM messages 
        WHERE session_id = (%s) AND mode = (%s)'''
        curs.execute(query, (session_id, 'base'))
        chat_history = curs.fetchall()
        curs.close()
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Session ID not found")
    return chat_history

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
        if request.use_notes:
            similar_chunks = find_similar_notes(conn, request_embedding)
        else:
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
    INSERT INTO messages (role, content, embedding, mode, created_at, session_id)
    VALUES 
    (%s, %s, %s, %s, %s, %s),
    (%s, %s, %s, %s, %s, %s)
    """, (
        'user', input_message, str(request_embedding), 'socratic', datetime.datetime.now(), session_id,
        'assistant', output_message, str(response_embedding), 'socratic', datetime.datetime.now(), session_id
    ))
    curs.close()
    
    return {"message": output_message, "session_id": session_id}

@app.get("/history/socratic/{session_id}")
async def fetch_socratic_history(session_id: str,
                        conn=Depends(get_db)):
    curs = conn.cursor()
    try:
        query = '''SELECT role, content, created_at
        FROM messages 
        WHERE session_id = (%s) AND mode = (%s)'''
        curs.execute(query, (session_id, 'socratic'))
        chat_history = curs.fetchall()
        curs.close()
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Session ID not found")
    return chat_history

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

###################################
# Notes Functionality
###################################
@app.post("/topic")
async def create_topic(title: str,
                        conn=Depends(get_db)):
    curs = conn.cursor()
    curs.execute("""
    INSERT INTO topics (title, created_at)
    VALUES (%s, %s)
    """, (title, datetime.datetime.now()))
    curs.close()
    return {'title':title}

@app.post("/topic/{topic_id}")
async def create_note_block(request: TopicBlockRequest,
                            conn=Depends(get_db)):
    curs = conn.cursor()
    note_embedding = base_chat.get_embeddings(request.note_content)
    curs.execute("""
    INSERT INTO topic_blocks(title, note_content, note_embedding, 
                 sot_attention, sot_intentionality, sot_difficulty, sot_content,
                 sot_emotion, created_at, topic_id)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s, %s, %s)
    """, (request.title, request.note_content, str(note_embedding), 
          request.sot_attention, request.sot_intentionality, request.sot_difficulty,
          request.sot_content, request.sot_emotion, datetime.datetime.now(),
          request.topic_id))
    curs.close()
    return {"topic_id":request.topic_id, "block_title":request.title}

@app.get("/topic")
async def fetch_all_topics(conn=Depends(get_db)):
    curs = conn.cursor()
    curs.execute("""
    SELECT *
    FROM topics
    """)
    results = curs.fetchall()
    curs.close()
    return results

@app.get("/topic/search")
async def search_topic(title:str,
                        conn=Depends(get_db)):
    curs = conn.cursor()
    curs.execute("""
    SELECT title, created_at
    FROM topics
    WHERE title LIKE %s
    """, (f"%%{title}%%", ))
    results = curs.fetchall()
    return [{
        "title": row[0],
        "created_at": row[1]
    }
    for row in results
    ]

@app.get("/topic/{topic_id}")
async def fetch_note_blocks(topic_id: int, conn=Depends(get_db)):
    curs = conn.cursor()
    curs.execute("""
    SELECT title, note_content, 
                sot_attention, sot_intentionality, sot_difficulty, sot_content,
                sot_emotion, created_at
    FROM topic_blocks
    WHERE topic_id = (%s)
    """, (topic_id, ))

    results = curs.fetchall()
    return [{
        "title": row[0],
        "note_content": row[1],
        "sot_attention": row[2],
        "sot_intentionality": row[3],
        "sot_difficulty": row[4],
        "sot_emotion": row[5],
        "sot_content": row[6],
        "created_at": row[7]
    }
    for row in results]


@app.post("/topic/{topic_id}/search")
async def search_topic_blocks(request: SearchRequest,
                          conn=Depends(get_db)):
    curs = conn.cursor()
    search_query = request.search_query
    request_embedding = base_chat.get_embeddings(request.search_query)
    ## Temporarily limits to top 10 search results, will add pagination in the future
    if request.keyword_toggle:
        curs.execute("""
        SELECT tb.title, tb.note_content, 
                tb.sot_attention, tb.sot_intentionality, tb.sot_difficulty, 
                tb.sot_content, tb.sot_emotion, tb.created_at
        FROM topic_blocks AS tb
        JOIN topics ON topics.topic_id = tb.topic_id
        WHERE tb.note_content LIKE %s
        ORDER BY tb.note_embedding <-> %s 
        LIMIT 10;
        """, (f'%%{search_query}%%', str(request_embedding)))
        results = curs.fetchall()
        curs.close()
    else:
        curs.execute("""
        SELECT tb.title, tb.note_content, 
                tb.sot_attention, tb.sot_intentionality, tb.sot_difficulty, 
                tb.sot_content, tb.sot_emotion, tb.created_at
        FROM topic_blocks AS tb
        JOIN topics ON topics.topic_id = tb.topic_id
        ORDER BY tb.note_embedding <-> %s 
        LIMIT 10;
        """, (str(request_embedding), ))
        results = curs.fetchall()
        curs.close()

    return [{
        "title": row[0],
        "note_content": row[1],
        "sot_attention": row[2],
        "sot_intentionality": row[3],
        "sot_difficulty": row[4],
        "sot_emotion": row[5],
        "sot_content": row[6],
        "created_at": row[7]
    }
    for row in results]


###################################
# Analytics Functionality
###################################

@app.post("/analytics/score/{block_id}")
async def score_block(block_id: int,
                      conn=Depends(get_db)):
    curs = conn.cursor()
    curs.execute("""
    SELECT title, note_content, 
            sot_attention, sot_intentionality, sot_difficulty, sot_content,
            sot_emotion, created_at
    FROM topic_blocks
    WHERE block_id = (%s)
    """, (block_id, ))
    result = curs.fetchone()
    request_query = f"""
    block_id: {block_id}
    title: {result[0]}
    note_content: {result[1]}
    sot_attention: {result[2]}
    sot_intentionality: {result[3]}
    sot_difficulty: {result[4]}
    sot_content: {result[5]}
    sot_emotion: {result[6]}
    """
    response = scorer_chat.chain.invoke({"input": request_query})
    raw = response.content
    clean = re.sub(r'```json|```', '', raw).strip()
    try:
        data = json.loads(clean)
        scores = data["block_scores"][0]  
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON")
    
    curs.execute("""
    INSERT INTO block_scores (block_id, attention_score, intentionality_score, 
                emotions, thought_type, temporal_orientation, thought_quality, reasoning)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING *
    """, (
        block_id,
        scores["attention_score"],
        scores["intentionality_score"],
        json.dumps(scores["emotions"]), 
        scores["thought_type"],
        scores["temporal_orientation"],
        scores["thought_quality"],
        scores["reasoning"]
    ))
    inserted = curs.fetchone()
    curs.close()
    return {
        "score_id": inserted[0],
        "block_id": inserted[1],
        "attention_score": inserted[2],
        "intentionality_score": inserted[3],
        "emotions": inserted[4],
        "thought_type": inserted[5],
        "temporal_orientation": inserted[6],
        "thought_quality": inserted[7],
        "reasoning": inserted[8]
    }

@app.get("/analytics/score/{block_id}")
async def get_block_score(block_id: int,
                      conn=Depends(get_db)):
    curs = conn.cursor()
    curs.execute("""
    SELECT  attention_score, intentionality_score, 
            emotions, thought_type, temporal_orientation, thought_quality, reasoning
    FROM block_scores
    WHERE block_id = (%s)
    """, (block_id, ))
    result = curs.fetchone()
    
    return {
        "attention_score": result[0],
        "intentionality_score": result[1],
        "emotions": result[2],
        "thought_type": result[3],
        "temporal_orientation": result[4],
        "thought_quality": result[5],
        "reasoning": result[6]
    }


@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest,
                        conn=Depends(get_db)):
    curs = conn.cursor()

    query = """
    SELECT block_id, title, note_content,
           sot_attention, sot_intentionality, sot_difficulty,
           sot_content, sot_emotion, created_at
    FROM topic_blocks 
    WHERE 1=1
    """
    # FILTER BLOCKS BASED ON REQUEST
    params = []
    if request.topic_ids:
        query += " AND topic_id = ANY(%s)"
        params.append(request.topic_ids)
    if request.date_from:
        query += " AND created_at >= %s"
        params.append(request.date_from)
    if request.date_to:
        query += " AND created_at <= %s"
        params.append(request.date_to)
    query += " ORDER BY created_at ASC"
    curs.execute(query, params)
    blocks = curs.fetchall()

    if not blocks:
        raise HTTPException(status_code=404, detail="No blocks found for the given scope")

    all_scores = []
    blocks_to_score = []

    # CHECKS IF REQUESTED BLOCKS FOR ANALYSIS HAVE ALREADY BEEN SCORED. 
    # IF NO (e.g. blocks_to_score is Not None), THEN REQUEST A SCORE.
    for block in blocks:
        block_id = block[0]
        curs.execute("SELECT attention_score, intentionality_score, emotions, thought_type, temporal_orientation, thought_quality, reasoning FROM block_scores WHERE block_id = %s", (block_id,))
        cached = curs.fetchone()
        if cached:
            all_scores.append({
                "block_id": block_id,
                "block_title": block[1],
                "attention_score": cached[0],
                "intentionality_score": cached[1],
                "emotions": cached[2],
                "thought_type": cached[3],
                "temporal_orientation": cached[4],
                "thought_quality": cached[5],
                "reasoning": cached[6],
                "difficulty": block[5],
                "created_at": str(block[8])
            })
        else:
            blocks_to_score.append(block)

    if blocks_to_score:
        batch_query = "\n\n".join([
            f"block_id: {b[0]}\ntitle: {b[1]}\nnote_content: {b[2]}\nsot_attention: {b[3]}\nsot_intentionality: {b[4]}\nsot_difficulty: {b[5]}\nsot_content: {b[6]}\nsot_emotion: {b[7]}"
            for b in blocks_to_score
        ])
        response = scorer_chat.chain.invoke({"input": batch_query})
        raw = response.content
        clean = re.sub(r'```json|```', '', raw).strip()
        try:
            scored_data = json.loads(clean)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Model returned invalid JSON during scoring")

        for scored in scored_data["block_scores"]:
            curs.execute("""
            INSERT INTO block_scores (block_id, attention_score, intentionality_score,
                        emotions, thought_type, temporal_orientation, thought_quality, reasoning)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                scored["block_id"],
                scored["attention_score"],
                scored["intentionality_score"],
                json.dumps(scored["emotions"]),
                scored["thought_type"],
                scored["temporal_orientation"],
                scored["thought_quality"],
                scored["reasoning"]
            ))
            original = next(b for b in blocks_to_score if b[0] == scored["block_id"])
            all_scores.append({
                **scored,
                "difficulty": original[5],
                "created_at": str(original[8])
            })

    # COMPUTE REQUESTED METRICS
    result = {"block_scores": all_scores}

    if "attention_trend" in request.metrics:
        result["attention_trend"] = [
            {"block_title": s["block_title"], "attention_score": s["attention_score"], "created_at": s["created_at"]}
            for s in all_scores
        ]

    if "emotion_pattern" in request.metrics:
        emotion_keys = ["anxiety", "curiosity", "frustration", "boredom", "confidence", "motivation"]
        result["emotion_pattern"] = {
            emotion: round(sum(s["emotions"][emotion] for s in all_scores) / len(all_scores), 2)
            for emotion in emotion_keys
        }

    if "difficulty_correlation" in request.metrics:
        result["difficulty_correlation"] = [
            {"block_title": s["block_title"], "difficulty": int(s["difficulty"]), "attention_score": s["attention_score"]}
            for s in all_scores
        ]

    if "intentionality_profile" in request.metrics:
        result["intentionality_profile"] = [
            {"block_title": s["block_title"], "intentionality_score": s["intentionality_score"], "thought_type": s["thought_type"]}
            for s in all_scores
        ]

    if "drift_ratio" in request.metrics:
        total = len(all_scores)
        spontaneous = sum(1 for s in all_scores if s["thought_type"] == "spontaneous")
        result["drift_ratio"] = {
            "spontaneous": spontaneous,
            "deliberate": total - spontaneous,
            "ratio": round(spontaneous / total, 2)
        }

    if "temporal_bias" in request.metrics:
        orientations = ["future", "past", "present"]
        result["temporal_bias"] = {
            o: sum(1 for s in all_scores if s["temporal_orientation"] == o)
            for o in orientations
        }

    if "progression" in request.metrics:
        result["progression"] = [
            {"block_title": s["block_title"], "attention_score": s["attention_score"], "intentionality_score": s["intentionality_score"], "created_at": s["created_at"]}
            for s in all_scores
        ]

    if "time_of_day_pattern" in request.metrics:
        result["time_of_day_pattern"] = [
            {"block_title": s["block_title"], "attention_score": s["attention_score"], "created_at": s["created_at"]}
            for s in all_scores
        ]

    # LLM CALL FOR QUALITATIVE METRICS
    needs_llm = any(m in request.metrics for m in ["sticking_points", "interventions", "summary"])
    if needs_llm:
        insight_query = json.dumps({"block_scores": all_scores})
        insight_response = insight_chat.chain.invoke({"input": insight_query})
        raw_insight = insight_response.content
        clean_insight = re.sub(r'```json|```', '', raw_insight).strip()
        try:
            insight_data = json.loads(clean_insight)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Model returned invalid JSON during insight generation")

        if "sticking_points" in request.metrics:
            result["sticking_points"] = insight_data.get("sticking_points", [])
        if "interventions" in request.metrics:
            result["interventions"] = insight_data.get("interventions", [])
        if "summary" in request.metrics:
            result["summary"] = insight_data.get("summary", "")

    curs.close()
    return result