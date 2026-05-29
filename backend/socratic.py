from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from db import DB_PASS
import psycopg

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="qwen3.5-2b",
    temperature=0.5,
    max_completion_tokens=500
)

embedder = OpenAIEmbeddings(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="text-embedding-embeddinggemma-300m-qat",
    check_embedding_ctx_length=False
)

prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a student studying assistant. You will ask the student questions 
     about a topic based on their understanding, and you will identify gaps in their explanations."""),
     MessagesPlaceholder(variable_name='history'),
     ("human","{input}")
])

chain = prompt | llm

store = {}

def get_history(session_id: str, conn):
    if session_id not in store:
        curs = conn.cursor()
        curs.execute(
            "SELECT role, content FROM messages WHERE session_id = %s ORDER BY created_at ASC",
            (session_id,)
        )
        messages = []
        for role, content in curs.fetchall():
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        store[session_id] = messages
        curs.close()
    return store[session_id]

def get_embeddings(text):
    embeddings = embedder.embed_query(text)
    return embeddings

