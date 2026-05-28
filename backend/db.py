from psycopg2 import pool
from psycopg2.extras import register_uuid
import os
from dotenv import load_dotenv
import psycopg
from langchain_postgres import PostgresChatMessageHistory

load_dotenv()

DB_PASS = os.getenv("DB_PASS")

register_uuid()

connection_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host="localhost",
    port=5432,
    database="postgres",  
    user="postgres",      
    password=DB_PASS
)

def get_db():
    conn = connection_pool.getconn()
    try:
        yield conn          
        conn.commit()       
    except Exception:
        conn.rollback()  
        raise   
    finally:
        connection_pool.putconn(conn)  


def create_tables():
    conn = connection_pool.getconn()
    try:
        curs = conn.cursor()
        curs.execute("""
        CREATE TABLE IF NOT EXISTS sessions(
            id UUID PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        curs.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id SERIAL PRIMARY KEY,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(768),
            created_at TIMESTAMP DEFAULT NOW(),
            session_id UUID NOT NULL REFERENCES sessions(id)
            );
        """)
        conn.commit()
        curs.close()
    except:
        raise ConnectionError("Could not instantiate table.")

    sync_connection = psycopg.connect(f"postgresql://postgres:{DB_PASS}@localhost:5432/postgres")
    PostgresChatMessageHistory.create_tables(sync_connection, "chat_history")
    sync_connection.close()

# we use a context manager to scope the cursor session
# with conn.cursor() as curs:

#     try:
#         # simple single row system query
#         curs.execute("SELECT version()")

#         # returns a single row as a tuple
#         single_row = curs.fetchone()

#         # use an f-string to print the single tuple returned
#         print(f"{single_row}")

#         # simple multi row system query
#         curs.execute("SELECT query, backend_type FROM pg_stat_activity")

#         # a default install should include this query and some backend workers
#         many_rows = curs.fetchmany(5)

#         # use the * unpack operator to print many_rows which is a Python list
#         print(*many_rows, sep = "\n")

#     # a more robust way of handling errors
#     except (Exception, psycopg2.DatabaseError) as error:
#         print(error)