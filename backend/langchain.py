from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

class BaseChat():
    def __init__(self, llm=None, embedder=None):
        self.llm = llm
        self.embedder = embedder
        if llm is None:
            self.llm = ChatOpenAI(
            base_url="http://127.0.0.1:1234/v1",
            api_key="lm-studio",
            model="qwen3.5-2b",
            temperature=0.5,
            max_completion_tokens=500
            )
        if embedder is None:
            self.embedder = OpenAIEmbeddings(
            base_url="http://127.0.0.1:1234/v1",
            api_key="lm-studio",
            model="text-embedding-embeddinggemma-300m-qat",
            check_embedding_ctx_length=False
            )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system","""You are a general student studying assistant. Respond to student
             inquiries in a gentle and respectful manner. If you decide you do not have sufficient
             context, mention it."""),
            MessagesPlaceholder(variable_name='history'),
            ("human","{input}")
        ])
        self.store = {}
        self.chain = self.prompt_template | self.llm
    
    def get_history(self, session_id: str, conn):
        if session_id not in self.store:
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
            self.store[session_id] = messages
            curs.close()
        return self.store[session_id]
    
    def get_embeddings(self, text):
        embeddings =self.embedder.embed_query(text)
        return embeddings
    

class SocracticChat(BaseChat):
    def __init__(self, llm=None, embedder=None):
        super().__init__(llm, embedder)
        self.prompt_template = ChatPromptTemplate.from_messages([
        ("system","""You are a student studying assistant. You will help students through the
        Socractic method. You will ask the student questions about a topic based on their 
        understanding, and you will identify gaps in their explanations."""),
        MessagesPlaceholder(variable_name='history'),
        ("human","{input}")
        ])
        self.chain = self.prompt_template | self.llm
        