from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    use_rag: bool = False

class UploadRequest(BaseModel):
    filepath: str
    session_id: str = "default"

class SearchRequest(BaseModel):
    search_query: str
    keyword_toggle: bool = False
    