from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    use_rag: bool = False
    use_notes: bool = False

class UploadRequest(BaseModel):
    filepath: str
    session_id: str = "default"

class SearchRequest(BaseModel):
    search_query: str
    keyword_toggle: bool = False

class TopicBlockRequest(BaseModel):
    title: str
    note_content: str
    sot_attention: str
    sot_intentionality: str
    sot_difficulty: str
    sot_emotion: str
    sot_content: str
    topic_id: str