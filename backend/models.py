from pydantic import BaseModel, Field
from enum import Enum
import datetime

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
    sot_difficulty: int
    sot_emotion: str
    sot_content: str
    topic_id: int


class Metric(str, Enum):
    classifications = "classifications"
    attention_trend = "attention_trend"
    emotion_pattern = "emotion_pattern"
    difficulty_correlation = "difficulty_correlation"
    intentionality_profile = "intentionality_profile"
    topic_comparison = "topic_comparison"
    time_of_day_pattern = "time_of_day_pattern"
    progression = "progression"
    sticking_points = "sticking_points"
    drift_ratio = "drift_ratio"
    temporal_bias = "temporal_bias"
    interventions = "interventions"
    summary = "summary"


class AnalyticsRequest(BaseModel):
    topic_ids: list[int] | None = None
    date_from: datetime.datetime | None = None
    date_to: datetime.datetime | None = None
    metrics: list[Metric] = list(Metric) 


