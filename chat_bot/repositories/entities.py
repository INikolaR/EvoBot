from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class RequestEntity:
    user_id: str
    chat_type: int
    user_text: str
    model_response: str
    request_time: datetime
    response_time: datetime
    rag_context: Optional[str] = None
    prev_request_id: Optional[int] = None

@dataclass
class UserEntity:
    user_id: str
    chat_type: int
    last_request_id: Optional[int] = None
