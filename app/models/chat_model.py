from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
