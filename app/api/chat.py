from fastapi import APIRouter, HTTPException
from app.models.chat_model import ChatRequest, ChatResponse
from app.services.chat_services import get_answer_for_session

router = APIRouter()

@router.post("/chat/", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        answer = get_answer_for_session(req.session_id, req.question)
        return ChatResponse(session_id=req.session_id, question=req.question, answer=answer)
    except Exception as e:
        # Log full error for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
