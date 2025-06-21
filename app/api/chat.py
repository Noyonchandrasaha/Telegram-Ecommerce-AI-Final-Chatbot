from fastapi import APIRouter, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from groq import Groq
from app.models.chat_model import ChatRequest, ChatResponse
from app.services.chat_services import get_answer_for_session
import os
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()
router = APIRouter()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# @router.post("/chat/", response_model=ChatResponse)
# def chat_endpoint(req: ChatRequest):
#     try:
#         answer = get_answer_for_session(req.session_id, req.question)
#         return ChatResponse(session_id=req.session_id, question=req.question, answer=answer)
#     except Exception as e:
#         # Log full error for debugging
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/audio")
async def chat_with_audio_or_text(
    session_id: str = Form(...),
    text: str = Form(None),
    audio: UploadFile = None
):
    try:
        if audio:
            contents = await audio.read()
            transcription = client.audio.transcriptions.create(
                file=(audio.filename, contents),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
            question = transcription.text

            # # Translate to English if not already
            # response = client.chat.completions.create(
            #     model="llama-3.3-70b-versatile",
            #     messages=[
            #         {
            #             "role": "system",
            #             "content": (
            #                 "You are a translator. Translate the input to fluent English. "
            #                 "If it's already in English, return it as-is. Do not add comments."
            #             ),
            #         },
            #         {"role": "user", "content": question}
            #     ],
            #     temperature=0.3
            # )
            # question = response.choices[0].message.content.strip()

        elif text:
            question = text
        else:
            raise HTTPException(status_code=400, detail="Provide either text or audio input.")

        # Send to LangChain-powered Q&A
        answer = get_answer_for_session(session_id=session_id, question=question)
        return {
            "session_id": session_id,
            "question": question,
            "answer": answer
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")