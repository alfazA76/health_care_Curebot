# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from chatbot import get_llm_response, refresh_vectorstore
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, HTTPException

from auth import router as auth_router, get_optional_user
from db import create_db_and_tables, get_session
from models import Chat, Message, User
from sqlmodel import Session, select

app = FastAPI(title="Health Bot API")

# This allows your frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str
    chat_id: Optional[int] = None

@app.post("/ask")
def ask_question(query: Query, user: Optional[User] = Depends(get_optional_user), session: Session = Depends(get_session)):
    response = get_llm_response(query.text, user_id=user.id if user else None)

    # Persist chat history if user is authenticated
    if user:
        chat: Optional[Chat] = None
        if query.chat_id:
            chat = session.get(Chat, query.chat_id)
            if not chat or chat.user_id != user.id:
                raise HTTPException(status_code=403, detail="Invalid chat_id")
        else:
            # Create a new chat with a simple title
            title = (query.text[:40] + "...") if len(query.text) > 40 else query.text
            chat = Chat(user_id=user.id, title=title)
            session.add(chat)
            session.commit()
            session.refresh(chat)

        # Save user message and assistant response
        session.add(Message(chat_id=chat.id, role="user", content=query.text))
        session.add(Message(chat_id=chat.id, role="assistant", content=response))
        session.commit()

        return {"answer": response, "chat_id": chat.id}

    return {"answer": response}

@app.post("/reindex")
def reindex_documents():
    count = refresh_vectorstore()
    return {"status": "ok", "chunks_indexed": count}

@app.get("/health")
def health():
    return {"status": "ok"}

# Include routers
from routes_chats import router as chats_router
from routes_files import router as files_router
from whatsapp import router as whatsapp_router
from routes_media import router as media_router

app.include_router(auth_router)
app.include_router(chats_router)
app.include_router(files_router)
app.include_router(whatsapp_router)
app.include_router(media_router)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()