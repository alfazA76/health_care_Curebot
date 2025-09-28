# routes_chats.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from auth import get_current_user
from db import get_session
from models import Chat, Message, User

router = APIRouter(prefix="/chats", tags=["chats"])


class ChatOut(BaseModel):
    id: int
    title: str

class MessageOut(BaseModel):
    id: int
    role: str
    content: str

class ChatCreate(BaseModel):
    title: str


@router.get("/", response_model=List[ChatOut])
def list_chats(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    chats = session.exec(select(Chat).where(Chat.user_id == user.id).order_by(Chat.created_at.desc())).all()
    return [ChatOut(id=c.id, title=c.title) for c in chats]


@router.post("/", response_model=ChatOut)
def create_chat(payload: ChatCreate, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    chat = Chat(user_id=user.id, title=payload.title)
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return ChatOut(id=chat.id, title=chat.title)


@router.get("/{chat_id}/messages", response_model=List[MessageOut])
def get_messages(chat_id: int, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    chat = session.get(Chat, chat_id)
    if not chat or chat.user_id != user.id:
        raise HTTPException(status_code=404, detail="Chat not found")
    msgs = session.exec(select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.asc())).all()
    return [MessageOut(id=m.id, role=m.role, content=m.content) for m in msgs]
