# models.py
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Relationship


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    phone: Optional[str] = Field(default=None, index=True)
    password_hash: str
    health_history: Optional[str] = Field(default=None)
    state: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None)
    address_line: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    chats: list["Chat"] = Relationship(back_populates="user")
    files: list["UserFile"] = Relationship(back_populates="user")


class Chat(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: Optional[User] = Relationship(back_populates="chats")
    messages: list["Message"] = Relationship(back_populates="chat")


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chat_id: int = Field(foreign_key="chat.id")
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    chat: Optional[Chat] = Relationship(back_populates="messages")


class UserFile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    path: str
    original_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: Optional[User] = Relationship(back_populates="files")
