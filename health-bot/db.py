# db.py
from contextlib import contextmanager
from typing import Iterator
import os
from sqlmodel import SQLModel, create_engine, Session

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./health.db")
engine = create_engine(DB_URL, echo=False)


def create_db_and_tables() -> None:
    from models import User, Chat, Message, UserFile  # noqa: F401
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
