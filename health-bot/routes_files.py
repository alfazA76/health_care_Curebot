# routes_files.py
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlmodel import Session

from auth import get_current_user
from db import get_session
from models import UserFile, User
import chroma_manager
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from PIL import Image
import pytesseract

router = APIRouter(prefix="/files", tags=["files"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    saved = []
    user_dir = UPLOAD_DIR / f"user_{user.id}"
    user_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        dest = user_dir / f.filename
        try:
            with dest.open("wb") as out:
                out.write(await f.read())
            rec = UserFile(user_id=user.id, path=str(dest), original_name=f.filename)
            session.add(rec)
            saved.append(f.filename)
        except Exception:
            raise HTTPException(status_code=500, detail=f"Failed to save {f.filename}")

    session.commit()
    
    # Index into per-user Chroma
    texts: List[str] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    for name in saved:
        p = user_dir / name
        try:
            suffix = p.suffix.lower()
            if suffix == ".pdf":
                loader = PyPDFLoader(str(p))
                docs = loader.load()
                texts.extend([d.page_content for d in splitter.split_documents(docs)])
            elif suffix in {".txt", ".md"}:
                loader = TextLoader(str(p), encoding="utf-8")
                docs = loader.load()
                texts.extend([d.page_content for d in splitter.split_documents(docs)])
            elif suffix == ".csv":
                df = pd.read_csv(p)
                concat_text = df.astype(str).to_csv(index=False)
                chunks = splitter.split_text(concat_text)
                texts.extend(chunks)
            elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                img = Image.open(p)
                ocr = pytesseract.image_to_string(img)
                if ocr.strip():
                    texts.extend(splitter.split_text(ocr))
        except Exception:
            # continue indexing others, but don't crash the request
            pass

    if texts:
        try:
            chroma_manager.add_texts(user.id, texts)
        except Exception:
            pass

    return {"status": "ok", "saved": saved, "chunks_indexed": len(texts)}


@router.get("/")
def list_files(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    from sqlmodel import select
    items = session.exec(select(UserFile).where(UserFile.user_id == user.id)).all()
    return [{"id": it.id, "name": it.original_name, "path": it.path} for it in items]


@router.delete("/{file_id}")
def delete_file(file_id: int, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    rec: Optional[UserFile] = session.get(UserFile, file_id)
    if not rec or rec.user_id != user.id:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        Path(rec.path).unlink(missing_ok=True)
    except Exception:
        pass
    session.delete(rec)
    session.commit()
    # Note: We are not removing vectors from Chroma here for simplicity.
    return {"status": "ok"}
