# chroma_manager.py
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PERSIST_DIR = Path(os.getenv("CHROMA_DIR", "./chroma")).resolve()
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Chroma client
client = chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings(allow_reset=True))

# Lazy-load embedding model (small and fast)
_EMB_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMB_MODEL_NAME)
    return _embedder


def _embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_embedder()
    embs = model.encode(texts, normalize_embeddings=True, batch_size=32, convert_to_numpy=True)
    return embs.tolist()


def _user_collection_name(user_id: int) -> str:
    return f"user_{user_id}"


def ensure_user_collection(user_id: int):
    name = _user_collection_name(user_id)
    try:
        col = client.get_collection(name)
    except Exception:
        col = client.create_collection(name)
    return col


def add_texts(user_id: int, texts: List[str], metadatas: Optional[List[dict]] = None) -> int:
    if not texts:
        return 0
    col = ensure_user_collection(user_id)
    embeddings = _embed_texts(texts)
    ids = [f"u{user_id}_d{i}" for i in range(col.count() + 1, col.count() + 1 + len(texts))]
    col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas or [{} for _ in texts])
    return len(texts)


def query_user_texts(user_id: int, query: str, k: int = 4) -> List[str]:
    try:
        col = ensure_user_collection(user_id)
        q_emb = _embed_texts([query])[0]
        res = col.query(query_embeddings=[q_emb], n_results=k)
        return res.get("documents", [[]])[0]
    except Exception:
        return []
