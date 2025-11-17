from sqlalchemy import text
from .db import SessionLocal, init_pgvector
from .utils import embed_texts

def upsert_documents(items: list[dict]):
    init_pgvector()
    contents = [it["content"] for it in items]
    embs = embed_texts(contents)
    with SessionLocal() as s:
        for it, emb in zip(items, embs):
            s.execute(
                text("INSERT INTO documents (title, content, metadata, embedding) VALUES (:t, :c, cast(:m as jsonb), :e)"),
                {"t": it.get("title",""), "c": it["content"], "m": (it.get("metadata") or {}), "e": emb.tolist()}
            )
        s.commit()
