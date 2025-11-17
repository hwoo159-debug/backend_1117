from sentence_transformers import SentenceTransformer
import os
_model = None

def get_embedder():
    global _model
    if _model is None:
        name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _model = SentenceTransformer(name)
    return _model

def embed_texts(texts: list[str]):
    model = get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs
