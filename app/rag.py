import os
import gc
import ctypes
import platform
import numpy as np
import requests  # 현재는 사용하지 않지만, 혹시 모를 확장을 위해 남겨둠
import json
import re
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import warnings
warnings.filterwarnings("ignore")

############################################################
# Memory Cleanup
############################################################

def aggressive_memory_cleanup():
    gc.collect()
    if platform.system() == "Linux":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except:
            pass

############################################################
# Groq LLM
############################################################

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

def call_llm_api(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "LLM 생성 오류가 발생했습니다."

############################################################
# 간단 로컬 임베딩 (Jina 완전 제거용)
############################################################

LOCAL_EMBED_DIM = int(os.getenv("LOCAL_EMBED_DIM", "512"))

_token_pattern = re.compile(r"[가-힣A-Za-z0-9]+")

def simple_tokenize(text: str) -> List[str]:
    text = "" if text is None else str(text)
    return _token_pattern.findall(text.lower())

def simple_embed(texts: List[str], dim: int = LOCAL_EMBED_DIM) -> np.ndarray:
    """
    외부 API 없이 동작하는 해시 기반 임베딩.
    - 단어를 토큰화한 뒤 hash(token) % dim 위치에 카운트를 더함
    - 마지막에 L2 normalize
    """
    n = len(texts)
    vecs = np.zeros((n, dim), dtype="float32")

    for i, t in enumerate(texts):
        tokens = simple_tokenize(t)
        for tok in tokens:
            h = hash(tok) % dim
            vecs[i, h] += 1.0

    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs = vecs / norms
    return vecs

############################################################
# Database Connector
############################################################

class DatabaseConnector:
    def __init__(self):
        raw_url = os.getenv("DATABASE_URL")
        if not raw_url:
            raise RuntimeError("환경변수 DATABASE_URL 이 없습니다.")

        if raw_url.startswith("postgres://"):
            raw_url = raw_url.replace("postgres://", "postgresql://", 1)

        parsed = urlparse(raw_url)
        self.config = {
            "host": parsed.hostname,
            "port": parsed.port,
            "database": parsed.path.lstrip("/"),
            "user": parsed.username,
            "password": parsed.password,
        }

        self.connection = None
        print(f"[DB] config = {self.config}")

    def connect(self):
        try:
            import psycopg2
            self.connection = psycopg2.connect(**self.config)
            print("[DB] 연결 성공")
            return True
        except Exception as e:
            print(f"[DB ERROR] {e}")
            return False

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def get_policy_chunks(self, limit: int = 20000):
        """
        DB_InsData.insurance_clauses 에서 약관 청크를 읽어온다.
        각 row가 곧 하나의 청크.
        """
        try:
            from psycopg2.extras import RealDictCursor
            cur = self.connection.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT
                    id          AS chunk_id,
                    product_id  AS product_id,
                    COALESCE(search_text, original_text) AS chunk_text,
                    embedding   AS embedding
                FROM insurance_clauses
                WHERE COALESCE(search_text, original_text) IS NOT NULL
                LIMIT %s
            """, (limit,))

            rows = cur.fetchall()
            cur.close()
            print(f"[DB] 보험약관 로드 {len(rows)}개")
            return rows
        except Exception as e:
            print(f"[DB 조회 오류] {e}")
            return []


############################################################
# Vector Store
############################################################

class VectorStore:
    def __init__(self):
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, D)

    def load(self, docs: List[Dict], embeddings: np.ndarray):
        self.documents = docs
        self.embeddings = embeddings
        print(f"[VS] 벡터스토어 구축: {len(docs)}개, dim={embeddings.shape[1]}")

    def search(self, query_embedding: np.ndarray, k: int = 5):
        if self.embeddings is None or len(self.embeddings) == 0:
            print("[VS] 임베딩 없음")
            return []

        q = query_embedding[0]

        # 혹시라도 차원이 다르면 pad/truncate 로 맞춰줌
        d_store = self.embeddings.shape[1]
        d_q = q.shape[0]
        if d_q < d_store:
            q = np.pad(q, (0, d_store - d_q))
        elif d_q > d_store:
            q = q[:d_store]

        sims: List[Tuple[int, float]] = []
        q_norm = np.linalg.norm(q) + 1e-8

        for i, emb in enumerate(self.embeddings):
            denom = q_norm * (np.linalg.norm(emb) + 1e-8)
            if denom == 0.0:
                sim = 0.0
            else:
                sim = float(np.dot(q, emb) / denom)
            sims.append((i, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in sims[:k]]


############################################################
# RAG SYSTEM
############################################################

class RAGSystem:
    def __init__(self):
        self.db = DatabaseConnector()
        self.vs = VectorStore()

    def initialize(self, limit: int = 20000):
        print("\n=== RAG 초기화 ===")

        if not self.db.connect():
            return False

        docs = self.db.get_policy_chunks(limit)
        if not docs:
            print("[DB] 약관 데이터가 없습니다.")
            self.db.disconnect()
            return False

        # DB embedding 컬럼은 현재 사용하지 않고,
        # chunk_text 기반 로컬 임베딩으로 일관되게 구축
        texts = [str(d["chunk_text"]) for d in docs]
        print(f"[EMBED] 청크 수: {len(texts)}개, dim={LOCAL_EMBED_DIM}")
        embeddings = simple_embed(texts, dim=LOCAL_EMBED_DIM)

        self.vs.load(docs, embeddings)

        self.db.disconnect()
        print("=== 초기화 완료 ===\n")
        return True

    def answer(self, question: str):
        # 질문도 동일한 로컬 임베딩으로 처리
        if self.vs.embeddings is None:
            return {"answer": "RAG가 초기화되지 않았습니다.", "sources": []}

        dim = self.vs.embeddings.shape[1]
        q_emb = simple_embed([question], dim=dim)
        retrieved = self.vs.search(q_emb, k=5)

        if not retrieved:
            return {"answer": "관련 정보 없음", "sources": []}

        context = "\n".join(
            f"[{i+1}] {doc['chunk_text'][:250]}"
            for i, (doc, _) in enumerate(retrieved)
        )

        prompt = f"""
다음은 보험약관의 일부입니다. 이를 참고해 질문에 명확하게 답하세요.

[약관]
{context}

[질문]
{question}

[규칙]
- 약관 내용을 요약해서 말할 것
- 질문에 직접적으로 답하기
- 불필요한 말 금지
- 결론 먼저 말하기

[최종 답변]
"""

        answer = call_llm_api(prompt)

        return {
            "answer": answer,
            "sources": [doc["chunk_id"] for doc, _ in retrieved]
        }


############################################################
# Singleton Access
############################################################

_rag_instance: Optional[RAGSystem] = None

def init_rag():
    global _rag_instance
    if _rag_instance is None:
        rag = RAGSystem()
        rag.initialize()
        _rag_instance = rag
    return True

def answer_with_rag(question: str):
    if _rag_instance is None:
        raise RuntimeError("RAG 초기화 필요")
    return _rag_instance.answer(question)
