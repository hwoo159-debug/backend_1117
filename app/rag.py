import os
import gc
import ctypes
import platform
import numpy as np
import requests
import json
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
# Jina Embedding API
############################################################

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBED_MODEL = os.getenv("JINA_EMBED_MODEL", "jina-embeddings-v2-base")

def embed_with_jina(texts: List[str]) -> np.ndarray:
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": texts,
        "model": JINA_EMBED_MODEL
    }

    try:
        res = requests.post(url, json=data, headers=headers)
        res.raise_for_status()
        vectors = [item["embedding"] for item in res.json()["data"]]
        return np.array(vectors, dtype="float32")
    except Exception as e:
        print(f"[EMBED ERROR] {e}")
        return np.zeros((len(texts), 1024), dtype="float32")


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
        self.documents = []
        self.embeddings = None

    def load(self, docs: List[Dict], embeddings: np.ndarray):
        self.documents = docs
        self.embeddings = embeddings
        print(f"[VS] 벡터스토어 구축: {len(docs)}개")

    def search(self, query_embedding: np.ndarray, k=5):
        if self.embeddings is None or len(self.embeddings) == 0:
            print("[VS] 임베딩 없음")
            return []

        q = query_embedding[0]
        sims = []
        for i, emb in enumerate(self.embeddings):
            sim = float(
                np.dot(q, emb) /
                (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-8)
            )
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

        # texts = [str(d["chunk_text"])[:800] for d in docs]
        # embeddings = embed_with_jina(texts) 주석처리
        
        emb_list = []
        valid_docs = []

        for d in docs:
            raw = d.get("embedding")
            if raw is None:
                continue
            if isinstance(raw, str):
                try:
                    vec = json.loads(raw)
                except Exception:
                    continue
            else:
                # 이미 리스트/array면 그대로
                vec = raw
            emb_list.append(vec)
            valid_docs.append(d)
        if not emb_list:
            print("[VS] embedding 데이터가 하나도 없습니다.")
            self.db.disconnect()
            return False

        embeddings = np.array(emb_list, dtype="float32")
        self.vs.load(docs, embeddings)

        self.db.disconnect()
        print(f"[VS] 벡터스토어 구축: {len(valid_docs)}개")
        print("=== 초기화 완료 ===\n")
        return True

    def answer(self, question: str):
        q_emb = embed_with_jina([question])
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
        rag.initialize()  # 기본 limit=1000
        _rag_instance = rag
    return True

def answer_with_rag(question: str):
    if _rag_instance is None:
        raise RuntimeError("RAG 초기화 필요")
    return _rag_instance.answer(question)
