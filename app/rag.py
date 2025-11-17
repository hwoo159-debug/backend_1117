# app/rag.py

import os
import gc
import torch
import ctypes
import platform
import numpy as np
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import warnings

warnings.filterwarnings("ignore")


############################################################
# 메모리 정리
############################################################

def aggressive_memory_cleanup():
    gc.collect()

    if platform.system() == "Linux":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


############################################################
# Database Connector  (env: DATABASE_URL 직접 파싱)
############################################################

class DatabaseConnector:
    def __init__(self):
        raw_url = os.getenv("DATABASE_URL")
        if not raw_url:
            raise RuntimeError("환경변수 DATABASE_URL 이 설정되어 있지 않습니다.")

        # postgres:// → postgresql:// 로 정규화
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
        print(f"[DB] config 사용: db={self.config['database']} host={self.config['host']}")

    def connect(self) -> bool:
        try:
            import psycopg2
            self.connection = psycopg2.connect(**self.config)
            print("[DB] 연결 성공")
            return True
        except Exception as e:
            print(f"[DB] 연결 오류: {e}")
            return False

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def get_policy_chunks(self, limit: int = 300):
        """
        policychunk 테이블 구조:
        - chunk_id (PK)
        - product_id
        - chunk_text  ← 약관 텍스트
        - embedding
        - metadata
        """
        try:
            from psycopg2.extras import RealDictCursor
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT chunk_id, product_id, chunk_text
                FROM policychunk
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            cursor.close()

            print(f"[DB] policychunk {len(rows)}개 로드됨")
            return rows

        except Exception as e:
            print(f"[DB] 문서 조회 오류: {e}")
            return []


############################################################
# Embedding Model  (BAAI/bge-m3)
############################################################

class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None

    def load(self):
        print("[EMB] 임베딩 모델 로딩…")
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(self.model_name, use_fp16=False)
        print("[EMB] 로드 완료")

    def encode(self, texts: List[str]):
        if self.model is None:
            self.load()

        out = self.model.encode(
            texts,
            batch_size=8,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"]

    def unload(self):
        print("[EMB] 언로드")
        del self.model
        self.model = None
        aggressive_memory_cleanup()


############################################################
# Vector Store (메모리 기반)
############################################################

class VectorStore:
    def __init__(self):
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def load(self, docs: List[Dict], embeddings: np.ndarray):
        self.documents = docs
        self.embeddings = embeddings
        print(f"[VS] {len(docs)}개 문서 벡터스토어 구축")

    def search(self, query_embedding: np.ndarray, k: int = 5):
        if self.embeddings is None or len(self.embeddings) == 0:
            print("[VS] 저장된 임베딩 없음")
            return []

        q = query_embedding[0]
        scores: List[Tuple[int, float]] = []

        for i, emb in enumerate(self.embeddings):
            emb_vec = emb[0] if emb.ndim > 1 else emb
            sim = float(
                np.dot(q, emb_vec)
                / (np.linalg.norm(q) * np.linalg.norm(emb_vec) + 1e-8)
            )
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in scores[:k]]


############################################################
# Language Model (Qwen)
############################################################

class LanguageModel:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        print("[LLM] 언어모델 로딩…")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32, trust_remote_code=True
        ).eval()
        print("[LLM] 로딩 완료")

    def generate(self, prompt: str, max_new_tokens: int = 350) -> str:
        if self.model is None:
            self.load()

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=800,
            )

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text.replace(prompt, "").strip()

        except Exception as e:
            print(f"[LLM] 생성 오류: {e}")
            return "답변 생성 중 오류가 발생했습니다."
        finally:
            aggressive_memory_cleanup()

    def unload(self):
        print("[LLM] 언로드")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        aggressive_memory_cleanup()


############################################################
# RAG SYSTEM
############################################################

class RAGSystem:
    def __init__(self):
        self.db = DatabaseConnector()
        self.emb = EmbeddingModel()
        self.llm = LanguageModel()
        self.vs = VectorStore()

    def initialize(self, limit: int = 300):
        print("\n=== RAG 초기화 시작 ===")
        if not self.db.connect():
            print("[RAG] DB 연결 실패")
            return False

        docs = self.db.get_policy_chunks(limit=limit)
        if not docs:
            print("[RAG] policychunk 비어 있음")
            self.db.disconnect()
            return False

        texts = [str(d["chunk_text"])[:800] for d in docs]

        self.emb.load()
        embeddings = self.emb.encode(texts)
        self.emb.unload()

        self.vs.load(docs, embeddings)

        self.db.disconnect()
        print("=== RAG 초기화 완료 ===\n")
        return True

    def answer(self, question: str):
        self.emb.load()
        q_emb = self.emb.encode([question])
        self.emb.unload()

        retrieved = self.vs.search(q_emb, k=5)
        if not retrieved:
            return {"answer": "관련 정보 없음", "sources": []}

        context = "\n".join(
            f"[{i+1}] {doc['chunk_text'][:250]}"
            for i, (doc, _) in enumerate(retrieved)
        )

        prompt = f"""
다음은 보험약관의 일부입니다. 이를 참고해 질문에 명확하고 간단하게 답하세요.
절대 약관 원문을 그대로 복사하거나 이어쓰지 마세요. 
핵심 내용만 짧게 정리해서 답변하세요.

[약관 발췌]
{context}

[질문]
{question}

[규칙]
- 약관 내용을 요약해서 말할 것
- 질문과 관련 없는 내용은 제거
- 문장을 새로 구성할 것
- 결론을 한 문단으로 제시할 것

[최종 답변]
"""


        answer = self.llm.generate(prompt, max_new_tokens=300)
        return {
            "answer": answer,
            "sources": [doc["chunk_id"] for doc, _ in retrieved],
        }


############################################################
# 전역 인스턴스 + 헬퍼
############################################################

_rag_instance: Optional[RAGSystem] = None


def init_rag():
    global _rag_instance
    if _rag_instance is None:
        rag = RAGSystem()
        rag.initialize(limit=300)
        _rag_instance = rag
    return True


def answer_with_rag(question: str):
    if _rag_instance is None:
        raise RuntimeError("RAG가 초기화되지 않았습니다. init_rag() 먼저 호출하세요.")
    return _rag_instance.answer(question)
