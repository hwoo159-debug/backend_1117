# app/rag.py

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
# Database Connector (policychunk 전용)
############################################################

class DatabaseConnector:
    def __init__(self, config: Dict):
        self.config = config
        self.connection = None

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
            cursor.execute("""
                SELECT chunk_id, product_id, chunk_text
                FROM policychunk
                LIMIT %s
            """, (limit,))
            rows = cursor.fetchall()
            cursor.close()

            print(f"[DB] {len(rows)}개 policychunk 로드됨")
            return rows

        except Exception as e:
            print(f"[DB] 문서 조회 오류: {e}")
            return []


############################################################
# Embedding Model
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

        result = self.model.encode(
            texts,
            batch_size=8,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return result["dense_vecs"]

    def unload(self):
        print("[EMB] 언로드")
        del self.model
        self.model = None
        aggressive_memory_cleanup()


############################################################
# Vector Store (메모리 기반 검색)
############################################################

class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = None

    def load(self, docs: List[Dict], embeddings: np.ndarray):
        self.documents = docs
        self.embeddings = embeddings
        print(f"[VS] {len(docs)}개 문서 벡터스토어 구축")

    def search(self, query_embedding: np.ndarray, k: int = 5):
        if self.embeddings is None or len(self.embeddings) == 0:
            print("[VS] 저장된 임베딩 없음")
            return []

        q = query_embedding[0]
        scores = []

        for i, emb in enumerate(self.embeddings):
            emb_vec = emb[0] if emb.ndim > 1 else emb
            sim = np.dot(q, emb_vec) / (np.linalg.norm(q) * np.linalg.norm(emb_vec))
            scores.append((i, float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = [(self.documents[idx], score) for idx, score in scores[:k]]
        return results


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

    def generate(self, prompt: str) -> str:
        if self.model is None:
            self.load()

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            text = text.replace(prompt, "").strip()
            return text
        except:
            return "답변 생성 오류"
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
# RAG SYSTEM (policychunk 최적화 버전)
############################################################

class RAGSystem:
    def __init__(self, db_config):
        self.db = DatabaseConnector(db_config)
        self.emb = EmbeddingModel()
        self.llm = LanguageModel()
        self.vs = VectorStore()

    def initialize(self, limit=300):
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
            f"[{i+1}] {d['chunk_text'][:250]}"
            for i, (d, _) in enumerate(retrieved)
        )

        prompt = f"""
보험약관 정보:
{context}

질문: {question}

답변:"""

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": [d["chunk_id"] for d, _ in retrieved]
        }


############################################################
# 전역 인스턴스 (FastAPI에서 사용)
############################################################

_rag_instance: Optional[RAGSystem] = None

def init_rag(db_config):
    global _rag_instance
    if _rag_instance is None:
        rag = RAGSystem(db_config)
        rag.initialize(limit=300)
        _rag_instance = rag
    return True

def answer_with_rag(question: str):
    if _rag_instance is None:
        raise RuntimeError("RAG 초기화 필요")
    return _rag_instance.answer(question)
