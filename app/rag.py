# app/rag.py
# 백엔드 서버용 RAG 모듈 (코랩 절차형 코드 → 함수형/모듈형 완전 재구성)

import gc
import torch
import ctypes
import platform
from typing import List, Dict, Tuple, Optional
import numpy as np
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')


############################################################
# 메모리 정리 기능
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
# Database Connector
############################################################

class DatabaseConnector:
    def __init__(self, config: Dict):
        self.config = config
        self.connection = None

    def connect(self) -> bool:
        try:
            import psycopg2
            self.connection = psycopg2.connect(**self.config)
            return True
        except Exception as e:
            print(f"[DB] 연결 오류: {e}")
            return False

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def get_all_documents(self, table_name: str = "policychunk", limit: int = 50):
        try:
            from psycopg2.extras import RealDictCursor
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)

            query = f"SELECT * FROM {table_name} LIMIT %s"
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            return rows

        except Exception as e:
            print(f"[DB] 문서 조회 오류: {e}")
            return []


############################################################
# Embedding Model (BAAI/bge-m3)
############################################################

class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None

    def load(self):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(self.model_name, use_fp16=False)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            self.load()
        outputs = self.model.encode(
            texts,
            batch_size=8,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return outputs["dense_vecs"]

    def unload(self):
        del self.model
        self.model = None
        aggressive_memory_cleanup()


############################################################
# Vector Store (numpy 기반 search)
############################################################

class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = np.array([])

    def add_documents_with_embeddings(self, docs: List[Dict], embeddings: np.ndarray):
        self.documents = docs
        self.embeddings = embeddings

    def search(self, query_embedding: np.ndarray, k: int = 5):
        if len(self.embeddings) == 0:
            return []

        query_embedding = query_embedding[0] if query_embedding.ndim > 1 else query_embedding

        similarities = []
        for idx, emb in enumerate(self.embeddings):
            emb_vec = emb[0] if emb.ndim > 1 else emb
            sim = np.dot(query_embedding, emb_vec) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb_vec)
            )
            similarities.append((idx, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [(self.documents[idx], score) for idx, score in similarities[:k]]
        return results


############################################################
# Language Model (Qwen 1.5B)
############################################################

class LanguageModel:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()

    def generate(self, prompt: str, max_new_tokens=500) -> str:
        if self.model is None:
            self.load()

        try:
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                text = text.replace(prompt, "").strip()
                return text

        except Exception as e:
            print(f"[LLM] 생성 오류: {e}")
            return "답변 생성 실패"

        finally:
            aggressive_memory_cleanup()

    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        aggressive_memory_cleanup()


############################################################
# RAG SYSTEM (핵심)
############################################################

class RAGSystem:
    def __init__(self, db_config: Dict):
        self.embedding_model = EmbeddingModel()
        self.llm = LanguageModel()
        self.vector_store = VectorStore()
        self.db = DatabaseConnector(db_config)

    def initialize(self, limit=50):
        if not self.db.connect():
            print("[RAG] DB 연결 실패")
            return False

        docs = self.db.get_all_documents(limit=limit)
        if not docs:
            print("[RAG] 문서 없음")
            return False

        texts = [str(d.get("chunk_text"))[:800] for d in docs]

        self.embedding_model.load()
        embeddings = self.embedding_model.encode_texts(texts)
        self.embedding_model.unload()

        self.vector_store.add_documents_with_embeddings(docs, embeddings)

        self.db.disconnect()
        return True

    def answer(self, question: str, k=5):
        self.embedding_model.load()
        query_emb = self.embedding_model.encode_texts([question])
        self.embedding_model.unload()

        retrieved = self.vector_store.search(query_emb, k=k)
        if not retrieved:
            return {"answer": "관련 정보 없음", "sources": []}

        context = ""
        for i, (doc, score) in enumerate(retrieved):
            snippet = str(doc.get("chunk_text"))[:250]
            context += f"[{i+1}] {snippet}\n"

        prompt = f"""
보험약관 정보:
{context}

질문: {question}

답변:"""

        self.llm.load()
        answer = self.llm.generate(prompt, max_new_tokens=300)
        self.llm.unload()

        return {
            "answer": answer,
            "sources": [doc.get("chunk_id", None) for doc, s in retrieved]
        }


############################################################
# 전역 인스턴스 (서버에서 import해서 사용)
############################################################

_rag_instance: Optional[RAGSystem] = None


def init_rag(db_config: Dict):
    global _rag_instance
    if _rag_instance is None:
        rag = RAGSystem(db_config)
        rag.initialize(limit=50)
        _rag_instance = rag
    return True


def answer_with_rag(question: str):
    global _rag_instance
    if _rag_instance is None:
        raise RuntimeError("RAG가 초기화되지 않음 (init_rag 먼저 호출)")

    return _rag_instance.answer(question, k=5)
