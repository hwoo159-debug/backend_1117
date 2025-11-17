# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict

from .rag import init_rag, answer_with_rag


###########################################################
# 요청/응답 스키마 (프론트 호환)
###########################################################

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class AskSource(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict]


###########################################################
# FastAPI 초기 설정
###########################################################

app = FastAPI(title="PolicyBot API (RAG Version)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 필요하면 도메인 특정 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###########################################################
# 서버 시작할 때 RAG 초기화
###########################################################

@app.on_event("startup")
def startup_event():
    # Railway 환경 변수에서 DB_URL을 가져오거나, config.py에서 구성
    import os

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("환경 변수 DATABASE_URL이 없습니다.")

    # PostgreSQL connection parsing
    from urllib.parse import urlparse
    parsed = urlparse(db_url)

    db_config = {
        "host": parsed.hostname,
        "port": parsed.port,
        "database": parsed.path[1:],
        "user": parsed.username,
        "password": parsed.password,
    }

    print("=== RAG 초기화 시작 ===")
    init_rag(db_config)
    print("=== RAG 초기화 완료 ===")


###########################################################
# 건강 체크
###########################################################

@app.get("/health")
def health():
    return {"status": "ok"}


###########################################################
# Ask 엔드포인트 (프론트에서 질문)
###########################################################

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):

    if not req.question or req.question.strip() == "":
        raise HTTPException(400, "질문이 비어 있습니다.")

    try:
        print(f"[ASK] 질문 수신: {req.question}")

        # RAG 호출
        result = answer_with_rag(req.question)

        # result = {"answer": "...", "sources": [chunk_id1, chunk_id2 ...]}

        # sources 가 chunk_id 리스트라면 다음과 같은 형태로 변환
        # → 프론트엔드 호환 위해 id만 전달
        sources = [{"id": s, "title": None, "score": None} for s in result["sources"]]

        return AskResponse(
            answer=result["answer"],
            sources=sources
        )

    except Exception as e:
        print(f"[ERROR] ask 처리 중 오류: {e}")
        raise HTTPException(500, "서버 오류가 발생했습니다.")
