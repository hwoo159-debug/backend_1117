# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AskRequest, AskResponse
from .rag import init_rag, answer_with_rag

app = FastAPI(title="PolicyBot RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # 서버 시작 시 RAG 한 번 초기화
    init_rag()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        print(f"[ASK] 질문 수신: {req.question}")
        result = answer_with_rag(req.question)
        return AskResponse(
            answer=result["answer"],
            sources=[{"id": s} for s in result["sources"]],
        )
    except Exception as e:
        print(f"[ERROR] /ask 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")
