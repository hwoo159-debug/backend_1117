# -*- coding: utf-8 -*-
"""
RAG + Reranker + Prompt (배포용 정리 버전)

- OpenAI 임베딩 + gpt-4o 사용
- 보험사 다양화 Reranker
- FastAPI 백엔드와 호환:
    - init_rag(limit: int = 0) -> bool
    - answer_with_rag(question: str) -> dict {answer, sources, ...}
"""

import os
import re
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from sentence_transformers import CrossEncoder
from urllib.parse import urlparse

# ==========================================
# 환경변수 / 기본 설정
# ==========================================

# 반드시 Railway에 설정할 것:
#   - DATABASE_URL
#   - OPENAI_API_KEY
#
# (코드 안에 키/URL 절대 하드코딩 금지)

LOCAL_EMBED_DIM = int(os.getenv("LOCAL_EMBED_DIM", "1536"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ==========================================
# 키워드 추출기
# ==========================================

class KeywordExtractor:
    """사용자 질문에서 의도와 핵심 키워드를 추출"""

    INSURANCE_KEYWORDS = {
        '질병': ['질병', '병', '증상', '치료', '수술', '진단'],
        '관절': ['관절', '탈구', '이형성', '고관절', '슬관절', '슬개골'],
        '안과': ['안과', '안질', '백내장', '망막', '결막'],
        '피부': ['피부', '알레르기', '염증', '습진', '감염'],
        '소화기': ['소화기', '장염', '위염', '구토', '설사'],
        '암': ['암', '종양', '악성', '신생물'],
        '가입': ['가입', '기가입', '보장', '커버'],
        '해지': ['해지', '환급', '수수료', '해약'],
        '면책': ['면책', '면책기간', '대기기간', '관찰기간'],
        '나이': ['나이', '연령', '세', '고령', '노령', '나이제한'],
        '예방접종': ['예방접종', '백신', '접종'],
        '선천적': ['선천적', '유전', '선천', '기존질환'],
        '임신': ['임신', '임신중', '임신예정'],
        '보험료': ['보험료', '월료', '연료', '가격', '요금', '비용'],
        '한도': ['한도', '보장한도', '한도액', '최대'],
        '자기부담': ['자기부담', '본인부담', '부담금'],
        '보상': ['보상', '지급', '청구', '배상'],
        '비교': ['비교', '차이', '다른', '어느', '어디가'],
        '추천': ['추천', '추천해', '어떤게', '더 좋은', '넓은'],
    }

    ANIMAL_KEYWORDS = {
        '개': ['강아지', '개', '견', '도그', 'dog'],
        '묘': ['고양이', '고양', '묘', '캣', 'cat'],
    }

    def extract(self, question: str) -> Dict:
        result = {
            'animal': None,
            'intent': [],
            'keywords': [],
            'topics': []
        }

        q_lower = question.lower()

        # 동물 추출
        for animal, keywords in self.ANIMAL_KEYWORDS.items():
            if any(k in q_lower for k in keywords):
                result['animal'] = animal
                break

        # 주제/키워드 추출
        found_topics = set()
        for topic, keywords in self.INSURANCE_KEYWORDS.items():
            for k in keywords:
                if k in q_lower:
                    found_topics.add(topic)
                    result['keywords'].append(k)

        result['topics'] = list(found_topics)
        return result


# ==========================================
# 유틸리티
# ==========================================

def simple_embed(texts: List[str], dim: int = LOCAL_EMBED_DIM) -> np.ndarray:
    """OpenAI text-embedding-3-small 사용"""
    if not texts:
        return np.zeros((0, dim), dtype="float32")
    texts = [t.replace("\n", " ") for t in texts]
    try:
        res = client.embeddings.create(input=texts, model="text-embedding-3-small")
        return np.array([d.embedding for d in res.data], dtype="float32")
    except Exception as e:
        print(f"[Embed Error] {e}")
        return np.zeros((len(texts), dim), dtype="float32")


def clean_answer(answer: str) -> str:
    answer = re.sub(r'\[문서\d+\]', '', answer)
    answer = re.sub(r'\(\s*\)', '[정보없음]', answer)
    answer = re.sub(r'\s{2,}', ' ', answer)
    return answer.strip()


def call_llm_api_v2(system_prompt: str, user_prompt: str,
                    temperature: float = 0.3) -> str:
    """gpt-4o 호출"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"응답 실패: {e}"


# ==========================================
# Reranker (보험사 다양화)
# ==========================================

class ImprovedReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # sentence-transformers / torch 필요
        print(f"[Reranker] 모델 로딩 중... ({model_name})")
        self.model = CrossEncoder(model_name)
        self.keyword_extractor = KeywordExtractor()
        print("[Reranker] 로딩 완료")

    def rerank(self, question: str, docs: List[Dict],
               top_k: int = 6, max_per_company: int = 2) -> List[Dict]:
        """
        보험사 다양성을 보장하는 Reranking

        Args:
            question: 사용자 질문
            docs: 초기 검색 문서 리스트
            top_k: 최종 반환 문서 개수
            max_per_company: 보험사당 최대 선택 개수
        """
        if not docs:
            return []

        # 1. 키워드 추출
        keywords = self.keyword_extractor.extract(question)

        # 2. 각 문서에 점수 부여
        for doc in docs:
            # 벡터 유사도 점수
            vector_score = float(doc.get('similarity_score', 0.5))

            # 키워드 매칭 점수
            doc_text = (
                (doc.get('text') or '') +
                (doc.get('title') or '') +
                (doc.get('clause_type') or '')
            ).lower()

            keyword_matches = sum(1 for k in keywords['keywords'] if k in doc_text)
            keyword_score = min(keyword_matches * 0.15, 1.0)

            # 동물 종 매칭 점수
            animal_score = 0.0
            tags = doc.get('tags', [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags) if tags.startswith('[') else [tags]
                except Exception:
                    tags = [tags]

            tags_str = str(tags).lower()
            if keywords['animal']:
                if keywords['animal'] == '개' and any(
                    x in tags_str or x in doc_text for x in ['강아지', '개', '견']
                ):
                    animal_score = 0.3
                elif keywords['animal'] == '묘' and any(
                    x in tags_str or x in doc_text for x in ['고양이', '묘']
                ):
                    animal_score = 0.3

            doc['combined_score'] = (vector_score * 0.5) + (keyword_score * 0.3) + (animal_score * 0.2)

        # 3. 점수 기준 정렬
        sorted_docs = sorted(docs, key=lambda x: x.get('combined_score', 0), reverse=True)

        # 4. 보험사별 그룹화
        company_groups: Dict[str, List[Dict]] = {}
        for doc in sorted_docs:
            company = doc.get('company_name') or '미정'
            company_groups.setdefault(company, []).append(doc)

        # 5. 보험사별 다양화 선택
        result: List[Dict] = []
        for company, docs_list in company_groups.items():
            selected = docs_list[:max_per_company]
            result.extend(selected)

        # 6. 다시 점수 기준 정렬
        result = sorted(result, key=lambda x: x.get('combined_score', 0), reverse=True)

        # 7. 상위 top_k 반환
        return result[:top_k]


# ==========================================
# DB 커넥터
# ==========================================

class DatabaseConnector:
    """
    Postgres/pgvector 하이브리드 검색
    - insurance_clauses, companies, products 테이블 사용
    - DATABASE_URL 환경변수 사용
    """

    def __init__(self):
        raw_url = os.getenv("DATABASE_URL")
        if not raw_url:
            raise RuntimeError("환경변수 DATABASE_URL이 설정되어 있지 않습니다.")

        # postgres:// → postgresql:// 변환 (Railway 호환)
        if raw_url.startswith("postgres://"):
            raw_url = raw_url.replace("postgres://", "postgresql://", 1)

        parsed = urlparse(raw_url)
        self.url = raw_url
        print(f"[DB] host={parsed.hostname}, db={parsed.path.lstrip('/')}")

    def hybrid_search(self, query_vec: np.ndarray, filters: Dict = None,
                      limit: int = 20) -> List[Dict]:
        """
        pgvector + 메타데이터 필터 기반 검색
        """
        try:
            with psycopg2.connect(self.url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    vec_str = "[" + ",".join(map(str, query_vec[0])) + "]"

                    query = """
                        SELECT
                            i.id,
                            c.name as company_name,
                            p.name as product_name,
                            i.title,
                            i.clause_type,
                            i.coverage_limit,
                            i.coverage_ratio,
                            i.waiting_period,
                            i.tags,
                            COALESCE(i.search_text, i.original_text) as text,
                            1 - (i.embedding <=> %s::vector) as similarity_score
                        FROM insurance_clauses i
                        LEFT JOIN companies c ON i.company_id = c.company_id
                        LEFT JOIN products p ON i.product_id = p.product_id
                        WHERE 1=1
                    """
                    params = [vec_str]

                    if filters:
                        if filters.get("product_ids"):
                            query += " AND i.product_id = ANY(%s)"
                            params.append(filters["product_ids"])
                        if filters.get("company_ids"):
                            query += " AND i.company_id = ANY(%s)"
                            params.append(filters["company_ids"])

                    query += f" ORDER BY i.embedding <=> %s::vector LIMIT {limit}"
                    params.append(vec_str)

                    cur.execute(query, params)
                    rows = cur.fetchall()
                    print(f"[DB] hybrid_search rows={len(rows)}")
                    return rows
        except Exception as e:
            print(f"[DB Error] {e}")
            return []


# ==========================================
# 프롬프트 생성기
# ==========================================

class PromptGenerator:
    @staticmethod
    def get_system_prompt(keywords: Dict) -> str:
        animal_context = ""
        if keywords.get('animal') == '개':
            animal_context = "\n\n[주의] 사용자는 강아지(개)를 키우고 있습니다. 개 관련 정보만 우선 제공하세요."
        elif keywords.get('animal') == '묘':
            animal_context = "\n\n[주의] 사용자는 고양이(묘)를 키우고 있습니다. 묘 관련 정보만 우선 제공하세요."

        topic_context = ""
        topics = keywords.get('topics', [])
        if '나이' in topics:
            topic_context += "\n- 반려동물의 나이/연령 기준은 보험사별로 다를 수 있으므로 반드시 비교하세요."
        if '선천적' in topics:
            topic_context += "\n- 기존 질환, 선천성 질환은 보험사별로 제외 범위가 다릅니다."
        if '비교' in topics or '추천' in topics:
            topic_context += "\n- 여러 보험사를 표 형식으로 비교해서 설명하세요."

        base_prompt = f"""당신은 국내 펫보험 전문 상담 AI입니다.

[사용자 프로필]
- 펫보험 가입을 고민하는 반려인
- 보험 제도, 반려동물 특성에 따른 보장 차이, 보험사 간 상품 비교에 관심 있음
- 주요 보험사 예시: 메리츠화재, DB손해보험, 현대해상, 삼성화재, 한화손해보험, KB손해보험
{animal_context}
{topic_context}

[필수 원칙]
1. 반드시 '보험사명 - 상품명'을 명시하여 답하세요.
2. 약관 원문에 없는 정보는 임의로 만들지 마세요.
3. 개/묘를 구분하여 답하세요.
4. 보장 범위가 다르면 반드시 비교하여 설명하세요.
5. 정확하고 신뢰성 있게, 하지만 친절한 말투로 설명하세요."""
        return base_prompt

    @staticmethod
    def get_user_prompt(question: str, docs: List[Dict], keywords: Dict) -> str:
        context_items = []
        for i, doc in enumerate(docs, 1):
            company = doc.get('company_name') or '미정'
            product = doc.get('product_name') or '미정'
            title = doc.get('title') or ''
            clause_type = doc.get('clause_type') or ''
            coverage_limit = doc.get('coverage_limit') or '미정'
            coverage_ratio = doc.get('coverage_ratio') or '미정'
            waiting_period = doc.get('waiting_period') or '미정'
            text = (doc.get('text') or '')[:1000]
            combined_score = float(doc.get('combined_score', 0.0))

            context_items.append(
                f"""[문서{i}] {company} - {product} (점수: {combined_score:.2f})
제목: {title}
유형: {clause_type}
보장한도: {coverage_limit}
보장비율: {coverage_ratio}
면책기간: {waiting_period}
내용: {text}
---"""
            )

        context_str = "\n".join(context_items)

        return f"""[검색된 약관들]
{context_str}

[사용자 질문]
{question}

[답변 지침]
1. 위 약관들을 근거로만 답하세요.
2. 본문에서 [문서N] 토큰은 제거하고 자연스러운 문장으로 다시 표현하세요.
3. 보장한도, 면책기간 등 구체적 수치는 약관에 기재된 내용만 사용하세요.
4. 여러 보험사가 동시에 언급되는 경우, 차이점을 명확히 비교해서 설명하세요.
"""


# ==========================================
# 메인 봇 (배포용)
# ==========================================

class ImprovedPetInsuranceBot:
    def __init__(self):
        self.db = DatabaseConnector()
        self.reranker = ImprovedReranker()
        self.keyword_extractor = KeywordExtractor()
        self.prompt_generator = PromptGenerator()

    def ask(self, question: str, filters: Optional[Dict] = None) -> Dict:
        """질문 처리 파이프라인"""

        question = (question or "").strip()
        if not question:
            return {
                "question": question,
                "answer": "질문이 비어 있습니다.",
                "docs": [],
                "keywords": {},
                "debug": {}
            }

        # 1. 키워드 추출
        keywords = self.keyword_extractor.extract(question)

        if not filters:
            filters = {}

        # 2. 벡터 임베딩 + 검색
        q_vec = simple_embed([question])
        initial_docs = self.db.hybrid_search(q_vec, filters, limit=20)

        if not initial_docs:
            return {
                "question": question,
                "answer": "죄송합니다. 관련 약관 정보를 찾을 수 없습니다.",
                "docs": [],
                "keywords": keywords,
                "debug": {"initial_docs_count": 0}
            }

        # 3. Reranking (보험사 다양화)
        top_docs = self.reranker.rerank(
            question,
            initial_docs,
            top_k=6,
            max_per_company=2
        )

        # 4. 프롬프트 생성
        system_prompt = self.prompt_generator.get_system_prompt(keywords)
        user_prompt = self.prompt_generator.get_user_prompt(question, top_docs, keywords)

        # 5. LLM 호출
        answer = call_llm_api_v2(system_prompt, user_prompt, temperature=0.2)
        answer = clean_answer(answer)

        return {
            "question": question,
            "answer": answer,
            "docs": top_docs,
            "keywords": keywords,
            "debug": {
                "initial_docs_count": len(initial_docs),
                "final_docs_count": len(top_docs),
                "topics": keywords.get('topics', []),
                "animal": keywords.get('animal')
            }
        }


# ==========================================
# FastAPI 연동용 Singleton + 래퍼
# ==========================================

_bot_instance: Optional[ImprovedPetInsuranceBot] = None


def init_rag(limit: int = 0) -> bool:
    """
    FastAPI에서 앱 시작 시 호출되는 초기화 함수.
    - 시그니처만 맞추면 됨. 여기선 limit는 사용하지 않음.
    """
    global _bot_instance
    if _bot_instance is None:
        print("[RAG] ImprovedPetInsuranceBot 초기화 중...")
        _bot_instance = ImprovedPetInsuranceBot()
        print("[RAG] 초기화 완료")
    return True


def answer_with_rag(question: str) -> Dict:
    """
    FastAPI /ask 엔드포인트에서 호출되는 함수.
    - 기존 코드와 동일하게 'answer' 키를 포함한 dict 리턴.
    """
    if _bot_instance is None:
        raise RuntimeError("RAG 시스템이 초기화되지 않았습니다. init_rag()를 먼저 호출하세요.")

    result = _bot_instance.ask(question)

    # 기존 백엔드/프론트 호환을 위해 최소한 아래 구조 유지:
    docs = result.get("docs", [])
    sources = [str(d.get("id")) for d in docs if d.get("id") is not None]

    return {
        "answer": result.get("answer", ""),
        "sources": sources,
        "keywords": result.get("keywords", {}),
        "debug": result.get("debug", {}),
    }
