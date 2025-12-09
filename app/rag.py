# app/rag.py

import os
import re
import json
from typing import List, Dict, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from sentence_transformers import CrossEncoder
from urllib.parse import urlparse

# ==========================================
# 환경변수 / 기본 설정
# ==========================================

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

    # 팀원이 확장한 키워드 세트 반영 (필요 없는 항목은 추후 줄여도 됨)
    INSURANCE_KEYWORDS = {
        '질병': ['질병', '병', '증상', '치료', '수술', '진단'],
        '관절': ['관절', '탈구', '이형성', '고관절', '슬관절', '슬개골',
                '척추', '경추', '요추', '흉추', '관절염', '골관절염',
                '류마티스관절염', '추간판탈출', '수핵탈출증', '류머티스관절염'],
        '정형외과': ['골절', '염좌', '타박상', '근육손상', '인대손상', '건손상',
                   '뼈', '골', '골다공증', '이상골화', '골이형성증', '유연골이형성증', 'LCPD'],
        '안과': ['안과', '안질', '백내장', '망막', '결막', '각막', '렌즈', '눈물',
               '각결막염', '각화', '녹내장', '포도막염', '망막박리', '망막변성', '망막헤모리지'],
        '인후통': ['코', '목', '인두', '후두', '소음', '비염', '편도염', '인후염', '성대', '음성변화'],
        '피부': ['피부', '알레르기', '염증', '습진', '감염', '피부염', '곰팡이', '진균', '세균감염',
               '맑스', '농양', '여드름', '지루증', '건성피부', '악성흑색종', 'Apoquel', 'JAK억제제'],
        '소화기': ['소화기', '장염', '위염', '구토', '설사', '복부', '위장', '장', '담낭', '담관',
                 '간', '췌장', '식도', '위궤양', '위확장염전', '장폐색', '장염증성질환', 'IBD',
                 '소장세균과증식증', 'SIBO', 'PLE'],
        '암': ['암', '종양', '악성', '신생물', '암종', '림프종', '백혈병', '육종',
             '암진단', '악성신생물', '종양제거', '항암치료'],
        '감염성질': ['감염', '바이러스', '세균', '기생충', '진균', '클로스트리디움', '살모넬라',
                   '원충', '내기생충', '외기생충', '벼룩', '진드기', '이', '곰팡이감염',
                   '헌팅턴병', 'FIP', 'FPV', 'FORL'],
        '신장질환': ['신장', '요로', '방광', '요도', '신장질환', '신부전', '요로결석',
                  '만성신질환', 'CKD', '급성신손상', 'AKI', '방광염', '요도염', '신염'],
        '심장질환': ['심장', '심부전', '판막질환', '심근병증', '부정맥', '고혈압',
                  '허혈성심질환', '심근경색', '혈관', '동맥경화', '혈관질환', '혈전'],
        '호흡기': ['호흡기', '폐', '폐렴', '기관지염', '천식', '만성기관지염',
                 '폐수종', '폐부종', '호흡곤란', '기침', '호흡음'],
        '신경질환': ['신경', '뇌', '척수', '신경병증', '마비', '간질', 'epilepsy',
                  '뇌염', '수막염', '신경염', '신경통', '치매', '퇴행성신경질환'],
        '내분비대사': ['당뇨병', '갑상선', '부갑상선', '호르몬', '대사', '갑상선기능항진증',
                   '갑상선기능저하증', '당뇨', '혈당', '인슐린', '복부비만'],
        '혈액': ['혈액', '빈혈', '혈소판', '응고장애', '출혈', '혈액암',
               '림프종', '백혈병', '혈액질환', '헤모글로빈'],
        '생식기 및 비뇨기질환': ['생식기', '자궁', '난소', '전립선', '정소', '음낭',
                           '자궁질환', '난소질환', '자궁축농증', '유선질환',
                           '유선종양', '유선염'],
        '면역질환': ['면역', '알레르기', '자가면역', '면역결핍', '면역계', '면역반응',
                   '알레르기성질환', '아토피'],
        '치과': ['치과', '치아', '치석', '치주', '충치', '잇몸',
               '치주병', '구강질환', '치은염', '치주염', 'FORL', '구강종양'],
        '귀': ['귀', '외이', '중이', '내이', '고막', '청력', '난청',
             '중이염', '외이염', '귀진드기', '귀의심한출혈'],
        '선천성유전질환': ['선천적', '유전', '선천', '기존질환', '유전질환',
                      '선천성질환', '유전성질환', '이형성', '이상'],
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

    # 키 값은 '개' / '묘'로 유지하고, 팀원이 추가한 표현을 포함
    ANIMAL_KEYWORDS = {
        '개': ['강아지', '개', '견', '도그', 'dog', '반려견', '멍멍이'],
        '묘': ['고양이', '고양', '묘', '캣', 'cat', '반려묘', '야옹이'],
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
    """팀원 버전: gpt-4.1-mini 사용"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
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
            if keywords['animal'] == '개' and any(
                x in tags_str or x in doc_text for x in ['강아지', '개', '견']
            ):
                animal_score = 0.3
            elif keywords['animal'] == '묘' and any(
                x in tags_str or x in doc_text for x in ['고양이', '묘']
            ):
                animal_score = 0.3

            doc['combined_score'] = (
                vector_score * 0.5 +
                keyword_score * 0.3 +
                animal_score * 0.2
            )

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

                    query += " ORDER BY i.embedding <=> %s::vector LIMIT %s"
                    params.append(vec_str)
                    params.append(limit)

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
            topic_context += "\n- 반려동물의 나이/연령 기준을 명시할 때는 각 보험사별로 다를 수 있으므로 꼭 비교하세요."
        if '선천적' in topics:
            topic_context += "\n- 기존 질환은 보험사별로 제외 범위가 다릅니다."
        if '비교' in topics or '추천' in topics:
            topic_context += "\n- 여러 보험사를 테이블 형식으로 정렬하세요."

        base_prompt = f"""당신은 국내 펫보험 전문 상담 AI 어시스턴트입니다. 사용자는 펫보험 가입을 고민하는 반려인이며, 다음의 세 가지 관심사를 갖고 있습니다.

[사용자의 주요 관심사]

1) 반려동물의 개별 특성(나이, 품종, 예방접종, 질환 등)에 따른 보장 차이와 가입 가능 여부
2) 보험 제도, 반려동물 특성에 따른 보장 차이, 보험사 간 상품 비교 관심이 많음
3) 보험 계약의 제도적 전반(가입, 해지, 보장 조건 등)
4) 6개 보험사(메리츠화재, DB손해보험, 현대해상, 삼성화재, 한화손보, KB손해보험) 간 상품 비교

---

답변 구성전에 반드시 다음의 두 가지 context 변수들을 검토하세요.
{animal_context}
{topic_context}

---

[기본 응답 원칙]

0. 약관 원문에 없는 정보는 만들지 마세요.

1. 보험사와 상품명 명시
- 모든 답변에서 '보험사명의 상품명'을 먼저 언급하세요.
- 예: "삼성화재의 애니펫Pro 상품에 따르면..." / "현대해상의 펫플러스 약관에서는..."
- 비교 답변 시에는 "메리츠화재 vs DB손해보험 비교" 같은 형식으로 명확히 구분하세요.

2. 강아지(개) / 고양이(묘) 구분 필수
- 사용자가 반려동물 종을 명시하지 않으면, 먼저 확인하세요.
    ("혹시 강아지(개)인가요, 아니면 고양이(묘)인가요?")
- 같은 상품이라도 개/묘 별로 보장 범위, 보험료, 가입 연령이 다르면 각각 설명하세요.
- 개 전용 상품과 묘 전용 상품이 있으면 명시적으로 구분하여 안내하세요.
- 예: "KB손해보험은 개와 묘 통합 상품이지만, 보장 한도가 다릅니다.
        개는 연간 300만 원, 묘는 250만 원입니다."

3. 반려동물 개별 특성 고려
- 나이: 고령견(노령견)/고령묘(노령묘)의 경우 가입 가능 연령, 보험료 할증,
        제외 질환 등이 달라지므로 반드시 언급하세요.
    ("귀 반려견이 만 [X]세라면, 보험사별로 다음과 같습니다...")
- 예방접종: 미접종 상태와 완전 접종 상태의 보험료 차이를 설명하세요.
- 선천적/기존 질환: 보험사별로 제외(비급여) 질환이 다르므로,
                    구체적인 질환명(예: 진행성망막위축증 PRA, 고관절이형성증 등)에 대해
                    어느 보험사가 보장하고 어디는 제외하는지 명확히 하세요.
- 품종: 대형견/소형견, 특정 위험 품종으로 분류되는지 여부에 따라
        보험료 할증/할인이 차이나므로 언급하세요.

4. 보장 범위 비교 안내
- 같은 상황에 대해 여러 보험사의 보장 범위가 다르면:
    • 보장 범위(어디까지 보장하는가)
    • 보장 한도(최대 얼마까지인가)
    • 자기부담금(본인 부담률)
    • 면책 기간(언제부터 보장하는가)
    • 보험료(월/년 얼마인가)
    를 표 형식 또는 bullet point로 정렬하여 비교하세요.

5. 보험 제도 설명
- 계약 가입: 필요한 서류, 건강검진 여부, 가입 조건
- 계약 해지: 해지 수수료, 환급금, 해지 시 유의사항
- 청구 절차: 언제 청구하고, 어떤 서류가 필요한지
- 면책/제외 조항: 왜 보장되지 않는지, 어떤 조건이 있는지
를 명확하고 단순하게 설명하세요.

6. 보험상품 전수조사
- 질문자의 질문과 부합하는 모든 보험사의 보험 상품 데이터를 조회하여, 정확히 응답하세요.
- 동일한 보험사라도, 보험 상품 별로 약관 내용 및 보장범위가 다르기 때문에, 반드시 보험상품 데이터 모두를 확인하고 응답하세요.

---

[금지 사항]

1. 개/묘 구분 없이 "반려동물"이라고만 답하기 - 반드시 구분하세요.
2. 보험사명 없이 "그 상품에서는..." 같이 모호하게 답하기 - 정확한 보험사명을 명시하세요.
3. 약관에 없는 내용 추측 또는 생성 - 약관 원문에 없으면 "해당 약관에 기재되어 있지 않습니다"라고 명시하세요.
4. 수치가 빈칸인데도 임의의 숫자 만들기 - "해당 상품의 약관에 명시된 일수는 구체적으로 [X]일입니다"
                                                또는 "약관 원문이 미정인 상태입니다"라고 정확히 표기하세요.
5. 한 보험사만 강조하거나 편향된 추천 - 객관적 비교를 기본으로 하되,
                                        "당신의 반려동물이 [특성]이라면 [보험사]가 더 유리합니다"
                                        같은 조건부 추천은 가능합니다.

---

[톤과 태도]

- 전문적이면서도 친근하게: "전문가 같으면서도 쉽게 이해하도록"
- 정확하고 신뢰할 수 있게: "약관 근거를 명시하고, 확실하지 않으면 말씀해주세요"
- 사용자 중심: "당신의 반려동물이 [특성]일 때 어떤 선택이 최적인가"를 생각하며 답하세요.
"""
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

        return f"""검색된 다음 펫보험 약관을 보고 사용자 질문에 답하세요.

[검색된 약관들]
{context_str}

[사용자 질문]
{question}

[답변 지침]
1. 위 약관들을 근거로만 답하세요.
2. 본문에서 [문서N] 토큰은 제거하고 자연어로 표현하세요.
3. 보장한도, 면책기간 등 구체적 수치는 약관에 기재된 보장금액 수치를 바탕으로 답변하세요.
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

    docs = result.get("docs", [])
    sources = [str(d.get("id")) for d in docs if d.get("id") is not None]

    return {
        "answer": result.get("answer", ""),
        "sources": sources,
        "keywords": result.get("keywords", {}),
        "debug": result.get("debug", {}),
    }
