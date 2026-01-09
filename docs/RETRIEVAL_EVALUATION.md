# Retrieval 평가 가이드

## 개요

ChromaDB 벡터 스토어의 검색 품질을 측정하는 Hit Rate 평가 시스템입니다.
벡터 검색, BM25 키워드 검색, 하이브리드 검색(Vector + BM25 + RRF + Reranker)을 비교 평가할 수 있습니다.

## 평가 방식

### Hit Rate @ K
- 질문에 대해 상위 K개 검색 결과 중 정답 법령 조항이 포함되어 있는지 확인
- 정답률 = (정답 문제 수 / 전체 문제 수) × 100%

### 매칭 로직
**엄격 매칭(Strict Matching)** 사용:
```
패턴: 제N조\s*[\(①②③④⑤⑥⑦⑧⑨⑩]
```
- `제17조(인과관계)` - 매칭 O
- `제17조 ①항목은...` - 매칭 O
- `제4조에 따라 제17조를...` - 매칭 X (참조는 제외)

## 검색 모드

### 1. Vector (기본값)
- ChromaDB 벡터 검색 (Dense Retrieval)
- 의미적 유사성 기반 검색

### 2. BM25
- Kiwi 토크나이저 기반 BM25 검색 (Sparse Retrieval)
- 키워드 매칭 기반 검색
- 한국어 형태소 분석 적용

### 3. Hybrid
- Vector + BM25 + RRF (Reciprocal Rank Fusion)
- Cross-Encoder 리랭커 적용 (BAAI/bge-reranker-v2-m3)
- 가장 정확한 검색 결과 기대

## 사용법

```bash
# 벡터 검색 평가 (기본값)
python evaluate_retrieval.py

# BM25 검색 평가
python evaluate_retrieval.py --mode bm25

# 하이브리드 검색 평가
python evaluate_retrieval.py --mode hybrid

# 리랭커 없이 하이브리드 검색
python evaluate_retrieval.py --mode hybrid --no-reranker

# 모든 모드 비교 평가
python evaluate_retrieval.py --mode all --quiet

# top-5로 실행
python evaluate_retrieval.py --top-k 5

# 결과 저장
python evaluate_retrieval.py --save
```

### 옵션
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--eval-set` | `eval_set_retrieval.json` | 평가 데이터 경로 |
| `--top-k` | 3 | 검색할 문서 수 |
| `--mode` | `vector` | 검색 모드: vector, bm25, hybrid, all |
| `--no-reranker` | False | 하이브리드에서 리랭커 비활성화 |
| `--quiet` | False | 상세 출력 숨김 |
| `--save` | False | 결과를 JSON으로 저장 |

## 평가 데이터셋

### eval_set_retrieval.json
```json
{
  "metadata": {
    "sample_size": 100,
    "filter": "법령으로 시작하는 ground_truth만"
  },
  "items": [
    {
      "question": "질문 내용",
      "ground_truth": "양곡관리법 제22조에 따르면...",
      "law_name": "양곡관리법",
      "article": "22조",
      "doc_type": "statute",
      "doc_id": "HS_B_000001_QA_001"
    }
  ]
}
```

### 데이터 필터링 기준
1. ground_truth가 법령명으로 시작하는 항목만 포함
2. ChromaDB에 존재하는 법령만 포함
3. `law_name`, `article` 필드가 미리 추출되어 있음

## 출력 예시

### 단일 모드 출력
```
✅ [정답] 문제 1 (statute)
   - 질문: 영리를 목적으로 불법감청설비탐지업을 하고자 하는 자가...
   - 목표: 통신비밀보호법제10조의3 (매칭: 제10조의3)
   - 검색된 문서들:
     ⭐[1] [제10조의3(불법감청설비탐지업)] 제10조의3(불법감청설비탐지업)...
       [2] [제15조(벌칙)] 제15조(벌칙) ① 다음 각 호의 어느 하나에...
       [3] [제9조(통신제한조치)] 제9조(통신제한조치의 집행에 관한...
--------------------------------------------------

==================================================
📊 [최종 결과 리포트] - Vector
==================================================
총 평가 문제 수: 100개
✅ 정답 (Hit): 67개
❌ 오답 (Miss): 33개
🏆 Hit Rate @ 3: 67.00%

📈 [유형별 Hit Rate]
  statute: 67/100 (67.0%)
==================================================
```

### 모든 모드 비교 출력 (--mode all)
```
============================================================
📊 [검색 모드 비교 요약]
============================================================
모드                 Hit Rate        정답/전체
------------------------------------------------------------
Vector               67.00%          67/100
BM25                 58.00%          58/100
Hybrid               72.00%          72/100
============================================================
```

## 한계점

### 알려진 제약사항
1. **법령명 미포함**: ChromaDB 청크에 법령명 메타데이터가 없어 조항 번호로만 매칭
2. **동일 조항 구분 불가**: 다른 법률의 같은 조항 번호 구분 불가 (예: 형법 제17조 vs 민법 제17조)
3. **데이터 불일치**: eval_set은 전체 데이터 기반, ChromaDB는 샘플 데이터만 포함

### 평가 결과 해석 시 주의
- **절대적 정확도 지표가 아님**: 상대적 비교 용도로 활용
- **개선 효과 측정 용도**: 임베딩 모델 변경, 청킹 전략 변경 시 전후 비교에 유용
- **심각한 문제 탐지**: Hit Rate가 극단적으로 낮으면 시스템 문제 신호

## 관련 파일

```
├── evaluate_retrieval.py      # 메인 평가 스크립트
├── eval_set_retrieval.json    # 평가 데이터셋
└── src/
    ├── vectorstore.py         # ChromaDB 벡터 스토어
    └── retriever.py           # BM25, RRF, Reranker, HybridRetriever
```

## 의존성

하이브리드 검색을 사용하려면 추가 패키지 설치 필요:
```bash
pip install rank-bm25 kiwipiepy
```
