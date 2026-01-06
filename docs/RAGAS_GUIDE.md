# RAGAS 평가 시스템 사용 가이드

## 개요

RAGAS(RAG Assessment)는 RAG 파이프라인의 품질을 자동으로 측정하는 프레임워크입니다.

## 평가 지표

| 지표 | 측정 대상 | 설명 |
|------|----------|------|
| **Faithfulness** | 답변 품질 | 답변이 검색된 컨텍스트에 충실한가? (환각 여부) |
| **Answer Relevancy** | 답변 품질 | 답변이 질문에 적절히 대응하는가? |
| **Context Precision** | 검색 품질 | 검색된 문서 중 관련 문서가 상위에 있는가? |
| **Context Recall** | 검색 품질 | 정답에 필요한 정보가 검색되었는가? |

## 설치

```bash
pip install -r requirements.txt
```

필요한 패키지:
- `ragas>=0.1.0`
- `datasets>=2.14.0`

## 사용법

### 기본 평가

```bash
# 50개 샘플로 평가 (기본값)
python scripts/run_evaluation.py

# 샘플 크기 지정
python scripts/run_evaluation.py --sample-size 100
python scripts/run_evaluation.py -n 100
```

### 문서 타입별 평가

```bash
# 특정 문서 타입만 평가
python scripts/run_evaluation.py --doc-types judgement
python scripts/run_evaluation.py --doc-types judgement statute
python scripts/run_evaluation.py -t judgement decision statute interpretation
```

사용 가능한 문서 타입:
- `judgement`: 판례
- `decision`: 결정문
- `statute`: 법령
- `interpretation`: 해석

### 검색 방식 비교

```bash
# 벡터 검색 vs 하이브리드 검색 비교
python scripts/run_evaluation.py --compare
python scripts/run_evaluation.py -c
```

### 결과 저장

```bash
# CSV 파일로 결과 저장
python scripts/run_evaluation.py --output results.csv
python scripts/run_evaluation.py -o evaluation_results.csv
```

## Python 코드에서 사용

```python
from src.vectorstore import VectorStore
from src.rag_chain import RAGChain
from src.evaluation import QADataLoader, RAGASEvaluator, run_evaluation

# 1. RAG 시스템 초기화
vectorstore = VectorStore()
rag = RAGChain(vectorstore)

# 2. 간단한 평가 실행
results = run_evaluation(
    rag_chain=rag,
    sample_size=50,
    doc_types=["judgement", "statute"]
)

# 3. 상세 평가
qa_loader = QADataLoader("data/Training/2_labeled_data")
qa_items = qa_loader.load_qa_files(max_per_type=20)

evaluator = RAGASEvaluator(rag)
results = evaluator.evaluate(qa_items, sample_size=50)

print(f"Faithfulness: {results['scores']['faithfulness']:.4f}")
print(f"Answer Relevancy: {results['scores']['answer_relevancy']:.4f}")
print(f"Context Precision: {results['scores']['context_precision']:.4f}")
print(f"Context Recall: {results['scores']['context_recall']:.4f}")
print(f"Overall: {results['scores']['overall']:.4f}")
```

## QA 데이터 구조

평가에 사용되는 QA 데이터는 `data/Training/2_labeled_data/` 경로에 있습니다:

```
data/Training/2_labeled_data/
├── TL_judgement_QA/      # 판례 QA (~7,000개)
├── TL_decision_QA/       # 결정문 QA (~5,400개)
├── TL_statute_QA/        # 법령 QA (~35,000개)
└── TL_interpretation_QA/ # 해석 QA (~65개)
```

각 JSON 파일 구조:
```json
{
    "info": {
        "precedId": "100024",
        "caseName": "사건명",
        ...
    },
    "label": {
        "input": "질문 내용",
        "output": "정답 (Ground Truth)"
    }
}
```

## 평가 결과 해석

| 점수 범위 | 의미 |
|----------|------|
| 0.8 ~ 1.0 | 우수 |
| 0.6 ~ 0.8 | 양호 |
| 0.4 ~ 0.6 | 개선 필요 |
| 0.0 ~ 0.4 | 미흡 |

## 주의사항

1. **API 비용**: RAGAS 평가는 LLM API를 호출하므로 비용이 발생합니다.
2. **샘플 크기**: 처음에는 작은 샘플(10~20개)로 테스트 후 점차 늘리세요.
3. **평가 시간**: 샘플당 수 초가 소요됩니다. 100개 샘플 기준 약 5~10분.

## 활용 사례

1. **검색 방식 비교**: 벡터 검색 vs 하이브리드 검색 성능 비교
2. **파라미터 튜닝**: chunk_size, top_k 등 최적값 찾기
3. **모델 비교**: 다른 LLM 간 성능 비교
4. **회귀 테스트**: 코드 변경 후 성능 저하 여부 확인
