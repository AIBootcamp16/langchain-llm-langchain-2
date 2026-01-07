# RAGAS 평가 시스템 가이드

## 실행 방법

```bash
cd src
python evaluation.py
```

## 필수 환경 변수

`.env` 파일에 다음 키가 필요합니다:

### RAGAS 평가용 (필수)
| 키 | 용도 | 필수 |
|----|------|------|
| `OPENAI_API_KEY` | RAGAS 평가용 LLM (gpt-4o-mini) | **O** |

### RAG 답변 생성용 (하나만 선택)
| 키 | 용도 | 비고 |
|----|------|------|
| `OPENROUTER_API_KEY` | OpenRouter (무료 모델 사용 가능) | 우선순위 1 |
| `OPENAI_API_KEY` | OpenAI GPT | 우선순위 2 |
| `SOLAR_API_KEY` | Upstage Solar | 우선순위 3 |

> **참고**: RAG는 위 키 중 하나만 있으면 자동으로 감지합니다.
> OpenAI 키만 있어도 RAG + RAGAS 평가 모두 가능합니다.

## 주의사항

### 1. OpenAI API 키 관련

- **시스템 환경변수 주의**: 터미널에 이미 `OPENAI_API_KEY`가 설정되어 있으면 `.env` 파일보다 우선됩니다
- 키 오류 시 확인:
  ```bash
  echo $OPENAI_API_KEY  # 시스템 환경변수 확인
  ```
- 다른 키가 설정되어 있다면 터미널을 새로 열거나:
  ```bash
  unset OPENAI_API_KEY
  ```

### 2. 평가 소요 시간

- **50개 샘플 기준**: 약 10-15분 소요
- RAG 실행 (50개) + RAGAS 평가 (4개 메트릭 × 50개)
- 중간에 중단하면 처음부터 다시 실행해야 함

### 3. 비용

- RAGAS 평가: **gpt-4o-mini** 사용 (저렴)
- 50개 샘플 평가 시 약 **$0.01 ~ $0.05**
- OpenAI 대시보드에서 Usage 확인 (반영에 1-2시간 소요)

### 4. 평가 세트

- `eval_set.json`: 고정된 50개 샘플 (프로젝트 루트)
- 동일한 샘플로 평가해야 결과 비교 가능
- 새 평가 세트 생성:
  ```bash
  python evaluation.py --test  # 데이터 확인만
  ```

## 평가 결과 해석

| 메트릭 | 설명 | 좋은 점수 |
|--------|------|-----------|
| **faithfulness** | 답변이 검색 결과에 충실한지 | > 0.8 |
| **answer_relevancy** | 답변이 질문에 적절한지 | > 0.8 |
| **context_precision** | 검색된 문서의 정확도 | > 0.9 |
| **context_recall** | 필요한 정보를 찾았는지 | > 0.8 |
| **overall** | 종합 점수 | > 0.8 |

## 현재 성능 (2025-01-07)

```
faithfulness: 0.7541
answer_relevancy: 0.6966
context_precision: 0.9492
context_recall: 0.8937
overall: 0.8234
```

## 문제 해결

### 401 Authentication Error
```
AuthenticationError: Incorrect API key provided
```
→ 시스템 환경변수에 다른 OpenAI 키가 있는지 확인

### LLM returned 1 generations instead of 3
→ 정상 작동 중. 무시해도 됨 (경고일 뿐)

### ModuleNotFoundError
→ 필요한 패키지 설치:
```bash
pip install ragas sentence-transformers langchain-openai
```
