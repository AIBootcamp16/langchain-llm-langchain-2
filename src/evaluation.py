"""
RAGAS 평가 시스템
- RAG 파이프라인 품질 자동 측정
- Training QA 데이터 기반 평가
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

import pandas as pd
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)  # 환경변수 덮어쓰기 활성화

# RAGAS용 LLM 설정 (OpenAI 직접 사용)
ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )
)


# RAGAS용 커스텀 Embedding 클래스 (SentenceTransformer 사용)
class SentenceTransformerEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


ragas_embeddings = SentenceTransformerEmbeddings()

# 메트릭 인스턴스 생성 (LLM + Embedding 명시적 설정)
faithfulness = Faithfulness(llm=ragas_llm)
answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
context_precision = ContextPrecision(llm=ragas_llm)
context_recall = ContextRecall(llm=ragas_llm)

# 프로젝트 루트 경로 (src/ 상위 디렉토리)
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class QAItem:
    """QA 데이터 아이템"""
    question: str
    ground_truth: str
    doc_type: str  # judgement, decision, statute, interpretation
    doc_id: str


class QADataLoader:
    """Training QA 데이터 로더"""

    EVAL_SET_PATH = PROJECT_ROOT / "eval_set.json"  # 고정 평가 세트 경로

    def __init__(self, qa_base_path: str = None):
        if qa_base_path is None:
            qa_base_path = PROJECT_ROOT / "data" / "Training" / "2_labeled_data"
        self.qa_base_path = Path(qa_base_path)
        self.qa_folders = {
            "judgement": "TL_judgement_QA",
            "decision": "TL_decision_QA",
            "statute": "TL_statute_QA",
            "interpretation": "TL_interpretation_QA"
        }

    def load_qa_files(
        self,
        doc_types: List[str] = None,
        max_per_type: int = None
    ) -> List[QAItem]:
        """
        QA 파일 로드

        Args:
            doc_types: 로드할 문서 타입 리스트 (None이면 전체)
            max_per_type: 타입별 최대 로드 개수 (None이면 전체)

        Returns:
            QAItem 리스트
        """
        if doc_types is None:
            doc_types = list(self.qa_folders.keys())

        qa_items = []

        for doc_type in doc_types:
            if doc_type not in self.qa_folders:
                print(f"알 수 없는 문서 타입: {doc_type}")
                continue

            folder_path = self.qa_base_path / self.qa_folders[doc_type]
            if not folder_path.exists():
                print(f"폴더 없음: {folder_path}")
                continue

            files = list(folder_path.glob("*.json"))

            if max_per_type and len(files) > max_per_type:
                files = random.sample(files, max_per_type)

            for file_path in files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    qa_items.append(QAItem(
                        question=data["label"]["input"],
                        ground_truth=data["label"]["output"],
                        doc_type=doc_type,
                        doc_id=data["info"].get("precedId", file_path.stem)
                    ))
                except Exception as e:
                    print(f"파일 로드 실패 {file_path}: {e}")

        print(f"총 {len(qa_items)}개 QA 로드 완료")
        return qa_items

    def get_stats(self) -> Dict[str, int]:
        """QA 데이터 통계"""
        stats = {}
        for doc_type, folder_name in self.qa_folders.items():
            folder_path = self.qa_base_path / folder_name
            if folder_path.exists():
                stats[doc_type] = len(list(folder_path.glob("*.json")))
            else:
                stats[doc_type] = 0
        return stats

    def create_eval_set(
        self,
        sample_size: int = 50,
        doc_types: List[str] = None,
        seed: int = 42,
        output_path: Path = None
    ) -> List[QAItem]:
        """
        고정 평가 세트 생성 및 저장

        Args:
            sample_size: 총 샘플 크기
            doc_types: 포함할 문서 타입 (None이면 전체)
            seed: 랜덤 시드 (재현성 보장)
            output_path: 저장 경로 (None이면 기본 경로)

        Returns:
            생성된 QAItem 리스트
        """
        random.seed(seed)
        output_path = output_path or self.EVAL_SET_PATH

        # 전체 QA 로드
        all_items = self.load_qa_files(doc_types=doc_types, max_per_type=None)

        if len(all_items) <= sample_size:
            sampled = all_items
        else:
            sampled = random.sample(all_items, sample_size)

        # JSON으로 저장
        eval_data = {
            "metadata": {
                "sample_size": len(sampled),
                "seed": seed,
                "doc_types": doc_types or list(self.qa_folders.keys()),
                "created_at": pd.Timestamp.now().isoformat()
            },
            "items": [
                {
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                    "doc_type": item.doc_type,
                    "doc_id": item.doc_id
                }
                for item in sampled
            ]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

        print(f"고정 평가 세트 생성 완료: {output_path}")
        print(f"  - 샘플 수: {len(sampled)}")
        print(f"  - 시드: {seed}")

        return sampled

    def load_eval_set(self, eval_path: Path = None) -> List[QAItem]:
        """
        저장된 고정 평가 세트 로드

        Args:
            eval_path: 평가 세트 경로 (None이면 기본 경로)

        Returns:
            QAItem 리스트
        """
        eval_path = eval_path or self.EVAL_SET_PATH

        if not eval_path.exists():
            raise FileNotFoundError(
                f"평가 세트 파일이 없습니다: {eval_path}\n"
                f"먼저 create_eval_set()으로 생성하세요."
            )

        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = [
            QAItem(
                question=item["question"],
                ground_truth=item["ground_truth"],
                doc_type=item["doc_type"],
                doc_id=item["doc_id"]
            )
            for item in data["items"]
        ]

        print(f"고정 평가 세트 로드: {eval_path}")
        print(f"  - 샘플 수: {len(items)}")
        print(f"  - 생성일: {data['metadata'].get('created_at', 'N/A')}")

        return items

    def has_eval_set(self, eval_path: Path = None) -> bool:
        """평가 세트 존재 여부 확인"""
        eval_path = eval_path or self.EVAL_SET_PATH
        return eval_path.exists()


class RAGASEvaluator:
    """RAGAS 기반 RAG 평가기"""

    def __init__(self, rag_chain):
        """
        Args:
            rag_chain: RAGChain 인스턴스
        """
        self.rag_chain = rag_chain
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def _run_rag_and_collect(
        self,
        qa_items: List[QAItem],
        use_hybrid: bool = False
    ) -> Dict[str, List]:
        """
        RAG 실행 및 결과 수집

        Args:
            qa_items: 평가할 QA 아이템들
            use_hybrid: 하이브리드 검색 사용 여부

        Returns:
            RAGAS 평가용 데이터 딕셔너리
        """
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for item in tqdm(qa_items, desc="RAG 실행 중"):
            # RAG 쿼리 실행
            result = self.rag_chain.query(
                question=item.question,
                n_results=5
            )

            questions.append(item.question)
            answers.append(result["answer"])
            ground_truths.append(item.ground_truth)

            # 검색 결과에서 직접 컨텍스트 가져오기
            search_results = self.rag_chain.vectorstore.search(
                query=item.question,
                n_results=5
            )
            context_list = [r["content"] for r in search_results]
            contexts.append(context_list)

        return {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

    def evaluate(
        self,
        qa_items: List[QAItem],
        use_hybrid: bool = False,
        sample_size: int = None
    ) -> Dict[str, Any]:
        """
        RAGAS 평가 실행

        Args:
            qa_items: 평가할 QA 아이템들
            use_hybrid: 하이브리드 검색 사용 여부
            sample_size: 샘플 크기 (None이면 전체)

        Returns:
            평가 결과 딕셔너리
        """
        if sample_size and len(qa_items) > sample_size:
            qa_items = random.sample(qa_items, sample_size)

        print(f"\n=== RAGAS 평가 시작 ({len(qa_items)}개 샘플) ===")
        print(f"하이브리드 검색: {'ON' if use_hybrid else 'OFF'}")

        # RAG 실행 및 데이터 수집
        eval_data = self._run_rag_and_collect(qa_items, use_hybrid)

        # RAGAS Dataset 생성
        dataset = Dataset.from_dict(eval_data)

        # 평가 실행
        print("\nRAGAS 평가 중...")
        results = evaluate(
            dataset,
            metrics=self.metrics
        )

        # 결과 정리 (RAGAS 버전에 따라 결과 형식이 다름)
        def get_score(results, key):
            val = results[key]
            if isinstance(val, list):
                # 리스트면 평균 계산
                valid_vals = [v for v in val if v is not None and not (isinstance(v, float) and v != v)]
                return sum(valid_vals) / len(valid_vals) if valid_vals else 0.0
            return float(val)

        scores = {
            "faithfulness": get_score(results, "faithfulness"),
            "answer_relevancy": get_score(results, "answer_relevancy"),
            "context_precision": get_score(results, "context_precision"),
            "context_recall": get_score(results, "context_recall"),
        }

        # 종합 점수 계산
        scores["overall"] = sum(scores.values()) / len(scores)

        return {
            "scores": scores,
            "sample_size": len(qa_items),
            "use_hybrid": use_hybrid,
            "detailed_results": results.to_pandas().to_dict() if hasattr(results, 'to_pandas') else None
        }

    def compare_methods(
        self,
        qa_items: List[QAItem],
        sample_size: int = 50
    ) -> pd.DataFrame:
        """
        벡터 검색 vs 하이브리드 검색 비교 평가

        Args:
            qa_items: 평가할 QA 아이템들
            sample_size: 샘플 크기

        Returns:
            비교 결과 DataFrame
        """
        if sample_size and len(qa_items) > sample_size:
            qa_items = random.sample(qa_items, sample_size)

        print(f"\n=== 검색 방식 비교 평가 ({len(qa_items)}개 샘플) ===\n")

        # 벡터 검색 평가
        print("1. 벡터 검색 평가")
        vector_results = self.evaluate(qa_items, use_hybrid=False)

        # 하이브리드 검색 평가 (hybrid_retriever가 있는 경우만)
        if self.rag_chain.hybrid_retriever:
            print("\n2. 하이브리드 검색 평가")
            hybrid_results = self.evaluate(qa_items, use_hybrid=True)
        else:
            print("\n2. 하이브리드 검색: 비활성화됨 (hybrid_retriever 없음)")
            hybrid_results = None

        # 결과 비교 DataFrame
        comparison = {
            "Metric": list(vector_results["scores"].keys()),
            "Vector Search": list(vector_results["scores"].values()),
        }

        if hybrid_results:
            comparison["Hybrid Search"] = list(hybrid_results["scores"].values())
            comparison["Improvement"] = [
                hybrid_results["scores"][m] - vector_results["scores"][m]
                for m in vector_results["scores"].keys()
            ]

        df = pd.DataFrame(comparison)

        print("\n=== 비교 결과 ===")
        print(df.to_string(index=False))

        return df


def run_evaluation(
    rag_chain,
    qa_base_path: str = None,
    sample_size: int = 50,
    doc_types: List[str] = None,
    compare: bool = False,
    use_fixed_eval_set: bool = True,
    create_if_missing: bool = True
):
    """
    평가 실행 헬퍼 함수

    Args:
        rag_chain: RAGChain 인스턴스
        qa_base_path: QA 데이터 경로
        sample_size: 평가 샘플 크기 (고정 세트 생성 시에만 사용)
        doc_types: 평가할 문서 타입 (None이면 전체)
        compare: 벡터 vs 하이브리드 비교 여부
        use_fixed_eval_set: 고정 평가 세트 사용 여부 (기본 True)
        create_if_missing: 고정 세트가 없으면 생성 (기본 True)
    """
    qa_loader = QADataLoader(qa_base_path)

    print("=== QA 데이터 통계 ===")
    stats = qa_loader.get_stats()
    for doc_type, count in stats.items():
        print(f"  {doc_type}: {count}개")

    # 고정 평가 세트 사용
    if use_fixed_eval_set:
        if qa_loader.has_eval_set():
            qa_items = qa_loader.load_eval_set()
        elif create_if_missing:
            print("\n고정 평가 세트가 없습니다. 새로 생성합니다...")
            qa_items = qa_loader.create_eval_set(
                sample_size=sample_size,
                doc_types=doc_types
            )
        else:
            raise FileNotFoundError(
                "고정 평가 세트가 없습니다. create_eval_set()으로 먼저 생성하세요."
            )
    else:
        # 기존 방식 (랜덤 샘플링) - 비권장
        print("\n⚠️  랜덤 샘플링 사용 중 - 일관된 평가를 위해 고정 세트 사용을 권장합니다.")
        qa_items = qa_loader.load_qa_files(doc_types=doc_types, max_per_type=sample_size)

    if not qa_items:
        print("평가할 QA 데이터가 없습니다.")
        return

    # 평가기 생성
    evaluator = RAGASEvaluator(rag_chain)

    if compare:
        # 비교 평가 (고정 세트 전체 사용)
        results = evaluator.compare_methods(qa_items, sample_size=len(qa_items))
    else:
        # 단일 평가 (고정 세트 전체 사용)
        results = evaluator.evaluate(qa_items, sample_size=len(qa_items))
        print("\n=== 평가 결과 ===")
        for metric, score in results["scores"].items():
            print(f"  {metric}: {score:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS RAG 평가 시스템")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (데이터 로드만)")
    parser.add_argument("--compare", action="store_true", help="벡터 vs 하이브리드 비교")
    parser.add_argument("--sample-size", type=int, default=50, help="평가 샘플 크기")
    args = parser.parse_args()

    if args.test:
        # 테스트 모드: 데이터 로드만
        print("=== RAGAS 평가 시스템 테스트 ===\n")
        qa_loader = QADataLoader()

        print("QA 데이터 통계:")
        stats = qa_loader.get_stats()
        for doc_type, count in stats.items():
            print(f"  {doc_type}: {count}개")

        print("\n샘플 QA 로드 (타입별 5개):")
        samples = qa_loader.load_qa_files(max_per_type=5)

        if samples:
            print(f"\n첫 번째 샘플:")
            print(f"  질문: {samples[0].question[:100]}...")
            print(f"  정답: {samples[0].ground_truth[:100]}...")
            print(f"  타입: {samples[0].doc_type}")
    else:
        # 실제 평가 실행
        print("=== RAGAS RAG 평가 시작 ===\n")

        from vectorstore import VectorStore
        from rag_chain import RAGChain

        # VectorStore 및 RAGChain 초기화
        print("VectorStore 로딩 중...")
        vs = VectorStore()

        print("RAGChain 초기화 중...")
        rag = RAGChain(vs)
        print(f"  - LLM: {rag.provider} / {rag.model}")

        # 평가 실행
        print("\n평가 시작...")
        results = run_evaluation(
            rag_chain=rag,
            sample_size=args.sample_size,
            compare=args.compare
        )
