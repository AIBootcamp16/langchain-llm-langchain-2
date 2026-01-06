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
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

load_dotenv()


@dataclass
class QAItem:
    """QA 데이터 아이템"""
    question: str
    ground_truth: str
    doc_type: str  # judgement, decision, statute, interpretation
    doc_id: str


class QADataLoader:
    """Training QA 데이터 로더"""

    def __init__(self, qa_base_path: str = "data/Training/2_labeled_data"):
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
                n_results=5,
                use_hybrid=use_hybrid
            )

            questions.append(item.question)
            answers.append(result["answer"])
            ground_truths.append(item.ground_truth)

            # 컨텍스트 수집 (검색된 문서 내용)
            context_list = []
            if "sources" in result:
                # RAGChain의 결과에서 컨텍스트 추출
                # sources에는 doc_id만 있으므로, 실제 content를 가져와야 함
                # 임시로 빈 컨텍스트 처리 (실제 구현 시 수정 필요)
                pass

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

        # 결과 정리
        scores = {
            "faithfulness": float(results["faithfulness"]),
            "answer_relevancy": float(results["answer_relevancy"]),
            "context_precision": float(results["context_precision"]),
            "context_recall": float(results["context_recall"]),
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
    qa_base_path: str = "data/Training/2_labeled_data",
    sample_size: int = 50,
    doc_types: List[str] = None,
    compare: bool = False
):
    """
    평가 실행 헬퍼 함수

    Args:
        rag_chain: RAGChain 인스턴스
        qa_base_path: QA 데이터 경로
        sample_size: 평가 샘플 크기
        doc_types: 평가할 문서 타입 (None이면 전체)
        compare: 벡터 vs 하이브리드 비교 여부
    """
    # QA 데이터 로드
    qa_loader = QADataLoader(qa_base_path)

    print("=== QA 데이터 통계 ===")
    stats = qa_loader.get_stats()
    for doc_type, count in stats.items():
        print(f"  {doc_type}: {count}개")

    qa_items = qa_loader.load_qa_files(doc_types=doc_types, max_per_type=sample_size)

    if not qa_items:
        print("평가할 QA 데이터가 없습니다.")
        return

    # 평가기 생성
    evaluator = RAGASEvaluator(rag_chain)

    if compare:
        # 비교 평가
        results = evaluator.compare_methods(qa_items, sample_size=sample_size)
    else:
        # 단일 평가
        results = evaluator.evaluate(qa_items, sample_size=sample_size)
        print("\n=== 평가 결과 ===")
        for metric, score in results["scores"].items():
            print(f"  {metric}: {score:.4f}")

    return results


if __name__ == "__main__":
    # 테스트용 실행
    print("=== RAGAS 평가 시스템 테스트 ===\n")

    # QA 데이터 로더 테스트
    qa_loader = QADataLoader()

    print("QA 데이터 통계:")
    stats = qa_loader.get_stats()
    for doc_type, count in stats.items():
        print(f"  {doc_type}: {count}개")

    # 샘플 로드 테스트
    print("\n샘플 QA 로드 (타입별 5개):")
    samples = qa_loader.load_qa_files(max_per_type=5)

    if samples:
        print(f"\n첫 번째 샘플:")
        print(f"  질문: {samples[0].question[:100]}...")
        print(f"  정답: {samples[0].ground_truth[:100]}...")
        print(f"  타입: {samples[0].doc_type}")
