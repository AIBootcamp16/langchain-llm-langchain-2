#!/usr/bin/env python3
"""
RAGAS 평가 실행 스크립트

사용법:
    python scripts/run_evaluation.py --sample-size 50
    python scripts/run_evaluation.py --sample-size 100 --doc-types judgement statute
    python scripts/run_evaluation.py --compare  # 벡터 vs 하이브리드 비교
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vectorstore import VectorStore
from rag_chain import RAGChain
from evaluation import QADataLoader, RAGASEvaluator, run_evaluation


def main():
    parser = argparse.ArgumentParser(description="RAGAS RAG 평가 실행")
    parser.add_argument(
        "--sample-size", "-n",
        type=int,
        default=50,
        help="평가 샘플 크기 (기본값: 50)"
    )
    parser.add_argument(
        "--doc-types", "-t",
        nargs="+",
        choices=["judgement", "decision", "statute", "interpretation"],
        default=None,
        help="평가할 문서 타입 (기본값: 전체)"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="벡터 검색 vs 하이브리드 검색 비교"
    )
    parser.add_argument(
        "--hybrid", "-H",
        action="store_true",
        help="하이브리드 검색 사용"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="결과 저장 경로 (CSV)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RAGAS RAG 평가 시스템")
    print("=" * 60)

    # 벡터 스토어 로드
    print("\n벡터 스토어 로드 중...")
    vectorstore = VectorStore()

    # RAG 체인 생성
    print("RAG 체인 생성 중...")
    rag = RAGChain(vectorstore)

    # 평가 실행
    results = run_evaluation(
        rag_chain=rag,
        qa_base_path="data/Training/2_labeled_data",
        sample_size=args.sample_size,
        doc_types=args.doc_types,
        compare=args.compare
    )

    # 결과 저장
    if args.output and results is not None:
        import pandas as pd
        if isinstance(results, pd.DataFrame):
            results.to_csv(args.output, index=False)
            print(f"\n결과 저장됨: {args.output}")
        elif isinstance(results, dict) and "scores" in results:
            df = pd.DataFrame([results["scores"]])
            df.to_csv(args.output, index=False)
            print(f"\n결과 저장됨: {args.output}")

    print("\n" + "=" * 60)
    print("평가 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
