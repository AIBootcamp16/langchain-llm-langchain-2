"""
Retrieval 정확도 평가 스크립트
- eval_set.json 기반으로 검색 Hit Rate 측정
- 정답(ground_truth)에 포함된 법령 조항이 검색 결과에 있는지 확인
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vectorstore import VectorStore


@dataclass
class EvalSample:
    """평가 샘플"""
    question: str
    ground_truth: str
    target_law: str  # 법령명+조항 (예: "형법제17조")
    target_article: str  # 조항만 (예: "17조")
    doc_type: str
    doc_id: str


def load_eval_set(eval_set_path: str = "eval_set_retrieval.json") -> List[EvalSample]:
    """
    eval_set_retrieval.json 파일을 로드하고 평가 샘플 생성
    (law_name, article 필드가 이미 추출되어 있는 데이터셋 사용)

    Args:
        eval_set_path: eval_set_retrieval.json 파일 경로
    """
    print(f"📂 평가 데이터 로드 중: {eval_set_path}")

    with open(eval_set_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = data.get("items", [])
    samples = []
    skipped = 0

    for item in items:
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        doc_type = item.get("doc_type", "")
        doc_id = item.get("doc_id", "")
        law_name = item.get("law_name", "")
        article = item.get("article", "")

        if not question or not law_name or not article:
            skipped += 1
            continue

        # law_name + 제 + article 형태로 target_law 생성
        target_law = f"{law_name}제{article}"
        target_article = f"제{article}"  # 조항만 (예: "제17조")

        samples.append(EvalSample(
            question=question,
            ground_truth=ground_truth,
            target_law=target_law,
            target_article=target_article,
            doc_type=doc_type,
            doc_id=doc_id
        ))

    print(f"  총 {len(items)}개 중 {len(samples)}개 평가 가능 (스킵: {skipped}개)")
    return samples


def run_evaluation(
    vectorstore: VectorStore,
    samples: List[EvalSample],
    top_k: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Retrieval 평가 실행

    Args:
        vectorstore: VectorStore 인스턴스
        samples: 평가 샘플 리스트
        top_k: 검색할 문서 개수
        verbose: 오답 상세 출력 여부
    """
    print(f"\n{'='*50}")
    print(f"🚀 Retrieval 평가 시작 (top_k={top_k})")
    print(f"{'='*50}\n")

    total_count = len(samples)
    correct_count = 0
    errors = []

    # doc_type별 통계
    type_stats = {}

    for i, sample in enumerate(samples):
        # 검색 수행
        try:
            results = vectorstore.search(sample.question, n_results=top_k)
        except Exception as e:
            print(f"❌ [ERROR] 문제 {i+1} 검색 중 에러: {e}")
            continue

        # Hit 여부 확인
        # ChromaDB 청크에는 법령명이 없을 수 있으므로 조항 번호로 매칭
        is_hit = False
        hit_doc_idx = -1
        retrieved_contents = []

        for idx, doc in enumerate(results):
            content = doc.get("content", "")

            # content에서 "제N조(제목)" 패턴 추출
            article_match = re.search(r"(제\d+조(?:의\d+)?)\s*\(([^)]+)\)", content)
            if article_match:
                article_info = f"{article_match.group(1)}({article_match.group(2)})"
            else:
                # 제N조만이라도 추출
                simple_match = re.search(r"(제\d+조(?:의\d+)?)", content)
                article_info = simple_match.group(1) if simple_match else "조항없음"

            # 조항 정보 + 내용 미리보기
            preview = f"[{article_info}] {content[:50]}..."
            retrieved_contents.append(preview)

            # 엄격 매칭: "제N조(제목)" 또는 "제N조 ①" 형태로 조항이 시작하는 경우만 인정
            # 단순히 "제4조에 따라..." 같은 참조는 제외
            strict_pattern = rf"{re.escape(sample.target_article)}\s*[\(①②③④⑤⑥⑦⑧⑨⑩]"
            if not is_hit and re.search(strict_pattern, content):
                is_hit = True
                hit_doc_idx = idx

        # doc_type별 통계 업데이트
        if sample.doc_type not in type_stats:
            type_stats[sample.doc_type] = {"total": 0, "correct": 0}
        type_stats[sample.doc_type]["total"] += 1

        if is_hit:
            correct_count += 1
            type_stats[sample.doc_type]["correct"] += 1
            if verbose:
                print(f"✅ [정답] 문제 {i+1} ({sample.doc_type})")
                print(f"   - 질문: {sample.question[:80]}...")
                print(f"   - 목표: {sample.target_law} (매칭: {sample.target_article})")
                print(f"   - 검색된 문서들:")
                for j, content in enumerate(retrieved_contents):
                    marker = "⭐" if j == hit_doc_idx else "  "
                    print(f"     {marker}[{j+1}] {content}")
                print("-" * 50)
        else:
            errors.append({
                "index": i + 1,
                "question": sample.question,
                "target": sample.target_law,
                "doc_type": sample.doc_type,
                "retrieved": retrieved_contents
            })

            if verbose and len(errors) <= 10:  # 처음 10개만 출력
                print(f"❌ [오답] 문제 {i+1} ({sample.doc_type})")
                print(f"   - 질문: {sample.question[:80]}...")
                print(f"   - 목표: {sample.target_law} (매칭: {sample.target_article})")
                print(f"   - 검색된 것: {retrieved_contents[0] if retrieved_contents else 'None'}")
                print("-" * 50)

    # 결과 계산
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    # 최종 리포트
    print(f"\n{'='*50}")
    print(f"📊 [최종 결과 리포트]")
    print(f"{'='*50}")
    print(f"총 평가 문제 수: {total_count}개")
    print(f"✅ 정답 (Hit): {correct_count}개")
    print(f"❌ 오답 (Miss): {total_count - correct_count}개")
    print(f"🏆 Hit Rate @ {top_k}: {accuracy:.2f}%")

    # doc_type별 결과
    print(f"\n📈 [유형별 Hit Rate]")
    for doc_type, stats in sorted(type_stats.items()):
        type_acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"  {doc_type}: {stats['correct']}/{stats['total']} ({type_acc:.1f}%)")

    print(f"{'='*50}")

    return {
        "total": total_count,
        "correct": correct_count,
        "errors": len(errors),
        "hit_rate": accuracy,
        "top_k": top_k,
        "type_stats": type_stats,
        "error_details": errors[:20]  # 처음 20개 오답만 저장
    }


def save_results(results: Dict, output_path: str = "retrieval_eval_results.json"):
    """결과를 JSON 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장됨: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval 평가")
    parser.add_argument("--eval-set", type=str, default="eval_set_retrieval.json",
                        help="평가 데이터 경로 (기본: eval_set_retrieval.json)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="검색할 문서 수")
    parser.add_argument("--quiet", action="store_true",
                        help="오답 상세 출력 안함")
    parser.add_argument("--save", action="store_true",
                        help="결과를 JSON으로 저장")
    args = parser.parse_args()

    samples = load_eval_set(args.eval_set)

    if not samples:
        print("평가할 샘플이 없습니다.")
        sys.exit(1)

    print("\n🔧 VectorStore 초기화 중...")
    vs = VectorStore()

    results = run_evaluation(
        vectorstore=vs,
        samples=samples,
        top_k=args.top_k,
        verbose=not args.quiet
    )

    if args.save:
        save_results(results)
