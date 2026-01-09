"""
고급 검색 파이프라인
- BM25 + Kiwi 토크나이저 (한국어 형태소 분석)
- RRF (Reciprocal Rank Fusion) 하이브리드 검색
- Cross-Encoder 리랭커
"""

import os
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

# 리랭커용 (선택적)
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


class KiwiBM25Retriever:
    """Kiwi 토크나이저 기반 BM25 검색기"""

    def __init__(self, documents: List[Dict[str, Any]] = None):
        """
        Args:
            documents: 문서 리스트 [{"content": str, "metadata": dict}, ...]
        """
        self.kiwi = Kiwi()
        self.documents = documents or []
        self.bm25 = None
        self.tokenized_corpus = []

        if documents:
            self.index(documents)

    def _tokenize(self, text: str) -> List[str]:
        """Kiwi를 사용한 한국어 토큰화 (명사, 동사, 형용사 추출)"""
        tokens = []
        result = self.kiwi.tokenize(text)

        # 의미 있는 품사만 추출 (NNG: 일반명사, NNP: 고유명사, VV: 동사, VA: 형용사)
        meaningful_tags = {'NNG', 'NNP', 'NNB', 'VV', 'VA', 'MAG', 'SL', 'SH', 'SN'}

        for token in result:
            if token.tag in meaningful_tags and len(token.form) > 1:
                tokens.append(token.form)

        return tokens

    def index(self, documents: List[Dict[str, Any]]):
        """문서 인덱싱"""
        self.documents = documents

        print(f"BM25 인덱싱 중... ({len(documents)}개 문서)")
        self.tokenized_corpus = [
            self._tokenize(doc["content"]) for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 인덱싱 완료")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 검색"""
        if not self.bm25:
            raise ValueError("문서가 인덱싱되지 않았습니다. index()를 먼저 호출하세요.")

        # 쿼리 토큰화
        tokenized_query = self._tokenize(query)

        # BM25 스코어 계산
        scores = self.bm25.get_scores(tokenized_query)

        # Top-K 인덱스
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 스코어가 0보다 큰 경우만
                results.append({
                    "id": self.documents[idx]["metadata"].get("chunk_id", str(idx)),
                    "content": self.documents[idx]["content"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": float(scores[idx]),
                    "retriever": "bm25"
                })

        return results


class RRFRetriever:
    """RRF (Reciprocal Rank Fusion) 하이브리드 검색기"""

    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF 상수 (기본값 60, 논문 권장값)
        """
        self.k = k

    def fuse(
        self,
        results_list: List[List[Dict[str, Any]]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        여러 검색 결과를 RRF로 병합

        Args:
            results_list: 검색 결과 리스트들 [[result1, result2, ...], [result1, result2, ...]]
            top_k: 반환할 결과 수

        Returns:
            RRF 스코어로 정렬된 병합 결과
        """
        rrf_scores = defaultdict(float)
        doc_map = {}  # id -> 문서 정보 저장

        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc["id"]

                # RRF 스코어 계산: 1 / (k + rank)
                rrf_scores[doc_id] += 1.0 / (self.k + rank)

                # 문서 정보 저장 (처음 등장한 정보 유지)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # RRF 스코어로 정렬
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 결과 구성
        fused_results = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = rrf_scores[doc_id]
            fused_results.append(doc)

        return fused_results


class Reranker:
    """Cross-Encoder 기반 리랭커"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Args:
            model_name: Cross-Encoder 모델명
                - BAAI/bge-reranker-v2-m3 (다국어, 권장)
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (영어)
        """
        if not RERANKER_AVAILABLE:
            raise ImportError(
                "Reranker를 사용하려면 sentence-transformers가 필요합니다. "
                "pip install sentence-transformers"
            )

        print(f"리랭커 모델 로드 중: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name
        print("리랭커 모델 로드 완료")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        문서 리랭킹

        Args:
            query: 검색 쿼리
            documents: 리랭킹할 문서 리스트
            top_k: 반환할 상위 문서 수

        Returns:
            리랭킹된 문서 리스트
        """
        if not documents:
            return []

        # Query-Document 쌍 생성
        pairs = [(query, doc["content"]) for doc in documents]

        # Cross-Encoder 스코어 계산
        scores = self.model.predict(pairs)

        # 스코어와 함께 정렬
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 상위 K개 반환
        reranked = []
        for doc, score in scored_docs[:top_k]:
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            reranked.append(doc_copy)

        return reranked


class HybridRetriever:
    """
    하이브리드 검색기: 벡터 검색 + BM25 + RRF + 리랭킹

    파이프라인:
    1. 벡터 검색 (Dense) - 의미적 유사성
    2. BM25 검색 (Sparse) - 키워드 매칭
    3. RRF 병합 - 두 결과 통합
    4. 리랭킹 (선택) - 최종 순위 조정
    """

    def __init__(
        self,
        vectorstore,
        documents: List[Dict[str, Any]] = None,
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        rrf_k: int = 60
    ):
        """
        Args:
            vectorstore: VectorStore 인스턴스 (Dense 검색용)
            documents: BM25 인덱싱용 문서 리스트
            use_reranker: 리랭커 사용 여부
            reranker_model: 리랭커 모델명
            rrf_k: RRF 상수
        """
        self.vectorstore = vectorstore
        self.bm25_retriever = KiwiBM25Retriever(documents) if documents else None
        self.rrf = RRFRetriever(k=rrf_k)
        self.reranker = None

        if use_reranker:
            try:
                self.reranker = Reranker(model_name=reranker_model)
            except Exception as e:
                print(f"리랭커 로드 실패: {e}. 리랭킹 없이 진행합니다.")

    def index_bm25(self, documents: List[Dict[str, Any]]):
        """BM25 인덱스 생성"""
        if self.bm25_retriever is None:
            self.bm25_retriever = KiwiBM25Retriever()
        self.bm25_retriever.index(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        use_reranker: bool = True,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 최종 반환 문서 수
            vector_top_k: 벡터 검색 후보 수
            bm25_top_k: BM25 검색 후보 수
            use_reranker: 리랭킹 사용 여부
            filter_type: 문서 타입 필터 (judgement, decision, statute, interpretation)

        Returns:
            검색 결과 리스트
        """
        results_list = []

        # 1. 벡터 검색 (Dense)
        vector_results = self.vectorstore.search(
            query=query,
            n_results=vector_top_k,
            filter_type=filter_type
        )

        # 포맷 통일
        formatted_vector_results = []
        for r in vector_results:
            formatted_vector_results.append({
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "score": 1 - r["distance"],  # distance를 similarity로 변환
                "retriever": "vector"
            })
        results_list.append(formatted_vector_results)

        # 2. BM25 검색 (Sparse)
        if self.bm25_retriever and self.bm25_retriever.bm25:
            bm25_results = self.bm25_retriever.search(query, top_k=bm25_top_k)

            # 타입 필터 적용
            if filter_type:
                bm25_results = [
                    r for r in bm25_results
                    if r["metadata"].get("type") == filter_type
                ]

            results_list.append(bm25_results)

        # 3. RRF 병합
        if len(results_list) > 1:
            fused_results = self.rrf.fuse(results_list, top_k=top_k * 2)
        else:
            fused_results = results_list[0][:top_k * 2]

        # 4. 리랭킹 (선택적)
        if use_reranker and self.reranker and fused_results:
            final_results = self.reranker.rerank(query, fused_results, top_k=top_k)
        else:
            final_results = fused_results[:top_k]

        return final_results

    def search_vector_only(
        self,
        query: str,
        n_results: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """벡터 검색만 수행 (기존 방식)"""
        return self.vectorstore.search(query, n_results, filter_type)

    def search_bm25_only(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """BM25 검색만 수행"""
        if not self.bm25_retriever:
            raise ValueError("BM25 인덱스가 생성되지 않았습니다.")
        return self.bm25_retriever.search(query, top_k)


# 편의 함수
def create_hybrid_retriever(
    vectorstore,
    documents: List[Dict[str, Any]] = None,
    use_reranker: bool = True
) -> HybridRetriever:
    """하이브리드 검색기 생성 헬퍼 함수"""
    return HybridRetriever(
        vectorstore=vectorstore,
        documents=documents,
        use_reranker=use_reranker
    )


if __name__ == "__main__":
    # 테스트
    print("=== BM25 + Kiwi 토크나이저 테스트 ===")

    # 테스트 문서
    test_docs = [
        {"content": "폭행죄는 형법 제260조에 규정되어 있으며, 타인의 신체에 대하여 폭행을 가한 자는 2년 이하의 징역에 처한다.", "metadata": {"chunk_id": "doc1", "type": "statute"}},
        {"content": "사기죄는 타인을 기망하여 재물을 편취하는 범죄로, 형법 제347조에 규정되어 있다.", "metadata": {"chunk_id": "doc2", "type": "statute"}},
        {"content": "피고인은 피해자를 주먹으로 때려 폭행하였으므로 폭행죄가 성립한다.", "metadata": {"chunk_id": "doc3", "type": "judgement"}},
    ]

    # BM25 검색기 테스트
    bm25_retriever = KiwiBM25Retriever(test_docs)
    results = bm25_retriever.search("폭행죄 처벌", top_k=2)

    print("\nBM25 검색 결과:")
    for r in results:
        print(f"  - {r['id']}: {r['content'][:50]}... (score: {r['score']:.4f})")

    print("\n=== RRF 테스트 ===")
    rrf = RRFRetriever()

    # 가상의 두 검색 결과
    vector_results = [
        {"id": "doc1", "content": "문서1", "metadata": {}, "score": 0.9},
        {"id": "doc2", "content": "문서2", "metadata": {}, "score": 0.8},
        {"id": "doc3", "content": "문서3", "metadata": {}, "score": 0.7},
    ]
    bm25_results = [
        {"id": "doc3", "content": "문서3", "metadata": {}, "score": 10.5},
        {"id": "doc1", "content": "문서1", "metadata": {}, "score": 8.2},
        {"id": "doc4", "content": "문서4", "metadata": {}, "score": 5.1},
    ]

    fused = rrf.fuse([vector_results, bm25_results], top_k=3)
    print("RRF 병합 결과:")
    for r in fused:
        print(f"  - {r['id']}: RRF score = {r['rrf_score']:.4f}")
