"""
형사법 RAG 데이터 로더
- CSV 파일에서 판례, 결정문, 법령, 해석 데이터 로드
- 청킹 및 메타데이터 추출
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


class LegalDataLoader:
    """법률 데이터 로더"""

    def __init__(self, data_dir: str = "data_sampled"):
        self.data_dir = Path(data_dir)
        self.data_types = {
            "judgement": "판례",
            "decision": "결정문",
            "statute": "법령",
            "interpretation": "해석"
        }

        # judgement 인적사항/메타성 라인 감지용 (공백/특수문자 변형 대응)
        self.judgement_party_patterns = [
            r"【\s*피\s*고\s*인\s*】",
            r"【\s*항\s*소\s*인\s*】",
            r"【\s*검\s*사\s*】",
            r"【\s*검.*】",  # 예: 【검○】
            r"【\s*변\s*호\s*인\s*】",
            r"【\s*변호인\s*】",
        ]

        # decision 인적사항/메타성 라인 감지용 (공백/철자 변형 대응)
        self.decision_party_patterns = [
            r"청\s+구\s+인",
            r"피\s+청\s+구\s+인",
            r"피\s+청\s+구",
            r"변\s+호\s+사",
            r"재\s+판\s+장",
            r"재\s+판\s+관",
        ]

    def load_csv(self, file_path: Path) -> Dict[str, Any]:
        """단일 CSV 파일 로드 및 파싱"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')

        # 컬럼명 정리
        df.columns = df.columns.str.strip()

        # 문장번호 기준 정렬 (원문 흐름 복원)
        if '문장번호' in df.columns:
            df = df.sort_values('문장번호')

        # 문서 ID 추출
        doc_id = file_path.stem  # 파일명에서 확장자 제거

        # 내용 합치기 (+ party_lines 제거)
        party_lines = []
        content_lines = []

        # judgement/decision 분기용 (폴더명 기반)
        data_type = file_path.parent.name

        if '내용' in df.columns:
            for v in df['내용'].dropna().astype(str).tolist():
                text = v.strip()
                if not text:
                    continue

                is_party = False

                if data_type == "judgement":
                    for pat in self.judgement_party_patterns:
                        if re.search(pat, text):
                            is_party = True
                            break

                elif data_type == "decision":
                    for pat in self.decision_party_patterns:
                        if re.search(pat, text):
                            is_party = True
                            break

                if is_party:
                    party_lines.append(text)
                else:
                    content_lines.append(text)
        else:
            # 모든 컬럼의 데이터를 합침
            content_lines = df.iloc[:, -1].dropna().astype(str).tolist()

        content = '\n'.join(content_lines)

        # 메타데이터 추출
        metadata = {
            "doc_id": doc_id,
            "file_path": str(file_path),
            "party_lines": party_lines
        }

        # 구분 정보가 있으면 추출 (문자열로 변환)
        if '구분' in df.columns:
            sections = df['구분'].unique().tolist()
            metadata["sections"] = ", ".join(str(s) for s in sections)

        return {
            "content": content,
            "metadata": metadata
        }

    def load_all(self) -> List[Dict[str, Any]]:
        """모든 데이터 로드"""
        documents = []

        for data_type, type_name in self.data_types.items():
            type_dir = self.data_dir / data_type

            if not type_dir.exists():
                print(f"[경고] {type_dir} 폴더가 없습니다.")
                continue

            csv_files = list(type_dir.glob("*.csv"))
            print(f"\n[{type_name}] {len(csv_files)}개 파일 로드 중...")

            for file_path in tqdm(csv_files, desc=type_name):
                try:
                    doc = self.load_csv(file_path)
                    doc["metadata"]["type"] = data_type
                    doc["metadata"]["type_name"] = type_name
                    documents.append(doc)
                except Exception as e:
                    print(f"[에러] {file_path}: {e}")

        print(f"\n총 {len(documents)}개 문서 로드 완료")
        return documents

    def chunk_document(self, doc: Dict[str, Any], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """문서를 청크로 분할"""
        content = doc["content"]
        metadata = doc["metadata"]

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]

            # 문장 단위로 자르기 (마지막 마침표 찾기)
            if end < len(content):
                last_period = chunk_text.rfind('.')
                if last_period > chunk_size // 2:
                    chunk_text = chunk_text[:last_period + 1]
                    end = start + last_period + 1

            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_idx"] = chunk_idx
            chunk_metadata["chunk_id"] = f"{metadata['doc_id']}_chunk_{chunk_idx}"

            chunks.append({
                "content": chunk_text.strip(),
                "metadata": chunk_metadata
            })

            start = end - overlap
            chunk_idx += 1

        return chunks

    def load_and_chunk(self, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """데이터 로드 및 청킹"""
        documents = self.load_all()

        all_chunks = []
        print(f"\n문서 청킹 중 (chunk_size={chunk_size}, overlap={overlap})...")

        for doc in tqdm(documents, desc="청킹"):
            chunks = self.chunk_document(doc, chunk_size, overlap)
            all_chunks.extend(chunks)

        print(f"총 {len(all_chunks)}개 청크 생성 완료")
        return all_chunks


if __name__ == "__main__":
    # 테스트
    loader = LegalDataLoader("data_sampled")
    chunks = loader.load_and_chunk()

    # 샘플 출력
    if chunks:
        print("\n=== 샘플 청크 ===")
        sample = chunks[0]
        print(f"ID: {sample['metadata']['chunk_id']}")
        print(f"타입: {sample['metadata']['type_name']}")
        print(f"내용 (앞 200자): {sample['content'][:200]}...")
