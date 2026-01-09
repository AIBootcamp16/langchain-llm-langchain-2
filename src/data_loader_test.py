import sys
from pathlib import Path
sys.path.append("src")

from data_loader import LegalDataLoader

def run_one(path: str):
    loader = LegalDataLoader("data_sampled")
    doc = loader.load_csv(Path(path))

    print("="*80)
    print("FILE:", path)
    print("doc_id:", doc["metadata"].get("doc_id"))
    print("type:", doc["metadata"].get("type"), doc["metadata"].get("type_name"))
    print("doc_type:", doc["metadata"].get("doc_type"))
    print("file_name:", doc["metadata"].get("file_name"))
    print("party_lines count:", len(doc["metadata"].get("party_lines", [])))
    print("-"*80)
    print("party_lines (first 10):")
    for line in doc["metadata"].get("party_lines", [])[:10]:
        print("  -", line)

    print("-"*80)
    content = doc["content"]
    print("content length:", len(content))
    print("content head (first 500 chars):")
    print(content[:500])
    print("="*80)

if __name__ == "__main__":
    # 샘플 경로들 중 하나를 골라서 테스트
    
    # judgement
    # run_one("data_sampled/judgement/HS_P_64524.csv")
    
    # decision
    # run_one("data_sampled/decision/HS_K_730.csv")
    
    # statute
    # run_one("data_sampled/statute/HS_B_000006.csv")
    
    # interpretation
    # run_one("data_sampled/interpretation/HS_H_311551.csv")
    # run_one("data_sampled/interpretation/HS_H_312229.csv")
