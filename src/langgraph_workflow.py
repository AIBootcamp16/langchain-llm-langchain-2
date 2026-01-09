# src/langgraph_workflow.py
"""
LangGraph workflow (minimal)
- retrieve node: VectorStore.search로 컨텍스트 수집
- generate node: RAGChain의 LLM으로 답변 생성
- hallucination_check node: 답변이 컨텍스트 근거 기반인지 간단 검사

주의:
- 지금은 "복잡한 루프/에이전트" 없이, 3노드 직선 + 검사만 구성합니다.
- 이후에 원하면: (검사 Fail -> query rewrite -> retrieve 재시도) 루프를 붙이면 됩니다.
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional, Literal

from langgraph.graph import StateGraph, END

from src.vectorstore import VectorStore
from src.rag_chain import RAGChain


# -----------------------------
# 1) State 정의
# -----------------------------
class GraphState(TypedDict, total=False):
    question: str
    n_results: int
    filter_type: Optional[str]

    # retrieval 결과
    documents: List[Dict[str, Any]]  # VectorStore.search 결과 리스트
    context: str  # LLM에 넣을 컨텍스트(문자열)

    # generation 결과
    answer: str

    # hallucination check 결과
    grounded: bool
    issues: List[str]
    final: str  # 최종 출력(안전장치 반영)


# -----------------------------
# 2) Helper: context formatting
# -----------------------------
def _format_context(search_results: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for i, r in enumerate(search_results, 1):
        md = r.get("metadata", {}) or {}
        type_name = md.get("type_name", "문서")
        doc_id = md.get("doc_id", "unknown")
        dist = r.get("distance", None)
        dist_txt = f" (dist={dist:.4f})" if isinstance(dist, float) else ""
        parts.append(f"[문서 {i}] ({type_name}) - {doc_id}{dist_txt}\n{r.get('content','')}")
    return "\n---\n".join(parts)


# -----------------------------
# 3) Node 구현
# -----------------------------
def retrieve_node(state: GraphState, *, vectorstore: VectorStore) -> GraphState:
    question = state["question"]
    n_results = int(state.get("n_results", 5))
    filter_type = state.get("filter_type")

    docs = vectorstore.search(question, n_results=n_results, filter_type=filter_type)
    ctx = _format_context(docs) if docs else ""

    return {
        "documents": docs,
        "context": ctx,
    }


def generate_node(state: GraphState, *, rag_chain: RAGChain) -> GraphState:
    """
    기존 RAGChain.query()를 그대로 쓰면 또 내부에서 retrieval을 다시 합니다.
    그래서 여기서는 RAGChain의 llm만 재사용해서,
    (question + state.context)로 답변을 한 번에 생성합니다.
    """
    question = state["question"]
    context = state.get("context", "")

    if not context.strip():
        return {
            "answer": "관련 문서를 찾을 수 없습니다.",
        }

    system = rag_chain.system_prompt.strip()
    prompt = f"""{system}

아래 [컨텍스트]만 근거로 답변하세요.
- 컨텍스트에 없는 내용은 추측하지 말고 '근거 부족'이라고 말하세요.
- 가능하면 조문/사건번호를 함께 제시하세요.

[질문]
{question}

[컨텍스트]
{context}
"""

    resp = rag_chain.llm.invoke(prompt)
    answer = getattr(resp, "content", str(resp)).strip()

    return {"answer": answer}


def hallucination_check_node(state: GraphState, *, rag_chain: RAGChain) -> GraphState:
    """
    매우 단순한 근거성 검사.
    - 컨텍스트에 근거 없는 단정(형량/조문/사실)이 섞였는지 점검
    - Fail이면 경고를 붙여서 final에 반영
    """
    question = state["question"]
    context = state.get("context", "")
    answer = state.get("answer", "")

    if not context.strip():
        return {
            "grounded": False,
            "issues": ["검색 컨텍스트가 비어 있어 근거 기반 답변이 불가능합니다."],
            "final": answer,
        }

    judge_prompt = f"""
너는 '법률 RAG 답변 검증' 심사관이다.
아래 [컨텍스트]에 근거해서 [답변]이 과장/환각/근거 없는 단정을 포함하는지 검사하라.

판정 기준:
- 답변의 핵심 주장(처벌 수위, 조문 번호, 요건, 사실관계)이 컨텍스트에 명시적으로 뒷받침되는가?
- 컨텍스트에 없는 조문/형량/사실을 만들어내면 FAIL.
- 일부만 불명확하면, 어떤 부분이 근거 부족인지 issues에 적어라.

출력은 반드시 아래 JSON 형식만:
{{
  "grounded": true/false,
  "issues": ["...","..."]
}}

[질문]
{question}

[컨텍스트]
{context}

[답변]
{answer}
""".strip()

    resp = rag_chain.llm.invoke(judge_prompt)
    raw = getattr(resp, "content", str(resp)).strip()

    # JSON 파싱을 최대한 안전하게
    grounded = True
    issues: List[str] = []

    try:
        import json

        # 코드블록 제거(혹시 모를 경우)
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.strip().strip("`")
            # ```json ... ``` 같은 형태면 앞줄 제거
            cleaned = cleaned.replace("json", "", 1).strip()

        data = json.loads(cleaned)
        grounded = bool(data.get("grounded", False))
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
    except Exception:
        # 파싱 실패 시 보수적으로 Fail 처리
        grounded = False
        issues = ["검증 JSON 파싱 실패(모델 출력 형식 불일치).", f"raw={raw[:300]}..."]

    if grounded:
        final = answer
    else:
        warn = "\n".join([f"- {x}" for x in issues]) if issues else "- 근거 부족/환각 가능성"
        final = (
            answer
            + "\n\n"
            + "⚠️ 근거 기반 검증에서 문제가 감지되었습니다. 아래 항목을 확인하세요:\n"
            + warn
        )

    return {
        "grounded": grounded,
        "issues": issues,
        "final": final,
    }


# -----------------------------
# 4) Graph 조립
# -----------------------------
def build_workflow(
    vectorstore: VectorStore,
    rag_chain: RAGChain,
):
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", lambda s: retrieve_node(s, vectorstore=vectorstore))
    graph.add_node("generate", lambda s: generate_node(s, rag_chain=rag_chain))
    graph.add_node("hallucination_check", lambda s: hallucination_check_node(s, rag_chain=rag_chain))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "hallucination_check")
    graph.add_edge("hallucination_check", END)

    return graph.compile()


def run_workflow(
    question: str,
    vectorstore: VectorStore,
    rag_chain: RAGChain,
    n_results: int = 5,
    filter_type: Optional[str] = None,
) -> GraphState:
    app = build_workflow(vectorstore=vectorstore, rag_chain=rag_chain)
    state: GraphState = {
        "question": question,
        "n_results": n_results,
        "filter_type": filter_type,
    }
    return app.invoke(state)
