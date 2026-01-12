"""
LangGraph workflow (minimal + safe retry)
- retrieve node: VectorStore.search로 컨텍스트 수집
- generate node: (question + context)로 답변 생성
- hallucination_check node: 답변이 컨텍스트 근거 기반인지 검사
- regenerate_strict node: 근거에 없는 내용은 제거하고 1회만 보수적으로 재생성

구성:
retrieve -> generate -> hallucination_check -> (grounded면 END, 아니면 regenerate_strict 1회) -> hallucination_check -> END
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional

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
    documents: List[Dict[str, Any]]
    context: str

    # generation 결과
    answer: str

    # hallucination check 결과
    grounded: bool
    issues: List[str]
    final: str

    # retry 제어
    retry_count: int
    max_retries: int


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

    return {"documents": docs, "context": ctx}


def generate_node(state: GraphState, *, rag_chain: RAGChain) -> GraphState:
    question = state["question"]
    context = state.get("context", "")

    if not context.strip():
        return {"answer": "관련 문서를 찾을 수 없습니다."}

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

    grounded = True
    issues: List[str] = []

    try:
        import json

        cleaned = raw.strip()
        # ```json ... ``` 방어
        if cleaned.startswith("```"):
            cleaned = cleaned.strip().strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        data = json.loads(cleaned)
        grounded = bool(data.get("grounded", False))
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
    except Exception:
        grounded = False
        issues = ["검증 JSON 파싱 실패(모델 출력 형식 불일치).", f"raw={raw[:300]}..."]

    if grounded:
        final = answer
    else:
        warn = "\n".join([f"- {x}" for x in issues]) if issues else "- 근거 부족/환각 가능성"
        final = (
            answer
            + "\n\n⚠️ 근거 기반 검증에서 문제가 감지되었습니다. 아래 항목을 확인하세요:\n"
            + warn
        )

    return {"grounded": grounded, "issues": issues, "final": final}


def regenerate_strict_node(state: GraphState, *, rag_chain: RAGChain) -> GraphState:
    """
    환각 감지 후, 근거에만 기반해서 보수적으로 1회 재생성
    """
    question = state["question"]
    context = state.get("context", "")
    retry_count = int(state.get("retry_count", 0))

    if not context.strip():
        return {"answer": "관련 문서를 찾을 수 없습니다.", "retry_count": retry_count + 1}

    strict_prompt = f"""
너는 법률 AI다.
아래 제공된 문서 내용에 **명시적으로 포함된 내용만** 사용해서 답변하라.
추론, 일반화, 추정은 금지한다.
근거가 부족하면 반드시 '답변 불가'라고 말하라.

[질문]
{question}

[근거 문서]
{context}
""".strip()

    resp = rag_chain.llm.invoke(strict_prompt)
    answer = getattr(resp, "content", str(resp)).strip()

    return {"answer": answer, "retry_count": retry_count + 1}


# -----------------------------
# 4) Graph 조립
# -----------------------------
def decide_after_hallucination(state: GraphState) -> str:
    if state.get("grounded", False):
        return "end"

    if int(state.get("retry_count", 0)) < int(state.get("max_retries", 0)):
        return "regenerate_strict"

    return "end"


def build_workflow(vectorstore: VectorStore, rag_chain: RAGChain):
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", lambda s: retrieve_node(s, vectorstore=vectorstore))
    graph.add_node("generate", lambda s: generate_node(s, rag_chain=rag_chain))
    graph.add_node("hallucination_check", lambda s: hallucination_check_node(s, rag_chain=rag_chain))
    graph.add_node("regenerate_strict", lambda s: regenerate_strict_node(s, rag_chain=rag_chain))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "hallucination_check")

    # ✅ 조건 분기 (핵심)
    graph.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination,
        {
            "regenerate_strict": "regenerate_strict",
            "end": END,
        },
    )

    # strict 재생성 후 다시 검증
    graph.add_edge("regenerate_strict", "hallucination_check")

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
        "retry_count": 0,
        "max_retries": 1,  # strict regenerate 1회만 허용
    }
    return app.invoke(state)
