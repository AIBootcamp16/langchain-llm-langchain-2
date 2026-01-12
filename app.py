"""
Justi-Q Streamlit 프론트엔드
형사법 RAG 시스템 웹 인터페이스
"""

import sys
import os
sys.path.append("src")

from dotenv import load_dotenv
load_dotenv()

# LangSmith 설정 (Streamlit 임포트 전에 설정)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "justi-q"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

import streamlit as st
from vectorstore import VectorStore
from rag_chain import RAGChain

# ✅ LangGraph workflow 추가
from langgraph_workflow import run_workflow


# 페이지 설정
st.set_page_config(
    page_title="Justi-Q 형사법 AI",
    page_icon="⚖️",
    layout="wide"
)

# 세션 상태 초기화
@st.cache_resource
def load_rag_system():
    """RAG 시스템 로드 (캐싱)"""
    vectorstore = VectorStore(
        collection_name="legal_documents",
        persist_dir="chroma_db"
    )
    rag_chain = RAGChain(vectorstore)
    return rag_chain


def main():
    # 헤더
    st.title("⚖️ Justi-Q 형사법 AI")
    st.markdown("형사법 관련 질문에 판례와 법령을 기반으로 답변해드립니다.")
    st.divider()

    # 사이드바
    with st.sidebar:
        st.header("설정")
        n_results = st.slider("검색 문서 수", min_value=3, max_value=10, value=5)

        st.divider()
        st.header("정보")
        st.markdown("""
        **데이터 출처:**
        - 판례 750건
        - 결정문 294건
        - 법령 898건
        - 해석 58건

        **모델:**
        - 임베딩: multilingual-e5-large
        - LLM: Llama 3.3 70B
        """)

    # RAG 시스템 로드
    try:
        rag = load_rag_system()
    except ValueError as e:
        if "OPENROUTER_API_KEY" in str(e):
            st.error("시스템 로드 실패: OPENROUTER_API_KEY 환경변수를 설정해주세요.")
            st.info("**Streamlit Cloud 배포 시:** Settings > Secrets에서 `OPENROUTER_API_KEY = \"your_key\"` 추가")
        else:
            st.error(f"시스템 로드 실패: {e}")
        return
    except Exception as e:
        error_msg = str(e)
        st.error(f"시스템 로드 실패: {error_msg}")
        st.info("먼저 `python main.py --index` 로 인덱싱을 실행해주세요.")
        return

    def _one_line_summary(src: dict, max_len: int = 70) -> str:
        # title이 있으면 title 우선, 없으면 본문 첫 줄/앞부분
        title = (src.get("title") or "").strip()
        if title:
            s = title
        else:
            content = (src.get("content") or "").strip()
            first_line = content.splitlines()[0].strip() if content else ""
            s = first_line if first_line else content[:max_len]
        s = s.replace("\n", " ").strip()
        return (s[:max_len] + "…") if len(s) > max_len else s

    def _render_sources(sources: list, key_prefix: str):
        if not sources:
            return
        st.markdown(f"📚 참고 문서 ({len(sources)}건)")
        for i, src in enumerate(sources, 1):
            doc_type = src.get("type", "문서")
            doc_id = src.get("doc_id", "unknown")
            sim = None
            if src.get("distance") is not None:
                sim = 1 - float(src["distance"])
            summary = _one_line_summary(src)

            # 한 줄 요약(항목) + 클릭하면 본문이 펼쳐지는 토글
            header = f"{i}. [{doc_type}] {doc_id} — {summary}"
            if sim is not None:
                header += f" (유사도: {sim:.2%})"

            with st.expander(header, expanded=False):
                # 읽기 전용 본문 표시: textarea 대신 markdown/code 사용
                content = src.get("content", "")
                if content:
                    st.code(content, language=None)
                else:
                    st.caption("본문 내용이 없습니다.")


    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 참고 문서"):
                    for src in message["sources"]:
                        st.markdown(f"- **[{src['type']}]** {src['doc_id']}")

            # ✅ (선택) grounded/issues 표시용
            if message.get("grounded") is False and message.get("issues"):
                with st.expander("⚠️ 근거 검증 이슈"):
                    for it in message["issues"]:
                        st.markdown(f"- {it}")

    # 사용자 입력
    if prompt := st.chat_input("형사법 관련 질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                # ✅ 여기만 핵심 변경: rag.query() -> LangGraph run_workflow()
                final_state = run_workflow(
                    question=prompt,
                    vectorstore=rag.vectorstore,   # RAGChain이 가진 vectorstore 재사용
                    rag_chain=rag,
                    n_results=n_results,
                    filter_type=None,
                )

            final_text = final_state.get("final", "")
            st.markdown(final_text)

            # ✅ 참고 문서 표시 (LangGraph state.documents 기반)
            docs = final_state.get("documents") or []
            sources = []
            for d in docs:
                md = d.get("metadata", {}) or {}
                sources.append({
                    "doc_id": md.get("doc_id", "unknown"),
                    "type": md.get("type_name", "문서"),
                    "distance": d.get("distance", None),
                })

            with st.expander("📚 참고 문서"):
                if not sources:
                    st.markdown("- (없음)")
                else:
                    for src in sources:
                        dist = src.get("distance")
                        if isinstance(dist, float):
                            st.markdown(f"- **[{src['type']}]** {src['doc_id']} (dist: {dist:.4f})")
                        else:
                            st.markdown(f"- **[{src['type']}]** {src['doc_id']}")

            # ✅ grounded / issues 표시
            grounded = final_state.get("grounded", None)
            issues = final_state.get("issues") or []
            if grounded is False:
                st.warning("근거 기반 검증에서 문제가 감지되었습니다.")
                with st.expander("⚠️ 근거 검증 이슈"):
                    for it in issues:
                        st.markdown(f"- {it}")

        # 어시스턴트 메시지 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_text,
            "sources": sources,
            "grounded": final_state.get("grounded", None),
            "issues": issues,
        })


if __name__ == "__main__":
    main()
