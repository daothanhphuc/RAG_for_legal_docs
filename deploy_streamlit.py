import streamlit as st
import uuid
from rag_utils import initial_retrieval, rerank, build_prompt, ask_llm


INITIAL_K = 10
RERANK_TOP_K = 5

st.set_page_config(page_title="Legal RAG Chat", layout="wide")
st.title("Legal Chatbot Supporter")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("retrieved_chunks", [])

with st.sidebar:
    st.markdown("### Controls")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.retrieved_chunks = []
        st.success("Cleared.")

# Display history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
query = st.chat_input("Ask a legal question:")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieval + rerank
    with st.spinner("Retrieving relevant chunks..."):
        initial = initial_retrieval(query, k=INITIAL_K)
        reranked = rerank(query, initial, top_k=RERANK_TOP_K)

    # Show retrieved and scores
    # st.subheader("Retrieved & Reranked Chunks")
    # for i, chunk in enumerate(reranked, 1):
    #     meta = chunk.metadata
    #     st.markdown(f"**[{i}] so_ky_hieu:** {meta.get('so_ky_hieu','')}  "
    #                 f"**chunk_index:** {meta.get('chunk_index')}  "
    #                 f"**init_score:** {chunk.score_initial:.4f}  "
    #                 f"**rerank_score:** {chunk.score_rerank:.4f}")
    #     st.markdown(f"> {chunk.chunk_text.strip()[:1000]}...")
    #     st.caption(f"trich_yeu: {meta.get('trich_yeu','')}  ")

    # Build prompt and call LLM
    prompt = build_prompt(query, reranked)
    messages = [
        {"role": "system", "content": "Bạn là một người trợ lý chuyên tìm kiếm và trả lời về thông tin văn bản hành chính."},
    ]
    # Thêm lịch sử hội thoại trước (đã có user+assistant)
    for msg in st.session_state.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": prompt})
    with st.spinner("Generating answer from OpenAI..."):
        answer = ask_llm(messages)

    st.subheader("Answer")
    if answer:
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    # Optionally show sources summary
    st.subheader("Sources Summary")
    for i, c in enumerate(reranked, 1):
        meta = c.metadata
        st.markdown(f"{i}. **{meta.get('so_ky_hieu','')}** chunk_index={meta.get('chunk_index')} "
                    f"(rerank_score={c.score_rerank:.4f}) — trich_yeu: {meta.get('trich_yeu','')}")