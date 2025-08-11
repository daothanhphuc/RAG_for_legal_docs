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
if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = None
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("retrieved_chunks", [])

with st.sidebar:
    st.markdown("### Controls")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.retrieved_chunks = []
        st.success("Cleared.")

    st.markdown("### Chọn câu trả lời để hỏi lại")
    assistant_msgs = [
        (i, msg["content"])
        for i, msg in enumerate(st.session_state.chat_history)
        if msg["role"] == "assistant"
    ]

    if assistant_msgs:
        for i, content in assistant_msgs:
            display_text = content if len(content) < 80 else content[:77] + "..."
            if st.button(f"Assistant #{i+1}: {display_text}", key=f"select_{i}"):
                st.session_state.selected_text = content
                st.success(f"Đã chọn câu trả lời #{i+1} để hỏi lại.")
    else:
        st.info("Chưa có câu trả lời nào từ assistant.")
    

# ===== Khi click vào câu trả lời trước =====
clicked_index = st.session_state.get("clicked_message_index")  # index câu trả lời được click
if clicked_index is not None:
    # Lấy câu hỏi trước đó từ chat_history
    prev_user_msg = None
    for i in range(clicked_index, -1, -1):
        if st.session_state.chat_history[i]["role"] == "user":
            prev_user_msg = st.session_state.chat_history[i]["content"]
            break
    if prev_user_msg:
        st.session_state.last_user_query = prev_user_msg  # lưu để nối ngữ cảnh

# Input box
query = st.chat_input("Nhập câu hỏi:")

if query:
    if st.session_state.last_user_query:
        full_query = f"{st.session_state.last_user_query}\nNgười dùng hỏi tiếp: {query}"
    else:
        full_query = query

    st.session_state.chat_history.append({"role": "user", "content": full_query})
    st.session_state.last_user_query = full_query  # Cập nhật câu hỏi gốc mới

    with st.chat_message("user"):
        st.markdown(full_query)

    # Retrieval + rerank
    with st.spinner("Retrieving relevant chunks..."):
        initial = initial_retrieval(full_query, k=INITIAL_K)
        reranked = rerank(full_query, initial, top_k=RERANK_TOP_K)

    # Build prompt and call LLM
    prompt = build_prompt(full_query, reranked)
    messages = [{"role": "system", "content": "Bạn là một trợ lý chuyên tìm kiếm và trả lời về thông tin văn bản hành chính."}]
    for msg in st.session_state.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    with st.spinner("Generating answer from OpenAI..."):
        answer = ask_llm(messages)

    if answer:
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    # Sources Summary
    st.subheader("Sources Summary")
    for i, c in enumerate(reranked, 1):
        meta = c.metadata
        st.markdown(
            f"{i}. **{meta.get('so_ky_hieu','')}** chunk_index={meta.get('chunk_index')} "
            f"(rerank_score={c.score_rerank:.4f}) — trich_yeu: {meta.get('trich_yeu','')}"
        )