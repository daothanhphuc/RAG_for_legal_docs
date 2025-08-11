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
if "selected_chat_index" not in st.session_state:
    st.session_state.selected_chat_index = None
# st.session_state.setdefault("chat_history", [])
# st.session_state.setdefault("retrieved_chunks", [])

with st.sidebar:
    st.markdown("### Lịch sử hội thoại")
    for idx, chat in enumerate(st.session_state.chat_history):
        if st.button(f"💬 {chat['answer'][:50]}...", key=f"select_{idx}"):
            st.session_state.selected_chat_index = idx
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.selected_chat_index = None
        st.success("Cleared.")

# Nếu đã chọn câu trả lời, hiển thị toàn bộ hội thoại tới thời điểm đó
if st.session_state.selected_chat_index is not None:
    st.markdown("### 📜 Hội thoại đã chọn")
    for i in range(st.session_state.selected_chat_index + 1):
        chat = st.session_state.chat_history[i]
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

# Input box
query = st.chat_input("Nhập câu hỏi:")

if query:
    if st.session_state.selected_chat_index is not None:
        # Lấy lại chunks từ câu trả lời đã chọn
        prev_chat = st.session_state.chat_history[st.session_state.selected_chat_index]
        top_chunks = prev_chat["chunks"]
        context_info = "Tiếp tục từ câu trả lời đã chọn."
        st.session_state.selected_chat_index = None
    else:
        # Retrieval + rerank mới
        with st.spinner("Retrieving relevant chunks..."):
            initial = initial_retrieval(query, k=INITIAL_K)
            top_chunks = rerank(query, initial, top_k=RERANK_TOP_K)
        context_info = "Câu hỏi mới."

    # Build prompt and call LLM
    prompt = build_prompt(query, top_chunks)
    messages = [
        {"role": "system", "content": "Bạn là một người trợ lý chuyên tìm kiếm và trả lời về thông tin văn bản hành chính."},
    ]
    # Thêm lịch sử hội thoại trước (đã có user+assistant)
    for chat in st.session_state.chat_history:
        messages.append({"role": "user", "content": chat["question"]})
        messages.append({"role": "assistant", "content": chat["answer"]})

    messages.append({"role": "user", "content": prompt})
    with st.spinner("Generating answer from OpenAI..."):
        answer = ask_llm(messages)

    st.subheader("Answer")
    st.session_state.chat_history.append({
        "question": query,
        "answer": answer,
        "chunks": top_chunks
    })

    # Hiển thị
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Hiển thị nguồn
    st.subheader("Sources Summary")
    st.write(f"**Nguồn:** {context_info}")
    for i, c in enumerate(top_chunks, 1):
        meta = c.metadata
        st.markdown(
            f"{i}. **{meta.get('so_ky_hieu','')}** chunk_index={meta.get('chunk_index')} "
            f"(rerank_score={c.score_rerank:.4f}) — trich_yeu: {meta.get('trich_yeu','')}"
        )