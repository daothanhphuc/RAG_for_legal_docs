import os
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from pymilvus import connections, Collection
import openai
from typing import List, Dict
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

connections.connect(alias="default", uri=os.getenv("MILVUS_URI"), 
                    token=os.getenv("MILVUS_TOKEN"), 
                    secure=True
)
collection = Collection(name="document_chunks", using="default")
collection.load()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # pretrained reranker
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

INITIAL_K = 10
RERANK_TOP_K = 5
@dataclass
class RetrievedChunk:
    score_initial: float
    score_rerank: float
    chunk_text: str
    metadata: Dict

def normalize(vec: np.ndarray) -> List[float]:
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm != 0 else vec.tolist()

def generate_hypothetical_doc(user_question: str) -> str:
    prompt = f"""
    Bạn là một trợ lý thông minh. Hãy viết một đoạn văn giải thích, mở rộng, và làm rõ câu hỏi dưới đây,
    cung cấp thêm thông tin hoặc giả định cần thiết để giúp việc tìm kiếm dữ liệu tốt hơn.

    Câu hỏi: {user_question}

    Đoạn văn giải thích:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()

def build_expr_test(filters: dict, extra_keyword: str = None) -> str:
    expr_parts = []
    for key, value in filters.items():
        if isinstance(value, str):
            expr_parts.append(f'{key} == "{value}"')
        elif isinstance(value, (int, float)):
            expr_parts.append(f"{key} == {value}")
    if extra_keyword:
        expr_parts.append(f'trich_yeu like "%{extra_keyword}%"')
    return " and ".join(expr_parts) if expr_parts else ""

def extract_extra_keyword_with_gpt(query: str) -> str | None:
    prompt = f"""
Bạn là một trợ lý trích xuất thực thể chính trong câu hỏi pháp lý tiếng Việt.
Hãy chỉ trả về tên tổ chức, chủ thể hoặc cụm từ quan trọng nhất liên quan đến câu hỏi sau đây,
dùng để lọc văn bản trong bộ dữ liệu pháp luật:

Câu hỏi: {query}

Tên tổ chức hoặc từ khóa chính:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là một trợ lý trích xuất thực thể chính xác."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=20,
        n=1,
        stop=None
    )
    keyword = response.choices[0].message.content.strip()
    return keyword

def initial_retrieval(query: str, k: int = INITIAL_K) -> List[RetrievedChunk]:
    hypothetic_doc = generate_hypothetical_doc(query)
    q_vec = embedder.encode(hypothetic_doc)
    q_vec = normalize(q_vec)
    semantic_text, filters = parse_vn_query(query)
    extra_keyword = extract_extra_keyword_with_gpt(query)

    results = collection.search(
        data=[q_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 50}},
        limit=k,
        expr= build_expr_test(filters, extra_keyword),
        output_fields=[
            "document_id",
            "chunk_index",
            "so_ky_hieu",
            "trich_yeu",
            "file_link_local",
            "chunk_text",
        ],
    )
    hits = results[0]
    chunks = []
    for hit in hits:
        md = hit.entity
        metadata = {
            "document_id": md.get("document_id"),
            "chunk_index": md.get("chunk_index"),
            "so_ky_hieu": md.get("so_ky_hieu"),
            "trich_yeu": md.get("trich_yeu"),
            "file_link_local": md.get("file_link_local"),
        }
        chunks.append(RetrievedChunk(
            score_initial=hit.score,
            score_rerank=0.0,  # placeholder
            chunk_text=md.get("chunk_text", ""),
            metadata=metadata
        ))
    return chunks

# Build hybrid search
def parse_vn_query(user_query: str) -> tuple[str, dict]:
    filters = {}
    match_year = re.search(r"(năm|ban hành năm)\s*(\d{4})", user_query, re.IGNORECASE)
    if match_year:
        filters["nam_ban_hanh"] = int(match_year.group(2))

    # add regex for số ký hiệu
    match_so_ky_hieu = re.search(r"(\d+/\d{4}/[A-ZĐ]+-[A-Z]+)", user_query)
    if match_so_ky_hieu:
        filters["so_ky_hieu"] = match_so_ky_hieu.group(1)

    semantic_text = user_query
    for val in filters.values():
        semantic_text = re.sub(str(val), "", semantic_text, flags=re.IGNORECASE)
    return semantic_text.strip(), filters

def hybrid_search(user_query: str, top_k=20):
    semantic_text, filters = parse_vn_query(user_query)
    query_vector = embedder.encode(semantic_text)
    expr = build_expr_test(filters)

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        expr=expr if expr else None,
        output_fields=[
            "document_id",
            "chunk_index",
            "so_ky_hieu",
            "trich_yeu",
            "file_link_local",
            "chunk_text",
        ]
    )

    candidates = []
    for hits in results:
        for hit in hits:
            candidates.append({
                "document_id": hit.entity.get("document_id"),
                "chunk_index": hit.entity.get("chunk_index"),
                "so_ky_hieu": hit.entity.get("so_ky_hieu"),
                "trich_yeu": hit.entity.get("trich_yeu"),
                "file_link_local": hit.entity.get("file_link_local"),
                "chunk_text": hit.entity.get("chunk_text"),
                "score": hit.distance
            })
    return candidates


def rerank(query: str, candidates: List[RetrievedChunk], top_k: int = RERANK_TOP_K) -> List[RetrievedChunk]:
    if not candidates:
        return []

    pairs = [[query, c.chunk_text] for c in candidates]
    rerank_scores = cross_encoder.predict(pairs)  # higher is better
    for c, s in zip(candidates, rerank_scores):
        c.score_rerank = float(s)

    sorted_chunks = sorted(candidates, key=lambda x: x.score_rerank, reverse=True)
    return sorted_chunks[:top_k]

def build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.metadata
        part = (
            f"[{i}] score_initial: {c.score_initial:.4f}, rerank_score: {c.score_rerank:.4f}\n"
            f"so_ky_hieu: {meta.get('so_ky_hieu','')}\n"
            f"trich_yeu: {meta.get('trich_yeu','')}\n"
            f"file: {meta.get('file_link_local','')}\n"
            f"chunk_index: {meta.get('chunk_index')}\n"
            f"Text: {c.chunk_text.strip()}\n"
        )
        parts.append(part)
    context = "\n---\n".join(parts)
    prompt = f"""
Bạn là một trợ lý pháp lý. Sử dụng top-K đoạn văn bản đã truy xuất (bao gồm điểm số, siêu dữ liệu, và nội dung của chúng) để trả lời câu hỏi của người dùng.
Trích dẫn nguyên văn nội dung từ các đoạn văn bản được cung cấp, kèm theo số thứ tự đoạn.
Không thêm bình luận hoặc suy luận ngoài thông tin đã cho.
Nếu không tìm được đoạn văn bản nào liên quan, hãy trả lời: "Tôi không biết dựa trên các tài liệu đã cung cấp."
Context:
{context}

Question: {question}

Câu trả lời của bạn:
"""
    return prompt


def ask_llm(messages: list) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages= messages,
        # messages=[
        #     {"role": "system", "content": "Bạn là một người trợ lý chuyên tìm kiếm và trả lời về thông tin văn bản hành chính."},
        #     {"role": "user", "content": prompt}
        # ],
        max_tokens=800
    )
    return response.choices[0].message.content.strip()
