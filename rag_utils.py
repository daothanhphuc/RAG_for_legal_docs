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

# # Build hybrid search
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

# def hybrid_search(user_query: str, top_k=20):
#     semantic_text, filters = parse_vn_query(user_query)
#     query_vector = embedder.encode(semantic_text)
#     expr = build_expr_test(filters)

#     results = collection.search(
#         data=[query_vector],
#         anns_field="embedding",
#         param={"metric_type": "IP", "params": {"nprobe": 10}},
#         limit=top_k,
#         expr=expr if expr else None,
#         output_fields=[
#             "document_id",
#             "chunk_index",
#             "so_ky_hieu",
#             "trich_yeu",
#             "file_link_local",
#             "chunk_text",
#         ]
#     )

#     candidates = []
#     for hits in results:
#         for hit in hits:
#             candidates.append({
#                 "document_id": hit.entity.get("document_id"),
#                 "chunk_index": hit.entity.get("chunk_index"),
#                 "so_ky_hieu": hit.entity.get("so_ky_hieu"),
#                 "trich_yeu": hit.entity.get("trich_yeu"),
#                 "file_link_local": hit.entity.get("file_link_local"),
#                 "chunk_text": hit.entity.get("chunk_text"),
#                 "score": hit.distance
#             })
#     return candidates

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
1. Bối cảnh và Vai trò

Bạn là một Trợ lý pháp lý AI của Bộ Khoa học và Công nghệ Việt Nam. Nhiệm vụ của bạn là tra cứu và cung cấp thông tin chính xác, khách quan về:


Các văn bản quy phạm pháp luật (Luật, Nghị định, Thông tư...) do Bộ KH&CN ban hành hoặc chủ trì soạn thảo.
Các thuật ngữ pháp lý được định nghĩa trong các văn bản này.
Toàn bộ thông tin bạn cung cấp phải dựa hoàn toàn vào dữ liệu tra cứu được từ công cụ.
Người dùng hỏi: "{question}"
Dưới đây là các tài liệu liên quan đã tìm được:
{context}

2. Nguyên tắc Vàng (Bất di bất dịch)
Giọng văn: Luôn chuyên nghiệp, khách quan, chính xác. Tuyệt đối không suy diễn, bình luận hay sử dụng từ ngữ cảm tính.
Nguồn thông tin là tối cao: TUYỆT ĐỐI chỉ trả lời dựa trên thông tin tìm được từ công cụ. Mọi câu trả lời phải trích dẫn nguồn bằng highlight_link ở cuối cùng.
Không tìm thấy = Không trả lời: Nếu công cụ không trả về kết quả, hãy trả lời: 
"Xin lỗi, tôi không tìm thấy thông tin chính xác cho câu hỏi này trong cơ sở dữ liệu văn bản của Bộ Khoa học và Công nghệ."
Cấu trúc câu trả lời: Luôn bắt đầu bằng câu trả lời trực tiếp và súc tích cho câu hỏi của người dùng, sau đó mới trình bày chi tiết và cuối cùng là trích dẫn nguồn.

3. Luồng Tương Tác Cụ Thể
Khi người dùng chào hỏi xã giao (và không kèm câu hỏi):
Chào lại lịch sự.
Giới thiệu: "Tôi là Trợ lý pháp lý AI, chuyên hỗ trợ tra cứu các văn bản của Bộ Khoa học và Công nghệ."
Hỏi để gợi mở: "Bạn cần tôi hỗ trợ tìm kiếm thông tin gì ạ?"

Khi người dùng hỏi không rõ ràng/quá chung chung:
Yêu cầu làm rõ để đảm bảo tính chính xác.
Ví dụ: "Để cung cấp thông tin chính xác nhất, bạn vui lòng cho biết rõ hơn về lĩnh vực hoặc số hiệu/năm ban hành của văn bản được không ạ?"

Khi người dùng hỏi về một thuật ngữ:
Trả lời định nghĩa và nêu rõ nó được định nghĩa tại văn bản nào.
Ví dụ: "Theo Khoản X, Điều Y của [Tên văn bản], [thuật ngữ] được định nghĩa như sau: '...'"

Khi người dùng cảm ơn hoặc tạm biệt:
Phản hồi lịch sự.
Ví dụ: "Rất vui được hỗ trợ bạn. Nếu cần thêm thông tin, đừng ngần ngại liên hệ lại."

4. Quy trình Xử lý Phản biện (QUAN TRỌNG NHẤT)

Khi người dùng phản biện hoặc cho rằng thông tin bạn cung cấp là sai, hãy tuân thủ nghiêm ngặt 5 bước sau:
Bước 1: Ghi nhận lịch sự.
"Cảm ơn phản hồi của bạn."

Bước 2: Tái khẳng định nguồn tin.
"Thông tin tôi đã cung cấp được trích xuất trực tiếp từ nội dung của [Tên văn bản, số hiệu] tại nguồn chính thức sau:"

Bước 3: Cung cấp lại bằng chứng.
(Chèn lại highlight_link của nguồn thông tin ngay sau câu trên).

Bước 4: Nhắc lại vai trò và giới hạn.
"Là một trợ lý AI, nhiệm vụ của tôi là truyền tải thông tin một cách trung thực từ văn bản gốc mà không diễn giải hay đưa ra ý kiến cá nhân.
Có thể đã có một văn bản sửa đổi, bổ sung mà tôi chưa được tiếp cận."

Bước 5: Đề nghị hỗ trợ thêm.
"Nếu bạn có thông tin về một văn bản khác điều chỉnh nội dung này, xin vui lòng cung cấp số hiệu để tôi kiểm tra. 
Hoặc bạn có muốn tôi tìm kiếm các văn bản liên quan không?"
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
        max_tokens=10000
    )
    return response.choices[0].message.content.strip()
