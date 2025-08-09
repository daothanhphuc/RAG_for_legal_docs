from pymilvus import MilvusClient
from pymilvus import (
    FieldSchema, CollectionSchema, DataType, utility, Collection, connections
)
from huggingface_hub import login
# from chonkie import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
import os
import json
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)
milvus_client = MilvusClient(
    uri=os.getenv("MILVUS_URI"), 
    token=os.getenv("MILVUS_TOKEN"),
    secure = True
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "document_chunks"
VECTOR_DIM = 384  # in case all-MiniLM-L6-v2
METRIC_TYPE = "COSINE"

connections.connect(
    alias="default", 
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN"),
    secure=True
)

def create_collection():
    if milvus_client.has_collection(COLLECTION_NAME):
        print(f"[INFO] Collection '{COLLECTION_NAME}' đã tồn tại. Xóa và tạo lại...")
        milvus_client.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="document_id", dtype=DataType.INT64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="so_ky_hieu", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="trich_yeu", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="file_link_local", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=8192),
    ]

    schema = CollectionSchema(fields=fields, description="Chunks of legal documents")

    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        description="Chunks of legal documents",
        consistency_level="Strong"  
    )

    collection = Collection(name=COLLECTION_NAME)  # kết nối tới collection vừa tạo

    index_params = {
        "index_type": "HNSW",
        "metric_type": METRIC_TYPE,
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Đã tạo index trên field 'embedding' với params: {index_params}")

def load_data(json_folder):
    docs = {
        "document_id": [],
        "chunk_index": [],
        "so_ky_hieu": [],
        "trich_yeu": [],
        "file_link_local": [],
        "chunk_text": []
    }

    for filename in os.listdir(json_folder):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(json_folder, filename), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            chunk_text = chunk.get("chunk_text", "").strip()
            if not chunk_text:
                continue

            docs["document_id"].append(chunk.get("document_id", -1))
            docs["chunk_index"].append(chunk.get("chunk_index", -1))
            docs["so_ky_hieu"].append(chunk.get("so_ky_hieu", ""))
            docs["trich_yeu"].append(chunk.get("trich_yeu", ""))
            docs["file_link_local"].append(chunk.get("file_link_local", ""))
            docs["chunk_text"].append(chunk_text)

    total = len(docs["chunk_text"])
    print(f"Tải xong {total} đoạn, chuẩn bị tạo embedding vectors")
    return docs

def fix_str_list(lst):
    return [str(x) if x is not None else "" for x in lst]

def insert_to_milvus_from_loaded(metadata_dict, batch_size=64):
    total = len(metadata_dict["chunk_text"])
    assert all(len(v) == total for v in metadata_dict.values()), "Metadata phải cùng độ dài"

    def fix_str(x):
        return str(x) if x is not None else ""

    print(f"Đang upload {total} vectors lên Milvus theo batch size={batch_size}...")

    for start in tqdm(range(0, total, batch_size), desc="Inserting"):
        end = min(start + batch_size, total)
        batch_texts = metadata_dict["chunk_text"][start:end]

        # normalize if using cosine similarity
        batch_embeddings = embedding_model.encode(
            batch_texts,
            convert_to_numpy=False  # trả về list of python lists
        )
        # Nếu bạn dùng cosine và muốn normalize thủ công (Milvus không auto normalize):
        def normalize(vec):
            arr = np.array(vec, dtype=float)
            norm = np.linalg.norm(arr)
            return (arr / norm).tolist() if norm != 0 else arr.tolist()

        batch_embeddings = [normalize(v) for v in batch_embeddings]

        # Lấy metadata
        batch_document_id = [int(x) for x in metadata_dict["document_id"][start:end]]
        batch_chunk_index = [int(x) for x in metadata_dict["chunk_index"][start:end]]
        batch_so_ky_hieu = [fix_str(x) for x in metadata_dict["so_ky_hieu"][start:end]]
        batch_trich_yeu = [fix_str(x) for x in metadata_dict["trich_yeu"][start:end]]
        batch_file_link_local = [fix_str(x) for x in metadata_dict["file_link_local"][start:end]]
        batch_chunk_text = [fix_str(x) for x in batch_texts]

        # Xây list of row dicts
        batch_rows = []
        for emb, doc_id, cidx, skh, ty, fll, ctx in zip(
            batch_embeddings,
            batch_document_id,
            batch_chunk_index,
            batch_so_ky_hieu,
            batch_trich_yeu,
            batch_file_link_local,
            batch_chunk_text,
        ):
            row = {
                "embedding": emb,
                "document_id": doc_id,
                "chunk_index": cidx,
                "so_ky_hieu": skh,
                "trich_yeu": ty,
                "file_link_local": fll,
                "chunk_text": ctx,
            }
            batch_rows.append(row)

        # Insert row-wise
        milvus_client.insert(
            collection_name=COLLECTION_NAME,
            data=batch_rows,
        )

    print(f" Hoàn tất insert {total} vectors lên Milvus.")

def build_index():
    # Load collection into memory for search
    try:
        collection = Collection(name=COLLECTION_NAME, using="default")
        collection.load()
        print(f"Collection '{COLLECTION_NAME}' loaded and ready for search.")
    except Exception as e:
        print(f"Error loading collection: {e}")
        return None
    return collection  

def search(query: str, collection: Collection, top_k: int = 5):
    query_emb = embedding_model.encode(query, convert_to_numpy=False)

    def normalize(vec):
        arr = np.array(vec, dtype=float)
        norm = np.linalg.norm(arr)
        return (arr / norm).tolist() if norm != 0 else arr.tolist()
    query_emb = normalize(query_emb)

    # Need more advanced search experiment here
    results = collection.search(
        data=[query_emb],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 50}},  
        limit=top_k,
        expr=None,  # filter expr like "document_id == 123"
        output_fields=[
            "document_id",
            "chunk_index",
            "so_ky_hieu",
            "trich_yeu",
            "file_link_local",
            "chunk_text",
        ],
    )

    # output: list of SearchResults cho mỗi query - only 1 here
    hits = results[0]
    formatted = []
    for hit in hits:
        formatted.append({
            "score": hit.score,
            "document_id": hit.entity.get("document_id"),
            "chunk_index": hit.entity.get("chunk_index"),
            "so_ky_hieu": hit.entity.get("so_ky_hieu"),
            "trich_yeu": hit.entity.get("trich_yeu"),
            "file_link_local": hit.entity.get("file_link_local"),
            "chunk_text": hit.entity.get("chunk_text"),
        })
    return formatted

if __name__ == "__main__":
    # create_collection()

    # metadata = load_data("chunked_json") 
    # insert_to_milvus_from_loaded(metadata, batch_size=64)

    stats = milvus_client.get_collection_stats("document_chunks")
    print(f"Tổng số vector trong collection: {stats['row_count']}")

    # build_index()
    # collection = build_index()
    # query_text = "Nghị định vào năm 1993"
    # results = search(query_text, collection, top_k=5)
    # for i, r in enumerate(results, 1):
    #     print(f"  Result {i} (score={r['score']:.4f}) ")
    #     print(f"so_ky_hieu: {r['so_ky_hieu']}")
    #     print(f"trich_yeu: {r['trich_yeu']}")
    #     print(f"chunk_text: {r['chunk_text'][:300]}...")
    #     print()