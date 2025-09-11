# Legal RAG Chat — Setup and Usage

A Retrieval-Augmented Generation (RAG) workflow for Vietnamese legal documents. It chunks markdown sources, embeds them with SentenceTransformers, indexes in Milvus/Zilliz, and serves a Streamlit chatbot that retrieves, reranks, and answers using OpenAI.

## Features
- Chunk markdown sources with token-aware rules
- Build embeddings via `all-MiniLM-L6-v2` and insert into Milvus
- Fast semantic search with COSINE + HNSW
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`)
- Streamlit chat UI with conversational memory

## Repository layout
- `deploy_streamlit.py` — Streamlit chatbot app
- `rag_utils.py` — retrieval, rerank, prompting, LLM calls
- `create_chunks.py` — build chunk metadata JSON from `.md` files (uses Supabase for metadata)
- `code/milvus_insert.py` — create Milvus collection and insert chunk embeddings
- `code/query_milvus.py` — example Milvus queries (legacy)
- `code/connect_minio.py` — simple MinIO download helper (optional)
- `chunked_json/` — output chunk metadata JSON files
- `fix_standard_texts/`, `standard_texts/`, `old_doc_texts/` — markdown sources
- `filelocal/filelocal/` — local copies of original files referenced by metadata

## Prerequisites
- Python 3.10+
- Milvus or Zilliz Cloud
- An OpenAI API key
- (Optional) Supabase project for document metadata lookup in `create_chunks.py`
- (Optional) MinIO server if you use object storage

## Installation (Windows PowerShell)
```powershell
cd F:\RAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
If you also use helpers in `code/`, install:
```powershell
pip install -r .\code\requirements.txt
```

## Environment variables
Create a `.env` file at the project root:
```ini
# Zilliz/Milvus
MILVUS_URI="your_zilliz_or_milvus_uri"      # e.g. grpcs://xxx.api.gcp.zillizcloud.com:443
MILVUS_TOKEN="your_token_if_applicable"     # e.g. db_admin:***** (Zilliz) or leave blank for local

# OpenAI
OPENAI_API_KEY="sk-..."

# Hugging Face (for model downloads if rate limited)
HF_TOKEN="hf_..."

# Supabase (only needed by create_chunks.py)
SUPA_URL="https://YOUR-PROJECT.supabase.co"
SUPA_KEY="service_or_anon_key"
```
Notes:
- `rag_utils.py` connects to Milvus using `MILVUS_URI` and `MILVUS_TOKEN` and loads the `document_chunks` collection.
- `rag_utils.py` uses OpenAI models `gpt-4o-mini` for HyDE and answering.

## Prepare chunk metadata (optional, if you need to regenerate)
`create_chunks.py` reads `.md` files in `fix_standard_texts/`, fetches metadata from Supabase, token-chunks them, and writes JSON to `chunked_json/`.

```powershell
# Ensure SUPA_URL, SUPA_KEY are set in .env
python .\create_chunks.py
# Output JSON files will appear in .\chunked_json\
```

## Build Milvus collection and insert embeddings
Use `code/milvus_insert.py`:
1) Create (or recreate) the collection and index
```python
# in code/milvus_insert.py
create_collection()
```
2) Load chunk metadata and insert in batches
```python
metadata = load_data("chunked_json")
insert_to_milvus_from_loaded(metadata, batch_size=64)
```
3) Verify collection stats
```python
stats = milvus_client.get_collection_stats("document_chunks")
print(stats["row_count"])  # total vectors
```
Run the script directly to test pieces or adapt the `__main__` block. The collection schema includes fields: `embedding`, `document_id`, `chunk_index`, `so_ky_hieu`, `trich_yeu`, `file_link_local`, `chunk_text`.

## Run the Streamlit chatbot
```powershell
# Activate venv and ensure .env is populated
streamlit run .\deploy_streamlit.py
```
The app:
- Retrieves top-K candidates from Milvus via `rag_utils.initial_retrieval`
- Reranks with `rag_utils.rerank` using a cross-encoder
- Builds a role-aligned prompt with sources
- Calls OpenAI to answer in Vietnamese and shows source summaries

## Troubleshooting
- Milvus connection: verify `MILVUS_URI` and `MILVUS_TOKEN`. For local Milvus, you may need host/port based connection; the app uses URI + secure=True.
- Large model downloads: first run will download `sentence-transformers` models and a cross-encoder. Set `HF_TOKEN` if you hit rate limits.
- Collection not found: be sure you created and populated `document_chunks` before running the app.
- OpenAI quota/errors: ensure `OPENAI_API_KEY` is valid and you have access to `gpt-4o-mini`.


