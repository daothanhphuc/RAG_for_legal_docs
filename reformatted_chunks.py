import os
import json
import re
from pathlib import Path
from tqdm import tqdm

TEXT_FOLDER = "output_texts"        
OUTPUT_FOLDER = "chunks_json"   
CHUNK_WORD_LIMIT = 400            


def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    return re.sub(r"\n{2,}", "\n\n", text.strip())

def smart_chunk(text: str, max_words=CHUNK_WORD_LIMIT):
    """Chia văn bản thành các đoạn (chunk) theo đoạn xuống dòng, giới hạn số từ"""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk.split()) + len(para.split()) <= max_words:
            current_chunk += "\n\n" + para
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def save_chunks_as_json(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def chunk_text_files(input_folder=TEXT_FOLDER, output_folder=OUTPUT_FOLDER):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    files = list(Path(input_folder).rglob("*.txt"))

    for file_path in tqdm(files, desc="Chunking files"):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        clean = clean_text(raw_text)
        chunks = smart_chunk(clean)
        output_file = Path(output_folder) / (file_path.stem + ".json")
        save_chunks_as_json(chunks, output_file)


if __name__ == "__main__":
    chunk_text_files()
