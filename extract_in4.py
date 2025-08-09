import os
from openai import OpenAI
# import pytesseract
import pandas as pd
# import rarfile
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
# from striprtf.striprtf import rtf_to_text

MILVUS_HOST = "10.2.44.244"
MILVUS_PORT = "19530"
COLLECTION_NAME = "dichvucong"

OUTPUT_FOLDER = "output_texts"

INPUT_FOLDER = r"F:\RAG\RAG\filelocal\filelocal"    
# OUTPUT_FOLDER = "test"  

STANDARD_FOLDER = "fix_standard_texts"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def correct_vietnamese_text(raw_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia phục hồi văn bản tiếng Việt bị mã hóa sai. Chỉ trả về văn bản đã được phục hồi, không thêm giải thích hoặc văn bản phụ."},
                {"role": "user", "content": f"Sửa và khôi phục tiếng Việt cho nội dung sau:\n\n{raw_text}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR with OpenAI API]: {e}"
    
def standard_text(extracted_text):
    try: 
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=  [{"role": "system", "content": """Bạn là chuyên gia phục hồi văn bản tiếng Việt thành định dạng chuẩn Markdown. 
                                             Chuyển toàn bộ nội dung sang định dạng Markdown (.md)** với quy tắc sau:
                                            - Tiêu đề chính: `# `
                                            - Tiêu đề phụ: `## `, `### `...
                                            - Danh sách: `- ` hoặc `* `
                                            - Đánh số: `1.`, `2.`, ...
                                            - In đậm: `**text**`, *nghiêng*: `*text*` nếu thấy hợp lý
                                             Không thêm giải thích, hướng dẫn hoặc bất kỳ văn bản nào khác."""},
                        {"role": "user", "content": f"Chuyển đổi thành định dạng Markdown cho nội dung sau:\n\n{extracted_text}"
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR with OpenAI API]: {e}"

# trích xuất text từ từng loại file
# def extract_text_from_file(file_path):
#     try:
#         suffix = file_path.suffix.lower()
#         if suffix == '.rtf':
#             with open(file_path, 'r', encoding='cp1252', errors='ignore') as f:
#                 return rtf_to_text(f.read())
            
#         elif suffix in ['.tif', '.tiff']:
#             return pytesseract.image_to_string(Image.open(file_path), lang='vie')
        
#         # elif suffix == '.xls':
#         #     import xlrd
#         #     df = pd.read_excel(file_path, engine='xlrd')
#         #     return df.astype(str).to_string(index=False)
        
#         elif suffix == '.rar':
#             extracted_texts = []
#             with rarfile.RarFile(file_path) as rf:
#                 rf.extractall("temp_extracted")
#                 for f in Path("temp_extracted").rglob("*.*"):
#                     extracted_texts.append(extract_text_from_file(f))
#             return "\n".join(extracted_texts)
#         else:
#             return ""
#     except Exception as e:
#         return f"[ERROR reading {file_path.name}]: {e}"

ERROR_FILE_LIST = "error_files.txt"  
def process_standard_files():
    os.makedirs(STANDARD_FOLDER, exist_ok=True)

    # Load danh sách file lỗi (chỉ lấy tên không có phần mở rộng .md)
    with open(ERROR_FILE_LIST, "r", encoding="utf-8") as f:
        error_md_names = {Path(line.strip()).stem for line in f if line.strip()}

    files = list(Path(OUTPUT_FOLDER).rglob("*.txt"))

    for file_path in files:
        stem = file_path.stem
        if stem not in error_md_names:
            continue  

        print(f"📑 Standardizing: {file_path.name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                print(f"⚠️ Empty file skipped: {file_path.name}")
                continue

            md_text = standard_text(text)

            output_file = Path(STANDARD_FOLDER) / (stem + ".md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(md_text)

            print(f"Markdown saved: {output_file.name}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            
def process_all_files():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    files = list(Path(INPUT_FOLDER).rglob("*.*"))

    for file_path in files:
        print(f"Processing: {file_path.name}")
        raw_text = extract_text_from_file(file_path)
        if not raw_text.strip():
            print(f"Empty or unreadable file: {file_path.name}")
            continue
        fixed_text = correct_vietnamese_text(raw_text)

        output_file = Path(OUTPUT_FOLDER) / (file_path.stem + ".txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(fixed_text)
        print(f"Done: {output_file.name}")

# def process_standard_files():
#     os.makedirs(STANDARD_FOLDER, exist_ok=True)
#     files = list(Path(OUTPUT_FOLDER).rglob("*.txt"))

#     for file_path in files:
#         print(f"Standardizing: {file_path.name}")
#         try:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 text = f.read()

#             if not text.strip():
#                 print(f"Empty file skipped: {file_path.name}")
#                 continue

#             md_text = standard_text(text)

#             output_file = Path(STANDARD_FOLDER) / (file_path.stem + ".md")
#             with open(output_file, "w", encoding="utf-8") as f:
#                 f.write(md_text)

#             print(f"Markdown saved: {output_file.name}")
#         except Exception as e:
#             print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    # process_all_files()
    process_standard_files()
    