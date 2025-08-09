import os
from supabase import create_client
from dotenv import load_dotenv
import time

load_dotenv()

url = os.getenv("SUPA_URL")
key = os.getenv("SUPA_KEY")
supabase = create_client(url, key)

# file_link_local = "0a2fba3b3570cabc7b56ab46fb5d54116cb0208d.rtf"
markdown_folder = "old_doc_texts"
print(markdown_folder)

for md_filename in os.listdir(markdown_folder):
    if not md_filename.endswith(".md"):
        continue

    base_filename = os.path.splitext(md_filename)[0]
    file_link_local = base_filename + ".doc"  # bảng file_links_local lưu tên file .rtf gốc
    md_path = os.path.join(markdown_folder, md_filename)

    # query document_id
    res = supabase.table("file_links_local")\
                  .select("document_id")\
                  .eq("file_link_local", file_link_local)\
                  .execute()

    if not res.data:
        print(f"Không tìm thấy document_id cho {file_link_local}")
        continue

    # read md file content
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
    except Exception as e:
        print(f"Lỗi khi đọc file {md_filename}: {e}")
        continue

    for row in res.data:
        document_id = row["document_id"]
        print(f"\nĐang xử lý file: {md_filename} → document_id: {document_id}")

        # ===== XÓA content cũ nếu có =====
        supabase.table("document_contents")\
            .update({"content": None})\
            .eq("document_id", document_id)\
            .eq("file_link_local", file_link_local)\
            .execute()
        print(f"Đã xoá content cũ")

        insert_data = {
            "document_id": document_id,
            "file_link_local": file_link_local,
            "content": md_content   # ✅ GHI TOÀN BỘ NỘI DUNG MD
        }

        try:
            supabase.table("document_contents").insert(insert_data).execute()
            print(f"Đã ghi nội dung: {md_filename} → document_id {document_id}")
        except Exception as e:
            print(f"Lỗi khi insert {md_filename}: {e}")

        time.sleep(1)
