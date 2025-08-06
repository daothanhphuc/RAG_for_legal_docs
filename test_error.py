import json
# from striprtf.striprtf import rtf_to_text

# with open("chunks_output/00a70c7078311b559662693ece6a1c8d259fbc6d.json", encoding="utf-8") as f:
#     chunks = json.load(f)

# for i, chunk in enumerate(chunks[:3]):
#     print(f"Chunk {i+1}:")
#     print(tcvn3_to_unicode(chunk))
#     print("-" * 50)





# from pathlib import Path

# ERROR_FOLDER = "output_texts"
# error_prefix = "[ERROR đọc"

# def count_vietnamese_error_files(folder_path):
#     folder = Path(folder_path)
#     error_files = []

#     for txt_file in folder.rglob("*.txt"):
#         try:
#             with open(txt_file, "r", encoding="utf-8") as f:
#                 first_line = f.readline()
#                 if first_line.strip().startswith(error_prefix):
#                     error_files.append(txt_file.name)
#         except Exception as e:
#             print(f"⚠️ Không thể đọc {txt_file.name}: {e}")

#     print(f"\n❌ Tổng cộng có {len(error_files)} file lỗi:")
#     for fname in error_files:
#         print(f" - {fname}")

#     return error_files

# if __name__ == "__main__":
#     count_vietnamese_error_files(ERROR_FOLDER)




import os
from pathlib import Path

# TARGET_FOLDER = r"C:\Users\phucdz\code\RAG\fix_standard_texts"  

# error_files = []

# # Duyệt toàn bộ file trong thư mục
# for filepath in Path(TARGET_FOLDER).rglob("*.md"):
#     try:
#         with open(filepath, "r", encoding="utf-8") as f:
#             content = f.read()
#             if "[ERROR" in content:
#                 error_files.append(filepath.name)   # Chỉ lấy tên file
#     except Exception as e:
#         print(f"⚠️ Không thể đọc file: {filepath.name}, lỗi: {e}")

# # Ghi kết quả vào file
# output_path = Path("error_files.txt")
# error_files = sorted(set(error_files)) # loại bỏ trùng lặp
# with open(output_path, "w", encoding="utf-8") as f:
#     for filename in error_files:
#         f.write(filename + "\n")

# print(f"✅ Tổng số file chứa '[ERROR': {len(error_files)}")
# print(f"📄 Danh sách đã lưu vào: {output_path.resolve()}")



##### DELETE ERROR FILES #####
##############################

TARGET_FOLDER = Path(r"F:\RAG\RAG\fix_standard_texts")
ERROR_FILE_LIST = Path("error_files.txt")

# Đọc danh sách tên file cần xóa
with open(ERROR_FILE_LIST, "r", encoding="utf-8") as f:
    filenames_to_delete = [line.strip() for line in f.readlines() if line.strip()]

deleted = []
not_found = []

for filename in filenames_to_delete:
    file_path = TARGET_FOLDER / filename
    if file_path.exists():
        try:
            file_path.unlink()
            deleted.append(filename)
        except Exception as e:
            print(f"⚠️ Lỗi khi xóa {filename}: {e}")
    else:
        not_found.append(filename)

# In kết quả
print(f"✅ Đã xóa {len(deleted)} file.")
if not_found:
    print(f"⚠️ {len(not_found)} file không tìm thấy:")
    for f in not_found:
        print(f" - {f}")


################# COPY ERROR FILES #################
####################################################

# from pathlib import Path

# TARGET_FOLDER = r"F:\RAG\RAG\fix_standard_texts"
# START_FILE = "ce6e903d0abed2b01092b6c1eaf34c69a83ca40f.md"
# OUTPUT_FILE = Path("error_files.txt")

# # Collect and sort all .md filenames
# all_md = sorted([p.name for p in Path(TARGET_FOLDER).rglob("*.md")])
# # print("First 2 files in all_md:", all_md[:2])

# if START_FILE not in all_md:
#     print(f"⚠️ File '{START_FILE}' not found in {TARGET_FOLDER}")
# else:
#     start_idx = all_md.index(START_FILE)
#     # include the start file and all after
#     files_after = all_md[start_idx:]

#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         for name in files_after:
#             f.write(name + "\n")

#     print(f"✅ {len(files_after)} filenames (starting from '{START_FILE}') written to {OUTPUT_FILE.resolve()}")