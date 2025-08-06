import os
import json
from datetime import datetime
from supabase import create_client, Client

# Cấu hình Supabase
SUPABASE_URL = "https://jfznazmrqxjmqnfkavmg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impmem5hem1ycXhqbXFuZmthdm1nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2NDYyNjEsImV4cCI6MjA2ODIyMjI2MX0.djD_Zmg7J43iFDJr8MOLpACy56HbHPWBbGOP7TCBzrE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def parse_date(date_str):
    for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except:
            continue
    return None

def upload_document(data):
    # Insert vào bảng documents nếu chưa tồn tại code
    detail = data.get("detail", {})
    code = data.get("code")
    # Kiểm tra code đã tồn tại chưa
    if code:
        check = supabase.table("documents").select("id").eq("code", code).limit(1).execute()
        if check.data and len(check.data) > 0:
            print(f"Đã tồn tại code: {code}, bỏ qua.")
            return

    date_val = parse_date(data.get("date"))
    ngay_ban_hanh_val = parse_date(detail.get("Ngày ban hành"))
    doc = {
        "code": code,
        "date": date_val.isoformat() if date_val else None,
        "substract": data.get("substract"),
        "detail_link": data.get("detail_link"),
        "so_ky_hieu": detail.get("Số ký hiệu"),
        "ngay_ban_hanh": ngay_ban_hanh_val.isoformat() if ngay_ban_hanh_val else None,
        "loai_van_ban": detail.get("Loại văn bản"),
        "co_quan_ban_hanh": detail.get("Cơ quan ban hành"),
        "nguoi_ky": detail.get("Người ký"),
        "trich_yeu": detail.get("Trích yếu"),
    }
    res = supabase.table("documents").insert(doc).execute()
    doc_id = res.data[0]['id']

    # Insert file_links
    for link in data.get("file_links", []):
        supabase.table("file_links").insert({
            "document_id": doc_id,
            "file_link": link
        }).execute()

    # Insert file_links_local
    for link in data.get("file_links_local", []):
        supabase.table("file_links_local").insert({
            "document_id": doc_id,
            "file_link_local": link
        }).execute()

def main():
    folder = r'C:\Users\phucdz\code\RAG\standard_texts'
    for fname in os.listdir(folder):
        if fname.endswith(".md"):
            with open(os.path.join(folder, fname), encoding="utf-8") as f:
                data = json.load(f)
                upload_document(data)
    print("Done uploading all documents.")

if __name__ == "__main__":
    main()