from supabase import create_client
import os

url = "https://jfznazmrqxjmqnfkavmg.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impmem5hem1ycXhqbXFuZmthdm1nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2NDYyNjEsImV4cCI6MjA2ODIyMjI2MX0.djD_Zmg7J43iFDJr8MOLpACy56HbHPWBbGOP7TCBzrE"
supabase = create_client(url, key)

# Ví dụ: truy vấn bảng documents
# response = supabase.table("documents").select("*").limit(3).execute()
# print(response.data)

# response1 = supabase.table("file_links").select("*").limit(1).execute()
# print(response1)

# response1 = supabase.table("file_links_local").select("*").limit(5).execute()
# print(response1)
response2 = supabase.table("document_contents").select("*").eq("document_id", 4191).limit(1).execute()
print(response2)

# response3 = supabase.table("document_contents").select("*").limit(5).execute()
# print(response3)



# response1 = supabase.table("file_links_local")\
#     .select("document_id")\
#     .eq("file_link_local", "5871533758c8ea7a5c6edc1f2fcebdcacad99b3d.xls")\
#     .limit(1)\
#     .execute()

# if response1.data:
#     print("Tồn tại trong bảng file_links_local")
#     print("document_id liên kết:", response1.data[0]["document_id"])
# else:
#     print("Không tìm thấy trong bảng file_links_local")