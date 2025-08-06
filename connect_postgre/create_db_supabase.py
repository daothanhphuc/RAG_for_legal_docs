import psycopg2
from dotenv import load_dotenv
import os

# Load biến môi trường từ file .env
load_dotenv()

PG_USER = os.getenv("user")
PG_PASSWORD = os.getenv("password")
PG_HOST = os.getenv("host")
PG_PORT = int(os.getenv("port", 5432))
PG_DBNAME = os.getenv("dbname")

conn = psycopg2.connect(
    host=PG_HOST,
    dbname=PG_DBNAME,
    user=PG_USER,
    password=PG_PASSWORD,
    port=PG_PORT
)

sql = """
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    code TEXT,
    date DATE,
    substract TEXT,
    detail_link TEXT,
    so_ky_hieu TEXT,
    ngay_ban_hanh DATE,
    loai_van_ban TEXT,
    co_quan_ban_hanh TEXT,
    nguoi_ky TEXT,
    trich_yeu TEXT
);

CREATE TABLE IF NOT EXISTS file_links (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    file_link TEXT
);

CREATE TABLE IF NOT EXISTS file_links_local (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    file_link_local TEXT
);

CREATE TABLE IF NOT EXISTS document_contents (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    file_link_local TEXT,
    content TEXT
);
"""

def main():
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print("Đã tạo bảng thành công.")

# if __name__ == "__main__":
    main()
