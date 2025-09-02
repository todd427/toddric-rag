import argparse, os, sqlite3

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS docs(
        doc_id   TEXT PRIMARY KEY,
        title    TEXT,
        source   TEXT,
        checksum TEXT,
        meta_json TEXT
    );""",
    """CREATE TABLE IF NOT EXISTS chunks(
        chunk_id  TEXT PRIMARY KEY,
        doc_id    TEXT,
        ord       INTEGER,
        text      TEXT,
        meta_json TEXT,
        FOREIGN KEY(doc_id) REFERENCES docs(doc_id)
    );""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
        text, chunk_id UNINDEXED, doc_id UNINDEXED, tokenize='porter'
    );""",
    """CREATE TABLE IF NOT EXISTS vec_chunks(
        chunk_id TEXT PRIMARY KEY,
        dim      INTEGER NOT NULL,
        vec      BLOB    NOT NULL
    );""",
    """CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ord);"""
]

def load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    try:
        conn.enable_load_extension(True)
        for lib in ('vector0','vec0','sqlite_vec','sqlite-vector'):
            try:
                conn.load_extension(lib)
                return True
            except Exception:
                continue
    except Exception:
        pass
    return False

def init_db(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        for s in SCHEMA:
            cur.execute(s)
        conn.commit()
    finally:
        conn.close()

def upsert_fts(conn: sqlite3.Connection, chunk_id: str, doc_id: str, text: str):
    conn.execute("DELETE FROM fts_chunks WHERE chunk_id=?", (chunk_id,))
    conn.execute("INSERT INTO fts_chunks (text, chunk_id, doc_id) VALUES (?, ?, ?)", (text, chunk_id, doc_id))

def upsert_vec(conn: sqlite3.Connection, chunk_id: str, vec, dim: int):
    conn.execute("REPLACE INTO vec_chunks (chunk_id, dim, vec) VALUES (?, ?, ?)", (chunk_id, dim, vec.tobytes()))

def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def main():
    ap = argparse.ArgumentParser(description="Initialize RAG SQLite database")
    ap.add_argument("--db", default="./store/rag.sqlite")
    args = ap.parse_args()
    init_db(args.db)
    print(f"Initialized DB at {args.db}")

if __name__ == "__main__":
    main()
