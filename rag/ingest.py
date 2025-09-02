import argparse, glob, json, os, re, sqlite3
from typing import Dict
from datetime import datetime, timezone

from . import db
from .chunkers import simple_chunks, checksum
from .embeddings import EmbeddingModel

def read_sources(src: str):
    paths = []
    if os.path.isdir(src):
        for root, _, files in os.walk(src):
            for f in files:
                paths.append(os.path.join(root, f))
    elif os.path.isfile(src):
        paths = [src]
    else:
        return
    for p in paths:
        if os.path.isdir(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in (".txt", ".md"):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                yield {"source": p, "title": os.path.basename(p), "text": f.read()}
        elif ext == ".jsonl":
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        text = obj.get("text") or obj.get("content") or ""
                        title = obj.get("title") or os.path.basename(p)
                        yield {"source": p, "title": title, "text": str(text)}
                    except Exception:
                        continue

def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def main():
    ap = argparse.ArgumentParser(description="Ingest into RAG store (SQLite + FTS5 + sqlite-vec)")
    ap.add_argument("--src", required=True, help="File or folder to ingest (.txt/.md/.jsonl)")
    ap.add_argument("--db", default="./store/rag.sqlite")
    ap.add_argument("--model", default=None, help="Override embedding model")
    ap.add_argument("--chunk", type=int, default=None, help="Target chunk size (chars)")
    ap.add_argument("--overlap", type=int, default=None, help="Chunk overlap (chars)")
    args = ap.parse_args()

    # Config (yaml optional)
    cfg = {
        "db_path": args.db,
        "emb_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk": {"size": 900, "overlap": 150, "min_len": 200},
    }
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        import yaml  # type: ignore
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                cfg.update(loaded)
    except Exception:
        pass

    db_path = args.db or cfg.get("db_path", "./store/rag.sqlite")
    emb_model_name = args.model or cfg.get("emb_model", "sentence-transformers/all-MiniLM-L6-v2")
    ch_size = args.chunk or cfg.get("chunk", {}).get("size", 900)
    ch_overlap = args.overlap or cfg.get("chunk", {}).get("overlap", 150)
    ch_min = cfg.get("chunk", {}).get("min_len", 200)

    db.init_db(db_path)
    conn = db.connect(db_path)
    vec_loaded = db.load_sqlite_vec(conn)
    print(f"DB ready at {db_path}. sqlite-vec loaded: {vec_loaded}")

    emb = EmbeddingModel(emb_model_name)

    batch_texts, batch_ids = [], []
    for doc in read_sources(args.src):
        raw = doc["text"]
        text = clean_text(raw)
        if not text:
            continue
        # doc identity + meta
        doc_id = checksum(doc["source"] + "::" + text[:1024])
        try:
            source_mtime = datetime.fromtimestamp(os.path.getmtime(doc["source"]), tz=timezone.utc).isoformat()
        except Exception:
            source_mtime = None
        meta_doc: Dict = {"ingested_at_utc": datetime.now(tz=timezone.utc).isoformat()}
        if source_mtime:
            meta_doc["source_mtime_utc"] = source_mtime

        conn.execute(
            "REPLACE INTO docs(doc_id,title,source,checksum,meta_json) VALUES (?,?,?,?,?)",
            (doc_id, doc.get("title") or doc_id, doc["source"], checksum(text), json.dumps(meta_doc, ensure_ascii=False)),
        )

        # chunk + stage embeddings
        for ch in simple_chunks(text, size=ch_size, overlap=ch_overlap, min_len=ch_min):
            ord_idx = ch["ord"]
            ctext = ch["text"]
            chunk_id = checksum(doc_id + f"#{ord_idx}" + ctext[:64])
            conn.execute(
                "REPLACE INTO chunks(chunk_id,doc_id,ord,text,meta_json) VALUES (?,?,?,?,?)",
                (chunk_id, doc_id, ord_idx, ctext, "{}"),
            )
            db.upsert_fts(conn, chunk_id, doc_id, ctext)
            batch_texts.append(ctext)
            batch_ids.append(chunk_id)

            if len(batch_texts) >= 64:
                vecs = emb.encode(batch_texts)
                dim = int(vecs.shape[1])
                for cid, vec in zip(batch_ids, vecs):
                    db.upsert_vec(conn, cid, vec, dim)
                conn.commit()
                batch_texts.clear()
                batch_ids.clear()

    if batch_texts:
        vecs = emb.encode(batch_texts)
        dim = int(vecs.shape[1])
        for cid, vec in zip(batch_ids, vecs):
            db.upsert_vec(conn, cid, vec, dim)
        conn.commit()

    print("Ingest complete.")

if __name__ == "__main__":
    main()
