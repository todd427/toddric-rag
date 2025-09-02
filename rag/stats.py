import argparse, sqlite3, json
ap = argparse.ArgumentParser()
ap.add_argument("--db", default="./store/rag.sqlite")
ap.add_argument("--limit", type=int, default=10)
a = ap.parse_args()
conn = sqlite3.connect(a.db); conn.row_factory = sqlite3.Row
docs = conn.execute("select count(*) as n from docs").fetchone()["n"]
chunks = conn.execute("select count(*) as n from chunks").fetchone()["n"]
vecs = conn.execute("select count(*) as n from vec_chunks").fetchone()["n"]
print(f"docs={docs}  chunks={chunks}  vectors={vecs}")
print("\nTop docs by chunks:")
for r in conn.execute("""select d.title, d.source, count(*) as chunks
                           from chunks c join docs d on d.doc_id=c.doc_id
                          group by c.doc_id order by chunks desc limit ?""", (a.limit,)):
    print(f"- {r['title']}  ({r['chunks']} chunks)")
print("\nRecent docs:")
for r in conn.execute("""select title, source, meta_json
                           from docs order by rowid desc limit ?""", (a.limit,)):
    meta = {}
    try: meta = json.loads(r["meta_json"] or "{}")
    except: pass
    print(f"- {r['title']}  :: ingested_at_utc={meta.get('ingested_at_utc')}")
