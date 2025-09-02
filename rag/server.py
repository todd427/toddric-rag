from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import Optional
from .retrieve import retrieve
from . import db as dbmod
from datetime import datetime
from zoneinfo import ZoneInfo

APP_TZ = ZoneInfo("Europe/Dublin")
app = FastAPI(title="toddric-rag", version="0.1.0")

@app.get("/healthz")
def healthz():
    now_local = datetime.now(tz=APP_TZ).isoformat()
    return {"ok": True, "time_local": now_local, "tz": "Europe/Dublin"}

@app.get("/rag")
def rag(q: str = Query(..., description="Query text"), k: int = 5, hybrid: bool = True, db: str = "./store/rag.sqlite"):
    try:
        results = retrieve(q, db_path=db, k=k, hybrid=hybrid)
        # enrich with doc meta (including created_at if present)
        conn = dbmod.connect(db)
        enriched = []
        for r in results:
            row = conn.execute("SELECT title, source, checksum, meta_json FROM docs WHERE doc_id=?", (r["doc_id"],)).fetchone()
            meta = {}
            if row and row["meta_json"]:
                try:
                    meta = json.loads(row["meta_json"])
                except Exception:
                    meta = {}
            # Render any UTC timestamps in local time if present
            def to_local(ts: Optional[str]):
                if not ts: return None
                try:
                    # Expect ISO with offset or 'Z'. Normalize 'Z' to '+00:00'
                    ts_norm = ts.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(ts_norm)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                    return dt.astimezone(APP_TZ).isoformat()
                except Exception:
                    return ts
            meta_local = meta.copy()
            for key in ("ingested_at_utc", "source_mtime_utc"):
                if key in meta:
                    meta_local[key.replace("_utc", "_local")] = to_local(meta[key])
            enriched.append({**r, "meta": meta_local})
        return JSONResponse(enriched)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
