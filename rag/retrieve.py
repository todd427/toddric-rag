import argparse
import re
import sqlite3
from typing import Dict, List

import numpy as np

from . import db


def _sanitize_fts(q: str) -> str:
    """
    Keep only alphanumeric tokens and join with spaces so we can safely embed
    into a literal FTS MATCH expression (no quotes/AND/OR/etc).
    """
    toks = re.findall(r"[A-Za-z0-9]+", q)
    return " ".join(toks) if toks else ""


def _bm25(conn: sqlite3.Connection, q: str, k: int = 8) -> List[Dict]:
    """
    Try FTS5 MATCH using a literal expression; if the SQLite build complains,
    fall back to a LIKE scan. Convert bm25() (lower-is-better) to higher-is-better.
    """
    expr = _sanitize_fts(q)
    rows = []
    if expr:
        try:
            sql = (
                "SELECT chunk_id, doc_id, text, bm25(fts_chunks) AS score "
                f"FROM fts_chunks WHERE fts_chunks MATCH '{expr}' "
                "ORDER BY score LIMIT ?"
            )
            rows = conn.execute(sql, (k * 4,)).fetchall()
        except sqlite3.OperationalError:
            rows = []
    if not rows:
        like = f"%{q.strip()}%"
        rows = conn.execute(
            "SELECT chunk_id, doc_id, text, 0.0 AS score "
            "FROM fts_chunks WHERE text LIKE ? LIMIT ?",
            (like, k * 4),
        ).fetchall()

    out = []
    for r in rows:
        s = -float(r["score"]) if "score" in r.keys() else 0.0
        out.append(
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "text": r["text"],
                "bm25": s,
            }
        )
    return out


def _vec(
    conn: sqlite3.Connection,
    q: str,
    k: int = 8,
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Dict]:
    try:
        from .embeddings import EmbeddingModel

        emb = EmbeddingModel(emb_model)
    except Exception:
        return []

    v = emb.encode([q])[0]
    dim = int(v.shape[0])

    rows = conn.execute("SELECT chunk_id, dim, vec FROM vec_chunks").fetchall()
    scored: List[tuple[str, float]] = []
    for r in rows:
        if int(r["dim"]) != dim:
            continue
        vec = np.frombuffer(r["vec"], dtype=np.float32)
        scored.append((r["chunk_id"], float(np.dot(v, vec))))
    scored.sort(key=lambda x: x[1], reverse=True)

    out: List[Dict] = []
    for cid, sc in scored[: k * 4]:
        row = conn.execute(
            "SELECT chunk_id, doc_id, text FROM chunks WHERE chunk_id=?", (cid,)
        ).fetchone()
        if row:
            out.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "vec": sc,
                }
            )
    return out


def fuse(
    bm25_list: List[Dict], vec_list: List[Dict], alpha: float = 0.6, k: int = 5
) -> List[Dict]:
    """
    Score fusion: normalize each list to 0..1 then weighted sum.
    Prefer diverse docs first (one per doc) until at least k//2, then fill.
    """
    by_id: Dict[str, Dict] = {}

    def norm(vals):
        if not vals:
            return {}
        arr = np.array(vals, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-6:
            return {i: 1.0 for i in range(len(vals))}
        return {i: (vals[i] - lo) / (hi - lo) for i in range(len(vals))}

    bm_vals = [d.get("bm25", 0.0) for d in bm25_list]
    vec_vals = [d.get("vec", 0.0) for d in vec_list]
    bm_norm, vec_norm = norm(bm_vals), norm(vec_vals)

    for i, d in enumerate(bm25_list):
        by_id.setdefault(d["chunk_id"], {**d, "vec": 0.0, "score": 0.0})
        by_id[d["chunk_id"]]["score"] += (1 - alpha) * bm_norm.get(i, 0.0)
    for i, d in enumerate(vec_list):
        by_id.setdefault(d["chunk_id"], {**d, "score": 0.0})
        by_id[d["chunk_id"]]["score"] += alpha * vec_norm.get(i, 0.0)

    merged = list(by_id.values())
    merged.sort(key=lambda x: x["score"], reverse=True)

    seen, result = set(), []
    for d in merged:
        if d["doc_id"] in seen and len(result) < k // 2:
            continue
        result.append(d)
        seen.add(d["doc_id"])
        if len(result) >= k:
            break
    return result


def _by_source(conn: sqlite3.Connection, rows: List[Dict], source_like: str) -> List[Dict]:
    """Filter candidates to docs whose title OR source LIKE %source_like%."""
    if not source_like:
        return rows
    ok = {
        r["doc_id"]
        for r in conn.execute(
            "SELECT doc_id FROM docs WHERE source LIKE ? OR title LIKE ?",
            (f"%{source_like}%", f"%{source_like}%"),
        )
    }
    return [r for r in rows if r["doc_id"] in ok]


def _by_kind(conn: sqlite3.Connection, rows: List[Dict], kind: str) -> List[Dict]:
    """Filter candidates by docs.meta_json.kind == kind (e.g., profile|wiki|memories|books)."""
    if not kind:
        return rows
    ok = {
        r["doc_id"]
        for r in conn.execute(
            "SELECT doc_id FROM docs WHERE json_extract(meta_json,'$.kind') = ?", (kind,)
        )
    }
    return [r for r in rows if r["doc_id"] in ok]


def retrieve(
    query: str,
    db_path: str,
    k: int = 5,
    hybrid: bool = True,
    alpha: float = 0.6,
    source_like: str = "",
    kind: str = "",
) -> List[Dict]:
    conn = db.connect(db_path)
    if hybrid:
        bm25_top = _bm25(conn, query, k=k)
        vec_top = _vec(conn, query, k=k)
        bm25_top = _by_source(conn, bm25_top, source_like)
        vec_top = _by_source(conn, vec_top, source_like)
        bm25_top = _by_kind(conn, bm25_top, kind)
        vec_top = _by_kind(conn, vec_top, kind)
        fused = fuse(bm25_top, vec_top, alpha=alpha, k=k)
    else:
        fused = _by_source(conn, _bm25(conn, query, k=k), source_like)
        fused = _by_kind(conn, fused, kind)[:k]
        fused = [{**d, "score": float(d.get("bm25", 0.0))} for d in fused]

    out: List[Dict] = []
    for r in fused:
        meta = conn.execute(
            "SELECT title, source FROM docs WHERE doc_id=?", (r["doc_id"],)
        ).fetchone()
        out.append(
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "title": meta["title"] if meta else None,
                "source": meta["source"] if meta else None,
                "text": r["text"],
                "score": float(r["score"]),
            }
        )
    conn.close()
    return out


def main():
    ap = argparse.ArgumentParser(description="Query RAG store")
    ap.add_argument("--db", default="./store/rag.sqlite")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--source-like", default="", help="Filter docs by title/source substring (e.g., 'age', 'wiki').")
    ap.add_argument("--kind", default="", help="Restrict to doc kind: profile|wiki|memories|books")
    a = ap.parse_args()

    res = retrieve(
        a.q,
        db_path=a.db,
        k=a.k,
        hybrid=a.hybrid,
        alpha=a.alpha,
        source_like=a.source_like,
        kind=a.kind,
    )
    for i, r in enumerate(res, 1):
        print(f'[{i}] {r["title"]} ({r["source"]}) :: score={r["score"]:.3f}')
        print(r["text"][:300])
        print("---")


if __name__ == "__main__":
    main()

