import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .retrieve import retrieve
from .embeddings import get_embedder


def load_jsonl(path: Path) -> Iterable[Dict]:
    """Yield JSON objects per line; skip blanks/bad lines."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                # Ignore malformed lines so eval keeps going
                continue


def tokens_all_in(text: str, tokens: List[str]) -> Tuple[bool, List[str]]:
    """Return (all_present, missing_tokens)."""
    tl = text.lower()
    missing = [t for t in tokens if t.lower() not in tl]
    return (len(missing) == 0, missing)


def tokens_any_in(text: str, tokens: List[str]) -> bool:
    """True if any token is present (case-insensitive)."""
    tl = text.lower()
    return any(t.lower() in tl for t in tokens)


def summarize_hits(hits: List[Dict], max_titles: int = 2) -> str:
    """Human-friendly summary of top hit titles/sources."""
    titles = []
    for h in hits[:max_titles]:
        t = h.get("title") or "<untitled>"
        s = h.get("source") or ""
        titles.append(f"{t} ({s})")
    return "; ".join(titles)


def main():
    ap = argparse.ArgumentParser(description="Simple RAG eval: hit-rate on must_include/any_of tokens.")
    ap.add_argument("--db", default="./store/rag.sqlite", help="Path to SQLite store")
    ap.add_argument("--qa", required=True, help="Path to JSONL: {question, must_include:[...], any_of:[...]} per line")
    ap.add_argument("--k", type=int, default=6, help="Top-K to retrieve")
    ap.add_argument("--alpha", type=float, default=0.6, help="Hybrid fusion weight (0..1)")
    ap.add_argument("--hybrid", action="store_true", help="Use hybrid (FTS + vectors)")
    ap.add_argument("--kind", default="", help="Restrict to doc kind: profile|wiki|memories|books")
    ap.add_argument("--source-like", default="", help="Restrict by source/title substring (SQL LIKE pattern)")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N questions")
    args = ap.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        raise SystemExit(f"QA file not found: {qa_path}")

    # Warm the embedder in THIS process so the first query isn't slow.
    get_embedder()

    total = 0
    passed = 0
    latencies: List[float] = []
    failures: List[Dict] = []

    for i, rec in enumerate(load_jsonl(qa_path), start=1):
        if args.limit and i > args.limit:
            break

        q = rec.get("question") or rec.get("q")
        if not q:
            continue

        # Per-record overrides (optional)
        must = rec.get("must_include") or []
        any_of = rec.get("any_of") or []
        kind = rec.get("kind") or args.kind
        source_like = rec.get("source_like") or args.source_like
        top_k = int(rec.get("k") or args.k)
        alpha = float(rec.get("alpha") or args.alpha)
        hybrid = bool(rec.get("hybrid") if rec.get("hybrid") is not None else args.hybrid)

        total += 1
        t0 = time.perf_counter()
        hits = retrieve(
            q,
            db_path=args.db,
            k=top_k,
            hybrid=hybrid,
            alpha=alpha,
            source_like=source_like,
            kind=kind,
        ) or []
        dt = time.perf_counter() - t0
        latencies.append(dt)

        combined = " ".join((h.get("text") or "") for h in hits)
        ok_all, missing = tokens_all_in(combined, must) if must else (True, [])
        ok_any = tokens_any_in(combined, any_of) if any_of else True
        ok_hits = len(hits) > 0
        ok = ok_all and ok_any and ok_hits

        status = "PASS" if ok else "FAIL"
        print(f"[{i:02d}] {status}  {dt*1000:.0f} ms  Q: {q}")
        print(f"     hits={len(hits)}  tops: {summarize_hits(hits)}")
        if not ok:
            if must:
                print(f"     missing(all): {missing}")
            if any_of:
                print(f"     any_of: {any_of}  present={tokens_any_in(combined, any_of)}")
        print("")

        if ok:
            passed += 1
        else:
            failures.append({
                "q": q,
                "missing": missing,
                "any_of": any_of,
                "tops": [h.get("title") for h in hits[:3]],
            })

    if total == 0:
        raise SystemExit("No questions found in the QA file.")

    n = len(latencies)
    avg_ms = (sum(latencies) / n) * 1000.0 if n else 0.0
    p95_idx = max(0, math.ceil(0.95 * n) - 1)
    p95_ms = sorted(latencies)[p95_idx] * 1000.0 if n else 0.0

    print("=" * 60)
    print(f"Total: {total}  Passed: {passed}  Hit-rate: {passed/total:.0%}")
    print(f"Latency: avg={avg_ms:.0f} ms  p95={p95_ms:.0f} ms")
    if failures:
        print("Failures (first 5):")
        for f in failures[:5]:
            print(f" - Q: {f['q']}  missing={f['missing']}  any_of={f['any_of']}  tops={f['tops']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

