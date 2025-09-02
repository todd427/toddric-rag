import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .retrieve import retrieve


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def tokens_all_in(text: str, tokens: List[str]) -> Tuple[bool, List[str]]:
    """Return (all_present, missing_tokens)."""
    tl = text.lower()
    missing = [t for t in tokens if t.lower() not in tl]
    return (len(missing) == 0, missing)


def summarize_hits(hits: List[Dict], max_titles: int = 2) -> str:
    titles = []
    for h in hits[:max_titles]:
        t = h.get("title") or "<untitled>"
        s = h.get("source") or ""
        titles.append(f"{t} ({s})")
    return "; ".join(titles)


def main():
    ap = argparse.ArgumentParser(description="Simple RAG eval: hit-rate on must_include tokens.")
    ap.add_argument("--db", default="./store/rag.sqlite", help="Path to SQLite store")
    ap.add_argument("--qa", required=True, help="Path to JSONL: {question, must_include:[...]}")
    ap.add_argument("--k", type=int, default=6, help="Top-K to retrieve")
    ap.add_argument("--alpha", type=float, default=0.6, help="Hybrid fusion weight")
    ap.add_argument("--hybrid", action="store_true", help="Use hybrid (FTS + vectors)")
    ap.add_argument("--kind", default="", help="Restrict to doc kind: profile|wiki|memories|books")
    ap.add_argument("--source-like", default="", help="Restrict by source/title substring")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N questions")
    args = ap.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        raise SystemExit(f"QA file not found: {qa_path}")

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
        must = rec.get("must_include") or []
        # allow record-level overrides
        kind = rec.get("kind") or args.kind
        source_like = rec.get("source_like") or args.source_like

        total += 1
        t0 = time.time()
        hits = retrieve(
            q,
            db_path=args.db,
            k=args.k,
            hybrid=args.hybrid,
            alpha=args.alpha,
            source_like=source_like,
            kind=kind,
        ) or []
        dt = time.time() - t0
        latencies.append(dt)

        combined = " ".join((h.get("text") or "") for h in hits)
        ok, missing = tokens_all_in(combined, must) if must else (len(hits) > 0, [])

        status = "PASS" if ok else "FAIL"
        print(f"[{i:02d}] {status}  {dt*1000:.0f} ms  Q: {q}")
        print(f"     hits={len(hits)}  tops: {summarize_hits(hits)}")
        if not ok:
            print(f"     missing: {missing}")
        print("")

        if ok:
            passed += 1
        else:
            failures.append({
                "q": q,
                "missing": missing,
                "tops": [h.get("title") for h in hits[:3]],
            })

    if total == 0:
        raise SystemExit("No questions found in the QA file.")

    rate = passed / total
    avg_ms = (sum(latencies) / len(latencies)) * 1000.0 if latencies else 0.0
    p95_ms = sorted(latencies)[int(0.95 * max(1, len(latencies)-1))] * 1000.0 if latencies else 0.0

    print("=" * 60)
    print(f"Total: {total}  Passed: {passed}  Hit-rate: {rate:.0%}")
    print(f"Latency: avg={avg_ms:.0f} ms  p95={p95_ms:.0f} ms")
    if failures:
        print("Failures (first 5):")
        for f in failures[:5]:
            print(f" - Q: {f['q']}  missing={f['missing']}  tops={f['tops']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

