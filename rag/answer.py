import argparse
import re
from typing import Iterable, List, Optional, Tuple

from .retrieve import retrieve

# By default, we prefer these sources for personal facts to avoid fiction bleed-through.
ALLOWED_HINTS = (
    "wiki-",        # ./data/wiki-*.txt
    "faq.jsonl",    # ./data/faq.jsonl
    "bio.md",       # ./data/bio.md
    "timeline.md",  # ./data/timeline.md
    "/age",         # ./data/age6.txt, age9.txt, etc.
    "age",          # also match filenames like age6.txt
)

# ------------ Extractors ------------

MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

DATE_PATTERNS = [
    rf"\b{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",  # January 3, 1970
    rf"\b\d{{1,2}}\s+{MONTH}\s+\d{{4}}\b",                    # 3 January 1970
    r"\b\d{4}-\d{2}-\d{2}\b",                                 # 1970-01-03
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",                           # 01/03/1970
    r"\b\d{1,2}[-.](?:\d{1,2}|[A-Za-z]{3})[-.]\d{2,4}\b",     # 3-May-1970 / 03-05-70
    r"\bborn\s+(?:on|in)\s+(?:the\s+)?\d{4}\b",               # born in 1970 (year-only)
]

PLACE_PATTERNS = [
    r"\bborn\s+in\s+([A-Za-z .,'-]+?)\b(?:[,.;]|$)",
    r"\bbirthplace\s*[:\-]\s*([A-Za-z .,'-]+?)(?:[,.;]|$)",
]

# Generic FAQ catcher: "Answer: <text>."
FAQ_ANSWER_PAT = re.compile(r"Answer\s*:\s*(.+?)(?:[.;]|$)", re.IGNORECASE)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def extract_date(text: str) -> Optional[str]:
    t = _norm_ws(text)
    for pat in DATE_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def extract_place(text: str) -> Optional[str]:
    t = _norm_ws(text)
    for pat in PLACE_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # as a fallback, try FAQ "Answer: City, Country"
    faq = FAQ_ANSWER_PAT.search(t)
    if faq:
        return faq.group(1).strip()
    return None


def extract_name(text: str) -> Optional[str]:
    t = _norm_ws(text)
    # Prefer FAQ-style answers
    faq = FAQ_ANSWER_PAT.search(t)
    if faq:
        return faq.group(1).strip()
    # Crude capitalized-name heuristic (two+ capitalized tokens)
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", t)
    if m:
        return m.group(1).strip()
    return None


def guess_intent(question: str) -> str:
    q = (question or "").lower()
    if ("married" in q or "wedding" in q) and "when" in q: return "date"
    if "divorc" in q and "when" in q: return "date"

    if "date of birth" in q or "dob" in q or ("born" in q and ("when" in q or "what" in q)):
        return "date"
    if "where" in q and ("born" in q or "birthplace" in q):
        return "place"
    if q.startswith("who ") or " who " in q:
        return "name"
    # Fallback to FAQ-like extraction if present
    if "question:" in q or "answer:" in q:
        return "faq"
    return "generic"


def pick_hits(hits: Iterable[dict], allow_any_source: bool) -> List[dict]:
    if allow_any_source:
        return list(hits)
    # Prefer verified personal sources; if none match, fall back to all.
    vetted = [h for h in hits if any(hint in (h.get("source") or "") for hint in ALLOWED_HINTS)]
    return vetted if vetted else list(hits)


def scan_for_answer(hits: List[dict], intent: str) -> Optional[Tuple[str, dict, str]]:
    """
    Return (answer, hit, snippet) or None.
    """
    for h in hits:
        txt = _norm_ws(h.get("text") or "")
        if not txt:
            continue

        ans = None
        if intent == "date":
            ans = extract_date(txt)
        elif intent == "place":
            ans = extract_place(txt)
        elif intent == "name":
            ans = extract_name(txt)
        elif intent == "faq":
            m = FAQ_ANSWER_PAT.search(txt)
            ans = m.group(1).strip() if m else None
        else:
            # Try FAQ first, then date/place/name as a loose fallback
            m = FAQ_ANSWER_PAT.search(txt)
            ans = m.group(1).strip() if m else (extract_date(txt) or extract_place(txt) or extract_name(txt))

        if ans:
            # make a short context window around the match if possible
            idx = txt.lower().find(ans.lower())
            if idx >= 0:
                lo = max(0, idx - 80)
                hi = min(len(txt), idx + len(ans) + 80)
                snippet = txt[lo:hi]
            else:
                snippet = txt[:160]
            return ans, h, snippet
    return None


def main():
    ap = argparse.ArgumentParser(description="Answer helper: retrieves then extracts a short answer with provenance.")
    ap.add_argument("--db", default="./store/rag.sqlite")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--kind", default="", help="Restrict to doc kind: profile|wiki|memories|books")
    ap.add_argument("--source-like", default="", help="Restrict by source/title substring (e.g., 'wiki', 'bio', 'age')")
    ap.add_argument("--allow-any-source", action="store_true", help="If set, do not restrict to verified personal sources.")
    args = ap.parse_args()

    hits = retrieve(
        args.q,
        db_path=args.db,
        k=args.k,
        hybrid=args.hybrid,
        alpha=args.alpha,
        source_like=args.source_like,
        kind=args.kind,
    ) or []

    # Order/filter hits
    hits = pick_hits(hits, allow_any_source=args.allow_any_source)

    intent = guess_intent(args.q)
    found = scan_for_answer(hits, intent)

    if found:
        ans, h, snippet = found
        title = h.get("title") or "<untitled>"
        src = h.get("source") or ""
        print(f"Answer: {ans}")
        print(f"From: {title}  ({src})")
        print(f"...{snippet}...")
        return

    # Fallback: show top hits context to help you refine profile data or filters
    print("⚠️ No direct answer found in the top hits.\n")
    if not args.allow_any_source:
        print("(Personal-sources filter was active. Re-run with --allow-any-source to broaden.)")
    for i, h in enumerate(hits[:3], 1):
        title = h.get("title") or "<untitled>"
        src = h.get("source") or ""
        preview = _norm_ws(h.get("text") or "")[:240]
        print(f"[{i}] {title}  ({src})")
        print(preview)
        print("---")


if __name__ == "__main__":
    main()

