import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

UA = "toddric-rag/0.1 (+contact: you@example.com)"  # set a real contact if you want

def fetch_json(url: str):
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))

def pick_page(pages: dict):
    # pages is a dict keyed by pageid strings; return the first page object
    for _, v in pages.items():
        return v
    return None

def pull(title: str, lang: str = "en") -> dict:
    t_enc = quote(title.replace(" ", "_"))
    base = f"https://{lang}.wikipedia.org/w/api.php"

    # 1) Plaintext extract
    url_extract = (
        f"{base}?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles={t_enc}"
    )
    j1 = fetch_json(url_extract)
    page1 = pick_page(j1.get("query", {}).get("pages", {})) or {}
    extract = page1.get("extract") or ""

    # 2) Canonical URL + lastrevid
    url_info = (
        f"{base}?action=query&prop=info|revisions&rvprop=ids&inprop=url|canonicalurl"
        f"&format=json&titles={t_enc}"
    )
    j2 = fetch_json(url_info)
    page2 = pick_page(j2.get("query", {}).get("pages", {})) or {}
    canonicalurl = page2.get("canonicalurl") or page2.get("fullurl") or ""
    lastrevid = page2.get("lastrevid")

    return {
        "title": page2.get("title") or page1.get("title") or title,
        "canonicalurl": canonicalurl,
        "lastrevid": lastrevid,
        "extract": extract,
    }

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def main():
    ap = argparse.ArgumentParser(description="Pull a Wikipedia page (plaintext) with attribution.")
    ap.add_argument("--title", default="Todd McCaffrey")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", default="./profile", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pull(args.title, args.lang)
    text = data.get("extract", "").strip()
    if not text:
        raise SystemExit(f"Could not fetch plaintext extract for '{args.title}' ({args.lang}).")

    meta = {
        "title": data.get("title") or args.title,
        "canonicalurl": data.get("canonicalurl"),
        "lastrevid": data.get("lastrevid"),
        "retrieved_utc": datetime.now(tz=timezone.utc).isoformat(),
        "license": "CC BY-SA 3.0/4.0 (Wikipedia content)",
    }

    fname = out_dir / f"wiki-{slugify(meta['title'])}.txt"
    header = (
        f"Source: Wikipedia — {meta['title']}\n"
        f"URL:    {meta['canonicalurl']}\n"
        f"RevID:  {meta['lastrevid']}\n"
        f"Retrieved: {meta['retrieved_utc']}\n"
        "License: CC BY-SA (https://creativecommons.org/licenses/by-sa/4.0/ "
        "or https://creativecommons.org/licenses/by-sa/3.0/)\n"
        "Notes: This text may be modified downstream (RAG chunking/paraphrase).\n\n"
    )
    fname.write_text(header + text + "\n", encoding="utf-8")
    print(f"wrote: {fname}")

if __name__ == "__main__":
    main()
