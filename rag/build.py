import argparse, os, shutil, glob
from pathlib import Path
ALLOWED = {".txt", ".md", ".jsonl"}

def gather(spec: str):
    paths=[]
    for token in filter(None,(t.strip() for t in spec.split(":"))):
        if any(ch in token for ch in "*?[]"): paths.extend(glob.glob(os.path.expanduser(token), recursive=True))
        else: paths.append(os.path.expanduser(token))
    return [Path(p) for p in paths]

def main():
    ap = argparse.ArgumentParser(description="Copy allowed files into ./data for ingest")
    ap.add_argument("--sources", default="", help="Colon-separated dirs/globs")
    ap.add_argument("--data", default="./data")
    a = ap.parse_args()
    out = Path(a.data); out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in gather(a.sources):
        if p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    src = Path(root)/f
                    if src.suffix.lower() in ALLOWED:
                        dst = out / src.name
                        if dst.resolve()!=src.resolve():
                            shutil.copy2(src, dst); copied += 1
        elif p.is_file() and p.suffix.lower() in ALLOWED:
            dst = out / p.name
            if dst.resolve()!=p.resolve():
                shutil.copy2(p, dst); copied += 1
    print(f"build: copied={copied} -> {out}")
if __name__ == "__main__": main()
