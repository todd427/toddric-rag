import argparse, re, subprocess, sys

def main():
    ap = argparse.ArgumentParser(description="Quality gate wrapper for rag.eval")
    ap.add_argument("--db", default="./store/rag.sqlite")
    ap.add_argument("--qa", default="./eval/memory_quiz.jsonl")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--min_hit_rate", type=float, default=0.90)
    ap.add_argument("--max_avg_ms", type=int, default=4000)
    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "rag.eval",
        "--db", args.db, "--qa", args.qa,
        "--k", str(args.k), "--alpha", str(args.alpha)
    ]
    if args.hybrid:
        cmd.append("--hybrid")

    print("Running:", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)

    # Parse the footer lines from rag.eval
    m_rate = re.search(r"Hit-rate:\s+(\d+)%", out.stdout)
    m_avg  = re.search(r"Latency:\s+avg=(\d+)\s+ms", out.stdout)
    if not (m_rate and m_avg):
        print("Gate: could not parse eval output.", file=sys.stderr)
        sys.exit(2)

    rate = int(m_rate.group(1)) / 100.0
    avg  = int(m_avg.group(1))

    ok = True
    if rate < args.min_hit_rate:
        print(f"Gate: FAIL hit-rate {rate:.0%} < {args.min_hit_rate:.0%}", file=sys.stderr)
        ok = False
    if avg > args.max_avg_ms:
        print(f"Gate: FAIL avg latency {avg}ms > {args.max_avg_ms}ms", file=sys.stderr)
        ok = False

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

