# toddric-rag\n\nTiny RAG engine (SQLite + FTS5 + sqlite-vec). See rag/ for code.\n
## Debug server

Start a tiny FastAPI server:

```bash
make run-server       # foreground (Ctrl+C to stop)
# or
make up               # background
make logs             # tail logs
make down             # stop
```

Then query:

- JSON API: `http://localhost:8077/rag?q=Where%20did%20Aunt%20Mary%20live?`

- Health: `http://localhost:8077/healthz`



**Time zones:** Timestamps are stored in UTC during ingest and rendered in **Europe/Dublin** on the API (Rose is running IST). 
