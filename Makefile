SHELL := /bin/bash
.ONESHELL:
VENV ?= ../venvs/ragEnv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
DB   ?= ./store/rag.sqlite
SRC  ?= ./data
Q    ?= test
PORT ?= 8077
FILES := $(wildcard $(SRC)/*.txt) $(wildcard $(SRC)/*.md) $(wildcard $(SRC)/*.jsonl)

.DEFAULT_GOAL := all

.PHONY: all help venv init db maybe_ingest ingest query clean run-server up down logs
.PHONY: ask

.PHONY: eval
eval:
	$(PY) -m rag.eval --db $(DB) --qa ./eval/memory_quiz.jsonl --k 12 --alpha 0.4 --hybrid

.PHONY: ci
ci:
	$(PY) -m rag.gate --db $(DB) --qa ./eval/memory_quiz.jsonl --k 12 --alpha 0.4 --hybrid --min_hit_rate 0.90 --max_avg_ms 4000

.PHONY: warm
warm:
	$(PY) -c "from rag.embeddings import get_embedder; get_embedder(); print('embedder warmed')"


ask:
	$(PY) -m rag.answer --db $(DB) --q "$(Q)" --source-like "memories"


all: venv init db build maybe_ingest
	@echo "✔ Ready. DB at $(DB)"
	@echo "Try: make query Q='Where did Aunt Mary live?' --hybrid"

help:
	@echo "Targets:"
	@echo "  all            - create venv, install, init DB, auto-ingest if files exist"
	@echo "  venv           - create virtualenv at $(VENV) if missing"
	@echo "  init           - install package in editable mode"
	@echo "  db             - init sqlite db at $(DB)"
	@echo "  ingest SRC=... - ingest files from SRC (txt/md/jsonl)"
	@echo "  query Q=...    - query the store (use --hybrid for BM25+vec)"
	@echo "  run-server     - FastAPI server at http://localhost:$(PORT)"
	@echo "  up/down/logs   - background server control"
	@echo "  clean          - remove the db file"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		python -m venv "$(VENV)"; \
		echo "Created venv at $(VENV)"; \
	else \
		echo "Using existing venv at $(VENV)"; \
	fi

init:
	$(PIP) install -e .

db:
	$(PY) -m rag.db --db $(DB)

maybe_ingest:
	@if [ -n "$(strip $(FILES))" ]; then \
		echo "Ingesting files from $(SRC)..."; \
		$(PY) -m rag.ingest --src $(SRC) --db $(DB); \
	else \
		echo "No .txt/.md/.jsonl in $(SRC). Skipping ingest."; \
		echo "Add files to $(SRC) and run: make ingest"; \
	fi

ingest:
	$(PY) -m rag.ingest --src $(SRC) --db $(DB)

query:
	$(PY) -m rag.retrieve --db $(DB) --q "$(Q)" --k 5 --hybrid

run-server:
	$(PY) -m uvicorn rag.server:app --host 0.0.0.0 --port $(PORT) --reload

up:
	@mkdir -p ./store
	@nohup $(PY) -m uvicorn rag.server:app --host 0.0.0.0 --port $(PORT) > ./store/server.log 2>&1 & echo $$! > ./store/server.pid
	@echo "Server started on http://localhost:$(PORT) (PID `cat ./store/server.pid`)"
	@echo "Tail logs with: make logs"

down:
	@if [ -f ./store/server.pid ]; then kill `cat ./store/server.pid` || true; rm -f ./store/server.pid; echo "Server stopped."; else echo "No PID file."; fi

logs:
	@tail -n 100 -f ./store/server.log

.PHONY: stats
stats:
	$(PY) -m rag.stats --db $(DB)

.PHONY: build
build:
	$(PY) -m rag.build --sources "$(SOURCES)" --data ./data
