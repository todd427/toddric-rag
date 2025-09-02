# -------- config --------
VENV            ?= ../venvs/ragEnv
PY              := $(VENV)/bin/python
PIP             := $(VENV)/bin/pip
DB              ?= ./store/rag.sqlite
DATA            ?= ./data

# Default SOURCES (edit as needed)
SOURCES ?= $(HOME)/toddric-extract/out/text:\
$(HOME)/toddric-extract/data/project:\
$(HOME)/toddric-extract/data/raw/memories/data:\
$(HOME)/toddric-extract/memories/data:\
$(PWD)/profile

# Query defaults
K        ?= 5
ALPHA    ?= 0.6
SRC_LIKE ?=
KIND     ?= profile

# Wikipedia defaults
WIKI_TITLE ?= Todd McCaffrey
WIKI_LANG  ?= en

.DEFAULT_GOAL := all

# -------- setup --------
.PHONY: venv init db build maybe_ingest stats all

venv:
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv "$(VENV)"; \
	fi
	@$(PIP) install -U pip setuptools wheel >/dev/null
	@$(PIP) install -e . >/dev/null
	@echo "Using venv at $(VENV)"

init:
	@mkdir -p $(DATA) ./store ./profile

db:
	@$(PY) -m rag.db --db $(DB)

build:
	@$(PY) -m rag.build --sources "$(SOURCES)" --data $(DATA)

maybe_ingest:
	@if ls -1 $(DATA)/*.txt $(DATA)/*.md $(DATA)/*.jsonl >/dev/null 2>&1; then \
		$(PY) -m rag.ingest --src $(DATA) --db $(DB); \
	else \
		echo "No .txt/.md/.jsonl in $(DATA). Skipping ingest."; \
	fi

stats:
	@$(PY) -m rag.stats --db $(DB)

all: venv init db build maybe_ingest stats

# -------- actions --------
.PHONY: query ask pull-wiki warm eval ci run-server

query:
	@$(PY) -m rag.retrieve --db $(DB) --q "$(Q)" --k $(K) --alpha $(ALPHA) --hybrid \
		--source-like "$(SRC_LIKE)" --kind "$(KIND)"

# Default ask: hybrid + prefer profile facts to avoid fiction bleed-through
ask:
	@$(PY) -m rag.answer --db $(DB) --q "$(Q)" --hybrid --kind $(KIND)

pull-wiki: venv
	@$(PY) -m rag.pull_wiki --title "$(WIKI_TITLE)" --lang "$(WIKI_LANG)" --out ./profile
	@$(PY) -m rag.build --sources "./profile" --data $(DATA)
	@$(PY) -m rag.ingest --src $(DATA) --db $(DB)
	@$(PY) -m rag.stats --db $(DB)

# Manual warm (useful for long-running servers). Note: separate process; eval/answer warm themselves.
warm:
	@$(PY) -c "from rag.embeddings import get_embedder; get_embedder(); print('embedder warmed')"

# Eval: now warms in-process (via your eval.py patch) and reports proper p95.
eval:
	@$(PY) -m rag.eval --db $(DB) --qa ./eval/memory_quiz.jsonl --k 12 --alpha 0.4 --hybrid

# Quality gate for CI
ci:
	@$(PY) -m rag.gate --db $(DB) --qa ./eval/memory_quiz.jsonl --k 12 --alpha 0.4 --hybrid \
		--min_hit_rate 0.90 --max_avg_ms 4000

# Optional server if/when you add it
run-server:
	@$(PY) -m rag.server --db $(DB) --host 0.0.0.0 --port 8000

