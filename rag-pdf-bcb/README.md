# RAG over PDFs: the Central Bank's Monetary Policy Report

Code for three Salada de Dados articles in the RAG series:

- Part 2, **"RAG over PDFs: What Gets Lost Between the Table and the Text"**
  ([English](https://saladadedados.com/en/blog/rag-pdf-tables-images) ·
  [Português](https://saladadedados.com/pt/blog/rag-pdf-tables-images)): four ways to turn a PDF into
  text, with small local models. Runs for free, with no API key.
- Part 3, **"Why the Chat Reads Your Screenshot and Your RAG Does Not"**
  ([English](https://saladadedados.com/en/blog/rag-pdf-page-as-image) ·
  [Português](https://saladadedados.com/pt/blog/rag-pdf-page-as-image)): the same search, with the page
  sent as an image to four large models through OpenRouter. Needs an API key and costs a few dollars.
- Part 4, **"RAG as a Tool: An Agent Found What the Fixed Search Missed"**
  ([English](https://saladadedados.com/en/blog/rag-mcp-agent) ·
  [Português](https://saladadedados.com/pt/blog/rag-mcp-agent)): the chunks in Postgres + pgvector,
  served by an MCP server with two tools, and an agent that decides how to search. Needs an API key.

All three use the Banco Central do Brasil's Monetary Policy Report (September 2026) in English and
Portuguese, and the same 33 questions with known answers: 12 in the text, 11 in tables and 10 only in charts.

## What's inside

| File | What it does |
|---|---|
| `download_rpm.py` | Downloads the report in Portuguese and English to `data/` and checks it is the same file the article used |
| `extract.py` | Extracts both PDFs with `pypdf`, `pymupdf4llm` and Docling, one `<!-- page N -->` marker per page, and times each one |
| `describe.py` | Describes every figure Docling found with a local vision model (`qwen2.5vl:3b`) and writes the Docling + vision version |
| `questions.json` | The 33 questions, in both languages, with the exact answers and the table, chart or page that holds each one |
| `targets.py` | Finds the page of every answer in each language and saves `targets.json` |
| `evaluate.py` | Splits, embeds and searches each extraction, asks a local LLM, and measures right page, answer in context and right answer |
| `vision.py` | Part 3: searches the `pypdf` text as above, then asks large models through OpenRouter twice, with the 3 chunks as text and with their pages as images |
| `index_pg.py` | Part 4: stores the `pypdf` chunks in Postgres + pgvector and creates a read-only role for the MCP server |
| `mcp_server.py` | Part 4: MCP server (stdio) with two read-only tools, `search_report(query)` and `view_page(page)` |
| `agent.py` | Part 4: the agent loop, with no framework: lists the server's tools, calls models through OpenRouter, at most 6 steps |
| `results/` | The results reported in the articles (`qwen2.5:3b`, the `qwen2.5:7b` rerun on table questions, extraction times, `results_vision.json` for part 3 and `results_agent.json` for part 4) |

## Running part 2

Requires Python 3.11+ and Docker. Everything runs on a CPU; a GPU only makes it faster.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

docker compose up -d               # Ollama on port 11434
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull qwen2.5vl:3b

python download_rpm.py
python extract.py                  # about 15 minutes on a laptop CPU, most of it Docling
python describe.py                 # about 2 hours per language on a CPU; cached, safe to stop and resume
python targets.py
python evaluate.py                 # every extractor, both languages, plus the no-RAG baseline
```

To rerun only the table questions with a larger model, as in the article:

```bash
docker compose exec ollama ollama pull qwen2.5:7b
LLM=qwen2.5:7b TYPES=table EXTRACTORS=pypdf,docling python evaluate.py
```

The embedding model, [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small),
and Docling's layout models are downloaded from Hugging Face on the first run.

`evaluate.py --no-llm` measures only the search (right page and answer in context), in a few minutes.

## Running part 3

Part 3 needs only the `pypdf` extraction, no Docker and no Docling, plus an [OpenRouter](https://openrouter.ai) key.

```bash
python download_rpm.py
python extract.py pypdf            # a few seconds
python targets.py

export OPENROUTER_API_KEY=...      # Windows PowerShell: $env:OPENROUTER_API_KEY = "..."
MODELS=google/gemini-4-argon IDS=t01,c01 python vision.py   # a test run: 2 questions, a few cents
python vision.py                   # all four models, both modes, both languages: about US$ 3.70
```

Set a spending limit on your OpenRouter key before the full run. The key is read from the environment
(or from a dotenv file given in `ENV_FILE`) and is never printed or saved. Replies, tokens and the cost
OpenRouter reports for each call are cached in `out/vision_cache.jsonl`, so reruns are free.
Model names and prices are from November 2026 and change often.

## Running part 4

Needs the `pypdf` extraction from part 3, Docker for Postgres, and an [OpenRouter](https://openrouter.ai) key.

```bash
docker compose up -d db            # Postgres + pgvector on port 5433
python index_pg.py                 # chunks into the database, plus the read-only role rpm_reader

export OPENROUTER_API_KEY=...      # Windows PowerShell: $env:OPENROUTER_API_KEY = "..."
MODELS=google/gemini-4-argon IDS=t01,c01 python agent.py   # a test run: 2 questions, a few cents
python agent.py                    # three models, both languages: about US$ 3
TOOL_DOCS=bare python agent.py     # the same with one-line tool descriptions: about US$ 3.30
```

`agent.py` starts `mcp_server.py` as a subprocess, one per model and language. Every finished
question is cached in `out/agent_cache.jsonl` with its tool calls, tokens and cost, so reruns and
regrading are free. `BUDGET` (US$, default 15) stops the run when the reported cost goes past it.
The server uses MCP SDK 2.x (`MCPServer`); v1 code with `FastMCP` will not run on it.

To use the server from another MCP client, run `DOC_LANG=en python mcp_server.py` over stdio.

## Notes

- Everything after the extractor is fixed: chunks of up to 400 tokens that never cross a page, the same
  embedding model, top 3, and the same LLM at temperature 0. If a result changes, it was the extractor.
  In part 3, the search is fixed too, and only the model and what it sees (text or page) change.
- Grading is strict: "2.8" does not count when the reply says "12.8" or "2.85".
- A reply that refuses ("I could not find…", "the excerpt does not provide…") counts as wrong even if it
  names the right answer, since large models often refuse by repeating the question. See `REFUSAL`
  in `evaluate.py`. This rule and the "958,000" answer for question x04 were added in `rag-pdf-v2`;
  part 2's English totals were regraded from the saved replies (see the correction note in the article).
  In `rag-pdf-v3`, the rule looks only at the first sentence, so a right answer that later says
  "the headers were not in the text" is not taken as a refusal. This changed no result in parts 2 and 3.
- LLM replies and figure descriptions are cached in `out/`, so reruns are free. Delete `out/` to start over.
- Docling's figure captions are often missing, so `describe.py` reads them from the PDF text layer,
  just above each figure.
- Results from a local LLM can change slightly with the hardware and the Ollama version.
- `pymupdf4llm` depends on PyMuPDF, which is licensed under the AGPL (or a commercial licence).
- In parts 3 and 4, the whole page leaves your machine. Fine for a public report; think twice for anything else.
- Results hold for this document and these questions. See the articles for the caveats.

## Data source

Banco Central do Brasil, *Relatório de Política Monetária* / *Monetary Policy Report*, volume 2, nr. 3,
September 2026. Reproduction permitted only if the source is stated as above. The PDFs are downloaded
by `download_rpm.py`, not stored in this repository.
