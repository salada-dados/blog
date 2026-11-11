# RAG chunking experiment on the GDPR

Code for the Salada de Dados article **"RAG in Practice: Does Chunk Size Change the Answer?"**
([English](https://saladadedados.com/en/blog/rag-chunking-experiment) ·
[Português](https://saladadedados.com/pt/blog/rag-chunking-experiment)).

It compares seven ways to split a document for RAG, on the GDPR in English and Portuguese,
using 32 questions with known answers. Everything runs locally and for free: no API key.

## What's inside

| File | What it does |
|---|---|
| `download_gdpr.py` | Downloads the GDPR from EUR-Lex (en and pt) and saves the 99 articles to `data/` |
| `chunking.py` | The chunking strategies: fixed size, fixed size with overlap, by article, by paragraph, by point |
| `questions.json` | The 32 questions, in both languages, with the article that answers each one |
| `evaluate.py` | Measures every strategy (top-3 hit rate, MRR, tokens sent to the LLM) and prints a table |
| `index.py` | Stores the by-point chunks in PostgreSQL + pgvector |
| `ask.py` | Searches pgvector and asks a local LLM (Ollama) to answer, citing the article |

## Running it

Requires Python 3.11+ and Docker.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python download_gdpr.py
python evaluate.py                 # the experiment; takes a few minutes on a CPU

docker compose up -d               # PostgreSQL + pgvector on port 5433, Ollama on 11434
docker compose exec ollama ollama pull qwen2.5:3b
python index.py
python ask.py en "A company leaked my data. How long does it have to notify the authority?"
```

The embedding model, [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small),
is downloaded from Hugging Face on the first run (about 470 MB).

## Notes

- The model reads at most 512 tokens per chunk and expects the `query: ` and `passage: ` prefixes.
- Search in `evaluate.py` is exact cosine similarity, the same thing pgvector does without an index.
- Results hold for this document and these questions. See the article for the caveats.

## Data source

GDPR: Regulation (EU) 2016/679 of the European Parliament and of the Council,
© European Union, [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Only the version published
in the Official Journal of the European Union is deemed authentic. The text is downloaded by
`download_gdpr.py`, not stored in this repository.
