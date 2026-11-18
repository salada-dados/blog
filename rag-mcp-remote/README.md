# The report's MCP server on the internet, for Claude on the web

The MCP server from [`rag-pdf-bcb`](../rag-pdf-bcb) (`mcp_server.py`), changed to run as a remote
connector: Streamable HTTP, on a Vercel function, with the database on Neon. Claude on the web,
on mobile and in Desktop can then use it with no local setup.

What changed from the local server, and why:

| Local (`mcp_server.py`) | Remote (`app.py`) | Why |
|---|---|---|
| stdio | Streamable HTTP, stateless, JSON responses | Claude's servers call it over HTTPS; any request may hit a new instance |
| PyTorch + sentence-transformers | ONNX Runtime + tokenizers (`embedding.py`) | PyTorch alone exceeds Vercel's 500 MB limit for a Python function |
| model from Hugging Face's cache | `e5-small-int8emb.onnx`, 183 MB, downloaded on the first search | only the vocabulary table is int8; same 3 chunks as the index in 66 of 66 questions |
| PDFs in `data/` | downloaded from the Banco Central do Brasil on first use, checksum verified | the report is not redistributed here |
| Postgres in Docker | Postgres + pgvector on Neon, read-only role | the function needs a database on the internet |
| `DOC_LANG` picks one report | `lang` argument on both tools | one connector for both editions |

Measured on the author's machine, from start to the first answer (one search, one page):
PyTorch server 1,285 MB and 19.5 s; this one 571 MB and 3.0 s (model and PDFs already on disk).

## Files

- `app.py`: the server (Vercel entrypoint: `app`).
- `embedding.py`: e5-small with ONNX Runtime, and the download-once cache.
- `build_model.py`: writes `model/e5-small-int8emb.onnx` from Hugging Face's fp32 ONNX file.
- `check_model.py`: compares a model's retrieved chunks with the index's, for the 33 questions.
- `copy_to_cloud.py`: copies the chunks, vectors included, to a cloud Postgres and creates the read-only role.

## Running it

On your machine, with the [`rag-pdf-bcb`](../rag-pdf-bcb) database running (part 4):

```bash
pip install -r requirements-dev.txt
python build_model.py     # prints the file's size and sha256
python check_model.py     # same 3 chunks in 66/66?
uvicorn app:app --port 8000
```

The server is at `http://localhost:8000/mcp`. Locally it reads the PDFs from `../rag-pdf-bcb/data`
and the model from `model/`.

On Vercel:

1. Create a Postgres database with pgvector (Neon's free plan is enough) and copy the chunks:
   `CLOUD_ADMIN_DSN=... READER_PASSWORD=... python copy_to_cloud.py`.
2. Publish `model/e5-small-int8emb.onnx` where the function can download it (we used a GitHub
   release asset: the file is too large for the repository and for a Vercel CLI upload).
3. Import this repository in Vercel with **Root Directory** `rag-mcp-remote`, and set
   `READER_DSN` (the `rpm_reader` address), `MODEL_URL` and `MODEL_SHA256`.
4. In Claude: **Customize → Connectors → Add custom connector**, with `https://<project>.vercel.app/mcp`.

## No authentication, on purpose

The report is public and both tools only read, so the server has no login: anyone with the
address can use it. It ran on Vercel's Hobby plan, which stops instead of billing past its
limits, and came down after the article. A server over private documents, or with any tool that
writes, needs OAuth.
