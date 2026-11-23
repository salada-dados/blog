# The report's MCP server on the internet, for Claude on the web

The MCP server from [`rag-pdf-bcb`](../rag-pdf-bcb) (`mcp_server.py`), changed to run as a remote
connector: Streamable HTTP, on a Vercel function, with the database on Neon. Claude on the web,
on mobile and in Desktop can then use it with no local setup.

What changed from the local server, and why:

| Local (`mcp_server.py`) | Remote (`app.py`) | Why |
|---|---|---|
| stdio | Streamable HTTP, stateless, JSON responses | Claude's servers call it over HTTPS; any request may hit a new instance |
| PyTorch + sentence-transformers | ONNX Runtime + tokenizers (`embedding.py`) | PyTorch alone exceeds Vercel's 500 MB limit for a Python function |
| model from Hugging Face's cache | `e5-small-int8emb.onnx`, 183 MB, downloaded on the first search and kept in memory | only the vocabulary table is int8; same 3 chunks as the index in 66 of 66 questions |
| PDFs in `data/` | downloaded from the Banco Central do Brasil on first use, checksum verified, kept in memory | the report is not redistributed here |
| Postgres in Docker | Postgres + pgvector on Neon, read-only role | the function needs a database on the internet |
| `DOC_LANG` picks one report | `lang` argument on both tools | one connector for both editions |
| no login (it runs on your machine) | OAuth with a password page, no database (`auth.py`) | on the internet, anyone with the address could use it |

Measured on the author's machine, from start to the first answer (one search, one page):
PyTorch server 1,285 MB and 19.5 s; this one 571 MB and 3.0 s (model and PDFs already on disk).
With both PDFs open and everything in memory, as on Vercel, it peaks at 926 MB.

Why memory and not `/tmp`: on Vercel, `/tmp` has limited space and also holds the Python dependencies,
installed there when the function starts. With the model on disk, 80 MB were left; with
pymupdf 2.x, which brings pymupdf4llm and its dependencies, the tokenizer no longer fit.

## Files

- `app.py`: the server (Vercel entrypoint: `app`).
- `auth.py`: OAuth with one password; clients, codes and tokens are signed messages, not stored.
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

Locally, set `AUTH_SECRET`, `ACCESS_PASSWORD` and, to test with a client other than Claude,
`ALLOWED_REDIRECTS` with its callback address. The server is at `http://localhost:8000/mcp`. Locally it reads the PDFs from `../rag-pdf-bcb/data`
and the model from `model/`.

On Vercel:

1. Create a Postgres database with pgvector (Neon's free plan is enough) and copy the chunks:
   `CLOUD_ADMIN_DSN=... READER_PASSWORD=... python copy_to_cloud.py`.
2. Publish `model/e5-small-int8emb.onnx` where the function can download it (we used a GitHub
   release asset: the file is too large for the repository and for a Vercel CLI upload).
3. Import this repository in Vercel with **Root Directory** `rag-mcp-remote`, and set
   `READER_DSN` (the `rpm_reader` address), `MODEL_URL`, `MODEL_SHA256`, `AUTH_SECRET`
   (`python -c "import secrets; print(secrets.token_urlsafe(32))"`) and `ACCESS_PASSWORD`.
4. In Claude: **Customize → Connectors → Add custom connector**, with `https://<project>.vercel.app/mcp`.
   Claude opens the password page once; after that it renews its token on its own.

## Authentication

Claude on the web accepts a custom connector with no authentication or with OAuth (a fixed token
in a header is a beta not every account has). So the server is its own OAuth authorization server,
and logging in means typing one password. The SDK provides the `/authorize`, `/token` and
`/register` endpoints; `auth.py` provides what they read and write.

Nothing is stored, because on Vercel each request may run on a different instance. The client
registration, the authorization code and the tokens are messages signed with `AUTH_SECRET`, which
any instance can check. The trade-offs:

- Tokens cannot be revoked one by one; changing `AUTH_SECRET` revokes them all.
- An authorization code could be exchanged twice within its 5 minutes; PKCE means only the client
  that started the login can exchange it.
- Only Claude's callback addresses may register, so codes go nowhere else.
- One password for everyone: fine for a demo or a small team, not for telling users apart.
  For that, put a real identity provider (Google, GitHub, your company's) behind `/authorize`.

The server ran on Vercel's Hobby plan, which stops instead of billing past its limits, and came
down after the article.
