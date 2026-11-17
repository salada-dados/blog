# Salada de Dados: code from the blog

Code and data for the experiments and tutorials published on [saladadedados.com](https://saladadedados.com).
Each folder is a self-contained project, with its own README and dependencies.

## Experiments and tutorials

| Folder | Article | Topic |
|---|---|---|
| [`rag-chunking-gdpr`](rag-chunking-gdpr) | RAG in Practice: Does Chunk Size Change the Answer? ([en](https://saladadedados.com/en/blog/rag-chunking-experiment) · [pt](https://saladadedados.com/pt/blog/rag-chunking-experiment)) | RAG, pgvector, embeddings |
| [`rag-pdf-bcb`](rag-pdf-bcb) | RAG over PDFs: What Gets Lost Between the Table and the Text ([en](https://saladadedados.com/en/blog/rag-pdf-tables-images) · [pt](https://saladadedados.com/pt/blog/rag-pdf-tables-images)) | RAG, PDF extraction, Docling, vision models |
| [`rag-pdf-bcb`](rag-pdf-bcb) (`vision.py`) | Why the Chat Reads Your Screenshot and Your RAG Does Not ([en](https://saladadedados.com/en/blog/rag-pdf-page-as-image) · [pt](https://saladadedados.com/pt/blog/rag-pdf-page-as-image)) | RAG, page images, large vision models, cost |
| [`rag-pdf-bcb`](rag-pdf-bcb) (`mcp_server.py`, `agent.py`) | RAG as a Tool: An Agent Found What the Fixed Search Missed ([en](https://saladadedados.com/en/blog/rag-mcp-agent) · [pt](https://saladadedados.com/pt/blog/rag-mcp-agent)) | RAG, MCP, agents, pgvector |

Older series with their own repository:

- [`langgraph-fastapi-example`](https://github.com/salada-dados/langgraph-fastapi-example): LangGraph agent with FastAPI and SSE streaming.

## Versions

Each article links to a tag (for example `rag-chunking-v1`), so the code you find always matches
what the article describes, even if the folder changes later.

## Getting a single folder

```bash
git clone --filter=blob:none --sparse https://github.com/salada-dados/blog.git
cd blog
git sparse-checkout set rag-chunking-gdpr
```
