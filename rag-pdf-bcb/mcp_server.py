"""MCP server for the Monetary Policy Report: search the text, look at a page (part 4).

Usage: run by agent.py (or any MCP client) as a stdio subprocess:
       DOC_LANG=en python mcp_server.py
Env:   DOC_LANG     pt or en (which version of the report)
       TOOL_DOCS    detailed (default) or bare: the tool descriptions, compared in the article
       READER_DSN   read-only connection (default: the rpm_reader role created by index_pg.py)

Two tools, both read-only:
  search_report(query)  the 3 most similar chunks, each with its page number
  view_page(page)       the page rendered as a PNG at 150 dpi, as in part 3

Over stdio, stdout IS the protocol: never print() here. Logs go to stderr.
"""
import os
import sys
from pathlib import Path

import pymupdf
import psycopg
from mcp.server import MCPServer
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
LANG = os.environ.get("DOC_LANG", "en")
DOCS = os.environ.get("TOOL_DOCS", "detailed")
DSN = os.environ.get("READER_DSN", "postgresql://rpm_reader:reader@localhost:5433/rag")
PDF = {"pt": "data/rpm202609p.pdf", "en": "data/rpm202609i.pdf"}[LANG]
K = 3
DPI = 150

model = SentenceTransformer("intfloat/multilingual-e5-small")
doc = pymupdf.open(HERE / PDF)
conn = psycopg.connect(DSN, autocommit=True)
register_vector(conn)

DESCRIPTIONS = {
    "bare": {
        "search_report": "Search the report.",
        "view_page": "Get a page of the report.",
    },
    "detailed": {
        "search_report": (
            f"Search the Banco Central do Brasil's Monetary Policy Report (September 2026, in "
            f"{'Portuguese' if LANG == 'pt' else 'English'}) and return the {K} most similar text chunks, "
            "each with its page number. The text comes from the PDF's text layer: tables lose their "
            "headers, and charts keep only their title, legend and axis labels, never their values. "
            "Write the query in the report's language, with the words the report would use. "
            "If the chunks do not answer the question, search again with other words."
        ),
        "view_page": (
            f"Return one page of the report (1 to {len(doc)}) as an image. Use it to read a chart or "
            "a table on a page that search_report found: the values of charts are only in the image."
        ),
    },
}[DOCS]

mcp = MCPServer("monetary-policy-report", instructions="Read-only access to one PDF report.")
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


@mcp.tool(description=DESCRIPTIONS["search_report"], annotations=READ_ONLY)
def search_report(query: str) -> str:
    vector = model.encode("query: " + query, normalize_embeddings=True)
    rows = conn.execute(
        "SELECT page, content FROM rpm_chunks WHERE lang = %s ORDER BY embedding <=> %s LIMIT %s",
        (LANG, vector, K),
    ).fetchall()
    return "\n\n".join(f"[page {page}]\n{content}" for page, content in rows)


@mcp.tool(description=DESCRIPTIONS["view_page"], annotations=READ_ONLY)
def view_page(page: int) -> Image:
    if not 1 <= page <= len(doc):
        # ToolError reaches the model with its message; any other exception arrives as a bare
        # "Error executing tool", and the model cannot tell what to fix.
        raise ToolError(f"page must be between 1 and {len(doc)}, got {page}")
    return Image(data=doc[page - 1].get_pixmap(dpi=DPI).tobytes("png"), format="png")


if __name__ == "__main__":
    print(f"rpm MCP server: {LANG}, {DOCS} tool docs", file=sys.stderr)
    mcp.run("stdio")
