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
from functools import cache
from pathlib import Path

import pymupdf
import psycopg
from mcp.server import MCPServer
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pgvector.psycopg import register_vector

HERE = Path(__file__).parent
LANG = os.environ.get("DOC_LANG", "en")
DOCS = os.environ.get("TOOL_DOCS", "detailed")
DSN = os.environ.get("READER_DSN", "postgresql://rpm_reader:reader@localhost:5433/rag")
PDF = {"pt": "data/rpm202609p.pdf", "en": "data/rpm202609i.pdf"}[LANG]
K = 3
DPI = 150



@cache
def embedder():
    """Loaded on the first search, not at startup: importing PyTorch and loading the model takes
    30 s or more, and MCP clients give a server about that long to start (30 s in Claude Code)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("intfloat/multilingual-e5-small")

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
        # Clients such as Claude Desktop load MCP tools on demand, by searching their descriptions.
        # Before rag-pdf-v5 this one did not name the report, so a search for "Banco Central relatório
        # política monetária" found search_report and left view_page out for nine searches.
        "view_page": (
            f"Return one page (1 to {len(doc)}) of the Banco Central do Brasil's Monetary Policy Report "
            "(Relatório de Política Monetária, September 2026) as an image. Use it to read a chart "
            "(gráfico) or a table (tabela) on a page that search_report found: the values of charts "
            "are only in the image. Load it together with search_report."
        ),
    },
}[DOCS]

INSTRUCTIONS = (
    "Read-only access to the Banco Central do Brasil's Monetary Policy Report (Relatório de Política "
    "Monetária, September 2026). Two tools work together: search_report finds the pages, and view_page "
    "shows a page as an image. Chart values exist only in the image, so after a search that points to "
    "a chart or a table, call view_page on that page."
)
mcp = MCPServer("monetary-policy-report", instructions=INSTRUCTIONS)
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


@mcp.tool(description=DESCRIPTIONS["search_report"], annotations=READ_ONLY)
def search_report(query: str) -> str:
    vector = embedder().encode("query: " + query, normalize_embeddings=True)
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
