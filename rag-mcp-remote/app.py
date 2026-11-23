"""The Monetary Policy Report MCP server, over HTTP, for Claude on the web (remote connector).

Same two tools as rag-pdf-bcb/mcp_server.py, with the changes a serverless host requires:
  - Streamable HTTP instead of stdio, stateless: each request may land on a different instance.
  - ONNX Runtime instead of PyTorch (see embedding.py).
  - The report's PDFs are downloaded from the Banco Central do Brasil on first use, not bundled.
  - Postgres is in the cloud (copy_to_cloud.py), reached through a read-only role.
  - Both languages in one server: the tools take a lang argument.

Local:  uvicorn app:app --port 8000      then point an MCP client at http://localhost:8000/mcp
Vercel: app.py is the entrypoint; see README.md.
Env:    READER_DSN, MODEL_URL + MODEL_SHA256 (or MODEL_PATH),
        AUTH_SECRET, ACCESS_PASSWORD, PUBLIC_URL (see auth.py)

Authentication: OAuth with a password page (auth.py). Without it, anyone who finds the address
uses the server, and its cost and limits, as much as they want.
"""
import functools
import os
import sys
import threading
import time
import traceback
from typing import Literal

import psycopg
import pymupdf
from mcp.server import MCPServer
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from pgvector.psycopg import register_vector
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from auth import SETTINGS, StatelessProvider, login
from embedding import download, embedder

DSN = os.environ.get("READER_DSN", "postgresql://rpm_reader:reader@localhost:5433/rag")
K = 3
DPI = 150
MAX_QUERY = 500  # characters; a search query, not a document

# Monetary Policy Report, volume 2, nr. 3, Banco Central do Brasil. Downloaded, not redistributed.
PDFS = {
    "pt": ("https://www.bcb.gov.br/content/ri/relatorioinflacao/202609/rpm202609p.pdf",
           "274c8ac6e603703dcb3d73dd758302cd10b8056580e4c44966773fab05db066b", 90),
    "en": ("https://www.bcb.gov.br/content/ri/inflationreport/202609/rpm202609i.pdf",
           "f9ed8609a7c5fda63ff22bdbbc9cddc37f22c9bb621f6b172a2b9af01316081a", 88),
}
Lang = Literal["pt", "en"]

_docs, _docs_lock = {}, threading.Lock()


def document(lang):
    with _docs_lock:
        if lang not in _docs:
            url, sha256, _ = PDFS[lang]
            local = os.path.join(os.path.dirname(__file__), "..", "rag-pdf-bcb", "data", url.rsplit("/", 1)[-1])
            _docs[lang] = pymupdf.open(stream=download(url, sha256, local), filetype="pdf")
        return _docs[lang]


INSTRUCTIONS = (
    "Read-only access to the Banco Central do Brasil's Monetary Policy Report (Relatório de Política "
    "Monetária, September 2026), in Portuguese (lang='pt') and English (lang='en'). Two tools work "
    "together: search_report finds the pages, and view_page shows a page as an image. Chart values "
    "exist only in the image, so after a search that points to a chart or a table, call view_page on "
    "that page, with the same lang. Page numbers differ between the two languages."
)
# Each request may hit a fresh instance, so nothing is kept between calls (stateless_http) and
# answers come back as plain JSON instead of an event stream (json_response).
mcp = MCPServer("monetary-policy-report", instructions=INSTRUCTIONS,
                auth=SETTINGS, auth_server_provider=StatelessProvider())
# The SDK adds /authorize, /token, /register and the .well-known metadata; /mcp then needs a token.
# The login page is ours: custom routes are public, as a login page must be.
mcp.custom_route("/login", methods=["GET", "POST"])(login)
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


def reported(tool):
    """For any unexpected error, write the traceback to the logs and tell the model what failed.
    Without this, the SDK answers a bare "Error executing tool" and logs nothing on Vercel: the
    first failure we hit there showed up in Claude only as "3 failed"."""
    @functools.wraps(tool)
    def run(*args, **kwargs):
        try:
            return tool(*args, **kwargs)
        except ToolError:
            raise
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            raise ToolError(f"server error, not your input: {type(error).__name__}: {error}"[:300]) from error
    return run


@mcp.tool(annotations=READ_ONLY, description=(
    f"Search the Banco Central do Brasil's Monetary Policy Report (Relatório de Política Monetária, "
    f"September 2026) and return the {K} most similar text chunks, each with its page number. "
    "lang='pt' searches the Portuguese edition, lang='en' the English one; use the language of the "
    "user's question. The text comes from the PDF's text layer: tables lose their headers, and charts "
    "keep only their title, legend and axis labels, never their values. Write the query in the "
    "report's language, with the words the report would use. If the chunks do not answer the "
    "question, search again with other words."
))
@reported
def search_report(query: str, lang: Lang = "pt") -> str:
    if len(query) > MAX_QUERY:
        raise ToolError(f"query is limited to {MAX_QUERY} characters, got {len(query)}")
    vector = embedder().encode("query: " + query)
    # One connection per call: a serverless instance may sleep for minutes between requests,
    # and the database (Neon) suspends after 5 idle minutes, so a kept connection would go stale.
    with psycopg.connect(DSN, autocommit=True) as conn:
        register_vector(conn)
        rows = conn.execute(
            "SELECT page, content FROM rpm_chunks WHERE lang = %s ORDER BY embedding <=> %s LIMIT %s",
            (lang, vector, K),
        ).fetchall()
    return "\n\n".join(f"[page {page}]\n{content}" for page, content in rows)


@mcp.tool(annotations=READ_ONLY, description=(
    "Return one page of the Banco Central do Brasil's Monetary Policy Report (Relatório de Política "
    "Monetária, September 2026) as an image: pages 1 to 90 for lang='pt', 1 to 88 for lang='en'. "
    "Use it to read a chart (gráfico) or a table (tabela) on a page that search_report found, with "
    "the same lang: the values of charts are only in the image. Load it together with search_report."
))
@reported
def view_page(page: int, lang: Lang = "pt") -> Image:
    pages = PDFS[lang][2]
    if not 1 <= page <= pages:
        raise ToolError(f"page must be between 1 and {pages} for lang='{lang}', got {page}")
    return Image(data=document(lang)[page - 1].get_pixmap(dpi=DPI).tobytes("png"), format="png")


@mcp.custom_route("/", methods=["GET"])
async def home(_: Request) -> PlainTextResponse:
    return PlainTextResponse(
        "MCP server for the Banco Central do Brasil's Monetary Policy Report (September 2026).\n"
        "Add <this address>/mcp as a custom connector in Claude (it asks for a password). Code: "
        "https://github.com/salada-dados/blog/tree/main/rag-mcp-remote\n"
    )


app = mcp.streamable_http_app(
    stateless_http=True,
    json_response=True,
    # The SDK turns on DNS rebinding protection when host is localhost (its default), and then
    # accepts only Host: localhost. On Vercel every request would get "421 Invalid Host header".
    # That protection is for servers on your own machine; this one is public by design.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
print(f"rpm MCP server ready, {time.process_time():.1f} s CPU at import", file=sys.stderr)
