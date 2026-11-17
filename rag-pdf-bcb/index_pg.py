"""Store the pypdf chunks of the report in PostgreSQL + pgvector, for the MCP server (part 4).

Usage: docker compose up -d db   then   python index_pg.py
Same chunks and embeddings as evaluate.py with pypdf (up to 400 tokens, never crossing a page,
multilingual-e5-small), so the agent's search starts where part 3's fixed search did.

Also creates a read-only role for the MCP server: it can SELECT the chunks and nothing else.
"""
import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from evaluate import OUT, chunk, model

HERE = Path(__file__).parent
DSN = os.environ.get("ADMIN_DSN", "postgresql://rag:rag@localhost:5433/rag")
READER_PASSWORD = os.environ.get("READER_PASSWORD", "reader")

with psycopg.connect(DSN, autocommit=True) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute("DROP TABLE IF EXISTS rpm_chunks")
    conn.execute("""
        CREATE TABLE rpm_chunks (
            id        serial PRIMARY KEY,
            lang      text NOT NULL,
            page      int  NOT NULL,
            content   text NOT NULL,
            embedding vector(384) NOT NULL
        )
    """)
    for lang in ("pt", "en"):
        chunks = chunk((OUT / f"pypdf_{lang}.md").read_text(encoding="utf-8"))
        vectors = model.encode(["passage: " + t for t, _ in chunks], normalize_embeddings=True, batch_size=32)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO rpm_chunks (lang, page, content, embedding) VALUES (%s, %s, %s, %s)",
                [(lang, page, text, vec) for (text, page), vec in zip(chunks, vectors)],
            )
        print(f"{lang}: {len(chunks)} chunks")

    # The MCP server connects as this role. Text from the document reaches the model, and the model
    # chooses the tool arguments, so the server must not be able to change anything.
    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = 'rpm_reader'").fetchone()
    if not exists:
        conn.execute(f"CREATE ROLE rpm_reader LOGIN PASSWORD '{READER_PASSWORD}'")
    conn.execute("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM rpm_reader")
    conn.execute("GRANT SELECT ON rpm_chunks TO rpm_reader")
    conn.execute("ALTER ROLE rpm_reader SET default_transaction_read_only = on")
    print("read-only role rpm_reader: SELECT on rpm_chunks only")
