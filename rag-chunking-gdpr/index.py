"""Store the GDPR in PostgreSQL + pgvector, cut by the winning strategy (by_point).

Usage: docker compose up -d   then   python index.py
"""
import json
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

import chunking as c

DSN = "postgresql://rag:rag@localhost:5433/rag"
HERE = Path(__file__).parent
model = SentenceTransformer(c.MODEL)

with psycopg.connect(DSN, autocommit=True) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("""
        CREATE TABLE chunks (
            id        serial PRIMARY KEY,
            lang      text NOT NULL,
            article   int  NOT NULL,
            content   text NOT NULL,
            embedding vector(384) NOT NULL
        )
    """)
    for lang in ("pt", "en"):
        articles = json.loads((HERE / f"data/articles_{lang}.json").read_text(encoding="utf-8"))
        chunks = c.by_point(articles, lang)
        vectors = model.encode(["passage: " + t for t, _ in chunks], normalize_embeddings=True, batch_size=32)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (lang, article, content, embedding) VALUES (%s, %s, %s, %s)",
                [(lang, min(arts), text, vec) for (text, arts), vec in zip(chunks, vectors)],
            )
        print(f"{lang}: {len(chunks)} chunks")
