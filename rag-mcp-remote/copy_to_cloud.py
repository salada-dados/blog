"""Copy the report's chunks from the local database (part 4) to a cloud Postgres, such as Neon.

Usage: CLOUD_ADMIN_DSN="postgresql://owner:...@...neon.tech/neondb?sslmode=require" \
       READER_PASSWORD="<long random password>" python copy_to_cloud.py
Then:  READER_DSN for the server = the same address, with rpm_reader and READER_PASSWORD.

Rows are copied as they are, vectors included: nothing is embedded again, so the cloud index is
the one the series measured. The server's role can only SELECT this table. Unlike the local
Docker database, this one is on the internet: the password must be long and random.
"""
import os

import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector

LOCAL = os.environ.get("ADMIN_DSN", "postgresql://rag:rag@localhost:5433/rag")
CLOUD = os.environ["CLOUD_ADMIN_DSN"]
PASSWORD = os.environ["READER_PASSWORD"]
if len(PASSWORD) < 24:
    raise SystemExit("READER_PASSWORD: use at least 24 random characters")

with psycopg.connect(LOCAL) as local:
    register_vector(local)
    rows = local.execute("SELECT id, lang, page, content, embedding FROM rpm_chunks ORDER BY id").fetchall()

with psycopg.connect(CLOUD, autocommit=True) as cloud:
    cloud.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(cloud)
    cloud.execute("DROP TABLE IF EXISTS rpm_chunks")
    cloud.execute("""
        CREATE TABLE rpm_chunks (
            id        int  PRIMARY KEY,
            lang      text NOT NULL,
            page      int  NOT NULL,
            content   text NOT NULL,
            embedding vector(384) NOT NULL
        )
    """)
    with cloud.cursor() as cur:
        cur.executemany("INSERT INTO rpm_chunks VALUES (%s, %s, %s, %s, %s)", rows)
    for lang, n in cloud.execute("SELECT lang, count(*) FROM rpm_chunks GROUP BY lang ORDER BY lang"):
        print(f"{lang}: {n} chunks")

    if not cloud.execute("SELECT 1 FROM pg_roles WHERE rolname = 'rpm_reader'").fetchone():
        cloud.execute(sql.SQL("CREATE ROLE rpm_reader LOGIN PASSWORD {}").format(PASSWORD))
    else:
        cloud.execute(sql.SQL("ALTER ROLE rpm_reader PASSWORD {}").format(PASSWORD))
    cloud.execute("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM rpm_reader")
    cloud.execute("GRANT USAGE ON SCHEMA public TO rpm_reader")
    cloud.execute("GRANT SELECT ON rpm_chunks TO rpm_reader")
    cloud.execute("ALTER ROLE rpm_reader SET default_transaction_read_only = on")
    print("read-only role rpm_reader: SELECT on rpm_chunks only")
