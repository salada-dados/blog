"""Does the ONNX model find the same chunks as the index? Run it before trusting a smaller model.

Usage: python check_model.py [model.onnx ...]   (default: the file build_model.py writes)
Needs: the part 4 database (rag-pdf-bcb: docker compose up -d db; python index_pg.py)
       and sentence-transformers, the model the index was built with.

For each of the 33 questions, in pt and en, it compares the query vector with the original
model's and the 3 chunks each one retrieves. The cosine alone is misleading: Hugging Face's
fully quantized model scores 0.996 on average and still retrieves other chunks in 21 of 66.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from embedding import TOKENIZER, Embedder, download

HERE = Path(__file__).parent
QUESTIONS = json.loads((HERE.parent / "rag-pdf-bcb" / "questions.json").read_text(encoding="utf-8"))
DSN = os.environ.get("READER_DSN", "postgresql://rpm_reader:reader@localhost:5433/rag")
MODELS = sys.argv[1:] or [str(HERE / "model" / "e5-small-int8emb.onnx")]

conn = psycopg.connect(DSN, autocommit=True)
register_vector(conn)
reference = SentenceTransformer("intfloat/multilingual-e5-small")


def top3(lang, vector):
    return [r[0] for r in conn.execute(
        "SELECT id FROM rpm_chunks WHERE lang = %s ORDER BY embedding <=> %s LIMIT 3", (lang, vector),
    ).fetchall()]


for path in MODELS:
    onnx_model = Embedder(Path(path).read_bytes(), download(*TOKENIZER))
    cosines, same_chunks, same_order = [], 0, 0
    for q in QUESTIONS:
        for lang in ("pt", "en"):
            text = "query: " + q[lang]
            a = reference.encode(text, normalize_embeddings=True)
            b = onnx_model.encode(text)
            cosines.append(float(a @ b))
            ta, tb = top3(lang, a), top3(lang, b)
            same_chunks += set(ta) == set(tb)
            same_order += ta == tb
    n = len(cosines)
    print(f"{Path(path).name}: cosine min {min(cosines):.6f}, mean {np.mean(cosines):.6f}; "
          f"same 3 chunks {same_chunks}/{n}, same order {same_order}/{n}")
