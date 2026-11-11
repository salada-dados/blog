"""Ask the GDPR a question: search pgvector, then let a local LLM answer citing the article.

Usage: python ask.py pt "Uma empresa vazou meus dados. Em quanto tempo ela precisa avisar a autoridade?"
"""
import sys

import psycopg
import requests
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from chunking import MODEL

DSN = "postgresql://rag:rag@localhost:5433/rag"
OLLAMA = "http://localhost:11434/api/chat"
LLM = "qwen2.5:3b"
K = 3

PROMPT = {
    "pt": "Responda à pergunta usando apenas os trechos do GDPR abaixo. Cite o número do artigo. "
          "Se a resposta não estiver nos trechos, diga que não encontrou.",
    "en": "Answer the question using only the GDPR excerpts below. Cite the article number. "
          "If the answer is not in the excerpts, say you could not find it.",
}

model = SentenceTransformer(MODEL)


def search(lang: str, question: str, k: int = K) -> list[tuple[int, str, float]]:
    vector = model.encode("query: " + question, normalize_embeddings=True)
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        return conn.execute(
            """SELECT article, content, 1 - (embedding <=> %s) AS similarity
               FROM chunks WHERE lang = %s
               ORDER BY embedding <=> %s LIMIT %s""",
            (vector, lang, vector, k),
        ).fetchall()


def answer(lang: str, question: str) -> str:
    excerpts = "\n\n".join(content for _, content, _ in search(lang, question))
    reply = requests.post(OLLAMA, json={
        "model": LLM,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": PROMPT[lang]},
            {"role": "user", "content": f"{excerpts}\n\n---\n\n{question}"},
        ],
    }, timeout=300)
    reply.raise_for_status()
    return reply.json()["message"]["content"]


if __name__ == "__main__":
    lang, question = sys.argv[1], sys.argv[2]
    for article, content, sim in search(lang, question):
        print(f"[{sim:.3f}] art. {article}: {content.splitlines()[1][:90]}...")
    print()
    print(answer(lang, question))
