"""Compare PDF extractors on the same questions, in Portuguese and English.

Usage: python evaluate.py [--no-llm]
       LLM=qwen2.5:7b TYPES=table EXTRACTORS=pypdf,docling python evaluate.py   (a subset, saved apart)
Everything except the extractor stays fixed: the same chunker (blocks of up to 400 tokens,
never crossing a page), the same embedding model, top 3, and the same local LLM.

Three things are measured per question:
  found     a chunk from the page that holds the answer is in the top 3
  in_ctx    the exact answer (the number or the name) is written in those 3 chunks
  correct   the LLM's reply contains the answer
LLM replies are cached in out/llm_cache.jsonl, so reruns are free.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

HERE = Path(__file__).parent
OUT = HERE / "out"
MODEL = "intfloat/multilingual-e5-small"
LLM = os.environ.get("LLM", "qwen2.5:3b")
OLLAMA = "http://localhost:11434/api/chat"
K = 3
MAX_TOKENS = 400  # e5-small reads at most 512 tokens, "passage: " included
EXTRACTORS = os.environ.get("EXTRACTORS", "pypdf,pymupdf4llm,docling,docling-vlm").split(",")
TYPES = os.environ.get("TYPES", "text,table,chart").split(",")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = SentenceTransformer(MODEL)


def n_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split(text: str, seps=("\n\n", "\n", " ")) -> list[str]:
    """The usual recursive splitter: try paragraphs, then lines, then words, and pack the
    pieces greedily up to MAX_TOKENS. It is what most tutorials (and assistants) reach for."""
    if n_tokens(text) <= MAX_TOKENS:
        return [text]
    sep, rest = seps[0], seps[1:]
    pieces = [p for p in text.split(sep) if p.strip()]
    if len(pieces) == 1:
        return split(text, rest) if rest else [tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:MAX_TOKENS])]
    chunks, current = [], ""
    for piece in pieces:
        candidate = current + sep + piece if current else piece
        if n_tokens(candidate) <= MAX_TOKENS:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if n_tokens(piece) <= MAX_TOKENS:
            current = piece
        else:
            chunks += split(piece, rest) if rest else [piece]
            current = ""
    if current:
        chunks.append(current)
    return chunks


def chunk(md: str) -> list[tuple[str, int]]:
    """Split each page on its own and remember the page of every chunk."""
    out = []
    parts = re.split(r"<!-- page (\d+) -->\n?", md)
    for page, text in zip(parts[1::2], parts[2::2]):
        out += [(c.strip(), int(page)) for c in split(text) if c.strip()]
    return out


def contains(text: str, answers: list[str]) -> bool:
    """The answer as a whole token: '2,8' must not match '12,8' or '2,85'."""
    for a in answers:
        if re.search(rf"(?<![\w,.]){re.escape(a)}(?![\d]|[,.]\d)", text, re.I):
            return True
    return False


PROMPT = {
    "pt": "Responda à pergunta usando apenas os trechos do Relatório de Política Monetária abaixo. "
          "Responda em uma frase curta, com o número ou o nome pedido. "
          "Se a resposta não estiver nos trechos, diga que não encontrou.",
    "en": "Answer the question using only the Monetary Policy Report excerpts below. "
          "Answer in one short sentence, with the number or the name asked for. "
          "If the answer is not in the excerpts, say you could not find it.",
}
NO_RAG = {
    "pt": "Responda à pergunta sobre a economia brasileira em uma frase curta, com o número ou o nome pedido.",
    "en": "Answer the question about the Brazilian economy in one short sentence, with the number or the name asked for.",
}

cache_file = OUT / "llm_cache.jsonl"
cache = {}
if cache_file.exists():
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cache[row["key"]] = row["reply"]


def ask(system: str, user: str) -> str:
    key = hashlib.sha256(f"{LLM}\n{system}\n{user}".encode()).hexdigest()
    if key not in cache:
        r = requests.post(OLLAMA, json={
            "model": LLM, "stream": False, "options": {"temperature": 0, "num_ctx": 4096},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, timeout=600)
        r.raise_for_status()
        cache[key] = r.json()["message"]["content"]
        with cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "reply": cache[key]}, ensure_ascii=False) + "\n")
    return cache[key]


def evaluate(chunks, questions, targets, lang, use_llm):
    texts = [t for t, _ in chunks]
    passages = model.encode(["passage: " + t for t in texts], normalize_embeddings=True, batch_size=32)
    queries = model.encode(["query: " + q[lang] for q in questions], normalize_embeddings=True)
    scores = queries @ passages.T
    rows = []
    for qi, q in enumerate(questions):
        top = np.argsort(-scores[qi])[:K]
        context = "\n\n".join(texts[i] for i in top)
        row = {
            "id": q["id"], "type": q["type"],
            "found": any(chunks[i][1] == targets[f"{lang}/{q['id']}"] for i in top),
            "in_ctx": contains(context, q["answers"]),
            "tokens": sum(n_tokens(texts[i]) for i in top),
        }
        if use_llm:
            row["reply"] = ask(PROMPT[lang], f"{context}\n\n---\n\n{q[lang]}")
            row["correct"] = contains(row["reply"], q["answers"])
        rows.append(row)
    return rows


def summary(rows, key):
    out = {}
    for t in ("text", "table", "chart"):
        sel = [r[key] for r in rows if r["type"] == t and key in r]
        out[t] = sum(sel) / len(sel) if sel else None
    return out


if __name__ == "__main__":
    use_llm = "--no-llm" not in sys.argv
    questions = [q for q in json.loads((HERE / "questions.json").read_text(encoding="utf-8")) if q["type"] in TYPES]
    targets = json.loads((HERE / "targets.json").read_text())
    results = {}
    for lang in ("pt", "en"):
        if use_llm and "EXTRACTORS" not in os.environ:
            rows = []
            for q in questions:
                reply = ask(NO_RAG[lang], q[lang])
                rows.append({"id": q["id"], "type": q["type"], "reply": reply, "correct": contains(reply, q["answers"])})
            results[f"{lang}/no-rag"] = {"rows": rows}
        for name in EXTRACTORS:
            md_file = OUT / f"{name}_{lang}.md"
            if not md_file.exists():
                continue
            chunks = chunk(md_file.read_text(encoding="utf-8"))
            rows = evaluate(chunks, questions, targets, lang, use_llm)
            results[f"{lang}/{name}"] = {"chunks": len(chunks), "rows": rows}
            print(f"{lang}/{name}: {len(chunks)} chunks", flush=True)

    header = "| lang | extractor | metric | text | table | chart |"
    print("\n" + header + "\n|---|---|---|---|---|---|")
    for key, res in results.items():
        for metric in ("found", "in_ctx", "correct"):
            s = summary(res["rows"], metric)
            if all(v is None for v in s.values()):
                continue
            cells = " | ".join("-" if v is None else f"{v:.0%}" for v in s.values())
            print(f"| {key.split('/')[0]} | {key.split('/')[1]} | {metric} | {cells} |")
    name = "results.json" if LLM == "qwen2.5:3b" else f"results_{LLM.replace(':', '-')}.json"
    (OUT / name).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
