"""Extrai o texto do PDF (linhas reconstruídas por coordenadas, para tabelas ficarem
em formato de linha), divide em trechos e gera embeddings. Saída: index/ (chunks.json, emb.npy)."""
import json, os, re
import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "rpm202609p.pdf")
OUT = os.path.join(HERE, "index")
EMB_MODEL = "intfloat/multilingual-e5-base"
HEADER = re.compile(r"^Setembro 2026\s+\\\s+Banco Central do Brasil.*$")


def page_lines(page):
    ws = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines, cur, cy = [], [], None
    for w in ws:
        y = (w[1] + w[3]) / 2
        if cy is None or abs(y - cy) > 3:
            if cur:
                lines.append(cur)
            cur, cy = [w], y
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    out = []
    for l in lines:
        t = " ".join(x[4] for x in sorted(l, key=lambda w: w[0]))
        if not HEADER.match(t):
            out.append(t)
    return out


def chunk(lines, size=1100, overlap=3):
    chunks, cur, n = [], [], 0
    i = 0
    while i < len(lines):
        cur.append(lines[i]); n += len(lines[i]) + 1; i += 1
        if n >= size:
            chunks.append("\n".join(cur))
            cur = cur[-overlap:]; n = sum(len(x) + 1 for x in cur)
    if cur and (not chunks or len("\n".join(cur)) > 200 or len(chunks) == 0):
        chunks.append("\n".join(cur))
    return chunks


def main():
    doc = pymupdf.open(PDF)
    items = []
    for pno, page in enumerate(doc):
        lines = page_lines(page)
        if not lines:
            continue
        # página inteira também é indexada (bom para tabelas/gráficos)
        for c in chunk(lines):
            items.append({"page": pno + 1, "text": c})
    os.makedirs(OUT, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL)
    emb = m.encode(["passage: " + it["text"] for it in items], batch_size=16,
                   normalize_embeddings=True, show_progress_bar=True)
    np.save(os.path.join(OUT, "emb.npy"), emb.astype(np.float32))
    with open(os.path.join(OUT, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(len(items), "trechos")


if __name__ == "__main__":
    main()
