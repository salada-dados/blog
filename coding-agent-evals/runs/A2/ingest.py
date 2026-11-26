"""Extrai o texto do RPM, divide em trechos, gera embeddings (bge-m3 via Ollama) e salva o índice."""
import json, re, sys
import numpy as np
import pymupdf
import requests

PDF = "rpm202609p.pdf"
OLLAMA = "http://localhost:11434"
EMB_MODEL = "bge-m3"
CHUNK = 1400
OVERLAP = 300


def clean(t):
    t = t.replace("ᐧ", "/").replace("ᐊ", "/")
    t = re.sub(r"Setembro 2026\s+\\\s+Banco Central do Brasil\s+\\\s+Relatório de Política Monetária\s+\\\s+\d+\s*", "", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


NUM = re.compile(r"^\[?-?\d[\d\.,]*\]?%?$")


def table_md(tab):
    """Tabela em duas formas: linhas cruas e linearizada (rótulo | cabeçalho: valor)."""
    raw = []
    for line in tab.to_markdown().splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"^\|[-|\s]+\|$", line):
            continue
        cells = [c.replace("<br>", " ").strip() for c in line[1:-1].split("|")]
        if any(cells):
            raw.append(cells)
    # 1a linha do markdown é cabeçalho sintético (Col1...) ou título; mantém se tiver conteúdo útil
    raw = [[("" if re.match(r"^Col\d+$", c) else c) for c in r] for r in raw]
    rows = [" | ".join(r) for r in raw]
    # cabeçalho = linhas antes da primeira linha com rótulo + valores numéricos
    first_data = next((i for i, r in enumerate(raw)
                       if r and r[0] and sum(bool(NUM.match(c)) for c in r[1:]) >= 2), None)
    lin = []
    if first_data:
        ncol = max(len(r) for r in raw)
        heads = [""] * ncol
        for h in raw[:first_data]:
            if h[0].startswith("Tabela") or sum(bool(c) for c in h) <= 1:
                continue
            last = ""
            for j in range(1, ncol):
                c = h[j] if j < len(h) else ""
                if c and not c.startswith("Col") and c != "%":
                    last = c
                    heads[j] = (heads[j] + " " + c).strip()
                elif last and re.match(r"^(19|20)\d\d$", last):
                    heads[j] = (heads[j] + " " + last).strip()  # propaga ano para a direita
        label_prev = ""
        for r in raw[first_data:]:
            label = r[0] or label_prev
            if not r[0]:
                continue
            if "Diferença" in label and label_prev:
                label = f"{label_prev} – {label}"
            else:
                label_prev = r[0]
            vals = []
            for j in range(1, len(r)):
                if not (r[j] and heads[j]):
                    continue
                hs, vs = heads[j].split(), r[j].split()
                if len(vs) > 1 and len(hs) == len(vs):  # células mescladas: "Nov Dez" / "0,41 0,68"
                    vals += [f"{h}: {v}" for h, v in zip(hs, vs)]
                else:
                    vals.append(f"{heads[j]}: {r[j]}")
            if vals:
                lin.append(f"{label} → " + "; ".join(vals))
    out = "\n".join(rows)
    if lin:
        out = "Valores por coluna:\n" + "\n".join(lin) + "\n\nTabela bruta:\n" + out
    return out


def split(text, size=CHUNK, overlap=OVERLAP):
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            k = text.rfind("\n", i + size // 2, j)
            if k > 0:
                j = k
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return [c for c in out if len(c) > 40]


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMB_MODEL, "input": texts}, timeout=600)
    r.raise_for_status()
    return r.json()["embeddings"]


def main():
    doc = pymupdf.open(PDF)
    chunks = []
    section = ""
    for pno, page in enumerate(doc, start=1):
        text = clean(page.get_text())
        # título do boxe/seção: primeira linha da página costuma ser o título
        first = text.split("\n", 1)[0][:120] if text else ""
        if re.match(r"^(\d\.\d|Boxe|[A-ZÁÉÍÓÚ][^.]{5,80}$)", first):
            section = first
        for c in split(text):
            chunks.append({"page": pno, "kind": "text", "section": section, "text": c})
        try:
            tabs = page.find_tables().tables
        except Exception:
            tabs = []
        for t in tabs:
            md = table_md(t)
            if len(md) > 60 and re.search(r"\d", md):
                # legenda: linha "Tabela ..." mais próxima no texto
                caps = re.findall(r"(Tabela [\w\.]+ – [^\n]+)", text)
                cap = " / ".join(caps)
                chunks.append({"page": pno, "kind": "table", "section": section,
                               "text": f"[Tabelas da página: {cap}]\n{md}"[:4500]})
    print(len(chunks), "chunks")
    vecs = []
    B = 16
    for i in range(0, len(chunks), B):
        batch = [f"{c['section']}\n{c['text']}" for c in chunks[i:i + B]]
        vecs.extend(embed(batch))
        print(i, end=" ", flush=True)
    V = np.array(vecs, dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    np.save("index_vecs.npy", V)
    json.dump(chunks, open("index_chunks.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("\nok")


if __name__ == "__main__":
    main()
