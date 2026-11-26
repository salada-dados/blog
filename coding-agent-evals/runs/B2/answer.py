"""Responde perguntas sobre o RPM de setembro de 2026.

Uso: python answer.py "pergunta"
Recuperação híbrida (BM25 + bge-m3) sobre texto, linhas de tabelas e séries de gráficos
extraídas do PDF; resposta gerada pelo qwen2.5 local via Ollama.
"""
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).parent
OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5:7b"
BUDGET = 3800  # caracteres de contexto

STOP = set("""a o as os de da do das dos e em no na nos nas um uma para por com que qual quais
quanto quanta foi é ser se ao aos à às sobre entre como mais menos ou seu sua seus suas ano
anos em relação qual foi está estava estavam""".split())


SYN = [(r"\bmm3m\b", "média móvel de três meses"), (r"\bdoze meses\b", "12 meses"),
       (r"\ba\.s\.", "ajuste sazonal"), (r"\btri\b", "trimestre")]


def norm(s):
    s = s.lower()
    for a, b in SYN:
        s = re.sub(a, b, s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", norm(s)) if t not in STOP]


class BM25:
    def __init__(self, docs, k1=1.4, b=0.75):
        self.docs = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / len(docs)
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q):
        out = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            out[i] = s
        return out


def embed(q):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": "bge-m3", "input": [q], "keep_alive": "30m"},
                      timeout=300)
    v = np.array(r.json()["embeddings"][0], dtype=np.float32)
    return v / np.linalg.norm(v)


def search_text(c):
    if c["kind"] == "chart":
        lines = c["text"].split("\n")
        return lines[0] + " " + " ".join(l.split(":")[0] for l in lines[1:]) + " gráfico"
    return c["text"]


def retrieve(question, chunks, emb, k=8):
    bm = BM25([toks(search_text(c)) for c in chunks])
    s_bm = bm.scores(toks(question))
    s_de = emb @ embed(question)
    rank_bm = np.argsort(-s_bm)
    rank_de = np.argsort(-s_de)
    rrf = np.zeros(len(chunks))
    for r, i in enumerate(rank_bm[:50]):
        rrf[i] += 1 / (60 + r)
    for r, i in enumerate(rank_de[:50]):
        rrf[i] += 1 / (60 + r)
    if CHART_Q.search(question):
        for i, c in enumerate(chunks):
            if c["kind"] == "chart":
                rrf[i] *= 1.6
    return [int(i) for i in np.argsort(-rrf)[:k]]


CHART_Q = re.compile(r"\b(em que ano|gr[aá]fico|pico|maior valor|menor (valor|n[ií]vel|crescimento)|qual economia|subiu mais|ca[ií]u mais|chegou perto|em meados)\b", re.I)


def compact_chart(text, question, limit=1400):
    """Mantém título e resumo das séries; corta a lista por ano se ficar grande."""
    lines = text.split("\n")
    out = [lines[0]]
    for l in lines[1:]:
        if len(l) > limit // max(1, len(lines) - 1) + 200:
            head, _, rest = l.partition(" Por ano: ")
            tail = ""
            m = re.search(r" Ano do pico.*$", rest)
            if m:
                tail = m.group(0)
                rest = rest[:m.start()]
            years = re.findall(r"\b(?:19|20)\d\d\b", question)
            items = rest.split("; ")
            if years:
                items = [x for x in items if any(x.startswith(y) for y in years)] or items
            l = head + (" Por ano: " + "; ".join(items) if items else "") + tail
        out.append(l)
    return "\n".join(out)


def build_context(question, chunks, ids):
    parts, used = [], 0
    for i in ids:
        c = chunks[i]
        t = c["text"]
        if c["kind"] == "chart":
            t = compact_chart(t, question)
        elif c["kind"] == "table":
            title, label, cells = (t.split(" | ", 2) + ["", ""])[:3]
            t = title + "\n" + "\n".join(f"- {label}, {cell}" for cell in cells.split("; "))
        t = f"[p. {c['page']}] {t}"
        if used + len(t) > BUDGET and parts:
            if used + 300 > BUDGET:
                break
            t = t[:BUDGET - used]
        parts.append(t)
        used += len(t)
    return "\n\n".join(parts)


PROMPT = """Você responde perguntas sobre o Relatório de Política Monetária (RPM) do Banco Central do Brasil de setembro de 2026, usando os trechos abaixo (texto, linhas de tabelas e séries extraídas de gráficos).

Trechos:
{ctx}

Pergunta: {q}

Instruções: responda em UMA frase curta em português, começando diretamente pela resposta, com o número exato (como aparece no trecho, com vírgula decimal e unidade) ou o nome pedido. Escolha o trecho que corresponde exatamente ao período, à variável e à comparação da pergunta. Não explique.
Resposta:"""


def generate(prompt):
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 60}}, timeout=590)
    r.raise_for_status()
    return r.json()["response"].strip()


def main():
    q = " ".join(sys.argv[1:]).strip()
    chunks = json.load(open(HERE / "data" / "chunks.json", encoding="utf-8"))
    emb = np.load(HERE / "data" / "emb.npy")
    ids = retrieve(q, chunks, emb)
    ctx = build_context(q, chunks, ids)
    if "--ctx" in sys.argv:
        print(ctx)
        return
    ans = generate(PROMPT.format(ctx=ctx, q=q))
    ans = ans.split("\n")[0].strip()
    print(ans)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
