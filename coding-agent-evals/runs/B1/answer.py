"""Responde perguntas sobre o RPM de setembro de 2026.

Uso: python answer.py "pergunta"
Recupera trechos (texto, linhas de tabela, gráficos) com busca híbrida BM25 + bge-m3 e
gera a resposta com qwen2.5 (texto) ou qwen2.5vl (imagem do gráfico) via Ollama local.
"""
import base64
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import requests
from rank_bm25 import BM25Okapi

HERE = Path(__file__).parent
IDX = HERE / "index"
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TEXT_MODEL = os.environ.get("RPM_TEXT_MODEL", "qwen2.5:7b")
VL_MODEL = "qwen2.5vl:3b"
EMB_MODEL = "bge-m3"
DEBUG = os.environ.get("RPM_DEBUG")

STOP = set("""a o as os de da do das dos e em no na nos nas um uma para por com que qual quais quanto quanta
quantos foi é ser se ao aos à às sua seu suas seus como mais ou entre relação sobre the""".split())


def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", norm(s)) if t not in STOP]


def embed(text):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMB_MODEL, "input": [text]}, timeout=300)
    r.raise_for_status()
    v = np.array(r.json()["embeddings"][0], dtype=np.float32)
    return v / np.linalg.norm(v)


EXPAND = [(r"doze meses", " 12m 12 meses"), (r"m[eé]dia m[oó]vel de tr[eê]s meses", " MM3M"),
          (r"trimestre anterior", " tri ante tri"), (r"interanual", " mesmo trimestre do ano anterior")]


def expand(q):
    for pat, add in EXPAND:
        if re.search(pat, q, re.I):
            q += add
    return q


def retrieve(question, chunks, emb, types=None, k=8):
    question = expand(question)
    idx = [i for i, c in enumerate(chunks) if types is None or c["type"] in types]
    docs = [tokens(chunks[i]["text"]) for i in idx]
    bm = BM25Okapi(docs)
    qt = tokens(question)
    s_bm = bm.get_scores(qt)
    qv = embed(question)
    s_em = emb[idx] @ qv
    r_bm = np.argsort(-s_bm)
    r_em = np.argsort(-s_em)
    score = np.zeros(len(idx))
    for rank, j in enumerate(r_bm):
        score[j] += 1 / (60 + rank)
    for rank, j in enumerate(r_em):
        score[j] += 1 / (60 + rank)
    order = np.argsort(-score)[:k]
    return [(chunks[idx[j]], score[j]) for j in order]


def ollama_chat(model, messages, num_predict=80, num_ctx=4096):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": model, "messages": messages, "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": num_ctx}}, timeout=900)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


CHART_Q = re.compile(r"gr[aá]fico|em que ano|pico|maior valor|menor n[ií]vel|menor valor|subiu mais|"
                     r"chegou perto|em meados|m[eé]dia m[oó]vel|qual economia|atingiu", re.I)

SYSTEM = ("Você responde perguntas sobre o Relatório de Política Monetária (RPM) do Banco Central do Brasil "
          "de setembro de 2026, usando apenas os trechos fornecidos. Linhas de tabela vêm no formato "
          "'Tabela | linha | coluna: valor; coluna: valor'; escolha a coluna que corresponde exatamente ao "
          "período e ao conceito perguntados. Responda em português com UMA frase curta que contenha o número "
          "(copiado exatamente como no trecho, com vírgula decimal e unidade) ou o nome pedido. "
          "Se a pergunta for 'para quanto subiu/caiu', responda o novo nível (o valor depois da mudança), "
          "não o tamanho da variação. Em frases com 'respectivamente', associe cada valor ao item na mesma ordem.")


MONTHS = {"janeiro": "jan", "fevereiro": "fev", "março": "mar", "marco": "mar", "abril": "abr", "maio": "mai",
          "junho": "jun", "julho": "jul", "agosto": "ago", "setembro": "set", "outubro": "out",
          "novembro": "nov", "dezembro": "dez"}
QUARTER = {"primeiro": "I", "segundo": "II", "terceiro": "III", "quarto": "IV",
           "1º": "I", "2º": "II", "3º": "III", "4º": "IV"}


def period_filter(question, text):
    """Numa linha de tabela, mantém só as células cuja coluna bate com o período da pergunta."""
    parts = text.split(" | ")
    if len(parts) < 3:
        return text
    q = question.lower()
    years = set(re.findall(r"\b(20\d\d)\b", q))
    months = {v for k, v in MONTHS.items() if re.search(rf"\b{k}\b", q)}
    quarters = {v for k, v in QUARTER.items() if re.search(rf"(?<!\w){k}\s+(tri|bim)", q)}
    if not (years or months or quarters):
        return text
    cells = [c.rsplit(": ", 1) for c in parts[-1].split("; ")]
    if any(len(c) != 2 for c in cells):
        return text

    def score(lab):
        toks = lab.lower().replace("-", " ").split()
        s = 0
        if years:
            s += 1 if any(y in lab for y in years) else 0
        if months:
            s += 2 if any(m in toks for m in months) else 0
        if quarters:
            s += 2 if any(t.upper() in quarters for t in lab.split()) else 0
        return s

    best = max(score(c[0]) for c in cells)
    if best == 0:
        return text
    keep = [c for c in cells if score(c[0]) == best]
    return " | ".join(parts[:-1]) + " | " + "; ".join(f"{a}: {b}" for a, b in keep)


def fmt_chunk(c, question=None):
    if question and c["type"] == "table_row":
        c = {**c, "text": period_filter(question, c["text"])}
    tag = {"text": "Texto", "table_row": "Tabela", "chart": "Gráfico", "table": "Tabela"}[c["type"]]
    return f"[{tag}, p. {c['page']}] {c['text']}"


def answer_text(question, hits, budget=5200):
    ctx, used = [], 0
    for c, _ in hits:
        t = fmt_chunk(c, question)
        if used + len(t) > budget:
            continue
        ctx.append(t)
        used += len(t)
    prompt = "Trechos do relatório:\n\n" + "\n\n".join(ctx) + f"\n\nPergunta: {question}\nResposta (uma frase curta):"
    return ollama_chat(TEXT_MODEL, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])


def answer_chart(question, chart):
    img = base64.b64encode((HERE / chart["img"]).read_bytes()).decode()
    prompt = (f"Este é o {chart['title']} do Relatório de Política Monetária do Banco Central do Brasil. "
              f"Observe o gráfico com atenção (eixos, legenda e séries) e responda em português, em uma frase "
              f"curta, com o número ou o nome pedido.\nPergunta: {question}")
    return ollama_chat(VL_MODEL, [{"role": "user", "content": prompt, "images": [img]}], num_ctx=8192)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Uso: python answer.py \"pergunta\"")
        return
    chunks = json.loads((IDX / "chunks.json").read_text(encoding="utf-8"))
    emb = np.load(IDX / "emb.npy")
    rows = retrieve(question, chunks, emb, types={"table_row", "table"}, k=4)
    texts = retrieve(question, chunks, emb, types={"text"}, k=6)
    hits = rows[:2] + texts[:3] + rows[2:] + texts[3:]
    if DEBUG:
        for c, s in hits:
            print(f"  {s:.4f} {fmt_chunk(c)[:160]}", file=sys.stderr)
    if CHART_Q.search(question):
        charts = retrieve(question, chunks, emb, types={"chart"}, k=3)
        if DEBUG:
            for c, s in charts:
                print(f"  CHART {s:.4f} {c['img']} {c['text'][:120]}", file=sys.stderr)
        print(answer_chart(question, charts[0][0]))
        return
    print(answer_text(question, hits))


if __name__ == "__main__":
    main()
