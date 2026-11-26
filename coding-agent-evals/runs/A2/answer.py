"""Responde perguntas sobre o RPM de setembro de 2026 (RAG local: bge-m3 + BM25 + qwen2.5 via Ollama)."""
import json, os, re, sys, unicodedata
import numpy as np
import requests
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA = "http://localhost:11434"
EMB_MODEL = "bge-m3"
LLM = os.environ.get("RPM_LLM", "qwen2.5:7b")
TOPK = 8

STOP = set("a o os as de da do das dos e em no na nos nas um uma para por com que qual quais é foi ser ao à se como mais".split())


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def tok(s):
    return [w for w in re.findall(r"\w+", norm(s)) if w not in STOP]


def ensure_index():
    if not os.path.exists(os.path.join(HERE, "index_vecs.npy")):
        cwd = os.getcwd()
        os.chdir(HERE)
        import ingest
        ingest.main()
        os.chdir(cwd)


def retrieve(q, chunks, V):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMB_MODEL, "input": [q], "keep_alive": "60m"}, timeout=300)
    qv = np.array(r.json()["embeddings"][0], dtype=np.float32)
    qv /= np.linalg.norm(qv)
    dense = V @ qv
    bm = BM25Okapi([tok(c["section"] + " " + c["text"]) for c in chunks])
    sparse = bm.get_scores(tok(q))
    score = np.zeros(len(chunks))
    for arr in (dense, sparse):
        for rank, i in enumerate(np.argsort(-arr)[:50]):
            score[i] += 1.0 / (60 + rank)
    order = np.argsort(-score)
    picked = list(order[:TOPK])
    return [chunks[i] for i in sorted(picked, key=lambda i: (chunks[i]["page"], i))]


PROMPT = """Você responde perguntas sobre o Relatório de Política Monetária (RPM) do Banco Central do Brasil de setembro de 2026, usando SOMENTE os trechos abaixo.
Os trechos podem conter tabelas extraídas do PDF (colunas separadas por "|" ou valores em sequência vertical, na mesma ordem dos cabeçalhos).

TRECHOS:
{ctx}

PERGUNTA: {q}

Instruções:
- Responda em português, em UMA frase completa, curta e direta (sujeito + verbo + valor, ex.: "A projeção de X para 2026 é de 1,8%."), contendo o número (com unidade, ex.: %, p.p., US$) ou o nome pedido. Nunca responda só com o número.
- Use exatamente os valores do texto; não arredonde nem invente.
- Se a pergunta for "para quanto" ou "qual o nível/valor", informe o valor final (nível), não apenas a variação.
- Se a informação não estiver nos trechos, diga "Não encontrei essa informação no relatório."
RESPOSTA:"""


def main():
    if len(sys.argv) < 2:
        print('uso: python answer.py "pergunta"')
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    ensure_index()
    chunks = json.load(open(os.path.join(HERE, "index_chunks.json"), encoding="utf-8"))
    V = np.load(os.path.join(HERE, "index_vecs.npy"))
    hits = retrieve(q, chunks, V)
    ctx = "\n\n".join(f"[p. {c['page']} | {c['section']}]\n{c['text']}" for c in hits)
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": LLM, "prompt": PROMPT.format(ctx=ctx, q=q), "stream": False, "keep_alive": "60m",
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 120},
    }, timeout=600)
    ans = r.json()["response"].strip().split("\n")[0].strip()
    if os.environ.get("RPM_DEBUG"):
        print(ctx, file=sys.stderr)
    print(ans)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
