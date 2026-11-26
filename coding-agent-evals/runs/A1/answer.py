"""Responde perguntas sobre o RPM de setembro/2026 do BCB.
Uso: python answer.py "pergunta"
Recuperação híbrida (BM25 + embeddings multilingual-e5) e geração com qwen2.5:7b no Ollama."""
import json, os, re, sys, unicodedata, warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import numpy as np
import requests
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = os.path.join(HERE, "index")
OLLAMA = "http://localhost:11434/api/chat"
LLM = os.environ.get("RPM_LLM", "qwen2.5:7b")
EMB_MODEL = "intfloat/multilingual-e5-base"
TOP_K = int(os.environ.get("RPM_TOPK", "5"))

STOP = set("""a o as os de da do das dos e em no na nos nas um uma para por com que qual quais
quanto quantos quanta quantas foi é ser se ao aos à às como sobre segundo relatório rpm
qual? era são está estava ou mais menos pelo pela pelos pelas seu sua seus suas""".split())


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def tok(s):
    return [t for t in re.findall(r"[a-z0-9]+(?:[,.][0-9]+)?", norm(s)) if t not in STOP]


def retrieve(q, chunks, emb):
    bm = BM25Okapi([tok(c["text"]) for c in chunks])
    s_bm = bm.get_scores(tok(q))
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL, device="cpu")
    qv = m.encode(["query: " + q], normalize_embeddings=True)[0]
    s_de = emb @ qv
    # Reciprocal rank fusion
    score = np.zeros(len(chunks))
    for s in (s_bm, s_de):
        for r, i in enumerate(np.argsort(-s)):
            score[i] += 1.0 / (60 + r)
    return list(np.argsort(-score)[:TOP_K])


SYSTEM = (
    "Você responde perguntas sobre o Relatório de Política Monetária (RPM) do Banco Central do Brasil "
    "de setembro de 2026, usando SOMENTE os trechos fornecidos. Responda em português, em UMA frase curta "
    "e direta (sujeito + verbo + valor, ex.: 'A projeção de X para 2026 é de 1,8%.') que contenha o número (com unidade, ex.: %, R$, US$, pontos) ou o nome pedido, exatamente "
    "como aparece no texto. Preste atenção a período, ano, trimestre e indicador pedidos. Tabelas aparecem "
    "linha a linha: o rótulo seguido dos valores na ordem das colunas do cabeçalho. "
    "Se a informação não estiver nos trechos, diga que não foi encontrada no relatório."
)


def ask_llm(q, ctx):
    user = f"Trechos do relatório:\n\n{ctx}\n\nPergunta: {q}\nResposta (uma frase curta):"
    r = requests.post(OLLAMA, json={
        "model": LLM, "stream": False,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "Trechos do relatório:\n\n[p. 1]\nA projeção de "
                      "crescimento do PIB para 2026 foi revisada de 2,0% para 1,8%.\n\nPergunta: Qual a "
                      "projeção de crescimento do PIB para 2026?\nResposta (uma frase curta):"},
                     {"role": "assistant", "content": "A projeção de crescimento do PIB para 2026 é de 1,8%."},
                     {"role": "user", "content": user}],
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 8192},
    }, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def main():
    if len(sys.argv) < 2:
        print('Uso: python answer.py "pergunta"'); sys.exit(1)
    q = " ".join(sys.argv[1:])
    with open(os.path.join(IDX, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    emb = np.load(os.path.join(IDX, "emb.npy"))
    ids = sorted(retrieve(q, chunks, emb), key=lambda i: i)  # ordem do documento
    ctx = "\n\n".join(f"[p. {chunks[i]['page']}]\n{chunks[i]['text']}" for i in ids)
    ans = ask_llm(q, ctx)
    ans = " ".join(ans.split())
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(ans)
    if os.environ.get("RPM_DEBUG"):
        print("páginas:", [chunks[i]["page"] for i in ids], file=sys.stderr)


if __name__ == "__main__":
    main()
