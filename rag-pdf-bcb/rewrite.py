"""Rewrite the question before the fixed search, with no agent (follow-up to part 4).

Usage: python rewrite.py              rewrite (prompt v2) + search only: right page, and what answering would cost
       ANSWER=1 python rewrite.py     also answer from the page images, as in part 3
       PROMPT=v1 python rewrite.py    the first, obvious rewrite prompt

In part 4, the agent's first search, with the question rewritten as keywords, found the page in
29 of the 54 cases the fixed search missed. Here the same models rewrite the question in one call,
the fixed search of part 3 runs on the rewritten query, and the answer comes from the page images
exactly as in vision.py. Answers for page sets part 3 already sent come from its cache, for free.
Rewrites are cached in out/rewrite_cache.jsonl.
"""
import hashlib
import json
import os
import time

import numpy as np
import pymupdf
import requests

from evaluate import K, OUT, chunk, is_correct, model, refused
from vision import DPI, HERE, KEY, PDFS, PROMPT_IMAGE, URL, ask, cache as vision_cache, page_png

MODELS = os.environ.get("MODELS", "anthropic/claude-sonnet-5.5,openai/gpt-6.1-sol,google/gemini-4-argon").split(",")
ANSWER = os.environ.get("ANSWER") == "1"

PROMPTS = {
    # v1: the obvious prompt. Models copied the report's name into every query (it is in every page
    # header, so it matches any page) and some wrote search-engine syntax ("A" OR "B").
    "v1": {
        "pt": "Reescreva a pergunta abaixo como uma consulta de busca no Relatório de Política Monetária do "
              "Banco Central (setembro de 2026, em português). Use as palavras-chave que o próprio relatório "
              "usaria. Responda só com a consulta.",
        "en": "Rewrite the question below as a search query for the Banco Central do Brasil's Monetary Policy "
              "Report (September 2026, in English). Use the keywords the report itself would use. "
              "Reply with the query only.",
    },
    # v2: written for a semantic search over chunks, after reading v1's queries.
    "v2": {
        "pt": "A busca compara o significado da consulta com trechos de um relatório econômico. Reescreva a "
              "pergunta abaixo como a frase curta (até 10 palavras) que estaria no trecho com a resposta, "
              "em português. Não cite o nome do relatório nem do Banco Central, não use aspas nem operadores "
              "como OR. Responda só com a consulta.",
        "en": "The search compares the meaning of the query with chunks of an economic report. Rewrite the "
              "question below as the short phrase (up to 10 words) that would appear in the chunk holding the "
              "answer, in English. Do not name the report or the central bank, and do not use quotes or "
              "operators such as OR. Reply with the query only.",
    },
}
VERSION = os.environ.get("PROMPT", "v2")
REWRITE = PROMPTS[VERSION]

cache_file = OUT / "rewrite_cache.jsonl"
cache = {}
if cache_file.exists():
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cache[row["key"]] = row


def rewrite(model_id, lang, q):
    key = hashlib.sha256(f"{model_id}\n{lang}\n{q['id']}\n{REWRITE[lang]}".encode()).hexdigest()
    if key not in cache:
        for attempt in range(4):
            try:
                r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"}, json={
                    "model": model_id, "temperature": 0, "max_tokens": 300, "usage": {"include": True},
                    "messages": [{"role": "system", "content": REWRITE[lang]}, {"role": "user", "content": q[lang]}],
                }, timeout=120)
                if r.status_code == 200 and "choices" in r.json():
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(10 * (attempt + 1))
        else:
            raise SystemExit(f"{model_id}: rewrite failed")
        data = r.json()
        cache[key] = {"key": key, "query": (data["choices"][0]["message"].get("content") or "").strip().strip('"'),
                      "cost": data.get("usage", {}).get("cost") or 0}
        with cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
    return cache[key]


def vision_key(model_id, lang, q, pages):
    return hashlib.sha256(f"{model_id}\nimage\n{lang}\n{q['id']}\n{pages}\n{DPI}".encode()).hexdigest()


if __name__ == "__main__":
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    targets = json.loads((HERE / "targets.json").read_text())
    part3 = json.loads((OUT / "results_vision.json").read_text(encoding="utf-8"))
    results, to_pay = {}, 0
    for lang in ("pt", "en"):
        chunks = chunk((OUT / f"pypdf_{lang}.md").read_text(encoding="utf-8"))
        passages = model.encode(["passage: " + t for t, _ in chunks], normalize_embeddings=True, batch_size=32)
        doc = pymupdf.open(PDFS[lang])
        for model_id in MODELS:
            rows = {}
            for q in questions:
                rw = rewrite(model_id, lang, q)
                scores = model.encode(["query: " + rw["query"]], normalize_embeddings=True) @ passages.T
                pages = sorted({chunks[i][1] for i in np.argsort(-scores[0])[:K]})
                row = {"id": q["id"], "type": q["type"], "query": rw["query"], "pages": pages,
                       "found": targets[f"{lang}/{q['id']}"] in pages, "rewrite_cost": rw["cost"]}
                key = vision_key(model_id, lang, q, pages)
                if key not in vision_cache:
                    to_pay += 1
                if ANSWER:
                    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_png(doc, p)}"}}
                               for p in pages] + [{"type": "text", "text": q[lang]}]
                    res = ask(model_id, PROMPT_IMAGE[lang], content, key)
                    row.update(reply=res["reply"], answer_cost=res["cost"] or 0, refused=refused(res["reply"]),
                               correct=is_correct(res["reply"], q["answers"]))
                rows[q["id"]] = row
            results[f"{lang}/{model_id}"] = rows
            p3 = part3[f"{lang}/{model_id}/image"]
            line = (f"{lang} {model_id}: found {sum(r['found'] for r in rows.values())}/33 "
                    f"(part 3: {sum(r['found'] for r in p3.values())}), rewrite ${sum(r['rewrite_cost'] for r in rows.values()):.3f}")
            if ANSWER:
                line += (f", correct {sum(r['correct'] for r in rows.values())}/33 (part 3: {sum(r['correct'] for r in p3.values())}), "
                         f"answer ${sum(r['answer_cost'] for r in rows.values()):.3f}")
            print(line, flush=True)
    if not ANSWER:
        print(f"\nanswer calls not in part 3's cache: {to_pay}")
    (OUT / f"results_rewrite_{VERSION}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
