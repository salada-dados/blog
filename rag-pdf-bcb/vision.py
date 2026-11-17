"""Search the text, answer from the page image: large models via OpenRouter.

Usage: python vision.py
       MODELS=anthropic/claude-sonnet-5.5 IDS=t01,c01 MODES=image python vision.py   (a subset)

The search is the same as in evaluate.py with pypdf (chunks of up to 400 tokens, e5-small, top 3).
Each model then answers every question twice:
  text   the 3 chunks as text, as the 3B model got them in the article
  image  the pages those 3 chunks came from, rendered at 150 dpi, plus the question
Comparing the two separates what comes from a larger model and what comes from seeing the page.

The key is read from OPENROUTER_API_KEY, or from a dotenv file given in ENV_FILE (variable
OPENROUTER_API_KEY or OPEN_ROUTER). It is never printed or saved. Replies, tokens and the cost
OpenRouter reports are cached in out/vision_cache.jsonl, so reruns are free.
"""
import base64
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pymupdf
import requests

from evaluate import K, OUT, PROMPT, chunk, is_correct, model, refused

HERE = Path(__file__).parent
URL = "https://openrouter.ai/api/v1/chat/completions"
DPI = 150
MODELS = os.environ.get("MODELS", ",".join([
    "anthropic/claude-sonnet-5.5",
    "openai/gpt-6.1-sol",
    "google/gemini-4-argon",
    "qwen/qwen3-vl-235b-a22b-instruct",
])).split(",")
MODES = os.environ.get("MODES", "text,image").split(",")
IDS = set(filter(None, os.environ.get("IDS", "").split(",")))
PDFS = {"pt": HERE / "data/rpm202609p.pdf", "en": HERE / "data/rpm202609i.pdf"}

PROMPT_IMAGE = {
    "pt": "Responda à pergunta usando apenas as páginas do Relatório de Política Monetária nas imagens. "
          "Responda em uma frase curta, com o número ou o nome pedido. "
          "Se a resposta não estiver nas páginas, diga que não encontrou.",
    "en": "Answer the question using only the Monetary Policy Report pages in the images. "
          "Answer in one short sentence, with the number or the name asked for. "
          "If the answer is not on the pages, say you could not find it.",
}


def api_key() -> str:
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return key
    env_file = os.environ.get("ENV_FILE")
    if env_file:
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() in ("OPENROUTER_API_KEY", "OPEN_ROUTER") and value.strip():
                return value.strip().strip('"').strip("'")
    raise SystemExit("Set OPENROUTER_API_KEY, or ENV_FILE pointing to a file that has it.")


KEY = api_key()

cache_file = OUT / "vision_cache.jsonl"
cache = {}
if cache_file.exists():
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cache[row["key"]] = row


def page_png(doc, page: int) -> str:
    pix = doc[page - 1].get_pixmap(dpi=DPI)
    return base64.b64encode(pix.tobytes("png")).decode()


def ask(model_id: str, system: str, content, key: str) -> dict:
    if key not in cache:
        start = time.perf_counter()
        for attempt in range(3):
            r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"}, json={
                "model": model_id, "temperature": 0, "max_tokens": 4000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "usage": {"include": True},
            }, timeout=300)
            if r.status_code < 500 and r.status_code != 429:
                break
            time.sleep(10 * (attempt + 1))
        if r.status_code != 200:
            raise SystemExit(f"{model_id}: HTTP {r.status_code} {r.text[:300]}")
        data = r.json()
        usage = data.get("usage", {})
        cache[key] = {
            "key": key, "model": model_id,
            "reply": data["choices"][0]["message"].get("content") or "",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost": usage.get("cost"),
            "seconds": round(time.perf_counter() - start, 1),
        }
        with cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
    return cache[key]


if __name__ == "__main__":
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    if IDS:
        questions = [q for q in questions if q["id"] in IDS]
    targets = json.loads((HERE / "targets.json").read_text())
    results_file = OUT / "results_vision.json"
    results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else {}

    for lang in ("pt", "en"):
        chunks = chunk((OUT / f"pypdf_{lang}.md").read_text(encoding="utf-8"))
        texts = [t for t, _ in chunks]
        passages = model.encode(["passage: " + t for t in texts], normalize_embeddings=True, batch_size=32)
        queries = model.encode(["query: " + q[lang] for q in questions], normalize_embeddings=True)
        scores = queries @ passages.T
        doc = pymupdf.open(PDFS[lang])

        for qi, q in enumerate(questions):
            top = np.argsort(-scores[qi])[:K]
            pages = sorted({chunks[i][1] for i in top})
            context = "\n\n".join(texts[i] for i in top)
            found = targets[f"{lang}/{q['id']}"] in pages
            for model_id in MODELS:
                for mode in MODES:
                    if mode == "text":
                        system, content = PROMPT[lang], f"{context}\n\n---\n\n{q[lang]}"
                    else:
                        system = PROMPT_IMAGE[lang]
                        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_png(doc, p)}"}}
                                   for p in pages] + [{"type": "text", "text": q[lang]}]
                    key = hashlib.sha256(f"{model_id}\n{mode}\n{lang}\n{q['id']}\n{pages}\n{DPI}".encode()).hexdigest()
                    res = ask(model_id, system, content, key)
                    row = {k: res[k] for k in ("reply", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost", "seconds")}
                    row.update(id=q["id"], type=q["type"], pages=pages, found=found,
                               refused=refused(res["reply"]),
                               correct=is_correct(res["reply"], q["answers"]))
                    results.setdefault(f"{lang}/{model_id}/{mode}", {})[q["id"]] = row
                    mark = "ok " if row["correct"] else "-- "
                    print(f"{mark}{lang} {q['id']} {model_id} {mode}: ${row['cost'] or 0:.4f} "
                          f"{row['prompt_tokens']}+{row['completion_tokens']} tok, {row['seconds']}s | {row['reply'][:90]!r}",
                          flush=True)
            results_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n| lang | model | mode | text | table | chart | total | cost |\n|---|---|---|---|---|---|---|---|")
    for name, rows in results.items():
        rows = list(rows.values())
        by = {t: [r for r in rows if r["type"] == t] for t in ("text", "table", "chart")}
        cells = " | ".join(f"{sum(r['correct'] for r in v)}/{len(v)}" for v in by.values())
        cost = sum(r["cost"] or 0 for r in rows)
        lang, mode = name.split("/")[0], name.split("/")[-1]
        print(f"| {lang} | {'/'.join(name.split('/')[1:-1])} | {mode} | {cells} | "
              f"{sum(r['correct'] for r in rows)}/{len(rows)} | ${cost:.3f} |")
