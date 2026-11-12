"""Describe every figure Docling found with a local vision model, and write docling-vlm_<lang>.md.

Usage: python describe.py
Each "<!-- image -->" placeholder in Docling's Markdown becomes the figure's caption plus a
text description. Descriptions are cached in out/descriptions_<lang>.json, so a crash or a
rerun does not pay for the same figure twice.
"""
import base64
import io
import json
import re
import time
from pathlib import Path

import pymupdf
import requests
from docling_core.types.doc import DoclingDocument, PictureItem

HERE = Path(__file__).parent
OUT = HERE / "out"
OLLAMA = "http://localhost:11434/api/chat"
VLM = "qwen2.5vl:3b"
PDFS = {"pt": HERE / "data/rpm202609p.pdf", "en": HERE / "data/rpm202609i.pdf"}
CAPTION = re.compile(r"^(Gráfico|Figure|Tabela|Table)\s+\d+(\.\d+)*\s*[–-]")
MIN_SIDE = 120  # px at images_scale=2: anything smaller is a logo or an icon, not a chart

PROMPT = {
    # A long, structured prompt made the 3B model invent a title and loop over a template for
    # 7 minutes. A short one, with a cap on output tokens, works and takes about a minute.
    "pt": "Descreva este gráfico em até 5 frases, em português: o que ele mede, o período, as séries da legenda "
          "e o que se destaca (máximos, mínimos, quem é maior ou menor, valores aproximados). Use só o que está visível.",
    "en": "Describe this chart in up to 5 sentences, in English: what it measures, the period, the series in the legend "
          "and what stands out (peaks, lows, which is higher or lower, approximate values). Use only what is visible.",
}


def describe(img, lang: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    reply = requests.post(OLLAMA, json={
        "model": VLM,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 300, "repeat_penalty": 1.15},
        "messages": [{"role": "user", "content": PROMPT[lang],
                      "images": [base64.b64encode(buf.getvalue()).decode()]}],
    }, timeout=900)
    reply.raise_for_status()
    return reply.json()["message"]["content"].strip()


def caption(item: PictureItem, doc: DoclingDocument, pdf: pymupdf.Document) -> str:
    """Docling keeps captions apart from the picture, and links them only sometimes (27 of 123
    figures in pt). Often it even treats the caption as part of the figure, and then the caption
    vanishes from the Markdown. So we read it from the PDF text layer, just above or inside the
    figure. A description without "Figure 1.2.14 – Unemployment rate" is much harder to find."""
    if text := item.caption_text(doc).strip():
        return text
    prov = item.prov[0]
    page = pdf[prov.page_no - 1]
    box = prov.bbox.to_top_left_origin(page_height=page.rect.height)
    area = pymupdf.Rect(box.l, box.t - 30, box.r, box.b)
    lines = []
    for block in page.get_text("dict", clip=area)["blocks"]:
        for line in block.get("lines", []):
            lines.append((line["bbox"][1], "".join(s["text"] for s in line["spans"]).strip()))
    lines.sort()
    for i, (_, text) in enumerate(lines):
        if CAPTION.match(text):
            # captions can wrap: "Gráfico 1.2.11 – Índice de Atividade Econômica do" + "Banco Central (IBC-Br)"
            nxt = lines[i + 1][1] if i + 1 < len(lines) else ""
            return text + (" " + nxt if nxt and not nxt[0].isdigit() and len(nxt) < 60 else "")
    return ""


if __name__ == "__main__":
    timings_file = OUT / "timings.json"
    timings = json.loads(timings_file.read_text())
    for lang in ("pt", "en"):
        doc = DoclingDocument.load_from_json(OUT / f"docling_{lang}.json")
        pdf = pymupdf.open(PDFS[lang])
        cache_file = OUT / f"descriptions_{lang}.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
        pages = {}
        for item, _ in doc.iterate_items():
            if not isinstance(item, PictureItem):
                continue
            page = item.prov[0].page_no
            img = item.get_image(doc)
            ref = item.self_ref
            if img is None or min(img.size) < MIN_SIDE:
                text = ""
            elif ref in cache:
                text = cache[ref]["text"]
            else:
                t0 = time.perf_counter()
                text = describe(img, lang)
                cache[ref] = {"text": text, "seconds": round(time.perf_counter() - t0, 1)}
                cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{lang} p{page} {ref}: {len(text)} chars")
            label = "Descrição do gráfico" if lang == "pt" else "Chart description"
            cap = caption(item, doc, pdf)
            pages.setdefault(page, []).append(f"[{label}] {cap}\n{text}".strip() if text else "<!-- image -->")
        timings[f"vlm/{lang}"] = round(sum(c["seconds"] for c in cache.values()), 1)
        timings_file.write_text(json.dumps(timings, indent=1))

        md = (OUT / f"docling_{lang}.md").read_text(encoding="utf-8")
        parts = md.split("<!-- image -->")
        flat = [d for p in sorted(pages) for d in pages[p]]
        assert len(parts) - 1 == len(flat), (len(parts) - 1, len(flat))
        out = parts[0] + "".join(d + rest for d, rest in zip(flat, parts[1:]))
        (OUT / f"docling-vlm_{lang}.md").write_text(out, encoding="utf-8")
        print(f"{lang}: {sum(1 for d in flat if d != '<!-- image -->')} figures described")
