"""Find the page that answers each question, in each language, and save targets.json.

Text questions carry their pages (page_pt, page_en), found by reading the report.
Tables and charts: the page with the caption ("Tabela 1.2.1 –" / "Table 1.2.1 –").
"""
import json
import re
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
PDFS = {"pt": "data/rpm202609p.pdf", "en": "data/rpm202609i.pdf"}
WORD = {"pt": {"table": "Tabela", "chart": "Gráfico"}, "en": {"table": "Table", "chart": "Figure"}}

questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
targets = {}
for lang, pdf in PDFS.items():
    pages = [p.get_text() for p in pymupdf.open(HERE / pdf)]
    for q in questions:
        page = q.get(f"page_{lang}")
        if page is None:
            cap = re.compile(rf"^{WORD[lang][q['type']]} {re.escape(q['ref'])}\s+[–-]", re.M)
            page = next(n for n, t in enumerate(pages, 1) if cap.search(t))
        targets[f"{lang}/{q['id']}"] = page
        if q["type"] != "chart" and not any(a in pages[page - 1] for a in q["answers"]):
            print(f"WARN {lang}/{q['id']}: answer not found on page {page}")
(HERE / "targets.json").write_text(json.dumps(targets, indent=1))
print(targets)
