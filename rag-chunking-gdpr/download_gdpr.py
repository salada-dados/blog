"""Download the GDPR from EUR-Lex (Portuguese and English) and save its 99 articles as JSON.

Usage: python download_gdpr.py
Writes data/articles_pt.json and data/articles_en.json.

Source: Regulation (EU) 2016/679, (c) European Union, https://eur-lex.europa.eu/eli/reg/2016/679/oj
Only the official Journal (OJ) version is authentic.
"""
import json
import re
from pathlib import Path

import requests
from lxml import html

URL = "https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/?uri=CELEX:32016R0679"
DATA = Path(__file__).parent / "data"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def paragraph_text(div) -> str:
    """Text of one numbered paragraph, with each point (a), (b)... on its own line."""
    lines = []
    for el in div.iter("p", "table"):
        if el.tag == "table":
            cells = [clean(td.text_content()) for td in el.iter("td")]
            lines.append(" ".join(c for c in cells if c))
        elif el.getparent().tag == "div" and "oj-normal" in (el.get("class") or ""):
            lines.append(clean(el.text_content()))
    return "\n".join(line for line in lines if line)


def parse(page: bytes) -> list[dict]:
    root = html.fromstring(page)
    articles = []
    for n in range(1, 100):
        art = root.get_element_by_id(f"art_{n}")
        title = clean(art.find_class("oj-sti-art")[0].text_content())
        numbered = [d for d in art if d.tag == "div" and re.fullmatch(r"\d{3}\.\d{3}", d.get("id") or "")]
        if not numbered:  # short articles have no numbered paragraphs
            body = html.Element("div")
            for child in list(art):
                cls = child.get("class") or ""
                if "oj-ti-art" not in cls and "eli-title" not in cls:
                    body.append(child)
            numbered = [body]
        paragraphs = [p for p in (paragraph_text(d) for d in numbered) if p]
        articles.append({"article": n, "title": title, "paragraphs": paragraphs})
    return articles


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for lang in ("pt", "en"):
        page = requests.get(URL.format(lang=lang.upper()), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        page.raise_for_status()
        articles = parse(page.content)
        out = DATA / f"articles_{lang}.json"
        out.write_text(json.dumps(articles, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{out}: {len(articles)} articles, {sum(len(a['paragraphs']) for a in articles)} paragraphs")
