"""Chunking strategies. Each one returns a list of (text, {article numbers it contains})."""
from transformers import AutoTokenizer

MODEL = "intfloat/multilingual-e5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def n_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def header(article: dict, lang: str) -> str:
    word = "Artigo" if lang == "pt" else "Article"
    return f"{word} {article['article']}: {article['title']}"


def _whole_text(articles, lang):
    """The whole regulation as one token stream, remembering which article owns each token."""
    ids, owner = [], []
    for a in articles:
        text = header(a, lang) + "\n" + "\n".join(a["paragraphs"]) + "\n"
        t = tokenizer.encode(text, add_special_tokens=False)
        ids += t
        owner += [a["article"]] * len(t)
    return ids, owner


def fixed_size(articles, lang, size, offset=0):
    """Cut every `size` tokens, ignoring the structure. `offset` moves the first cut."""
    ids, owner = _whole_text(articles, lang)
    starts = ([0] if offset else []) + list(range(offset, len(ids), size))
    ends = starts[1:] + [len(ids)]
    return [(tokenizer.decode(ids[s:e]), set(owner[s:e])) for s, e in zip(starts, ends)]


def fixed_overlap(articles, lang, size, overlap):
    """The usual recipe: windows of `size` tokens that move forward by size - overlap."""
    ids, owner = _whole_text(articles, lang)
    step = size - overlap
    return [(tokenizer.decode(ids[s:s + size]), set(owner[s:s + size])) for s in range(0, len(ids), step)]


def by_article(articles, lang):
    return [(header(a, lang) + "\n" + "\n".join(a["paragraphs"]), {a["article"]}) for a in articles]


def by_paragraph(articles, lang):
    return [(header(a, lang) + "\n" + p, {a["article"]}) for a in articles for p in a["paragraphs"]]


def by_point(articles, lang, limit=400):
    """Like by_paragraph, but a paragraph longer than `limit` tokens becomes one chunk per point,
    each one keeping the article title and the paragraph's opening sentence."""
    chunks = []
    for a in articles:
        head = header(a, lang)
        for p in a["paragraphs"]:
            lines = p.split("\n")
            if n_tokens(p) <= limit or len(lines) == 1:
                chunks.append((head + "\n" + p, {a["article"]}))
            else:
                intro, points = lines[0], lines[1:]
                chunks += [(head + "\n" + intro + "\n" + pt, {a["article"]}) for pt in points]
    return chunks
