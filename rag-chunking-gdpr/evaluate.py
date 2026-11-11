"""Compare chunking strategies on the GDPR, in Portuguese and English.

Usage: python evaluate.py
For each question we know which article holds the answer. A strategy scores a hit when
a chunk from that article is among the top 3 results. Search is exact cosine similarity,
the same thing pgvector does without an index.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import chunking as c

K = 3
HERE = Path(__file__).parent
model = SentenceTransformer(c.MODEL)
MAX_TOKENS = model.max_seq_length  # 512: anything after this is silently ignored

OFFSETS = (0, 0.25, 0.5, 0.75)  # where the first fixed-size cut falls, as a fraction of the size
STRATEGIES = {
    **{f"fixed-{s}": [lambda a, l, s=s, o=o: c.fixed_size(a, l, s, int(s * o)) for o in OFFSETS]
       for s in (128, 256, 512)},
    "fixed-500+64": [lambda a, l: c.fixed_overlap(a, l, 500, 64)],
    "fixed-500+128": [lambda a, l: c.fixed_overlap(a, l, 500, 128)],
    "by-article": [c.by_article],
    "by-paragraph": [c.by_paragraph],
    "by-point": [c.by_point],
}


def evaluate(chunks, questions, lang):
    texts = [t for t, _ in chunks]
    # e5 was trained with these prefixes; without them, search gets worse
    passages = model.encode(["passage: " + t for t in texts], normalize_embeddings=True, batch_size=32)
    queries = model.encode(["query: " + q[lang] for q in questions], normalize_embeddings=True)
    scores = queries @ passages.T

    hits, rr, sent, misses = 0, 0.0, 0, []
    for qi, q in enumerate(questions):
        order = np.argsort(-scores[qi])
        target = set(q["articles"])
        rank = next(r + 1 for r, ci in enumerate(order) if chunks[ci][1] & target)
        rr += 1 / rank
        if rank <= K:
            hits += 1
        else:
            misses.append({"question": q[lang], "rank": rank, "top1": sorted(chunks[order[0]][1])})
        sent += sum(c.n_tokens(texts[ci]) for ci in order[:K])

    lengths = [c.n_tokens(t) for t in texts]
    read = [len(c.tokenizer.encode("passage: " + t)) for t in texts]
    return {
        "chunks": len(chunks),
        "median_tokens": int(np.median(lengths)),
        "over_limit": sum(r > MAX_TOKENS for r in read),
        "unread": sum(max(0, r - MAX_TOKENS) for r in read) / sum(lengths),
        "hit_at_3": hits / len(questions),
        "mrr": rr / len(questions),
        "tokens_to_llm": sent / len(questions),
        "misses": misses,
    }


if __name__ == "__main__":
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    results = {}
    print("| lang | strategy | chunks | over 512 | unread | hit@3 | MRR | tokens to LLM |")
    print("|---|---|---|---|---|---|---|---|")
    for lang in ("pt", "en"):
        articles = json.loads((HERE / f"data/articles_{lang}.json").read_text(encoding="utf-8"))
        for name, runs in STRATEGIES.items():
            rs = [evaluate(fn(articles, lang), questions, lang) for fn in runs]
            results[f"{lang}/{name}"] = rs
            hit = [r["hit_at_3"] for r in rs]
            spread = f" ({min(hit):.0%}–{max(hit):.0%})" if len(rs) > 1 else ""
            r = rs[0]
            print(f"| {lang} | {name} | {r['chunks']} | {r['over_limit']} | {r['unread']:.0%} | "
                  f"{np.mean(hit):.0%}{spread} | {np.mean([x['mrr'] for x in rs]):.2f} | "
                  f"{np.mean([x['tokens_to_llm'] for x in rs]):.0f} |")
    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
