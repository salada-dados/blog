"""Gera embeddings (bge-m3 via Ollama) para data/chunks.json -> data/emb.npy."""
import json
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).parent
OLLAMA = "http://localhost:11434"


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": "bge-m3", "input": texts,
                                                  "keep_alive": "30m"}, timeout=600)
    r.raise_for_status()
    v = np.array(r.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed_text(c):
    t = c["text"]
    if c["kind"] == "chart":
        # título + nomes das séries (os números não ajudam na busca semântica)
        lines = t.split("\n")
        names = [l.split(":")[0] for l in lines[1:]]
        t = lines[0] + " " + " ".join(names)
    t = t.replace("MM3M", "média móvel de três meses").replace("12m", "12 meses")
    return t[:1500]


if __name__ == "__main__":
    chunks = json.load(open(HERE / "data" / "chunks.json", encoding="utf-8"))
    texts = [embed_text(c) for c in chunks]
    vecs = []
    for i in range(0, len(texts), 32):
        vecs.append(embed(texts[i:i + 32]))
        print(i, flush=True)
    np.save(HERE / "data" / "emb.npy", np.vstack(vecs))
