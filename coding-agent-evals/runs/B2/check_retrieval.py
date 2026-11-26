"""Diagnóstico: a resposta esperada aparece no contexto recuperado? (sem LLM)"""
import json
import sys

import numpy as np

import answer
from eval import contains

sys.stdout.reconfigure(encoding="utf-8")
qs = json.load(open("dev.json", encoding="utf-8"))
chunks = json.load(open("data/chunks.json", encoding="utf-8"))
emb = np.load("data/emb.npy")
hit = 0
for q in qs:
    ids = answer.retrieve(q["question"], chunks, emb)
    ctx = answer.build_context(q["question"], chunks, ids)
    ok = contains(ctx, q["answers"])
    hit += ok
    print("OK " if ok else "-- ", q["id"], len(ctx), [chunks[i]["page"] for i in ids], q["question"][:70])
print(hit, "/", len(qs))
