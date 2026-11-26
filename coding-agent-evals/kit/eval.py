"""Avalia o sistema nas perguntas de dev.json.

Uso: python eval.py
Para cada pergunta, roda `python answer.py "pergunta"` e confere se a resposta traz o número
ou o nome esperado. Uma resposta que começa dizendo que não encontrou conta como erro.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REFUSAL = re.compile(r"não (encontrei|foi possível|está)|não (há|consta|aparece)", re.I)


def contains(text, answers):
    """A resposta como um número ou nome inteiro: '2,8' não vale para '12,8' nem '2,85'."""
    return any(re.search(rf"(?<![\w,.]){re.escape(a)}(?![\d]|[,.]\d)", text, re.I) for a in answers)


def refused(reply):
    first = re.split(r"(?<=[.!?])\s+", reply.strip(), maxsplit=1)[0]
    return bool(REFUSAL.search(first))


def answer(question):
    r = subprocess.run([sys.executable, "answer.py", question], cwd=HERE, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=600,
                       env={**os.environ, "PYTHONUTF8": "1"})
    return r.stdout.strip() or r.stderr.strip()[-300:]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    questions = json.loads((HERE / "dev.json").read_text(encoding="utf-8"))
    by_type, start = {}, time.time()
    for q in questions:
        reply = answer(q["question"])
        ok = contains(reply, q["answers"]) and not refused(reply)
        by_type.setdefault(q["type"], []).append(ok)
        print(f"{'OK ' if ok else 'ERR'} {q['id']} | {q['question']}\n    esperado: {q['answers'][0]} | resposta: {reply[:200]}")
    total = sum(sum(v) for v in by_type.values())
    print("\n" + " | ".join(f"{t}: {sum(v)}/{len(v)}" for t, v in by_type.items()))
    print(f"TOTAL: {total}/{len(questions)} em {time.time() - start:.0f}s")
