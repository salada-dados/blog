"""Split the 33 questions of the RAG series into the ones the agent may see and the ones it never sees.

Writes kit/dev.json (22 questions, Portuguese only, given to condition B) and holdout.json
(11 questions, kept out of every run folder). Same mix of types in both: text, table and chart.
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE.parent / "rag-pdf-bcb" / "questions.json"
HOLDOUT = {"text": 4, "table": 4, "chart": 3}

questions = json.loads(SOURCE.read_text(encoding="utf-8"))
rng = random.Random(2026)
held = set()
for kind, n in HOLDOUT.items():
    ids = sorted(q["id"] for q in questions if q["type"] == kind)
    held |= set(rng.sample(ids, n))


def slim(q):
    return {"id": q["id"], "type": q["type"], "question": q["pt"], "answers": q["answers"]}


dev = [slim(q) for q in questions if q["id"] not in held]
holdout = [slim(q) for q in questions if q["id"] in held]
(HERE / "kit" / "dev.json").write_text(json.dumps(dev, ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "holdout.json").write_text(json.dumps(holdout, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"dev {len(dev)}, holdout {len(holdout)}:", sorted(held))
