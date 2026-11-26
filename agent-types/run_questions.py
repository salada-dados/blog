"""The 33 short questions of the RAG series (pt and en), answered by each agent type.

Usage: ENV_FILE=path/to/.env python run_questions.py
       ARCHS=deep_agent IDS=t01,c01 python run_questions.py      (a subset)

workflow and react already ran with the same model in the RAG series (parts 3 and 4), so by default
only plan_execute and deep_agent run here; run_questions.py --summary adds those earlier results to
the table. Every finished question is cached in results/questions_cache.jsonl, so reruns are free.
BUDGET (US$, default 8) stops new runs once the cached cost goes past it.
"""
import asyncio
import hashlib
import json
import os
import sys

from common import MODEL, OUT, RAG, Run, report_tools

sys.path.insert(0, str(RAG))
from evaluate import is_correct, refused  # noqa: E402  (grading exactly as in the RAG series)

import agents  # noqa: E402

ARCHS = os.environ.get("ARCHS", "plan_execute,deep_agent").split(",")
IDS = set(filter(None, os.environ.get("IDS", "").split(",")))
BUDGET = float(os.environ.get("BUDGET", "8"))
VERSION = "v1"  # change it when a prompt or a limit changes, so the cache does not mix versions

cache_file = OUT / "questions_cache.jsonl"
cache = {}
if cache_file.exists():
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cache[row["key"]] = row


def spent() -> float:
    return sum(r["cost"] for r in cache.values())


async def solve(arch, q, lang, tools):
    run = Run()
    extra = {}
    if arch == "deep_agent":
        reply, state = await agents.deep_agent(q[lang], lang, tools, run)
        extra = {"todos": state["todos"], "files": list(state["files"]), "subagents": state["subagents"]}
    else:
        reply = await getattr(agents, arch)(q[lang], lang, tools, run)
    return {"reply": reply, **run.summary(), **extra}


async def run_arch(arch, lang, questions):
    async with report_tools(lang) as tools:
        for q in questions:
            key = hashlib.sha256(f"{MODEL}\n{arch}\n{lang}\n{q['id']}\n{VERSION}".encode()).hexdigest()
            if key not in cache:
                if spent() > BUDGET:
                    print(f"budget of US$ {BUDGET} reached, stopping {arch} {lang}", flush=True)
                    return
                try:
                    res = await solve(arch, q, lang, tools)
                except Exception as e:  # one failed question must not stop the others
                    print(f"!! {arch} {lang} {q['id']}: {e!r}"[:300], flush=True)
                    continue
                cache[key] = {"key": key, "model": MODEL, "arch": arch, "lang": lang, "id": q["id"], **res}
                with cache_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
            r = cache[key]
            mark = "ok " if is_correct(r["reply"], q["answers"]) else "-- "
            print(f"{mark}{arch} {lang} {q['id']}: {r['llm_calls']} llm, {r['tool_calls']} tools, "
                  f"${r['cost']:.4f}, {r['seconds']}s | {r['reply'][:70]!r}", flush=True)


def summary(questions):
    """One table with the four types; workflow and react come from the RAG series' results."""
    targets = json.loads((RAG / "targets.json").read_text())
    rows = {}
    old_wf = json.loads((RAG / "results/results_vision.json").read_text(encoding="utf-8"))
    old_re = json.loads((RAG / "results/results_agent.json").read_text(encoding="utf-8"))
    for lang in ("pt", "en"):
        for q in questions:
            w = old_wf[f"{lang}/{MODEL}/image"][q["id"]]
            rows.setdefault("workflow", []).append({
                "correct": w["correct"], "found": w["found"], "llm_calls": 1, "tool_calls": 1 + len(w["pages"]),
                "tokens": w["prompt_tokens"] + w["completion_tokens"], "cost": w["cost"], "seconds": w["seconds"]})
            a = old_re[f"{lang}/{MODEL}/detailed"][q["id"]]
            rows.setdefault("react", []).append({
                "correct": a["correct"], "found": a["found"], "llm_calls": a["steps"], "tool_calls": len(a["calls"]),
                "tokens": a["prompt_tokens"] + a["completion_tokens"], "cost": a["cost"], "seconds": a["seconds"]})
            for arch in ("plan_execute", "deep_agent"):
                key = hashlib.sha256(f"{MODEL}\n{arch}\n{lang}\n{q['id']}\n{VERSION}".encode()).hexdigest()
                if key in cache:
                    r = cache[key]
                    rows.setdefault(arch, []).append({
                        "correct": is_correct(r["reply"], q["answers"]), "found": targets[f"{lang}/{q['id']}"] in r["searched"],
                        "llm_calls": r["llm_calls"], "tool_calls": r["tool_calls"],
                        "tokens": r["prompt_tokens"] + r["completion_tokens"], "cost": r["cost"], "seconds": r["seconds"],
                        "subagents": r.get("subagents", 0), "todos": bool(r.get("todos")), "files": bool(r.get("files"))})
    print(f"\nmodel: {MODEL}\n")
    print("| type | right | page found | llm calls/q | tool calls/q | tokens/q | US$ total | s/q |")
    print("|---|---|---|---|---|---|---|---|")
    for arch in ("workflow", "react", "plan_execute", "deep_agent"):
        rs = rows.get(arch, [])
        if not rs:
            continue
        n = len(rs)
        print(f"| {arch} | {sum(r['correct'] for r in rs)}/{n} | {sum(r['found'] for r in rs)} | "
              f"{sum(r['llm_calls'] for r in rs)/n:.1f} | {sum(r['tool_calls'] for r in rs)/n:.1f} | "
              f"{sum(r['tokens'] for r in rs)/n:,.0f} | {sum(r['cost'] for r in rs):.2f} | {sum(r['seconds'] for r in rs)/n:.1f} |")
    deep = rows.get("deep_agent", [])
    if deep:
        print(f"\ndeep_agent: to-do list in {sum(r['todos'] for r in deep)}/{len(deep)}, files in "
              f"{sum(r['files'] for r in deep)}/{len(deep)}, subagents {sum(r['subagents'] for r in deep)} in total")
    (OUT / "questions_summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


async def main():
    questions = json.loads((RAG / "questions.json").read_text(encoding="utf-8"))
    if IDS:
        questions = [q for q in questions if q["id"] in IDS]
    if "--summary" not in sys.argv:
        await asyncio.gather(*(run_arch(a, lang, questions) for a in ARCHS for lang in ("pt", "en")))
    summary(questions)
    print(f"\nspent in this folder (cached runs): US$ {spent():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
