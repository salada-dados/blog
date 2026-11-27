"""The long tasks (tasks.json), each done by every agent type, several times, graded by a judge.

Usage: ENV_FILE=path/to/.env python run_tasks.py
       ARCHS=react TASKS=L1 REPEATS=1 python run_tasks.py      (a subset)

A long answer varies from run to run even at temperature 0, so each task runs REPEATS times (default 3).
A judge model from another provider (JUDGE, default Claude Sonnet 5.5) checks the answer against
each rubric item and replies true or false; it sees only the rubric and the answer, never the agent
type. Runs and grades are cached in results/tasks_cache.jsonl and results/judge_cache.jsonl.
BUDGET (US$, default 8) stops new runs once the cached cost of the runs goes past it.
"""
import asyncio
import hashlib
import json
import os
import sys

import agents
from common import MODEL, OUT, Run, chat, report_tools

HERE = OUT.parent
ARCHS = os.environ.get("ARCHS", "workflow,react,plan_execute,deep_agent").split(",")
TASKS = set(filter(None, os.environ.get("TASKS", "").split(",")))
REPEATS = int(os.environ.get("REPEATS", "3"))
BUDGET = float(os.environ.get("BUDGET", "8"))
JUDGE = os.environ.get("JUDGE", "anthropic/claude-sonnet-5.5")
VERSION = "v1"
# Long tasks get more room than the short questions: more steps for the loops, more for the subagents.
LIMITS = {"react": {"max_steps": 15}, "plan_execute": {"step_limit": 4, "replans": 1},
          "deep_agent": {"max_steps": 25, "sub_steps": 6}, "workflow": {}}
# L4 has 20 questions: every type gets room for all of them.
LIMITS_L4 = {"react": {"max_steps": 40}, "plan_execute": {"step_limit": 4, "replans": 1, "max_plan": 12, "max_replan": 8},
             "deep_agent": {"max_steps": 50, "sub_steps": 8}, "workflow": {}}

JUDGE_PROMPT = """You grade an answer about the Banco Central do Brasil's Monetary Policy Report (September 2026).
For each numbered rubric item, decide whether the answer states it. The answer may be in Portuguese
(decimal comma) or English. A number must match the item (within any range the item allows) and be
attached to the right thing and year; a number that appears for something else does not count.
Reply in JSON: {"items": [true, false, ...]}, one boolean per item, in order.

Rubric:
{rubric}

Answer:
{answer}"""


def load(name):
    path = OUT / name
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["key"]] = row
    return path, rows


runs_file, runs = load("tasks_cache.jsonl")
judge_file, grades = load("judge_cache.jsonl")


def append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def grade(task, lang, answer):
    key = hashlib.sha256(f"{JUDGE}\n{task['id']}\n{lang}\n{answer}".encode()).hexdigest()
    if key not in grades:
        rubric = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(task["rubric"]))
        run = Run()
        items = None
        for _ in range(3):  # a reply that is not the JSON asked for is asked again, never graded as all false
            msg = await chat(run, [{"role": "user", "content": JUDGE_PROMPT.replace("{rubric}", rubric).replace("{answer}", answer)}],
                             model=JUDGE, max_tokens=1000)
            text = msg.get("content") or ""
            try:
                items = json.loads(text[text.find("{"): text.rfind("}") + 1])["items"]
            except (ValueError, KeyError):
                continue
            if len(items) == len(task["rubric"]):
                break
            items = None
        if items is None:
            raise RuntimeError(f"judge gave no usable grade for {task['id']} {lang}")
        items = [bool(x) for x in items]
        grades[key] = {"key": key, "items": items, "cost": run.cost}
        append(judge_file, grades[key])
    return grades[key]["items"]


async def run_arch(arch, lang, tasks):
    async with report_tools(lang) as tools:
        for task in tasks:
            for rep in range(REPEATS):
                key = hashlib.sha256(f"{MODEL}\n{arch}\n{lang}\n{task['id']}\n{rep}\n{VERSION}".encode()).hexdigest()
                if key not in runs:
                    if sum(r["cost"] for r in runs.values()) > BUDGET:
                        print(f"budget of US$ {BUDGET} reached, stopping {arch} {lang}", flush=True)
                        return
                    run = Run()
                    extra = {}
                    limits = (LIMITS_L4 if task["id"] == "L4" else LIMITS)[arch]
                    try:
                        if arch == "deep_agent":
                            answer, state = await agents.deep_agent(task[lang], lang, tools, run, "report", **limits)
                            extra = {"todos": state["todos"], "files": state["files"], "subagents": state["subagents"]}
                        else:
                            answer = await getattr(agents, arch)(task[lang], lang, tools, run, "report", **limits)
                    except Exception as e:
                        print(f"!! {arch} {lang} {task['id']} #{rep}: {e!r}"[:300], flush=True)
                        continue
                    runs[key] = {"key": key, "model": MODEL, "arch": arch, "lang": lang, "task": task["id"],
                                 "rep": rep, "answer": answer, **run.summary(), **extra}
                    append(runs_file, runs[key])
                r = runs[key]
                try:
                    items = await grade(task, lang, r["answer"])
                except RuntimeError as e:
                    print(f"!! {e}", flush=True)
                    continue
                print(f"{arch} {lang} {task['id']} #{rep}: {sum(items)}/{len(items)}, {r['llm_calls']} llm, "
                      f"{r['tool_calls']} tools, ${r['cost']:.3f}, {r['seconds']}s", flush=True)


def summary(tasks):
    print(f"\nmodel: {MODEL}, judge: {JUDGE}\n")
    print("| type | task | rubric items (mean of runs) | llm calls | tool calls | tokens | US$/run | s/run |")
    print("|---|---|---|---|---|---|---|---|")
    table = {}
    for arch in ARCHS:
        for task in tasks:
            rs = [r for r in runs.values() if r["arch"] == arch and r["task"] == task["id"] and r["model"] == MODEL]
            if not rs:
                continue
            scores = []
            for r in rs:
                key = hashlib.sha256(f"{JUDGE}\n{task['id']}\n{r['lang']}\n{r['answer']}".encode()).hexdigest()
                if key in grades:
                    scores.append(sum(grades[key]["items"]))
            n, total = len(rs), len(task["rubric"])
            row = {"score": sum(scores) / max(len(scores), 1), "of": total, "runs": n,
                   "llm_calls": sum(r["llm_calls"] for r in rs) / n, "tool_calls": sum(r["tool_calls"] for r in rs) / n,
                   "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs) / n,
                   "cost": sum(r["cost"] for r in rs) / n, "seconds": sum(r["seconds"] for r in rs) / n,
                   "scores": scores}
            table[f"{arch}/{task['id']}"] = row
            print(f"| {arch} | {task['id']} | {row['score']:.1f}/{total} ({n} runs: {scores}) | {row['llm_calls']:.1f} | "
                  f"{row['tool_calls']:.1f} | {row['tokens']:,.0f} | {row['cost']:.3f} | {row['seconds']:.0f} |")
    (OUT / f"tasks_summary_{MODEL.split('/')[-1]}.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"\nruns: US$ {sum(r['cost'] for r in runs.values()):.2f}, judge: US$ {sum(g['cost'] for g in grades.values()):.2f}")


async def main():
    tasks = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
    if TASKS:
        tasks = [t for t in tasks if t["id"] in TASKS]
    if "--summary" not in sys.argv:
        await asyncio.gather(*(run_arch(a, lang, tasks) for a in ARCHS for lang in ("pt", "en")))
    summary(tasks)


if __name__ == "__main__":
    asyncio.run(main())
