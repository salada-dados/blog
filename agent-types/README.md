# Agent types: workflow, ReAct, plan-and-execute and deep agent

Code for the first article in the **Agents in Practice** series, **"ReAct, Plan-and-Execute or Deep
Agent? One Task, Four Architectures"**
([English](https://saladadedados.com/en/blog/agent-architectures) ·
[Português](https://saladadedados.com/pt/blog/agent-architectures)).

The same two tools, the same report and the same model, solved four ways, all written by hand with
no agent framework:

| Type | Who decides the next step | In `agents.py` |
|---|---|---|
| Workflow | The code: one search, the pages found as images, one answer | `workflow` |
| ReAct | The model, one step at a time, in a loop | `react` |
| Plan-and-execute | The model writes a plan; each step runs as a small loop; a replanner answers or adds steps | `plan_execute` |
| Deep agent | The model, with a to-do list, subagents with their own context, and files as memory | `deep_agent` |

The tools come from the RAG series (part 4): an MCP server over the Banco Central do Brasil's
Monetary Policy Report (September 2026), with `search_report(query)` and `view_page(page)`.
This folder reuses [`../rag-pdf-bcb`](../rag-pdf-bcb) for the server, the report, the 33 questions and
their grading.

## What's inside

| File | What it does |
|---|---|
| `common.py` | The model call through OpenRouter, the MCP tools, and the bookkeeping of tokens, cached tokens, cost and tool calls |
| `agents.py` | The four agent types |
| `run_questions.py` | The 33 short questions of the RAG series, in pt and en, with plan-and-execute and the deep agent. Workflow and ReAct come from parts 3 and 4, with the same model |
| `tasks.json` | Four long tasks with a rubric each: L1 projections (11 items), L2 revisions (9), L3 labor market from text and charts (8), L4 a 20-question briefing (10 tables, 10 charts) |
| `run_tasks.py` | Every long task, by every type, 3 times per language, graded item by item by a judge model from another provider |
| `results/` | Every run, cached: the answer, tool calls, tokens, cached tokens, cost and time (`questions_cache.jsonl`, `tasks_cache.jsonl`), the judge's grades (`judge_cache.jsonl`) and the tables in the article (`*_summary*.json`) |

## Running it

First set up [`../rag-pdf-bcb`](../rag-pdf-bcb) up to part 4: the venv, `docker compose up -d`,
`python download_rpm.py`, `python targets.py` and `python index_pg.py`. Then, from this folder:

```bash
export OPENROUTER_API_KEY=...      # Windows PowerShell: $env:OPENROUTER_API_KEY = "..."
ARCHS=deep_agent IDS=t01,c01 python run_questions.py      # a test run: 2 questions, a few cents
python run_questions.py                                   # plan-and-execute and deep agent: about US$ 2
python run_tasks.py                                       # GPT-6.1 Sol, 4 tasks: about US$ 9 plus the judge
MODEL=openai/gpt-6-luna python run_tasks.py               # the small model: about US$ 0.70
python run_tasks.py --summary                             # the tables, from the cache, free
```

The key can also come from a dotenv file given in `ENV_FILE`; it is never printed or saved. Everything
is cached in `results/`, so a rerun only pays for what is not there yet. `BUDGET` (US$) stops new runs
once the cached cost goes past it. A long answer varies even at temperature 0, which is why each task
runs 3 times per language; the judge also varies by about one rubric item on the same answer.

The deep agent's files live in a Python dict, not on disk: the agent never writes to your machine.

Results in the article: GPT-6.1 Sol and GPT-6 Luna, through OpenRouter, in November 2026. Judge:
Claude Sonnet 5.5. Prices and model versions change; the cost columns are what OpenRouter reported then.
