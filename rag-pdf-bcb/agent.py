"""An agent that answers the 33 questions with the MCP server's tools (part 4).

Usage: python agent.py
       MODELS=google/gemini-4-argon IDS=t01,c01 python agent.py        (a subset)
       TOOL_DOCS=bare MODELS=google/gemini-4-argon python agent.py     (the bare tool descriptions)

No framework: a loop of at most MAX_STEPS model calls. The model gets the question and the tools
the MCP server lists; it decides what to search, which pages to look at and when to answer.
Same questions, grading and models as part 3, so the fixed pipeline and the agent can be compared.

The key is read as in vision.py (OPENROUTER_API_KEY, or ENV_FILE). Every finished question is
cached in out/agent_cache.jsonl with its tool calls, tokens and cost, so reruns are free.
BUDGET (US$, default 15) stops the run if the cost reported by OpenRouter goes past it.
"""
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from mcp import Client, StdioServerParameters

from evaluate import OUT, is_correct, refused
from vision import KEY, URL

HERE = Path(__file__).parent
MODELS = os.environ.get("MODELS", "anthropic/claude-sonnet-5.5,openai/gpt-6.1-sol,google/gemini-4-argon").split(",")
TOOL_DOCS = os.environ.get("TOOL_DOCS", "detailed")
IDS = set(filter(None, os.environ.get("IDS", "").split(",")))
MAX_STEPS = 6
BUDGET = float(os.environ.get("BUDGET", "15"))

SYSTEM = {
    "pt": "Você responde perguntas sobre o Relatório de Política Monetária do Banco Central (setembro de 2026), "
          "usando as ferramentas para consultá-lo. Responda só com o que encontrar no relatório, em uma frase "
          "curta, com o número ou o nome pedido. Se não encontrar, diga que não encontrou.",
    "en": "You answer questions about the Banco Central do Brasil's Monetary Policy Report (September 2026), "
          "using the tools to consult it. Answer only with what you find in the report, in one short sentence, "
          "with the number or the name asked for. If you cannot find it, say you could not find it.",
}
LAST_STEP = {
    "pt": "Você chegou ao limite de consultas. Responda agora com o que já encontrou.",
    "en": "You have reached the limit of lookups. Answer now with what you have found.",
}

cache_file = OUT / "agent_cache.jsonl"
cache = {}
if cache_file.exists():
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cache[row["key"]] = row
spent = sum(r["cost"] for r in cache.values())


def chat(model_id, messages, tools):
    body = {"model": model_id, "temperature": 0, "max_tokens": 4000, "messages": messages,
            "usage": {"include": True}}
    if tools:
        body["tools"] = tools
    error = ""
    for attempt in range(4):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=300)
            if r.status_code == 200 and "choices" in r.json():
                return r.json()
            error = f"HTTP {r.status_code} {r.text[:300]}"
        except (requests.RequestException, ValueError) as e:  # network errors, a cut-off JSON body
            error = repr(e)[:300]
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{model_id}: {error}")


async def solve(client, tools, model_id, lang, q):
    """One question: loop until the model answers without calling a tool, or MAX_STEPS."""
    global spent
    messages = [{"role": "system", "content": SYSTEM[lang]}, {"role": "user", "content": q[lang]}]
    calls, searched, viewed = [], set(), set()
    tokens_in = tokens_out = 0
    cost = 0.0
    start = time.perf_counter()
    reply, hit_limit = "", False
    for step in range(MAX_STEPS + 1):
        last = step == MAX_STEPS
        if last:
            hit_limit = True
            messages.append({"role": "user", "content": LAST_STEP[lang]})
        data = await asyncio.to_thread(chat, model_id, messages, None if last else tools)
        usage = data.get("usage", {})
        tokens_in += usage.get("prompt_tokens") or 0
        tokens_out += usage.get("completion_tokens") or 0
        cost += usage.get("cost") or 0
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            reply = msg.get("content") or ""
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        images = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await client.call_tool(name, args)
            calls.append({"tool": name, "args": args, "error": result.is_error})
            texts = [c.text for c in result.content if c.type == "text"]
            pics = [c for c in result.content if c.type == "image"]
            if name == "search_report" and not result.is_error:
                searched |= {int(p) for t in texts for p in re.findall(r"\[page (\d+)\]", t)}
            if name == "view_page" and not result.is_error:
                viewed.add(args.get("page"))
            # A tool message carries only text in this API. The image goes in a user message right
            # after the tool results, and the tool message says so; otherwise the model never sees it.
            note = f"(page {args.get('page')} follows as an image)" if pics else ""
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "\n".join(texts) or note})
            images += [{"type": "image_url", "image_url": {"url": f"data:{p.mime_type};base64,{p.data}"}} for p in pics]
        if images:
            messages.append({"role": "user", "content": [{"type": "text", "text": "Requested page images:"}] + images})
    spent += cost
    return {"reply": reply, "steps": step + 1, "calls": calls, "searched": sorted(searched),
            "viewed": sorted(p for p in viewed if p is not None), "hit_limit": hit_limit,
            "prompt_tokens": tokens_in, "completion_tokens": tokens_out, "cost": cost,
            "seconds": round(time.perf_counter() - start, 1)}


async def run_model(model_id, lang, questions, targets, results):
    params = StdioServerParameters(command=sys.executable, args=[str(HERE / "mcp_server.py")],
                                   env={**os.environ, "DOC_LANG": lang, "TOOL_DOCS": TOOL_DOCS})
    async with Client(params) as client:
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": t.input_schema}}
                 for t in (await client.list_tools()).tools]
        for q in questions:
            key = hashlib.sha256(f"{model_id}\n{lang}\n{q['id']}\n{TOOL_DOCS}\n{MAX_STEPS}".encode()).hexdigest()
            if key not in cache:
                if spent > BUDGET:
                    print(f"budget of US$ {BUDGET} reached, stopping {model_id} {lang}", flush=True)
                    return
                cache[key] = {"key": key, **await solve(client, tools, model_id, lang, q)}
                with cache_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
            res = cache[key]
            target = targets[f"{lang}/{q['id']}"]
            row = {k: v for k, v in res.items() if k != "key"}
            row.update(id=q["id"], type=q["type"], found=target in res["searched"], viewed_target=target in res["viewed"],
                       refused=refused(res["reply"]), correct=is_correct(res["reply"], q["answers"]))
            results.setdefault(f"{lang}/{model_id}/{TOOL_DOCS}", {})[q["id"]] = row
            mark = "ok " if row["correct"] else "-- "
            print(f"{mark}{lang} {q['id']} {model_id}: {row['steps']} steps, {len(row['calls'])} calls, "
                  f"viewed {row['viewed']}, ${row['cost']:.4f} | {row['reply'][:80]!r}", flush=True)


async def main():
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    if IDS:
        questions = [q for q in questions if q["id"] in IDS]
    targets = json.loads((HERE / "targets.json").read_text())
    results_file = OUT / "results_agent.json"
    results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else {}
    await asyncio.gather(*(run_model(m, lang, questions, targets, results) for m in MODELS for lang in ("pt", "en")))
    results_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n| lang | model | docs | text | table | chart | total | found | calls/q | cost |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, rows in sorted(results.items()):
        rows = list(rows.values())
        by = {t: [r for r in rows if r["type"] == t] for t in ("text", "table", "chart")}
        cells = " | ".join(f"{sum(r['correct'] for r in v)}/{len(v)}" for v in by.values())
        lang, *model, docs = name.split("/")
        print(f"| {lang} | {'/'.join(model)} | {docs} | {cells} | {sum(r['correct'] for r in rows)}/{len(rows)} | "
              f"{sum(r['found'] for r in rows)} | {sum(len(r['calls']) for r in rows) / len(rows):.1f} | "
              f"${sum(r['cost'] for r in rows):.3f} |")
    print(f"\nspent so far (all cached runs): US$ {spent:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
