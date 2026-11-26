"""Four ways to answer with the same two tools, written by hand, no framework.

  workflow         the code decides every step: search once, look at the pages found, answer
  react            the model decides every step, in a loop (the agent from part 4)
  plan_execute     the model writes a plan first; each step runs as a small loop; a replanner
                   decides whether to answer or add steps
  deep_agent       a to-do list, subagents with their own context, and files as memory, in the
                   style of Claude Code and LangChain's deepagents

Every function takes the task, the language, the report tools and a Run that adds up tokens and
cost, and returns the final answer as text. STYLE picks short answers (the 33 questions) or a
report (the long tasks).
"""
import json

from common import Run, chat, parse_args, tool_spec

SYSTEM = {
    "short": {
        "pt": "Você responde perguntas sobre o Relatório de Política Monetária do Banco Central (setembro de 2026). "
              "Responda só com o que encontrar no relatório, em uma frase curta, com o número ou o nome pedido. "
              "Se não encontrar, diga que não encontrou.",
        "en": "You answer questions about the Banco Central do Brasil's Monetary Policy Report (September 2026). "
              "Answer only with what you find in the report, in one short sentence, with the number or the name "
              "asked for. If you cannot find it, say you could not find it.",
    },
    "report": {
        "pt": "Você escreve análises sobre o Relatório de Política Monetária do Banco Central (setembro de 2026). "
              "Use só o que encontrar no relatório, com os números exatos e a página de cada um. "
              "Se algo pedido não estiver no relatório, diga que não encontrou.",
        "en": "You write analyses of the Banco Central do Brasil's Monetary Policy Report (September 2026). "
              "Use only what you find in the report, with the exact numbers and the page of each. "
              "If something asked for is not in the report, say you could not find it.",
    },
}
LAST_STEP = {
    "pt": "Você chegou ao limite de passos. Responda agora com o que já encontrou.",
    "en": "You have reached the step limit. Answer now with what you have found.",
}


async def loop(run: Run, agent: str, messages: list, specs: list, execute, max_steps: int, lang: str) -> str:
    """The ReAct loop: call the model; if it asks for tools, run them and call it again; stop when
    it answers without a tool, or force an answer at max_steps. execute(name, args) returns the
    text for the tool message and any images, which go in a user message right after (a tool
    message carries only text in this API)."""
    for step in range(max_steps + 1):
        last = step == max_steps
        if last:
            messages.append({"role": "user", "content": LAST_STEP[lang]})
        msg = await chat(run, messages, None if last else specs)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content") or ""
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        images = []
        for tc in tool_calls:
            text, pics = await execute(tc["function"]["name"], parse_args(tc))
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
            images += pics
        if images:
            messages.append({"role": "user", "content": [{"type": "text", "text": "Requested page images:"}] + images})
    return ""


# --- 1. workflow ------------------------------------------------------------------------------

async def workflow(task: str, lang: str, tools, run: Run, style: str = "short") -> str:
    """Fixed steps, as in part 3: one search with the task itself, then the pages of the 3 chunks
    as images, then one answer. No decision is left to the model."""
    found, _ = await tools.call(run, "workflow", "search_report", {"query": task})
    pages = sorted({int(line[6:-1]) for line in found.splitlines() if line.startswith("[page ")})
    images = []
    for p in pages:
        _, pics = await tools.call(run, "workflow", "view_page", {"page": p})
        images += pics
    msg = await chat(run, [{"role": "system", "content": SYSTEM[style][lang]},
                           {"role": "user", "content": images + [{"type": "text", "text": task}]}])
    return msg.get("content") or ""


# --- 2. react ---------------------------------------------------------------------------------

async def react(task: str, lang: str, tools, run: Run, style: str = "short", max_steps: int = 6) -> str:
    async def execute(name, args):
        return await tools.call(run, "react", name, args)
    messages = [{"role": "system", "content": SYSTEM[style][lang]}, {"role": "user", "content": task}]
    return await loop(run, "react", messages, tools.specs, execute, max_steps, lang)


# --- 3. plan and execute ----------------------------------------------------------------------

PLANNER = {
    "pt": "Antes de consultar o relatório, escreva um plano para a tarefa abaixo. O executor terá duas "
          "ferramentas: search_report (busca trechos do texto, com a página) e view_page (mostra a página "
          "como imagem; os valores de gráficos só aparecem na imagem). Cada passo deve ser uma consulta "
          "independente que o executor consiga fazer sozinho. Use o menor número de passos possível, no "
          "máximo {max_plan}. Responda em JSON: {\"steps\": [\"...\"]}",
    "en": "Before looking at the report, write a plan for the task below. The executor will have two tools: "
          "search_report (finds text chunks, with their page) and view_page (shows a page as an image; chart "
          "values are only in the image). Each step must be a self-contained lookup the executor can do on its "
          "own. Use as few steps as possible, at most {max_plan}. Reply in JSON: {\"steps\": [\"...\"]}",
}
EXECUTOR = {
    "pt": "Você executa um passo de um plano sobre o Relatório de Política Monetária do Banco Central "
          "(setembro de 2026). Use as ferramentas e responda com o que encontrou para este passo: os números "
          "exatos e a página de cada um. Se não encontrar, diga que não encontrou.",
    "en": "You carry out one step of a plan about the Banco Central do Brasil's Monetary Policy Report "
          "(September 2026). Use the tools and reply with what you found for this step: the exact numbers "
          "and the page of each. If you cannot find it, say so.",
}
REPLANNER = {
    "pt": "Abaixo estão a tarefa, o plano e o que cada passo encontrou. Se já dá para responder, responda "
          "em JSON {\"answer\": \"...\"}. Se falta algo, responda {\"steps\": [\"...\"]} só com os passos "
          "que faltam (no máximo {max_replan}).",
    "en": "Below are the task, the plan and what each step found. If you can answer now, reply in JSON "
          "{\"answer\": \"...\"}. If something is missing, reply {\"steps\": [\"...\"]} with only the "
          "missing steps (at most {max_replan}).",
}


def _json(msg: dict) -> dict:
    text = (msg.get("content") or "").strip()
    text = text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def plan_execute(task: str, lang: str, tools, run: Run, style: str = "short",
                       step_limit: int = 4, replans: int = 1, max_plan: int = 5,
                       max_replan: int = 3) -> str:
    """Plan, run each step as a small ReAct loop with its own context, then let a replanner answer
    or add steps (at most `replans` times). Only the text each step returns goes back to the
    planner: the page images stay inside the step that looked at them."""
    system = SYSTEM[style][lang]
    plan = _json(await chat(run, [{"role": "system", "content": system},
                                  {"role": "user", "content": PLANNER[lang].replace("{max_plan}", str(max_plan)) + f"\n\n{task}"}],
                            json_mode=True)).get("steps") or [task]
    plan = [str(s) for s in plan][:max_plan]
    done = []  # (step, result)

    async def execute(name, args):
        return await tools.call(run, "plan_execute", name, args)

    for round_ in range(replans + 1):
        for step in plan:
            context = "\n".join(f"- {s}: {r}" for s, r in done)
            prompt = f"Task: {task}\n\nPrevious steps:\n{context or '(none)'}\n\nThis step: {step}"
            messages = [{"role": "system", "content": EXECUTOR[lang]}, {"role": "user", "content": prompt}]
            done.append((step, await loop(run, "plan_execute", messages, tools.specs, execute, step_limit, lang)))
        notes = "\n\n".join(f"Step: {s}\nFound: {r}" for s, r in done)
        final = round_ == replans
        ask = (f"{task}\n\n{notes}\n\n" + (LAST_STEP[lang] if final else REPLANNER[lang].replace("{max_replan}", str(max_replan))))
        msg = await chat(run, [{"role": "system", "content": system}, {"role": "user", "content": ask}],
                         json_mode=not final)
        if final:
            return msg.get("content") or ""
        decision = _json(msg)
        if "answer" in decision or not decision.get("steps"):
            return str(decision.get("answer") or msg.get("content") or "")
        plan = [str(s) for s in decision["steps"]][:max_replan]
    return ""


# --- 4. deep agent ----------------------------------------------------------------------------

DEEP = {
    "pt": """Você é um agente de pesquisa sobre o Relatório de Política Monetária do Banco Central (setembro de 2026).

Como trabalhar:
1. Para tarefas com mais de um passo, comece com write_todos: uma lista curta de tarefas. Atualize os status (pending, in_progress, completed) conforme avança.
2. Delegue cada pesquisa a um subagente com task. O subagente tem as ferramentas do relatório e um contexto próprio; ele devolve só um resumo, sem encher o seu contexto com páginas inteiras. Descreva a pesquisa de forma completa: o subagente não vê esta conversa.
3. Guarde o que for encontrado em arquivos com write_file e releia com read_file antes de responder.
4. Para uma consulta única e simples, você pode usar search_report e view_page direto.
5. Quando terminar, responda sem chamar ferramentas.""",
    "en": """You are a research agent for the Banco Central do Brasil's Monetary Policy Report (September 2026).

How to work:
1. For tasks with more than one step, start with write_todos: a short to-do list. Update the statuses (pending, in_progress, completed) as you go.
2. Delegate each piece of research to a subagent with task. The subagent has the report tools and its own context; it returns only a summary, without filling your context with whole pages. Describe the research completely: the subagent does not see this conversation.
3. Save what you find in files with write_file and read them back with read_file before answering.
4. For a single, simple lookup you may use search_report and view_page directly.
5. When you are done, answer without calling any tool.""",
}
SUBAGENT = {
    "pt": "Você é um subagente de pesquisa. Use search_report e view_page para cumprir a pesquisa pedida no "
          "Relatório de Política Monetária do Banco Central (setembro de 2026) e responda com um resumo curto: "
          "os números exatos e a página de cada um. Se não encontrar, diga que não encontrou.",
    "en": "You are a research subagent. Use search_report and view_page to carry out the requested research in "
          "the Banco Central do Brasil's Monetary Policy Report (September 2026) and reply with a short summary: "
          "the exact numbers and the page of each. If you cannot find it, say so.",
}
DEEP_TOOLS = [
    tool_spec("write_todos", "Write or replace the to-do list for this task.",
              {"todos": {"type": "array", "items": {"type": "object", "properties": {
                  "content": {"type": "string"},
                  "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
                  "required": ["content", "status"]}}}, ["todos"]),
    tool_spec("task", "Start a research subagent with its own context and the report tools. It returns a summary.",
              {"description": {"type": "string", "description": "The full research request."}}, ["description"]),
    tool_spec("write_file", "Write a text file (overwrites).",
              {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    tool_spec("read_file", "Read a text file.", {"path": {"type": "string"}}, ["path"]),
    tool_spec("ls", "List the files.", {}, []),
]


async def deep_agent(task: str, lang: str, tools, run: Run, style: str = "short",
                     max_steps: int = 20, sub_steps: int = 6) -> tuple[str, dict]:
    """Returns the answer and what the agent left behind (to-do list, files, subagent count)."""
    state = {"todos": [], "files": {}, "subagents": 0}

    async def sub_execute(name, args):
        return await tools.call(run, "subagent", name, args)

    async def execute(name, args):
        if name in ("search_report", "view_page"):
            return await tools.call(run, "deep_agent", name, args)
        run.calls.append({"agent": "deep_agent", "tool": name, "args": args if name != "write_file" else
                          {"path": args.get("path"), "chars": len(args.get("content") or "")}, "error": False})
        if name == "write_todos":
            state["todos"] = args.get("todos") or []
            return "To-do list updated:\n" + "\n".join(f"[{t.get('status')}] {t.get('content')}" for t in state["todos"]), []
        if name == "write_file":
            state["files"][args.get("path", "notes.md")] = args.get("content") or ""
            return f"Wrote {args.get('path')}", []
        if name == "read_file":
            return state["files"].get(args.get("path"), f"No file {args.get('path')}"), []
        if name == "ls":
            return "\n".join(state["files"]) or "(no files)", []
        if name == "task":
            state["subagents"] += 1
            messages = [{"role": "system", "content": SUBAGENT[lang]},
                        {"role": "user", "content": args.get("description") or ""}]
            return await loop(run, "subagent", messages, tools.specs, sub_execute, sub_steps, lang) or "(no answer)", []
        return f"Unknown tool {name}", []

    messages = [{"role": "system", "content": DEEP[lang] + "\n\n" + SYSTEM[style][lang]},
                {"role": "user", "content": task}]
    answer = await loop(run, "deep_agent", messages, DEEP_TOOLS + tools.specs, execute, max_steps, lang)
    return answer, state
