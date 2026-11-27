"""What every agent type shares: the model call, the MCP tools and the bookkeeping.

The tools are the ones from the RAG series, part 4 (../rag-pdf-bcb/mcp_server.py): search_report finds
chunks of the Banco Central's Monetary Policy Report, view_page returns a page as an image.

The key is read from OPENROUTER_API_KEY, or from a dotenv file given in ENV_FILE (variables
OPENROUTER_API_KEY, OPEN_ROUTER or OPENROUTER_CHAVE). It is never printed or saved.
"""
import asyncio
import json
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from mcp import Client, StdioServerParameters

HERE = Path(__file__).parent
RAG = HERE.parent / "rag-pdf-bcb"
OUT = HERE / "results"  # every run is cached here, with its answer, tool calls, tokens and cost
OUT.mkdir(exist_ok=True)
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("MODEL", "openai/gpt-6.1-sol")


def api_key() -> str:
    names = ("OPENROUTER_API_KEY", "OPEN_ROUTER", "OPENROUTER_CHAVE")
    for name in names:
        if key := os.environ.get(name):
            return key
    if env_file := os.environ.get("ENV_FILE"):
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() in names and value.strip():
                return value.strip().strip('"').strip("'")
    raise SystemExit("Set OPENROUTER_API_KEY, or ENV_FILE pointing to a file that has it.")


KEY = api_key()


class Run:
    """Tokens, cost, model calls and tool calls of one task, across every agent and subagent."""

    def __init__(self):
        self.prompt_tokens = self.completion_tokens = self.cached_tokens = self.llm_calls = 0
        self.cost = 0.0
        self.calls = []        # every tool call: {"agent", "tool", "args", "error"}
        self.searched = set()  # pages that search_report returned
        self.viewed = set()    # pages that view_page showed
        self.start = time.perf_counter()

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "tool_calls": len(self.calls), "calls": self.calls,
                "searched": sorted(self.searched), "viewed": sorted(self.viewed),
                "prompt_tokens": self.prompt_tokens, "cached_tokens": self.cached_tokens,
                "completion_tokens": self.completion_tokens,
                "cost": round(self.cost, 6), "seconds": round(time.perf_counter() - self.start, 1)}


def _post(body: dict) -> dict:
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
    raise RuntimeError(f"{body['model']}: {error}")


async def chat(run: Run, messages: list, tools: list | None = None, json_mode: bool = False,
               max_tokens: int = 4000, model: str = MODEL) -> dict:
    """One model call. Returns the assistant message and adds its tokens and cost to the run."""
    body = {"model": model,"temperature": 0, "max_tokens": max_tokens, "messages": messages,
            "usage": {"include": True}}
    if tools:
        body["tools"] = tools
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = await asyncio.to_thread(_post, body)
    usage = data.get("usage", {})
    run.prompt_tokens += usage.get("prompt_tokens") or 0
    run.completion_tokens += usage.get("completion_tokens") or 0
    run.cached_tokens += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    run.cost += usage.get("cost") or 0
    run.llm_calls += 1
    return data["choices"][0]["message"]


def tool_spec(name: str, description: str, properties: dict, required: list) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


class ReportTools:
    """The MCP server's two tools, called over stdio, as in part 4."""

    def __init__(self, client: Client, specs: list):
        self.client, self.specs = client, specs

    async def call(self, run: Run, agent: str, name: str, args: dict) -> tuple[str, list]:
        """Returns the text for the tool message and the images to send after it."""
        result = await self.client.call_tool(name, args)
        run.calls.append({"agent": agent, "tool": name, "args": args, "error": result.is_error})
        texts = [c.text for c in result.content if c.type == "text"]
        pics = [c for c in result.content if c.type == "image"]
        if name == "search_report" and not result.is_error:
            run.searched |= {int(p) for t in texts for p in re.findall(r"\[page (\d+)\]", t)}
        if name == "view_page" and not result.is_error and isinstance(args.get("page"), int):
            run.viewed.add(args["page"])
        images = [{"type": "image_url", "image_url": {"url": f"data:{p.mime_type};base64,{p.data}"}} for p in pics]
        return "\n".join(texts) or (f"(page {args.get('page')} follows as an image)" if pics else ""), images


@asynccontextmanager
async def report_tools(lang: str):
    params = StdioServerParameters(command=sys.executable, args=[str(RAG / "mcp_server.py")],
                                   env={**os.environ, "DOC_LANG": lang, "TOOL_DOCS": "detailed"})
    async with Client(params) as client:
        specs = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": t.input_schema}}
                 for t in (await client.list_tools()).tools]
        yield ReportTools(client, specs)


def parse_args(tc: dict) -> dict:
    try:
        return json.loads(tc["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return {}
