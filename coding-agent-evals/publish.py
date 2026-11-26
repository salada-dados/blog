"""Prepare the published copy of the four runs: session logs without personal data, and the agents' code.

Usage: python publish.py

- logs-raw/<id>.jsonl (not in the repository) -> logs/<id>.jsonl: the user's folder name becomes USER,
  the startup event keeps only the run settings (it also lists the account's connectors and skills),
  and the plan's rate-limit events are dropped.
- C:/agent-runs/<id> -> runs/<id>: what the agent wrote (code, README and its own eval logs), without
  the PDF, the kit files, the .venv and the generated indexes.
"""
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).parent
RUNS = Path("C:/agent-runs")
USER = re.compile(r'(Users(?:\\\\|\\|/))[^\\/"]+', re.I)  # C:\Users\name, C:/Users/name, JSON-escaped too
OWNER = re.compile(r"([d-][rwx-]{9} +\d+ )[^ ]+( +\d+ )")  # file owner in `ls -l` output
INIT_KEEP =("type", "subtype", "cwd", "session_id", "model", "permissionMode", "claude_code_version", "tools")
KEEP = {".py", ".md", ".log", ".txt"}
SKIP = {"eval.py", "dev.json"}


def clean_log(src: Path, dst: Path):
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        e = json.loads(line)
        if e.get("type") == "rate_limit_event":
            continue
        if e.get("subtype") == "init":
            e = {k: e[k] for k in INIT_KEEP if k in e}
        text = USER.sub(r"\1USER", json.dumps(e, ensure_ascii=False))
        out.append(OWNER.sub(r"\1USER\2", text))
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")


def copy_run(run_id: str):
    src, dst = RUNS / run_id, HERE / "runs" / run_id
    if dst.exists():
        shutil.rmtree(dst)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if not p.is_file() or ".venv" in rel.parts or "__pycache__" in rel.parts:
            continue
        if p.suffix in KEEP and p.name not in SKIP and p.stat().st_size < 50_000:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            (dst / rel).write_text(USER.sub(r"\1USER", p.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    for raw in sorted((HERE / "logs-raw").glob("*.jsonl")):
        clean_log(raw, HERE / "logs" / raw.name)
        copy_run(raw.stem)
    for p in [*(HERE / "results").glob("*.json"), *(HERE / "logs").glob("grade_*.txt")]:
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:  # grade_*.txt captured from a Windows console
            text = raw.decode("cp1252")
        p.write_text(USER.sub(r"\1USER", text), encoding="utf-8")
