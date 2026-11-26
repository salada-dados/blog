"""Prepare a clean folder and let Claude Code build the RAG in it, unattended.

Usage: python run.py A 1      (condition A or B, repetition number)

The folder lives outside every repository (RUNS), so the agent cannot stumble on the full
question set. Condition B also gets kit/dev.json and kit/eval.py. The whole session is saved
as stream-json in logs/, and the hashes of the files the agent must not touch in manifests/.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RUNS = Path("C:/agent-runs")
PDF = HERE.parent / "rag-pdf-bcb" / "data" / "rpm202609p.pdf"
MODEL = "claude-opus-5-5"
# Opus 5.5 needs Claude Code 2.1.280+; CLAUDE_BIN points to a newer binary than the one on PATH
CLAUDE = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
# The prompt says about 60 minutes (A) or 2 hours (B: each eval.py pass takes ~40 min on a CPU);
# the process is killed 15 minutes after that.
TIME_LIMIT = {"A": 75 * 60, "B": 135 * 60}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(cond: str, n: str):
    run_id = f"{cond}{n}"
    folder = RUNS / run_id
    if folder.exists():
        sys.exit(f"{folder} already exists; every run starts from an empty folder")
    folder.mkdir(parents=True)
    shutil.copy(PDF, folder)
    if cond == "B":
        for name in ("dev.json", "eval.py"):
            shutil.copy(HERE / "kit" / name, folder)

    (HERE / "manifests").mkdir(exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)
    version = subprocess.run([CLAUDE, "--version"], capture_output=True, text=True).stdout.strip()
    manifest = {"run": run_id, "model": MODEL, "claude_code": version, "folder": str(folder),
                "files": {p.name: sha(p) for p in folder.iterdir()}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    prompt = (HERE / "prompts" / f"{cond}.md").read_text(encoding="utf-8")
    cmd = [CLAUDE, "-p", prompt, "--model", MODEL,
           "--output-format", "stream-json", "--verbose",
           "--dangerously-skip-permissions",
           "--setting-sources", "project"]  # no user settings, hooks or plugins
    log = HERE / "logs" / f"{run_id}.jsonl"
    start = time.time()
    with log.open("w", encoding="utf-8") as f:
        try:
            subprocess.run(cmd, cwd=folder, stdout=f, stderr=subprocess.STDOUT, timeout=TIME_LIMIT[cond])
            manifest["timed_out"] = False
        except subprocess.TimeoutExpired:
            manifest["timed_out"] = True
    manifest["seconds"] = round(time.time() - start)
    (HERE / "manifests" / f"{run_id}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{run_id}: {manifest['seconds']}s, timed out: {manifest['timed_out']}, log {log}")


if __name__ == "__main__":
    cond, n = sys.argv[1].upper(), sys.argv[2]
    assert cond in ("A", "B")
    main(cond, n)
