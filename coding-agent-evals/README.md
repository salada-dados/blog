# Coding agents with and without an eval

Code for the Salada de Dados article **"Give Your Coding AI a Way to Know It Got It Right"**
([English](https://saladadedados.com/en/blog/coding-agent-evals) ·
[Português](https://saladadedados.com/pt/blog/coding-agent-evals)).

Claude Code (Opus 5.5) builds a question-answering RAG over the Banco Central do Brasil's Monetary
Policy Report (September 2026), unattended, in an empty folder. Two conditions, two runs each:

- **A**: only the request ([`prompts/A.md`](prompts/A.md)), about 60 minutes.
- **B**: the same request plus 22 questions with their answers ([`kit/dev.json`](kit/dev.json)), a
  script that scores the system on them ([`kit/eval.py`](kit/eval.py)) and the instruction to iterate
  ([`prompts/B.md`](prompts/B.md)), about 2 hours.

Every run is then graded on all 33 questions of the RAG series: the 22 the agent could see and 11 it
never saw ([`holdout.json`](holdout.json)).

| | A1 | A2 | B1 | B2 |
|---|---|---|---|---|
| Development questions (22) | 13 | 14 | 19 | 17 |
| Held-out questions (11) | 4 | 8 | 9 | 6 |
| Time | 29 min | 58 min | 66 min | 82 min |
| Equivalent API cost | US$ 1.31 | US$ 1.43 | US$ 3.41 | US$ 4.51 |

## What's inside

| Path | What it is |
|---|---|
| `split.py` | Draws the 22 development and 11 held-out questions from `../rag-pdf-bcb/questions.json` (seed 2026, same mix of text, table and chart questions in both) |
| `prompts/` | The requests given to Claude Code, in Portuguese, as in the article |
| `kit/` | What condition B gets besides the PDF: `dev.json` and `eval.py` |
| `holdout.json` | The 11 questions no run ever saw |
| `run.py` | Creates an empty run folder, copies the PDF (and the kit for B), runs `claude -p` and records the session |
| `grade.py` | Scores a run on the 33 questions and looks for signs of gaming: changed kit files, questions or answers written in the code, time, cost, turns, eval runs and paths outside the folder |
| `publish.py` | Makes the published copies below: removes the user name and the account details from the session logs, and copies what the agent wrote |
| `manifests/` | One per run: model, Claude Code version, hashes of the files the agent got, start time, duration |
| `logs/` | The four sessions as `stream-json` (`<run>.jsonl`) and the output of `grade.py` (`grade_<run>.txt`) |
| `results/` | Every answer of every run, right or wrong, plus the session numbers |
| `runs/` | What each agent wrote: its code, its `README.md` and, for B, its own eval logs. Without the PDF, the kit, the `.venv` and the generated indexes |

The 33 questions, their answers and the scoring rule come from part 2 of the RAG series, in
[`../rag-pdf-bcb`](../rag-pdf-bcb).

## Running it again

Requires Python 3.11+, Claude Code logged in to an account that can use
Opus 5.5 (Claude Code 2.1.280 or later), and [Ollama](https://ollama.com) on port 11434 with the three
models the request allows:

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:3b

cd ../rag-pdf-bcb && python download_rpm.py && cd ../coding-agent-evals

python split.py            # writes kit/dev.json and holdout.json (already in the repository)
python run.py A 3          # condition A, run 3: creates C:/agent-runs/A3 (change RUNS in run.py)
python grade.py A3         # all 33 questions through the run's answer.py, about 30 to 60 minutes on a CPU
```

`run.py` uses the `claude` on your `PATH`; set `CLAUDE_BIN` to use another binary.

**Read this before running it.** `run.py` starts Claude Code with `--dangerously-skip-permissions`:
the agent runs any command without asking. We ran it on a personal computer, in a folder outside every
repository, and watched the logs; even so, one run killed two Python processes it did not check. Prefer
a virtual machine or a container.

Each run uses your Claude plan (or API credits). The numbers in the table are the cost Claude Code
reports for the session at API prices; the four runs added up to US$ 10.66.

## Limitations

Two runs per condition, one model, one document, one computer without a GPU. Condition B had twice
the time of condition A, because each pass of `eval.py` takes 18 to 40 minutes on a CPU. The article
discusses what these limits mean for the results.
