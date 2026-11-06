# MewCode Lite

[中文版 → README.md](README.md)

A minimal terminal coding agent **written from scratch, with no agent
framework**. Give it a task in the terminal and it reads files, writes
files, searches code and runs commands, step by step, until the task is
done.

> Status: learning project, v0.1. The goal is to implement the core
> mechanisms of an agent by hand and be able to explain them — not to
> ship a full product. What is implemented is listed below; what is not
> is stated honestly under "Limitations & roadmap".

## Implemented

| Mechanism | Where | Notes |
|---|---|---|
| ReAct-style main loop | `mewcode/agent.py` | model reply → execute tool_calls → feed results back → ask again, until a final text answer |
| Tool layer (4 tools) | `mewcode/tools.py` | `read_file` / `write_file` / `grep_search` / `run_shell`, OpenAI-style function schemas |
| Streaming tool_calls assembly | `mewcode/llm.py` `assemble_stream` | in a streamed response one tool call arrives as many fragments (arguments are string shards grouped by `index`); this stitches them back together |
| Structured error feedback | `mewcode/agent.py` `_execute` | tool failures (bad JSON arguments, missing file, blocked command) never crash the run; they are returned as `{"ok": false, "error": ...}` so the model can fix its arguments and retry |
| Max-steps limit | `Agent(max_steps=...)` | stops a model that would otherwise spin forever |
| Repeated-call detection | `Agent(repeat_limit=...)` | same tool + same arguments beyond a limit = stuck loop; the run stops and says why |
| Safety boundaries | `mewcode/tools.py` | file tools are confined to the workspace (`../` escapes rejected); shell has a destructive-command blocklist; non read-only commands require human confirmation |
| LLM client abstraction | `mewcode/llm.py` `LLMClient` | the loop depends only on a protocol; includes an OpenAI-compatible client (standard library only) and a scripted FakeLLM for offline use |

## Quick start

Python 3.10+. The runtime itself has **zero third-party dependencies**
(standard library only); pytest is needed only for the tests.

```bash
git clone https://github.com/yang-cloud-0/mewcode.git
cd mewcode
python3 -m venv .venv && source .venv/bin/activate
pip install pytest        # tests only
```

### 1. Offline demo (no API key, no network)

```bash
python -m mewcode --demo
```

The demo drives the **real** agent loop and tools with a scripted fake
model: it creates `hello.py` in a temp directory and actually runs it.

### 2. Tests

```bash
pytest -q
# 19 passed
```

Coverage: tool read/write and path-escape rejection, grep, shell safety
policy, streaming tool_calls fragment assembly, error-feedback recovery
in the main loop, max-steps and repeated-call detection.

### 3. Real model

Works with any OpenAI-format chat-completions endpoint
(OpenAI / DeepSeek / Qwen / ...):

```bash
export MEWCODE_API_KEY=your-key
export MEWCODE_MODEL=gpt-4o-mini
export MEWCODE_BASE_URL=https://api.openai.com/v1
python -m mewcode "Create a bubble-sort script in this directory and run it to verify"
```

Non read-only shell commands ask for confirmation (y/N) on the terminal
before they run.

## How it works (one minute)

```
user task
  │
  ▼
┌───────────────┐   tool_calls    ┌───────────────┐
│  agent loop   │ ──────────────▶ │  tool layer   │
│ (agent.py)    │ ◀────────────── │ (tools.py)    │
└───────────────┘  {"ok": ...}    └───────────────┘
  │  ▲
  │  │ streamed fragments are assembled in llm.py
  ▼  │
  model API (OpenAI-compatible)
```

In one sentence: **the model decides what to do next, the loop executes
it and feeds the truth back, and the tool layer does the work while
holding the safety boundary.**

## Limitations & roadmap

Honestly, the following are **not** implemented yet:

- [ ] Other provider protocols (e.g. Anthropic) — OpenAI-compatible only
- [ ] Context compression / token budgeting (long tasks can hit the context limit)
- [ ] A todo/planning tool and a plan mode for multi-file tasks
- [ ] An evaluation set (fixed tasks measuring pass rate / average steps)
- [ ] Session persistence and resume

Next steps: a small evaluation set first (fix-a-bug / add-a-function /
add-a-test tasks), then context compression. A RAG contract-review agent
is planned as a separate second project.

## Layout

```
mewcode/
├── mewcode/
│   ├── agent.py    # main loop: step limit, repeat detection, structured error feedback
│   ├── llm.py      # client protocol, streaming tool_calls assembly, FakeLLM
│   ├── tools.py    # 4 tools + workspace confinement + shell safety policy
│   └── cli.py      # CLI entry point and offline demo
├── tests/          # 19 pytest tests
├── README.md       # Chinese version
└── README.en.md    # this file
```

## License

MIT
