"""Command-line entry point.

Usage:
    python -m mewcode "your task"        # real model (needs an API key)
    python -m mewcode --demo             # offline scripted demo, no key

Real mode reads its configuration from the environment:
    MEWCODE_API_KEY   (or OPENAI_API_KEY)
    MEWCODE_MODEL     (default: gpt-4o-mini)
    MEWCODE_BASE_URL  (default: https://api.openai.com/v1)
    MEWCODE_WORKSPACE (default: current directory)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from .agent import Agent
from .llm import FakeLLM, OpenAICompatibleClient, tool_call_reply, AssistantReply
from .tools import ToolBox, Workspace


def _tty_confirm(command: str) -> bool:
    answer = input(f"\n[confirm] run shell command? {command}\n[y/N] ").strip().lower()
    return answer in ("y", "yes")


def run_demo() -> int:
    """Offline demo: a scripted model writes hello.py, runs it, and the
    real agent loop + real tools do all the work. No network, no key."""
    tmp = Path(tempfile.mkdtemp(prefix="mewcode_demo_"))
    print(f"demo workspace: {tmp}")
    script = [
        tool_call_reply("write_file", {
            "path": "hello.py",
            "content": "print('hello from MewCode demo')\n",
        }),
        tool_call_reply("run_shell", {"command": "python3 hello.py"}),
        AssistantReply(content="Done: created hello.py and ran it successfully."),
    ]
    toolbox = ToolBox(Workspace(tmp), confirm=lambda cmd: True)
    agent = Agent(FakeLLM(script), toolbox)
    result = agent.run("Create hello.py that prints a greeting, then run it.")
    print(f"steps={result.steps} stopped_reason={result.stopped_reason}")
    print(f"final: {result.final_text}")
    produced = tmp / "hello.py"
    assert produced.is_file() and "hello from MewCode" in produced.read_text()
    print("demo OK: hello.py exists in the demo workspace")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mewcode", description="MewCode Lite — minimal terminal coding agent")
    parser.add_argument("task", nargs="?", help="the coding task to perform")
    parser.add_argument("--demo", action="store_true", help="run the offline scripted demo (no API key needed)")
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args(argv)

    if args.demo:
        return run_demo()
    if not args.task:
        parser.print_help()
        return 2

    api_key = os.environ.get("MEWCODE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "No API key found. Set MEWCODE_API_KEY (or OPENAI_API_KEY), "
            "or run with --demo to see the offline demo.",
            file=sys.stderr,
        )
        return 2

    client = OpenAICompatibleClient(
        api_key=api_key,
        model=os.environ.get("MEWCODE_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("MEWCODE_BASE_URL", "https://api.openai.com/v1"),
    )
    workspace = Workspace(os.environ.get("MEWCODE_WORKSPACE", "."))
    agent = Agent(client, ToolBox(workspace, confirm=_tty_confirm), max_steps=args.max_steps)
    result = agent.run(args.task)
    print(result.final_text)
    print(f"[steps={result.steps}, stopped_reason={result.stopped_reason}]", file=sys.stderr)
    return 0 if result.stopped_reason == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
