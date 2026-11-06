"""The agent loop — the heart of MewCode Lite.

A plain ReAct-style loop, no framework:

    while steps < max_steps:
        reply = llm.chat(messages, TOOL_SCHEMAS)
        if reply has no tool_calls:  -> final answer, stop
        for each tool_call:
            run the tool
            append the result (or a structured error) as a tool message
        continue

Three guardrails live here, because the loop is the only place that
sees the whole conversation:

  1. max_steps — the model cannot loop forever; the run stops and
     reports that it hit the limit.
  2. Repeated-call detection — calling the same tool with the same
     arguments again and again is the classic failure mode of small
     agents. After `repeat_limit` identical calls the loop stops and
     says why, instead of burning tokens.
  3. Structured error feedback — a tool that raises (bad JSON arguments,
     missing file, blocked command, ...) does NOT crash the run. The
     error is sent back to the model as a JSON tool message:
         {"ok": false, "error": "..."}
     so the model can correct its arguments and retry. Successful
     results are wrapped as {"ok": true, "result": "..."} so the model
     can tell the two apart reliably.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .llm import AssistantReply, LLMClient
from .tools import TOOL_SCHEMAS, ToolBox, ToolError

SYSTEM_PROMPT = """You are MewCode, a terminal coding assistant.
You work inside one workspace directory. Use the provided tools to read
and modify files and to run commands. Prefer reading before writing.
When the task is done, answer in plain text without tool calls."""


@dataclass
class AgentResult:
    final_text: str
    steps: int
    stopped_reason: str  # "done" | "max_steps" | "repeated_calls"
    transcript: list[dict] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        toolbox: ToolBox,
        max_steps: int = 20,
        repeat_limit: int = 3,
    ):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.repeat_limit = repeat_limit
        self._tools = {
            "read_file": toolbox.read_file,
            "write_file": toolbox.write_file,
            "grep_search": toolbox.grep_search,
            "run_shell": toolbox.run_shell,
        }

    # ------------------------------------------------------------- one call
    def _execute(self, name: str, arguments_json: str) -> str:
        """Run one tool call; always returns a JSON string, never raises."""
        fn = self._tools.get(name)
        if fn is None:
            return json.dumps({"ok": False, "error": f"unknown tool: {name}"})
        try:
            args = json.loads(arguments_json or "{}")
            if not isinstance(args, dict):
                raise ToolError("tool arguments must be a JSON object")
            result = fn(**args)
            return json.dumps({"ok": True, "result": result})
        except ToolError as e:
            return json.dumps({"ok": False, "error": str(e)})
        except json.JSONDecodeError as e:
            return json.dumps({"ok": False, "error": f"invalid JSON arguments: {e}"})
        except TypeError as e:
            return json.dumps({"ok": False, "error": f"bad arguments for {name}: {e}"})
        except Exception as e:  # last-resort guard: feedback, not a crash
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

    @staticmethod
    def _call_key(name: str, arguments_json: str) -> str:
        return hashlib.sha256(f"{name}\n{arguments_json}".encode()).hexdigest()

    # -------------------------------------------------------------- main loop
    def run(self, user_task: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_task},
        ]
        seen_calls: dict[str, int] = {}
        steps = 0
        while steps < self.max_steps:
            steps += 1
            reply: AssistantReply = self.llm.chat(messages, TOOL_SCHEMAS)
            messages.append({
                "role": "assistant",
                "content": reply.content or "",
                "tool_calls": [
                    {
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in reply.tool_calls
                ] or None,
            })
            if not reply.tool_calls:
                return AgentResult(reply.content, steps, "done", messages)

            for tc in reply.tool_calls:
                key = self._call_key(tc.name, tc.arguments)
                seen_calls[key] = seen_calls.get(key, 0) + 1
                if seen_calls[key] > self.repeat_limit:
                    return AgentResult(
                        f"Stopped: tool {tc.name} was called with identical "
                        f"arguments {seen_calls[key]} times in a row — the "
                        "model appears to be stuck in a loop.",
                        steps, "repeated_calls", messages,
                    )
                result_json = self._execute(tc.name, tc.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                })
        return AgentResult(
            "Stopped: reached the maximum number of steps "
            f"({self.max_steps}) before the task was finished.",
            steps, "max_steps", messages,
        )
