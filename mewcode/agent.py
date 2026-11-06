"""Agent 主循环 —— MewCode Lite 的心脏。

一个不依赖任何框架的 ReAct 式循环：

    while steps < max_steps:
        reply = llm.chat(messages, TOOL_SCHEMAS)
        如果 reply 没有 tool_calls → 这是最终回答，停止
        对每个 tool_call：
            执行工具
            把结果（或结构化错误）作为 tool 消息追加回对话
        继续循环

三个护栏放在这一层，因为只有主循环能看到完整对话：

  1. 最大步数（max_steps）：模型不能无限打转，
     到达上限就停下并说明原因。
  2. 重复调用检测：用同一工具、同样的参数一遍遍调用，
     是小型 Agent 最经典的翻车方式。相同的调用超过
     `repeat_limit` 次后，循环停下并说明原因，而不是继续烧 token。
  3. 结构化错误反馈：工具出错（参数 JSON 坏了、文件不存在、
     命令被拦截……）不会让运行崩溃，错误会被转成 JSON 工具消息
     回传给模型：{"ok": false, "error": "..."}，
     让模型自己改参数重试。成功结果包成
     {"ok": true, "result": "..."}，模型能可靠地区分两者。
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

    # ---------------------------------------------------------- 单次工具执行
    def _execute(self, name: str, arguments_json: str) -> str:
        """执行一次工具调用。永远返回 JSON 字符串，永不抛异常。"""
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
        except Exception as e:  # 最后一道防线：回传错误，而不是崩溃
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

    @staticmethod
    def _call_key(name: str, arguments_json: str) -> str:
        return hashlib.sha256(f"{name}\n{arguments_json}".encode()).hexdigest()

    # -------------------------------------------------------------- 主循环
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
