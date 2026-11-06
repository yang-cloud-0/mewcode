"""MewCode Lite 的模型客户端层。

设计：Agent 主循环只和 `LLMClient` 协议打交道
（`chat(messages, tools) -> AssistantReply`），不关心背后是哪家模型。
两个实现：

  - OpenAICompatibleClient：对接任何 OpenAI 兼容的 chat-completions
    接口（OpenAI、DeepSeek、通义、本地部署服务等），只用 Python
    标准库。它请求的是流式（streaming）响应，再用下面的
    `assemble_stream` 把分片拼装回完整回复。
  - FakeLLM：按脚本回放的假模型，供 demo 和测试使用，
    让整个 Agent 可以离线、无需 API key 运行。

为什么流式对工具调用是关键：stream=true 时模型不会一次性返回
完整的 tool_call，而是返回很多个 SSE 小块，一个工具调用被拆散在
多个块里——函数名可能只出现一次，而 `arguments` 是按 `index`
分组、一点点到达的字符串碎片。`assemble_stream` 做的就是把这些
碎片拼回完整的工具调用。手写 Agent 最常见的 bug 就出在这里，
所以把它做成纯函数并配了专门的测试。
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，保持模型产出的原样

    def parsed_arguments(self) -> dict:
        """解析参数 JSON。格式不合法时抛 ValueError，
        由 Agent 主循环转成结构化错误反馈给模型。"""
        args = json.loads(self.arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
        return args


@dataclass
class AssistantReply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> AssistantReply:
        ...


# --------------------------------------------------------------------------
# 流式拼装
# --------------------------------------------------------------------------

def assemble_stream(chunks: Iterable[dict]) -> AssistantReply:
    """把一次流式 chat-completion 响应拼装成完整的 AssistantReply。

    `chunks` 是每条 SSE `data:` 行解码后的 JSON 对象（不含最后的
    `[DONE]`）。工具调用碎片按 `index` 分组拼接，文本碎片按顺序连接。
    """
    content_parts: list[str] = []
    by_index: dict[int, dict] = {}
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = by_index.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]
    tool_calls = [
        ToolCall(id=slot["id"] or f"call_{idx}", name=slot["name"], arguments=slot["arguments"])
        for idx, slot in sorted(by_index.items())
        if slot["name"]
    ]
    return AssistantReply(content="".join(content_parts), tool_calls=tool_calls)


def iter_sse_json(lines: Iterable[str]):
    """从原始 SSE 行（`data: {...}`）逐条解出 JSON 对象。"""
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        yield json.loads(payload)


# --------------------------------------------------------------------------
# 真实客户端（OpenAI 兼容，只用标准库）
# --------------------------------------------------------------------------

class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict]) -> AssistantReply:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            lines = (raw.decode("utf-8", errors="replace") for raw in resp)
            return assemble_stream(iter_sse_json(lines))


# --------------------------------------------------------------------------
# 脚本化假客户端（供 demo 与测试使用，完全离线）
# --------------------------------------------------------------------------

class FakeLLM:
    """按固定脚本回放回复，每次 chat() 调用消耗一条。

    每条脚本可以是一个 AssistantReply，也可以是一个接收当前
    对话、返回 AssistantReply 的可调用对象——用可调用形式，测试就
    能模拟“模型看到工具错误反馈后，下一轮自己修正参数”的行为。
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> AssistantReply:
        self.calls.append(messages)
        if not self.script:
            return AssistantReply(content="(fake model: script exhausted)")
        item = self.script.pop(0)
        return item(messages) if callable(item) else item


def tool_call_reply(name: str, arguments: dict, call_id: Optional[str] = None) -> AssistantReply:
    """小工具：在 demo 和测试里快速构造一个只含一次工具调用的回复。"""
    return AssistantReply(tool_calls=[
        ToolCall(id=call_id or f"call_{name}", name=name, arguments=json.dumps(arguments))
    ])
