"""LLM client layer for MewCode Lite.

Design: the agent loop only talks to the `LLMClient` protocol
(`chat(messages, tools) -> AssistantReply`). Two implementations:

  - OpenAICompatibleClient: talks to any OpenAI-compatible chat-completions
    endpoint (OpenAI, DeepSeek, Qwen, local servers, ...) using only the
    Python standard library. It requests *streaming* responses and
    reassembles them with `assemble_stream` (see below).
  - FakeLLM: a scripted stand-in used by the demo and the test-suite, so
    the whole agent can run offline with no API key and no network.

Why streaming matters for tool calls: with stream=true the model does not
return one finished tool_call. It returns many small SSE chunks, and a
single tool call is split across chunks — the function name may arrive
once, while `arguments` arrives as many string fragments keyed by an
`index`. `assemble_stream` is the piece that stitches those fragments
back into complete tool calls. Getting this wrong is the most common bug
in hand-written agents, which is why it is a pure function with tests.
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
    arguments: str  # raw JSON string, exactly as the model produced it

    def parsed_arguments(self) -> dict:
        """Parse the arguments JSON. Raises ValueError on malformed JSON —
        the agent loop turns that into structured error feedback."""
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
# Streaming assembly
# --------------------------------------------------------------------------

def assemble_stream(chunks: Iterable[dict]) -> AssistantReply:
    """Reassemble one streamed chat-completion into an AssistantReply.

    `chunks` are the decoded JSON objects of each SSE `data:` line
    (excluding the final `[DONE]`). Tool-call fragments are grouped by
    their `index`; content fragments are concatenated in order.
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
    """Yield decoded JSON objects from raw SSE lines (`data: {...}`)."""
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        yield json.loads(payload)


# --------------------------------------------------------------------------
# Real client (OpenAI-compatible, standard library only)
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
# Scripted fake client (demo + tests, fully offline)
# --------------------------------------------------------------------------

class FakeLLM:
    """Plays back a fixed script of replies, one per chat() call.

    Each script item is either an AssistantReply or a callable that
    receives the current messages and returns an AssistantReply — the
    callable form lets a test simulate a model that *reacts* to tool
    error feedback by fixing its arguments on the next turn.
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
    """Helper for building a one-tool-call reply in demos and tests."""
    return AssistantReply(tool_calls=[
        ToolCall(id=call_id or f"call_{name}", name=name, arguments=json.dumps(arguments))
    ])
