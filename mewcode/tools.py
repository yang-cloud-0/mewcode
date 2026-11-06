"""MewCode Lite 的工具层。

模型可以调用的 4 个工具：
  - read_file   ：读取工作区内的 UTF-8 文本文件
  - write_file  ：写入/覆盖工作区内的 UTF-8 文本文件
  - grep_search ：在工作区内的文本文件里做正则搜索
  - run_shell   ：在工作区里执行 shell 命令（有安全防护，见下文）

安全规则在这里强制执行，而不是指望模型自觉：
  1. 文件工具被限制在工作区根目录内，任何逃出根目录的路径
     （例如 "../secret.txt"）都会被拒绝。
  2. run_shell 直接拒绝明显有破坏性的命令。
  3. run_shell 在执行“只读白名单”之外的命令前，必须取得人工确认。
     测试和 demo 里注入确认回调；交互式 CLI 里在终端弹窗询问。

每个工具成功时返回普通字符串；失败时抛出 ToolError，
由 Agent 主循环转成结构化 JSON 错误回传给模型，
让模型自己修正参数重试，而不是让整个运行直接崩溃。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable, Optional


class ToolError(Exception):
    """工具调用失败。错误信息需要回传给模型，让它自己修正。"""


# 无论谁确认都不允许执行的命令，直接拒绝。
BLOCKED_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\brm\s+-rf\s+~",
    r"\bmkfs\b",
    r"\bdd\s+.*of=/dev/",
    r":\(\)\s*\{",          # fork bomb
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bchmod\s+-R\s+777\s+/",
]

# 只读命令白名单：可以免人工确认直接执行。
SAFE_COMMAND_PREFIXES = (
    "ls", "cat", "head", "tail", "grep", "rg", "find", "pwd", "echo",
    "python3 --version", "python --version", "git status", "git log",
    "git diff", "wc", "tree",
)

MAX_READ_CHARS = 20_000
MAX_GREP_MATCHES = 100
DEFAULT_TIMEOUT = 30


class Workspace:
    """把所有文件访问限制在一个根目录（工作区）之内。"""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel_path: str) -> Path:
        """解析模型给出的路径，拒绝逃出工作区根目录的路径。"""
        p = (self.root / rel_path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path escapes workspace: {rel_path!r}")
        return p


class ToolBox:
    """持有工作区与确认策略，并实现 4 个工具。"""

    def __init__(
        self,
        workspace: Workspace,
        confirm: Optional[Callable[[str], bool]] = None,
    ):
        # confirm(command) 返回 True 表示人工批准执行。
        # 默认策略：只有白名单里的只读命令可以免确认执行。
        self.ws = workspace
        self.confirm = confirm

    # ------------------------------------------------------------------ files
    def read_file(self, path: str) -> str:
        p = self.ws.resolve(path)
        if not p.is_file():
            raise ToolError(f"file not found: {path!r}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "\n... [truncated]"
        return text

    def write_file(self, path: str, content: str) -> str:
        p = self.ws.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    # ------------------------------------------------------------------- grep
    def grep_search(self, pattern: str, path: str = ".") -> str:
        base = self.ws.resolve(path)
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"invalid regex {pattern!r}: {e}") from e
        matches: list[str] = []
        files = [base] if base.is_file() else sorted(
            f for f in base.rglob("*")
            if f.is_file() and ".git" not in f.parts and "__pycache__" not in f.parts
        )
        for f in files:
            try:
                lines = f.read_text(encoding="utf-8", errors="strict").splitlines()
            except (UnicodeDecodeError, OSError):
                continue  # skip binary / unreadable files
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    rel = f.relative_to(self.ws.root)
                    matches.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(matches) >= MAX_GREP_MATCHES:
                        return "\n".join(matches) + "\n... [more matches omitted]"
        return "\n".join(matches) if matches else "no matches"

    # ------------------------------------------------------------------ shell
    def run_shell(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        for pat in BLOCKED_PATTERNS:
            if re.search(pat, command):
                raise ToolError(f"command blocked for safety: {command!r}")
        if not self._is_safe(command):
            if self.confirm is None or not self.confirm(command):
                raise ToolError(
                    f"command needs human confirmation and was not approved: {command!r}"
                )
        try:
            proc = subprocess.run(
                command, shell=True, cwd=self.ws.root,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {timeout}s: {command!r}")
        out = (proc.stdout or "") + (proc.stderr or "")
        out = out.strip() or "(no output)"
        if len(out) > MAX_READ_CHARS:
            out = out[:MAX_READ_CHARS] + "\n... [truncated]"
        return f"exit_code={proc.returncode}\n{out}"

    @staticmethod
    def _is_safe(command: str) -> bool:
        cmd = command.strip()
        return any(cmd == p or cmd.startswith(p + " ") for p in SAFE_COMMAND_PREFIXES)


# 提供给模型的工具定义（OpenAI 风格 function schema）。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the workspace root."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (create or overwrite) a UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search text files in the workspace with a regular expression. Returns file:line matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "description": "Optional subdirectory or file to search in (default: whole workspace)."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the workspace. Non read-only commands need human confirmation; destructive ones are blocked.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The shell command to run."}},
                "required": ["command"],
            },
        },
    },
]
