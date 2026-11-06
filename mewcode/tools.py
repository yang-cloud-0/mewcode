"""Tool layer for MewCode Lite.

Four tools the model can call:
  - read_file   : read a UTF-8 text file inside the workspace
  - write_file  : write/overwrite a UTF-8 text file inside the workspace
  - grep_search : regex search across text files inside the workspace
  - run_shell   : run a shell command in the workspace (guarded, see below)

Safety rules enforced here (not left to the model):
  1. File tools are confined to the workspace root. Any path that escapes
     the root (e.g. "../secret.txt") is rejected.
  2. run_shell rejects obviously destructive commands outright.
  3. run_shell asks for human confirmation before running anything that
     is not on a small read-only allowlist. In tests / demo a confirmation
     callback is injected; in the interactive CLI it prompts on the tty.

Every tool returns a plain string. On failure the tools in this module
raise ToolError; the agent loop converts that into a structured JSON
error message fed back to the model, so the model can fix its arguments
and retry instead of the run crashing.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable, Optional


class ToolError(Exception):
    """A tool call failed in a way the model should hear about and fix."""


# Commands that are never allowed, no matter who confirms.
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

# Read-only commands that may run without human confirmation.
SAFE_COMMAND_PREFIXES = (
    "ls", "cat", "head", "tail", "grep", "rg", "find", "pwd", "echo",
    "python3 --version", "python --version", "git status", "git log",
    "git diff", "wc", "tree",
)

MAX_READ_CHARS = 20_000
MAX_GREP_MATCHES = 100
DEFAULT_TIMEOUT = 30


class Workspace:
    """Confines all file access to one root directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel_path: str) -> Path:
        """Resolve a model-supplied path, rejecting escapes from the root."""
        p = (self.root / rel_path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path escapes workspace: {rel_path!r}")
        return p


class ToolBox:
    """Holds the workspace + confirmation policy and implements the tools."""

    def __init__(
        self,
        workspace: Workspace,
        confirm: Optional[Callable[[str], bool]] = None,
    ):
        # confirm(command) -> True means "yes, run it".
        # Default policy: only allowlisted read-only commands run unattended.
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


# OpenAI-style tool schemas handed to the model.
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
