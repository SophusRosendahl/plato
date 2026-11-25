"""Runs the agent's Python code in a persistent namespace.

Code-first: the model's only action is "run this Python". Tools are plain
functions in the namespace; variables persist across steps within a wake.
`stop()` is an interrupt (an exception); `reply()` is not — it just records.

SPIKE NOTE: this runs `exec()` in-process. That is fine for a demo you run on
your own machine and NOT a sandbox. A subprocess/Docker/E2B executor with the
same `run(code) -> ExecResult` shape is the obvious next step.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import datetime
import io
import json
import sys
import threading
import traceback
from dataclasses import dataclass

AGENT_FILENAME = "<agent>"
ECHO_HELPER = "__echo_expr__"
MAX_ECHO_CHARS = 20_000


class StopRun(Exception):
    def __init__(self, summary: str = "") -> None:
        super().__init__(summary)
        self.summary = summary


@dataclass
class ExecResult:
    status: str  # "continue" | "stop" | "error"
    output: str
    summary: str = ""


# ── output capture, per thread ───────────────────────────────────────────
# The agent's prints must become its tool output, but only the agent's: a web
# server thread, a CLI input loop or a UI callback printing at the same moment
# must reach the real terminal. `contextlib.redirect_stdout` swaps sys.stdout for
# the whole process, so instead sys.stdout is wrapped once in a router that sends
# each thread's writes to that thread's capture buffer, if it has one.
_capture = threading.local()


class _Router(io.TextIOBase):
    def __init__(self, real) -> None:
        self._real = real

    def _target(self):
        return getattr(_capture, "buf", None) or self._real

    def write(self, s: str) -> int:
        return self._target().write(s)

    def flush(self) -> None:
        self._target().flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _install_router() -> None:
    if not isinstance(sys.stdout, _Router):
        sys.stdout = _Router(sys.stdout)
    if not isinstance(sys.stderr, _Router):
        sys.stderr = _Router(sys.stderr)


@contextlib.contextmanager
def uncaptured():
    """Run a block with real output even if called from inside agent code,
    e.g. a channel delivering a message to a terminal UI."""
    prev = getattr(_capture, "buf", None)
    _capture.buf = None
    try:
        yield
    finally:
        _capture.buf = prev


def _echo_value(value) -> None:
    """REPL-style echo of a bare top-level expression. exec() would discard the
    value, and the model reads empty output as "nothing there" — a silent false
    negative. None stays silent so acting tools don't print noise."""
    if value is None:
        return
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception as e:  # a __repr__ that raises must not kill the block
        text = f"<unprintable {type(value).__name__}: {e}>"
    if len(text) > MAX_ECHO_CHARS:
        text = text[:MAX_ECHO_CHARS] + f"\n… [truncated — {len(text)} chars total]"
    print(text)


def _compile_with_echo(source: str):
    tree = ast.parse(source, filename=AGENT_FILENAME)
    tree.body = [
        ast.copy_location(
            ast.Expr(ast.Call(ast.Name(ECHO_HELPER, ast.Load()), [s.value], [])), s
        )
        if isinstance(s, ast.Expr) and not isinstance(s.value, ast.Constant)
        else s
        for s in tree.body
    ]
    ast.fix_missing_locations(tree)
    return compile(tree, AGENT_FILENAME, "exec")


def _normalize(code: str) -> str:
    code = code.replace("\r\n", "\n").strip("\n")
    # Models sometimes wrap the code in a markdown fence inside the tool call.
    if code.startswith("```"):
        lines = code.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def _format_error(exc: BaseException) -> str:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == AGENT_FILENAME]
    where = f" (line {frames[-1].lineno})" if frames else ""
    if isinstance(exc, SyntaxError):
        where = f" (line {exc.lineno})"
    return f"{type(exc).__name__}{where}: {exc}"


class Executor:
    def __init__(self, namespace: dict, output_cap: int = 8_000) -> None:
        self.output_cap = output_cap
        self.env: dict = {
            "__builtins__": builtins,
            "json": json,
            "datetime": datetime,
            ECHO_HELPER: _echo_value,
        }
        self.env.update(namespace)

    def run(self, code: str) -> ExecResult:
        buf = io.StringIO()
        status, summary = "continue", ""
        try:
            compiled = _compile_with_echo(_normalize(code))
            _install_router()
            _capture.buf = buf
            try:
                exec(compiled, self.env)
            finally:
                _capture.buf = None
        except StopRun as s:
            status, summary = "stop", s.summary
        except Exception as e:
            status = "error"
            buf.write(("\n" if buf.getvalue() else "") + "[error] " + _format_error(e))
        out = buf.getvalue()
        if len(out) > self.output_cap:
            out = out[: self.output_cap] + f"\n… [output truncated — {len(out)} chars total]"
        return ExecResult(status, out, summary)
