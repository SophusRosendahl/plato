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
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compiled, self.env)
        except StopRun as s:
            status, summary = "stop", s.summary
        except Exception as e:
            status = "error"
            buf.write(("\n" if buf.getvalue() else "") + "[error] " + _format_error(e))
        out = buf.getvalue()
        if len(out) > self.output_cap:
            out = out[: self.output_cap] + f"\n… [output truncated — {len(out)} chars total]"
        return ExecResult(status, out, summary)
