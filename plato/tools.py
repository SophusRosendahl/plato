"""Tool registry: plain Python functions the agent can call from its code.

A tool is just a function. Registering it does two things: binds it into the
namespace the agent's code runs in, and adds a one-line entry (signature + first
docstring line) to the catalogue the model sees. There is no separate schema to
keep in sync with the code.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    fn: Callable
    group: str

    @property
    def summary(self) -> str:
        doc = inspect.getdoc(self.fn) or ""
        return doc.strip().splitlines()[0] if doc.strip() else ""

    @property
    def signature(self) -> str:
        try:
            sig = inspect.signature(self.fn)
        except (TypeError, ValueError):
            return f"{self.name}(...)"
        params = []
        for p in sig.parameters.values():
            if p.default is inspect.Parameter.empty:
                params.append(p.name)
            else:
                params.append(f"{p.name}={p.default!r}")
        return f"{self.name}({', '.join(params)})"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, fn: Callable, *, name: str | None = None, group: str = "Tools") -> Callable:
        tool_name = name or fn.__name__
        self._tools[tool_name] = Tool(tool_name, fn, group)
        return fn

    def tool(self, fn: Callable | None = None, *, name: str | None = None, group: str = "Tools"):
        """Decorator form: `@registry.tool` or `@registry.tool(group="Web")`."""
        if fn is not None:
            return self.add(fn, name=name, group=group)
        return lambda f: self.add(f, name=name, group=group)

    def extend(self, other: "ToolRegistry") -> None:
        self._tools.update(other._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def namespace(self) -> dict[str, Callable]:
        return {name: t.fn for name, t in self._tools.items()}

    def describe(self) -> str:
        groups: dict[str, list[Tool]] = {}
        for t in self._tools.values():
            groups.setdefault(t.group, []).append(t)
        lines: list[str] = []
        for group, tools in groups.items():
            lines.append(f"  -- {group} --")
            width = max(len(t.signature) for t in tools)
            for t in tools:
                lines.append(f"  {t.signature.ljust(width)}  {t.summary}".rstrip())
            lines.append("")
        return "\n".join(lines).rstrip()


# A module-level registry for user tools, so `from plato import tool` works
# without the caller managing a registry object.
default_registry = ToolRegistry()


def tool(fn: Callable | None = None, *, name: str | None = None, group: str = "Tools"):
    """Register a function as an agent tool (module-level registry)."""
    return default_registry.tool(fn, name=name, group=group)
