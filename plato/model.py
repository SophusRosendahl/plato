"""Model adapters. The harness needs one thing: given messages + the single
`run` tool, return the model's (thoughts, code).

`OpenAICompatModel` works with any OpenAI-compatible chat API (OpenAI,
Fireworks, Together, OpenRouter, vLLM, Ollama…). `ScriptedModel` replays a
fixed script — used by the tests to exercise the loop and guards with no
network and no key.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Protocol

try:
    import openai
except ImportError:  # optional: only needed for live models
    openai = None


@dataclass
class ModelTurn:
    code: str | None
    thoughts: str = ""
    text: str = ""
    call_id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:10])
    usage: dict = field(default_factory=dict)
    # The provider's own form of this turn ({"provider": ..., "content": ...}),
    # kept so an adapter can replay it unchanged (e.g. Claude's thinking blocks).
    native: dict | None = None


class Model(Protocol):
    def complete(self, messages: list[dict], tool: dict) -> ModelTurn: ...


_XML_CODE = re.compile(r'<parameter name="code">(.*?)</parameter>', re.DOTALL)
_XML_THOUGHTS = re.compile(r'<parameter name="thoughts">(.*?)</parameter>', re.DOTALL)


def _parse_args(raw: str) -> tuple[str, str | None]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return "", None
    return str(args.get("thoughts", "")), args.get("code")


class OpenAICompatModel:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> None:
        if openai is None:
            raise RuntimeError("pip install openai to use OpenAICompatModel")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, messages: list[dict], tool: dict) -> ModelTurn:
        kwargs = dict(
            model=self.model,
            messages=[{k: v for k, v in m.items() if not k.startswith("_")} for m in messages],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool["function"]["name"]}},
            max_tokens=self.max_tokens,
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        content = msg.content or ""
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            thoughts, code = _parse_args(tc.function.arguments)
            return ModelTurn(code=code, thoughts=thoughts, text=content, call_id=tc.id, usage=usage)
        # Some models emit the call as XML in the content instead.
        m = _XML_CODE.search(content)
        if m:
            t = _XML_THOUGHTS.search(content)
            return ModelTurn(code=m.group(1).strip(), thoughts=t.group(1).strip() if t else "",
                             text="", usage=usage)
        return ModelTurn(code=None, text=content, usage=usage)


Step = str | ModelTurn | Callable[[list[dict]], "str | ModelTurn"]


class ScriptedModel:
    """Replays `steps` in order. A step is a code string, a ModelTurn, or a
    callable(messages) -> either. When the script runs out, `default` repeats
    (or an IndexError is raised if there is none)."""

    def __init__(self, steps: list[Step], default: Step | None = None) -> None:
        self.steps = list(steps)
        self.default = default
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], tool: dict) -> ModelTurn:
        self.calls.append([dict(m) for m in messages])
        if self.steps:
            step = self.steps.pop(0)
        elif self.default is not None:
            step = self.default
        else:
            raise IndexError("ScriptedModel ran out of steps")
        if callable(step):
            step = step(messages)
        return step if isinstance(step, ModelTurn) else ModelTurn(code=step)
