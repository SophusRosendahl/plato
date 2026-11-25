"""Model selection for the demos: your real model if configured, else an
offline stand-in so every demo runs (and can be tested) without an API key.

    export PLATO_MODEL=accounts/fireworks/models/kimi-k2p6
    export PLATO_BASE_URL=https://api.fireworks.ai/inference/v1
    export PLATO_API_KEY=...

Any OpenAI-compatible endpoint works.
"""

from __future__ import annotations

import os
import re
import time

from plato import ModelTurn, OpenAICompatModel

_HINT = re.compile(r"^→ Reply in: (.+)$")
_ITEM = re.compile(r"^\[\d\d:\d\d\] ([^:]+): (.*)$")
_ID = re.compile(r"\b(e[0-9a-f]{10})\b")


def make_model(offline_delay: float | None = None):
    name = os.environ.get("PLATO_MODEL")
    if not name:
        delay = offline_delay if offline_delay is not None else float(os.environ.get("PLATO_OFFLINE_DELAY", "2"))
        return OfflineModel(delay=delay)
    return OpenAICompatModel(
        name,
        base_url=os.environ.get("PLATO_BASE_URL"),
        api_key=os.environ.get("PLATO_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )


def describe(model) -> str:
    if isinstance(model, OfflineModel):
        return "offline stand-in model (set PLATO_MODEL for real answers)"
    return f"model {model.model}"


class OfflineModel:
    """A deterministic stand-in that drives the harness the way a model would,
    without understanding anything: it acknowledges, 'thinks' for `delay`
    seconds per step (so you can type follow-ups mid-run), answers every item
    on its `→ Reply in:` line, then stops. Real behaviour needs a real model."""

    def __init__(self, delay: float = 2.0) -> None:
        self.delay = delay
        self._todo: list[tuple[str, str]] = []
        self._answered_any = False

    def complete(self, messages: list[dict], tool: dict) -> ModelTurn:
        last = messages[-1]
        tools = tool["function"]["description"]
        items = _items(last["content"]) if last["role"] == "user" else []
        first_step = last["role"] == "user" and not any(m["role"] == "tool" for m in messages)

        if first_step:
            self._todo, self._answered_any = list(items), False
            return ModelTurn(code=_ack(self._todo, tools), thoughts="Acknowledge, then think.")
        if last["role"] == "user" and not items:                     # a harness correction
            return ModelTurn(code='stop("done")', thoughts="Wrap up.")
        time.sleep(self.delay)                                       # "thinking"; type a follow-up now
        self._todo += items
        if not self._todo:
            return ModelTurn(code='stop("Answered.")', thoughts="Everything is answered.")
        code = "\n".join(_answer(hint, text, also=self._answered_any) for hint, text in self._todo)
        self._todo, self._answered_any = [], True
        return ModelTurn(code=code, thoughts="Answer what is open.")


def _items(content: str) -> list[tuple[str, str]]:
    """(reply hint, "label: text") for each inbox item that names where to answer."""
    out, lines = [], content.splitlines()
    for i, line in enumerate(lines):
        m = _HINT.match(line.strip())
        if m and i + 1 < len(lines):
            item = _ITEM.match(lines[i + 1].strip())
            if item:
                out.append((m.group(1), f"{item.group(1)}: {item.group(2)}"))
    return out


def _ack(items: list[tuple[str, str]], tools: str) -> str:
    acks = []
    for hint, labelled in items:
        label, _, text = labelled.partition(": ")
        target = _ID.search(label) or _ID.search(hint)
        if hint.startswith("reply("):
            acks.append(f"reply({('Thinking about “' + _short(text) + '”…')!r})")
        elif target and "react(" in tools:
            acks.append(f"react({target.group(1)!r}, '👀')")
    return "\n".join(acks) or "print('thinking')"


def _answer(hint: str, labelled: str, also: bool) -> str:
    text = labelled.partition(": ")[2]
    answer = (f"(offline model) {'Also got' if also else 'I read'}: “{_short(text)}”. "
              "Set PLATO_MODEL to give me a real brain.")
    if "'...'" in hint:
        return hint.replace("'...'", repr(answer))
    return re.sub(r"\bmessage\b", repr(answer).replace("\\", "\\\\"), hint, count=1)


def _short(text: str, n: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"
