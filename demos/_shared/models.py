"""Model selection for the demos: your real model if configured, else an
offline stand-in so every demo runs (and can be tested) without an API key.

    export PLATO_MODEL=anthropic:claude-opus-5-5     # + ANTHROPIC_API_KEY
    export PLATO_MODEL=openai:gpt-5                  # + OPENAI_API_KEY
    export PLATO_MODEL=ollama:llama3.3               # local, no key

Any provider `plato.connect` knows, or any OpenAI-compatible base URL.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from plato import Limits, ModelTurn, connect

_HINT = re.compile(r"^→ Reply in: (.+)$")
_ITEM = re.compile(r"^\[\d\d:\d\d\] ([^:]+): (.*)$")
_ID = re.compile(r"\b(e[0-9a-f]{10})\b")


def make_model(offline_delay: float | None = None):
    name = os.environ.get("PLATO_MODEL")
    if not name:
        delay = offline_delay if offline_delay is not None else float(os.environ.get("PLATO_OFFLINE_DELAY", "2"))
        return OfflineModel(delay=delay)
    return connect(name)


# Pessimistic default prices (per million tokens, input/output): roughly the top
# tier of the priciest Gemini models, so the meter over-counts on cheaper models
# and a cap trips before real spend reaches it. Set PLATO_PRICE_PER_MTOK="in,out"
# to the real prices of the model you use for a closer count.
DEFAULT_PRICE_PER_MTOK = (2.50, 15.0)


def spend_limits(model) -> dict:
    """Agent kwargs that cap spend for a real model: PLATO_MONTHLY_BUDGET (default
    100) and PLATO_DAILY_BUDGET (default 5), in the currency of your prices, on one
    usage file shared by every demo and run (PLATO_USAGE_FILE, default
    ~/.plato/usage.json). The offline model costs nothing and gets no limits."""
    if isinstance(model, OfflineModel):
        return {}
    raw = os.environ.get("PLATO_PRICE_PER_MTOK")
    price = tuple(float(x) for x in raw.split(",")) if raw else DEFAULT_PRICE_PER_MTOK
    limits = Limits(monthly_cost=float(os.environ.get("PLATO_MONTHLY_BUDGET", "100")) or None,
                    daily_cost=float(os.environ.get("PLATO_DAILY_BUDGET", "5")) or None,
                    price_per_mtok=price)
    usage_file = os.environ.get("PLATO_USAGE_FILE") or Path.home() / ".plato" / "usage.json"
    return {"limits": limits, "usage_file": usage_file}


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
        elif "status='review'" in hint:  # a work item: pick it up first
            acks.append(hint.replace("'...'", repr("Picking this up.")).replace("'review'", "'in_progress'"))
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
