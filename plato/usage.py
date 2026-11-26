"""Spend limits: daily and monthly caps on tokens and/or money, checked before
every model call.

Usage is recorded per day in a JSON file (by default the workspace's
`usage.json`; point several agents at one file to give them a shared budget),
so a cap survives restarts. Token counts come from each model turn's `usage`, whichever provider
reported it; money is computed from your prices, because providers don't return
it and prices change.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

try:
    import fcntl  # POSIX: lets several processes share one usage file safely
except ImportError:  # Windows: in-process locking only
    fcntl = None


@dataclass(frozen=True)
class Limits:
    daily_tokens: int | None = None           # input + output tokens per day
    daily_cost: float | None = None           # in your currency, needs `price_per_mtok`
    monthly_tokens: int | None = None         # per calendar month
    monthly_cost: float | None = None         # per calendar month, needs `price_per_mtok`
    price_per_mtok: tuple[float, float] | None = None  # (input, output) per million tokens

    def __post_init__(self) -> None:
        if (self.daily_cost is not None or self.monthly_cost is not None) and self.price_per_mtok is None:
            raise ValueError("a cost limit needs price_per_mtok=(input, output) to know what tokens cost")

    @property
    def active(self) -> bool:
        return any(v is not None for v in (self.daily_tokens, self.daily_cost, self.monthly_tokens, self.monthly_cost))


def token_counts(usage: dict) -> tuple[int, int]:
    """(input, output) tokens from an OpenAI- or Anthropic-shaped usage dict."""
    if not usage:
        return 0, 0
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    inp = sum(int(usage.get(k) or 0) for k in
              ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return inp, int(usage.get("output_tokens") or 0)


class Usage:
    def __init__(self, path: Path, limits: Limits | None = None) -> None:
        self.path = Path(path)
        self.limits = limits or Limits()
        self._lock = threading.Lock()

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path.with_suffix(".lock"), "w") as lock:
            if fcntl:
                fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def today(self) -> dict:
        return self._load().get(date.today().isoformat(), {"input": 0, "output": 0, "calls": 0})

    def this_month(self) -> dict:
        prefix, total = date.today().isoformat()[:7], {"input": 0, "output": 0, "calls": 0}
        for day, counts in self._load().items():
            if day.startswith(prefix):
                for k in total:
                    total[k] += counts.get(k, 0)
        return total

    def record(self, usage: dict) -> None:
        inp, out = token_counts(usage)
        with self._locked():
            data = self._load()
            day = data.setdefault(date.today().isoformat(), {"input": 0, "output": 0, "calls": 0})
            day["input"] += inp
            day["output"] += out
            day["calls"] += 1
            self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def cost(self, day: dict) -> float | None:
        if self.limits.price_per_mtok is None:
            return None
        pin, pout = self.limits.price_per_mtok
        return day["input"] / 1e6 * pin + day["output"] / 1e6 * pout

    def exceeded(self) -> str | None:
        """Why today's limit is reached, or None if there's room left."""
        if not self.limits.active:
            return None
        day = self.today()
        tokens = day["input"] + day["output"]
        if self.limits.daily_tokens is not None and tokens >= self.limits.daily_tokens:
            return f"today's token limit ({self.limits.daily_tokens:,}) is used up ({tokens:,} tokens)"
        cost = self.cost(day)
        if self.limits.daily_cost is not None and cost is not None and cost >= self.limits.daily_cost:
            return f"today's spend limit ({self.limits.daily_cost:.2f}) is used up ({cost:.2f})"
        month = self.this_month()
        month_tokens = month["input"] + month["output"]
        if self.limits.monthly_tokens is not None and month_tokens >= self.limits.monthly_tokens:
            return f"this month's token limit ({self.limits.monthly_tokens:,}) is used up ({month_tokens:,} tokens)"
        month_cost = self.cost(month)
        if self.limits.monthly_cost is not None and month_cost is not None and month_cost >= self.limits.monthly_cost:
            return f"this month's spend limit ({self.limits.monthly_cost:.2f}) is used up ({month_cost:.2f})"
        return None
