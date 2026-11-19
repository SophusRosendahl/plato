"""plato — an event-driven, code-first agent harness (spike)."""

from plato.agent import Agent, Config, WakeResult
from plato.channels import Channel
from plato.events import Event, EventLog
from plato.model import ModelTurn, OpenAICompatModel, ScriptedModel
from plato.should_respond import Policy, Rule
from plato.tails import FileReference, PayloadField, RecentEvents
from plato.tools import ToolRegistry, tool
from plato.workspace import Workspace

__all__ = [
    "Agent",
    "Channel",
    "Config",
    "Event",
    "EventLog",
    "FileReference",
    "ModelTurn",
    "OpenAICompatModel",
    "PayloadField",
    "Policy",
    "RecentEvents",
    "Rule",
    "ScriptedModel",
    "ToolRegistry",
    "WakeResult",
    "Workspace",
    "tool",
]
