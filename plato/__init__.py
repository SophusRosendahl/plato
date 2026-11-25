"""plato — an event-driven, code-first agent harness (spike)."""

from plato.agent import Agent, Config, WakeResult
from plato.channels import WITHHELD, Channel
from plato.chat import Chat
from plato.events import Event, EventLog
from plato.model import ModelTurn, OpenAICompatModel, ScriptedModel
from plato.should_respond import Policy, Rule
from plato.tails import FileReference, InReplyTo, PayloadField, RecentEvents
from plato.tools import ToolRegistry, tool
from plato.workspace import Workspace

__all__ = [
    "Agent",
    "Channel",
    "Chat",
    "Config",
    "Event",
    "EventLog",
    "FileReference",
    "InReplyTo",
    "ModelTurn",
    "OpenAICompatModel",
    "PayloadField",
    "Policy",
    "RecentEvents",
    "Rule",
    "ScriptedModel",
    "ToolRegistry",
    "WITHHELD",
    "WakeResult",
    "Workspace",
    "tool",
]
