"""plato: an event-driven, code-first agent harness."""

from plato.agent import Agent, Config, WakeResult
from plato.anthropic_model import AnthropicModel
from plato.channels import WITHHELD, Channel
from plato.chat import Chat
from plato.connect import connect
from plato.events import Event, EventLog
from plato.model import ModelTurn, OpenAICompatModel, ScriptedModel
from plato.should_respond import Policy, Rule
from plato.tails import FileReference, InReplyTo, PayloadField, RecentEvents
from plato.tools import ToolRegistry, tool
from plato.workspace import Workspace

__all__ = [
    "Agent",
    "AnthropicModel",
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
    "connect",
    "tool",
]
