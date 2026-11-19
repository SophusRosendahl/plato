"""plato — an event-driven, code-first agent harness (spike)."""

from plato.agent import Agent, Config, WakeResult
from plato.channels import Channel, CommentsChannel
from plato.model import ModelTurn, OpenAICompatModel, ScriptedModel
from plato.tools import ToolRegistry, tool
from plato.workspace import Workspace

__all__ = [
    "Channel",
    "CommentsChannel",
    "Agent", "Config", "WakeResult", "ModelTurn", "OpenAICompatModel",
    "ScriptedModel", "ToolRegistry", "tool", "Workspace",
]
