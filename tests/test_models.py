"""Model adapters and connect(): parsing, the Claude translation, native replay.
No network: the SDK clients are replaced with fakes that record what they're sent."""

import json
from types import SimpleNamespace

import pytest

from plato import Agent
from plato.anthropic_model import AnthropicModel, from_anthropic, to_anthropic
from plato.connect import OPENAI_COMPATIBLE, connect
from plato.model import OpenAICompatModel


class Block(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


def claude_response(*blocks, stop_reason="tool_use"):
    usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5})
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=usage)


def tool_use(code, thoughts="", id="toolu_1"):
    return Block(type="tool_use", id=id, name="run", input={"thoughts": thoughts, "code": code})


class FakeClaude:
    def __init__(self, responses):
        self.responses, self.sent = list(responses), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        return self.responses.pop(0)


# ── connect() ────────────────────────────────────────────────────────────
def test_connect_anthropic_uses_the_native_adapter(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    m = connect("anthropic:claude-opus-5-5", effort="high")
    assert isinstance(m, AnthropicModel) and m.model == "claude-opus-5-5" and m.effort == "high"
    assert connect("claude:").model == "claude-opus-5-5"              # sensible default


def test_connect_openai_compatible_presets(monkeypatch):
    monkeypatch.delenv("PLATO_API_KEY", raising=False)
    monkeypatch.delenv("PLATO_BASE_URL", raising=False)
    m = connect("ollama:llama3.1:8b")                                # model ids may contain colons
    assert isinstance(m, OpenAICompatModel) and m.model == "llama3.1:8b"
    assert str(m.client.base_url).startswith("http://localhost:11434/v1")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert str(connect("groq:some-model").client.base_url).startswith("https://api.groq.com/openai/v1")


def test_connect_any_openai_compatible_url():
    m = connect("http://localhost:8000/v1:my-model")
    assert m.model == "my-model" and str(m.client.base_url).startswith("http://localhost:8000/v1")


def test_connect_explains_what_is_missing(monkeypatch):
    monkeypatch.delenv("PLATO_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="set MISTRAL_API_KEY"):
        connect("mistral:mistral-large-latest")
    with pytest.raises(ValueError, match="unknown provider 'nope'"):
        connect("nope:x")
    with pytest.raises(ValueError, match="name a model"):
        connect("openrouter:")


def test_presets_are_well_formed():
    for name, p in OPENAI_COMPATIBLE.items():
        assert p.base_url is None or p.base_url.startswith(("http://", "https://")), name


# ── the Claude translation ───────────────────────────────────────────────
HISTORY = [
    {"role": "system", "content": "SYSTEM"},
    {"role": "user", "content": "wake context"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
        "name": "run", "arguments": json.dumps({"thoughts": "t", "code": "reply('hi')"})}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "[delivered via chat]"},
    {"role": "user", "content": "[harness] correction"},
]


def test_history_translates_to_messages_api():
    system, msgs = to_anthropic(HISTORY)
    assert system == "SYSTEM"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "user"]
    assert msgs[1]["content"] == [{"type": "tool_use", "id": "call_1", "name": "run",
                                   "input": {"thoughts": "t", "code": "reply('hi')"}}]
    assert msgs[2]["content"][0] == {"type": "tool_result", "tool_use_id": "call_1",
                                     "content": "[delivered via chat]"}


def test_native_content_is_replayed_verbatim():
    native = [{"type": "thinking", "thinking": "", "signature": "SIG"},
              {"type": "tool_use", "id": "toolu_9", "name": "run", "input": {"code": "x"}}]
    hist = HISTORY[:2] + [{**HISTORY[2], "_native": {"provider": "anthropic", "content": native}}] + HISTORY[3:]
    assert to_anthropic(hist)[1][1]["content"] == native


def test_response_parses_to_a_turn_and_keeps_native_blocks():
    turn = from_anthropic(claude_response(
        Block(type="thinking", thinking="", signature="SIG"),
        tool_use("reply('hi')", thoughts="greet", id="toolu_7")))
    assert (turn.code, turn.thoughts, turn.call_id) == ("reply('hi')", "greet", "toolu_7")
    assert turn.native["content"][0] == {"type": "thinking", "thinking": "", "signature": "SIG"}


def test_text_only_and_refusals_become_no_tool_call():
    assert from_anthropic(claude_response(Block(type="text", text="Sure!"), stop_reason="end_turn")).code is None
    refused = from_anthropic(claude_response(stop_reason="refusal"))
    assert refused.code is None and "declined" in refused.text


def test_request_shape():
    fake = FakeClaude([claude_response(tool_use("stop()"))])
    AnthropicModel("claude-opus-5-5", client=fake, effort="medium").complete(
        HISTORY, {"type": "function", "function": {"name": "run", "description": "D", "parameters": {
            "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}})
    req = fake.sent[0]
    assert req["tool_choice"] == {"type": "auto"}                    # forced choice 400s on current Claude
    assert req["tools"][0]["strict"] is True and req["tools"][0]["input_schema"]["additionalProperties"] is False
    assert req["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert req["fallbacks"] == "default" and req["output_config"] == {"effort": "medium"}
    assert set(req["betas"]) == {"thinking-binding-controls-2026-08-01", "server-side-fallback-2026-07-01"}
    assert req["system"] == "SYSTEM"


def test_older_models_can_opt_out_of_adaptive_thinking_and_fallbacks():
    fake = FakeClaude([claude_response(tool_use("stop()"))])
    AnthropicModel("claude-haiku-4-5", client=fake, thinking=False, fallbacks=None).complete(
        HISTORY, {"type": "function", "function": {"name": "run", "description": "", "parameters": {"type": "object"}}})
    assert "thinking" not in fake.sent[0] and "betas" not in fake.sent[0] and "fallbacks" not in fake.sent[0]


def test_agent_runs_on_claude_and_replays_its_thinking(tmp_path):
    fake = FakeClaude([
        claude_response(Block(type="thinking", thinking="", signature="S1"),
                        tool_use("reply('On it.')", id="toolu_1")),
        claude_response(Block(type="thinking", thinking="", signature="S2"),
                        tool_use("reply('Done.'); stop('ok')", id="toolu_2")),
    ])
    agent = Agent(tmp_path / "ws", AnthropicModel(client=fake))
    agent.send("hello")
    r = agent.wake()
    assert r.outcome == "stopped" and r.replies == ["On it.", "Done."]
    second = fake.sent[1]["messages"]
    assert second[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "S1"}
    assert second[2]["content"][0]["tool_use_id"] == "toolu_1"


def test_openai_adapter_strips_private_fields():
    m = OpenAICompatModel("m", base_url="http://localhost:1/v1", api_key="x")
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        msg = SimpleNamespace(content="", tool_calls=[SimpleNamespace(
            id="c1", function=SimpleNamespace(arguments=json.dumps({"code": "stop()"})))])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    m.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    hist = HISTORY[:2] + [{**HISTORY[2], "_native": {"provider": "anthropic", "content": []}}] + HISTORY[3:]
    assert m.complete(hist, {"type": "function", "function": {"name": "run"}}).code == "stop()"
    assert all(not k.startswith("_") for msg in sent["messages"] for k in msg)
