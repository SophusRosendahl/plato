"""Claude through Anthropic's own Messages API (the official `anthropic` SDK).

plato keeps its conversation in the OpenAI chat shape; this adapter translates
it to the Messages API on the way out and back on the way in:

    system message      → the top-level `system` parameter
    assistant + call    → `tool_use` blocks (replayed verbatim from the response,
                          thinking blocks included, when this adapter produced it)
    tool message        → a `tool_result` block in a user turn

Current Claude models reject forced `tool_choice`, so the `run` tool is offered
with `tool_choice: auto` and a strict schema; the system prompt tells the model
to always act through it, and the harness already nudges a reply that doesn't.

plato trims long histories from the front within a wake. Thinking blocks are
bound to the conversation that produced them, so a trimmed history would be
rejected; `prefix_mismatch_behavior: "drop_block"` makes the API drop those
stale blocks instead of failing the request.

Refusal fallbacks are on by default (`fallbacks="default"`): if the model
declines, the API re-runs the request on a fallback model within the same call.
Pass `fallbacks=None` to turn that off.

For Claude on Amazon Bedrock, Google Vertex AI or Microsoft Foundry, pass that
platform's client (e.g. `anthropic.AnthropicBedrockMantle(aws_region=...)`).
"""

from __future__ import annotations

import json

from plato.model import ModelTurn

DEFAULT_MODEL = "claude-opus-5-5"
BETA_BLOCK_BINDING = "thinking-binding-controls-2026-08-01"
BETA_FALLBACK = "server-side-fallback-2026-07-01"


class AnthropicModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        client=None,
        max_tokens: int = 16_000,
        effort: str | None = None,
        thinking: bool = True,
        fallbacks: str | list | None = "default",
    ) -> None:
        if client is None:
            try:  # optional dependency, loaded only when Claude is used
                import anthropic  # noqa: PLC0415
            except ImportError as e:
                raise RuntimeError("pip install anthropic to use AnthropicModel") from e
            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort        # low | medium | high | xhigh | max; None = the model's default
        self.thinking = thinking    # False for older models that don't take adaptive thinking
        self.fallbacks = fallbacks

    def complete(self, messages: list[dict], tool: dict) -> ModelTurn:
        system, converted = to_anthropic(messages)
        betas = []
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": converted,
            "tools": [anthropic_tool(tool)],
            "tool_choice": {"type": "auto"},
        }
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
            betas.append(BETA_BLOCK_BINDING)
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            kwargs["fallbacks"] = self.fallbacks
            betas.append(BETA_FALLBACK)
        if betas:
            kwargs["betas"] = betas
        response = self.client.beta.messages.create(**kwargs)
        return from_anthropic(response)


def anthropic_tool(tool: dict) -> dict:
    fn = tool["function"]
    schema = dict(fn["parameters"])
    schema["additionalProperties"] = False  # required for strict tool use
    return {"name": fn["name"], "description": fn["description"], "input_schema": schema, "strict": True}


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI-shaped plato history → (system, Messages API turns)."""
    system_parts, out = [], []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_parts.append(m["content"])
        elif role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"] or "(no output)"}]})
        elif role == "assistant":
            native = m.get("_native")
            if native and native.get("provider") == "anthropic":
                out.append({"role": "assistant", "content": native["content"]})
                continue
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            for call in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                               "input": json.loads(call["function"]["arguments"] or "{}")})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no reply)"}]})
    if out and out[0]["role"] != "user":  # the first turn must be the user's
        out.insert(0, {"role": "user", "content": "(continuing)"})
    return "\n\n".join(system_parts), out


def from_anthropic(response) -> ModelTurn:
    """A Messages API response → the harness's ModelTurn (plus the native
    content, so the next request can replay it unchanged)."""
    blocks = list(response.content or [])
    texts = [b.text for b in blocks if getattr(b, "type", None) == "text"]
    call = next((b for b in blocks if getattr(b, "type", None) == "tool_use"), None)
    usage = response.usage.model_dump() if getattr(response, "usage", None) is not None else {}
    native = {"provider": "anthropic", "content": [b.model_dump(exclude_none=True) for b in blocks]}
    if getattr(response, "stop_reason", None) == "refusal":
        return ModelTurn(code=None, text="[the model declined this request]", usage=usage)
    if call is None:
        return ModelTurn(code=None, text="\n".join(texts), usage=usage, native=native)
    args = call.input if isinstance(call.input, dict) else {}
    return ModelTurn(code=args.get("code"), thoughts=str(args.get("thoughts", "")), text="\n".join(texts),
                     call_id=call.id, usage=usage, native=native)
