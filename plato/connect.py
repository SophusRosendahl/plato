"""Connect a model with one string: `connect("provider:model")`.

Two standard interfaces cover almost every model:

- **OpenAI Chat Completions** with tool calling, spoken by OpenAI and by most
  other providers and local servers. Each preset below is just a base URL and
  the environment variable that holds its key.
- **Anthropic Messages**, Claude's own API, via the official SDK
  (`plato.anthropic_model.AnthropicModel`).

    connect("anthropic:claude-opus-5-5")
    connect("openai:gpt-5")
    connect("ollama:llama3.3")                       # local, no key
    connect("openrouter:meta-llama/llama-4-maverick")
    connect("http://localhost:8000/v1:my-model")     # any OpenAI-compatible server

`PLATO_MODEL` uses the same syntax, so every demo and the CLI accept it.
`PLATO_API_KEY` / `PLATO_BASE_URL` override a preset's key and URL.

Other routes in: a LiteLLM proxy (`litellm:`) fronts 100+ providers behind the
OpenAI interface; Claude on Bedrock, Vertex AI or Foundry works by passing that
platform's Anthropic client to `AnthropicModel(client=...)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from plato.anthropic_model import AnthropicModel
from plato.model import OpenAICompatModel


@dataclass(frozen=True)
class Preset:
    base_url: str | None      # None = the SDK's default (api.openai.com)
    key_env: str | None       # None = no key needed (local servers)
    note: str = ""


# OpenAI-compatible endpoints. Base URLs are each provider's documented
# OpenAI-compatible endpoint; check the provider's docs if one moves.
OPENAI_COMPATIBLE: dict[str, Preset] = {
    "openai": Preset(None, "OPENAI_API_KEY"),
    "gemini": Preset("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "mistral": Preset("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "deepseek": Preset("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "xai": Preset("https://api.x.ai/v1", "XAI_API_KEY"),
    "groq": Preset("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": Preset("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": Preset("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "openrouter": Preset("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "hundreds of models, one key"),
    "litellm": Preset("http://localhost:4000/v1", "LITELLM_API_KEY", "a LiteLLM proxy you run"),
    "ollama": Preset("http://localhost:11434/v1", None, "local"),
    "lmstudio": Preset("http://localhost:1234/v1", None, "local"),
    "vllm": Preset("http://localhost:8000/v1", None, "local / self-hosted"),
}
NATIVE = {"anthropic", "claude"}


def providers() -> list[str]:
    return sorted(NATIVE | set(OPENAI_COMPATIBLE))


def connect(spec: str, **options):
    """Build a model from "provider:model" (or "http(s)://base-url:model").
    Extra keyword options go to the adapter (e.g. `effort="high"` for Claude,
    `temperature=0.2` for OpenAI-compatible models)."""
    key = os.environ.get("PLATO_API_KEY")
    if ":" not in spec and not spec.startswith(("http://", "https://")):
        # A bare model id: an OpenAI-compatible endpoint from PLATO_BASE_URL (or OpenAI itself).
        base = os.environ.get("PLATO_BASE_URL")
        return OpenAICompatModel(spec, base_url=base,
                                 api_key=key or os.environ.get("OPENAI_API_KEY") or "not-needed", **options)
    provider, model = _split(spec)
    if provider in NATIVE:
        return AnthropicModel(model or "claude-opus-5-5", api_key=key, **options)
    if provider.startswith(("http://", "https://")):
        return OpenAICompatModel(model, base_url=provider, api_key=key or "not-needed", **options)
    if provider not in OPENAI_COMPATIBLE:
        raise ValueError(f"unknown provider {provider!r}; known: {', '.join(providers())} "
                         "(or give an OpenAI-compatible base URL)")
    preset = OPENAI_COMPATIBLE[provider]
    if not model:
        raise ValueError(f"name a model, e.g. {provider}:<model-id>")
    key = key or (os.environ.get(preset.key_env) if preset.key_env else None)
    if preset.key_env and not key:
        raise ValueError(f"{provider} needs an API key: set {preset.key_env} (or PLATO_API_KEY)")
    return OpenAICompatModel(model, base_url=os.environ.get("PLATO_BASE_URL") or preset.base_url,
                             api_key=key or "not-needed", **options)


def _split(spec: str) -> tuple[str, str]:
    spec = spec.strip()
    if spec.startswith(("http://", "https://")):
        base, _, model = spec.rpartition(":")      # the model id comes after the last colon
        if base.endswith(("http", "https")) or "/" not in base:
            raise ValueError("write a base-URL spec as http(s)://host[:port]/path:model-id")
        return base, model
    provider, _, model = spec.partition(":")  # model ids may contain colons (ollama tags)
    return provider.lower(), model
