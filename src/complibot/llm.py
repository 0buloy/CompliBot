"""Pluggable LLM providers behind one tiny interface.

    provider.complete(system, user, json_schema=None) -> LLMResponse     (single call, JSON out)
    provider.chat(system, messages, tools) -> ChatResponse               (tool calling, for the agent)

- AnthropicProvider: Claude via the official `anthropic` SDK. Uses structured
  outputs (output_config.format = json_schema) so the reply is valid JSON, and
  the server-side refusal fallback so a declined request is retried on a
  fallback model inside the same call.
- OpenAIProvider: OpenAI chat completions with a strict json_schema.
- FakeProvider: deterministic, offline, no API key. Lets the whole pipeline
  (and CI) run for free; it "answers" by quoting the best-overlapping source.

Model names always come from environment variables (see config.py).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .config import Settings

# USD per 1M tokens (input, output). Used for the cost column in eval reports.
PRICES_PER_M: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "fake": (0.0, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_PER_M.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)


class LLMError(RuntimeError):
    """Any provider failure (network, auth, rate limit, refusal, truncation)."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse: ...


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences or chatter."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in model output") from None
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output is JSON but not an object")
    return value


# ------------------------------------------------------------------ tool calling
#
# Provider-neutral conversation format used by the agent:
#   {"role": "user", "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [ToolCall...], "raw": <provider message>}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}
# Each provider converts it to its own wire format. For Claude we send the assistant's
# original content blocks back unchanged ("raw"), which keeps thinking blocks valid.


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ChatResponse:
    text: str
    tool_calls: list[ToolCall]
    raw: object  # provider-native assistant message, replayed verbatim on the next turn
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str = ""

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)


def _anthropic_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    results: list[dict] = []  # all tool results of one turn go back in ONE user message
    for m in messages:
        if m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if m.get("is_error"):
                block["is_error"] = True
            results.append(block)
            continue
        if results:
            out.append({"role": "user", "content": results})
            results = []
        if m["role"] == "assistant":
            out.append({"role": "assistant", "content": m["raw"] if m.get("raw") is not None else m["content"]})
        else:
            out.append({"role": "user", "content": m["content"]})
    if results:
        out.append({"role": "user", "content": results})
    return out


def _openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        elif m["role"] == "assistant":
            msg: dict = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                                     for c in m["tool_calls"]]
            out.append(msg)
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


# --------------------------------------------------------------------- Anthropic


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, effort: str = "medium", fallbacks: str = "default", max_tokens: int = 4000,
                 client=None, timeout: float = 60.0, max_retries: int = 3):
        if client is None:
            import anthropic  # lazy: optional dependency

            # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
            # We deliberately don't add a second retry layer on top (retries would multiply).
            client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        self.client = client
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.max_tokens = max_tokens

    def build_request(self, system: str, user: str, json_schema: dict | None) -> dict:
        output_config: dict = {}
        if self.effort:
            output_config["effort"] = self.effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        req: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if output_config:
            req["output_config"] = output_config
        return req

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse:
        import anthropic

        req = self.build_request(system, user, json_schema)
        t0 = time.perf_counter()
        try:
            if self.fallbacks and self.fallbacks != "off":
                resp = self.client.beta.messages.create(
                    **req, betas=["server-side-fallback-2026-07-01"], fallbacks=self.fallbacks
                )
            else:
                resp = self.client.messages.create(**req)
        except anthropic.RateLimitError as e:
            raise LLMError(f"Anthropic rate limit: {e.message}", retryable=True) from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}", retryable=e.status_code >= 500) from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API", retryable=True) from e
        latency = (time.perf_counter() - t0) * 1000

        if resp.stop_reason == "refusal":
            raise LLMError("Model declined the request (stop_reason=refusal)")
        if resp.stop_reason == "max_tokens":
            raise LLMError("Model output was truncated (max_tokens); raise COMPLIBOT_MAX_OUTPUT_TOKENS")
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=resp.model or self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency,
        )


    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> ChatResponse:
        import anthropic

        req = {
            "model": self.model, "max_tokens": self.max_tokens, "system": system,
            "messages": _anthropic_messages(messages),
            "tools": [{**t, "strict": True} for t in tools],
            "tool_choice": {"type": "auto"},  # forced tool choice is rejected by current models
        }
        if self.effort:
            req["output_config"] = {"effort": self.effort}
        t0 = time.perf_counter()
        try:
            if self.fallbacks and self.fallbacks != "off":
                resp = self.client.beta.messages.create(**req, betas=["server-side-fallback-2026-07-01"],
                                                        fallbacks=self.fallbacks)
            else:
                resp = self.client.messages.create(**req)
        except anthropic.RateLimitError as e:
            raise LLMError(f"Anthropic rate limit: {e.message}", retryable=True) from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}", retryable=e.status_code >= 500) from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API", retryable=True) from e
        if resp.stop_reason == "refusal":
            raise LLMError("Model declined the request (stop_reason=refusal)")
        if resp.stop_reason == "max_tokens":
            raise LLMError("Model output was truncated (max_tokens)")
        calls = []
        for b in resp.content:
            if b.type == "tool_use":
                args = b.input if isinstance(b.input, dict) else json.loads(b.input)
                calls.append(ToolCall(id=b.id, name=b.name, args=args))
        return ChatResponse(
            text="".join(b.text for b in resp.content if b.type == "text"), tool_calls=calls, raw=resp.content,
            model=resp.model or self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=(time.perf_counter() - t0) * 1000, stop_reason=resp.stop_reason or "",
        )


# ------------------------------------------------------------------------ OpenAI


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, max_tokens: int = 4000, client=None, timeout: float = 60.0, max_retries: int = 3):
        if not model:
            raise ValueError("OPENAI_MODEL is not set")
        if client is None:
            from openai import OpenAI  # lazy: optional dependency

            client = OpenAI(timeout=timeout, max_retries=max_retries)
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse:
        import openai

        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": self.max_tokens,
        }
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "complibot_output", "schema": json_schema, "strict": True},
            }
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            raise LLMError("OpenAI rate limit", retryable=True) from e
        except openai.APIStatusError as e:
            raise LLMError(f"OpenAI API error {e.status_code}", retryable=e.status_code >= 500) from e
        except openai.APIConnectionError as e:
            raise LLMError("Could not reach the OpenAI API", retryable=True) from e
        latency = (time.perf_counter() - t0) * 1000
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("Model output was truncated (length)")
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model or self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
        )


    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> ChatResponse:
        import openai

        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=_openai_messages(system, messages), max_completion_tokens=self.max_tokens,
                tools=[{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                         "parameters": t["input_schema"], "strict": True}}
                       for t in tools],
            )
        except openai.RateLimitError as e:
            raise LLMError("OpenAI rate limit", retryable=True) from e
        except openai.APIStatusError as e:
            raise LLMError(f"OpenAI API error {e.status_code}", retryable=e.status_code >= 500) from e
        except openai.APIConnectionError as e:
            raise LLMError("Could not reach the OpenAI API", retryable=True) from e
        choice = resp.choices[0]
        calls = [ToolCall(id=c.id, name=c.function.name, args=json.loads(c.function.arguments or "{}"))
                 for c in (choice.message.tool_calls or [])]
        usage = resp.usage
        return ChatResponse(
            text=choice.message.content or "", tool_calls=calls, raw=None, model=resp.model or self.model,
            input_tokens=usage.prompt_tokens if usage else 0, output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.perf_counter() - t0) * 1000, stop_reason=choice.finish_reason or "",
        )


# -------------------------------------------------------------------------- Fake

SOURCE_RE = re.compile(r'<source id="(?P<id>[^"]+)"[^>]*>\n?(?P<text>.*?)\n?</source>', re.S)
QUESTION_RE = re.compile(r"<question>\s*(?P<q>.*?)\s*</question>", re.S)


def fake_answer(system: str, user: str, json_schema: dict | None) -> str:
    """Offline stand-in for an LLM answering from sources.

    Picks the source sharing the most (stemmed) words with the question. If it
    covers at least half the question's content words it quotes that source's
    first two sentences and cites it; otherwise it refuses.
    """
    from .retrieve import tokenize  # local import avoids a cycle

    qm = QUESTION_RE.search(user)
    question = qm.group("q") if qm else user
    q_tokens = set(tokenize(question))
    best_id, best_text, best_overlap = None, "", 0.0
    for m in SOURCE_RE.finditer(user):
        overlap = len(q_tokens & set(tokenize(m.group("text")))) / max(1, len(q_tokens))
        if overlap > best_overlap:
            best_id, best_text, best_overlap = m.group("id"), m.group("text"), overlap
    if best_id is None or best_overlap < 0.5:
        return json.dumps({"answer": "I don't know based on the provided documents.", "citations": [], "refused": True})
    body = re.sub(r"^\S+\s+", "", best_text.strip(), count=1)  # drop the clause number
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(body.split()))
    answer = " ".join(sentences[:2]) + f" [{best_id}]"
    return json.dumps({"answer": answer, "citations": [best_id], "refused": False})


class FakeProvider:
    name = "fake"
    model = "fake"

    def __init__(self, responder: Callable[[str, str, dict | None], str] = fake_answer, chat_policy=None):
        self.responder = responder
        # chat_policy(system, messages, tools) -> (text, [ToolCall]): a scripted planner for agent tests
        self.chat_policy = chat_policy
        self.calls: list[tuple[str, str]] = []  # recorded for test assertions

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        self.calls.append((system, user))
        text = self.responder(system, user, json_schema)
        return LLMResponse(
            text=text,
            model="fake",
            input_tokens=(len(system) + len(user)) // 4,
            output_tokens=len(text) // 4,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> ChatResponse:
        if self.chat_policy is None:
            raise LLMError("FakeProvider has no chat_policy for tool calling")
        t0 = time.perf_counter()
        text, calls = self.chat_policy(system, messages, tools)
        self.calls.append((system, json.dumps(messages[-1], default=str)[:2000]))
        return ChatResponse(text=text, tool_calls=calls, raw=None, model="fake", input_tokens=len(system) // 4,
                            output_tokens=len(text) // 4, latency_ms=(time.perf_counter() - t0) * 1000,
                            stop_reason="tool_use" if calls else "end_turn")


def get_provider(settings: Settings, provider: str | None = None, model: str | None = None) -> LLMProvider:
    kind = (provider or settings.llm_provider).lower()
    if kind == "fake":
        return FakeProvider()
    if kind == "anthropic":
        return AnthropicProvider(
            model=model or settings.anthropic_model,
            effort=settings.anthropic_effort,
            fallbacks=settings.anthropic_fallbacks,
            max_tokens=settings.max_output_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    if kind == "openai":
        return OpenAIProvider(model=model or settings.openai_model, max_tokens=settings.max_output_tokens,
                              timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries)
    raise ValueError(f"Unknown LLM provider {kind!r} (use fake | anthropic | openai)")
