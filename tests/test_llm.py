from types import SimpleNamespace

import pytest

from complibot.answer import ANSWER_SCHEMA
from complibot.config import Settings
from complibot.llm import AnthropicProvider, FakeProvider, LLMError, estimate_cost, get_provider, parse_json_object


class FakeAnthropicMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def anthropic_response(text='{"answer": "x", "citations": [], "refused": true}', stop="end_turn"):
    return SimpleNamespace(
        stop_reason=stop,
        model="claude-opus-5-5",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
    )


def make_client(resp):
    beta = FakeAnthropicMessages(resp)
    plain = FakeAnthropicMessages(resp)
    return SimpleNamespace(beta=SimpleNamespace(messages=beta), messages=plain), beta, plain


def test_anthropic_request_shape_uses_structured_output_and_fallback():
    client, beta, _ = make_client(anthropic_response())
    p = AnthropicProvider("claude-opus-5-5", effort="medium", fallbacks="default", client=client)
    out = p.complete("sys", "user", json_schema=ANSWER_SCHEMA)
    kw = beta.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": ANSWER_SCHEMA}
    assert kw["output_config"]["effort"] == "medium"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in kw  # Opus 5.5 rejects disabled thinking; we leave it adaptive
    assert out.text.startswith("{") and out.input_tokens == 1000
    assert out.cost_usd == pytest.approx(1000 * 4 / 1e6 + 100 * 20 / 1e6)


def test_anthropic_without_fallback_uses_plain_endpoint():
    client, beta, plain = make_client(anthropic_response())
    AnthropicProvider("claude-haiku-5-5", effort="", fallbacks="off", client=client).complete("s", "u")
    assert beta.kwargs is None and "output_config" not in plain.kwargs


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_anthropic_bad_stop_reasons_raise(stop):
    client, _, _ = make_client(anthropic_response(stop=stop))
    with pytest.raises(LLMError):
        AnthropicProvider("claude-opus-5-5", client=client).complete("s", "u")


def test_parse_json_object_variants():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('Sure! ```json\n{"a": 2}\n``` done') == {"a": 2}
    assert parse_json_object('Here: {"a": 3} thanks') == {"a": 3}
    with pytest.raises(ValueError):
        parse_json_object("no json here")
    with pytest.raises(ValueError):
        parse_json_object("[1, 2]")


def test_cost_table():
    assert estimate_cost("fake", 10_000, 10_000) == 0
    assert estimate_cost("unknown-model", 10, 10) == 0


def test_get_provider():
    assert isinstance(get_provider(Settings()), FakeProvider)
    with pytest.raises(ValueError):
        get_provider(Settings(llm_provider="nope"))
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        get_provider(Settings(llm_provider="openai", openai_model=""))
