from contextlib import contextmanager

from complibot.answer import CompliBot
from complibot.guardrails import Guardrails, RegexRedactor
from complibot.llm import FakeProvider
from complibot.tracing import Tracer, _mask_with


class RecordingClient:
    def __init__(self):
        self.events = []

    @contextmanager
    def _cm(self, name, kw):
        rec = {"name": name, **kw, "updates": []}
        self.events.append(rec)

        class Obs:
            def update(self, **u):
                rec["updates"].append(u)

        yield Obs()

    def start_as_current_observation(self, name, as_type="span", **kw):
        return self._cm(name, {"as_type": as_type, **kw})

    def flush(self):
        pass


def test_pipeline_emits_spans_without_raw_pii(retriever, settings):
    client = RecordingClient()
    bot = CompliBot(settings, retriever, FakeProvider(), Guardrails(RegexRedactor()), Tracer(client))
    bot.ask("I'm S1234567D. How many business days to stop using data after consent is withdrawn?")
    names = [e["name"] for e in client.events]
    assert names == ["ask", "input-guardrail", "retrieve", "answer-llm"]
    gen = next(e for e in client.events if e["name"] == "answer-llm")
    assert gen["as_type"] == "generation"
    assert any("usage_details" in u for u in gen["updates"])
    assert "S1234567D" not in repr(client.events)


def test_noop_tracer_and_broken_client_never_raise():
    with Tracer(None).observe("x") as obs:
        obs.update(output=1)

    class Broken:
        def start_as_current_observation(self, **kw):
            raise RuntimeError("langfuse down")

    with Tracer(Broken()).observe("x") as obs:
        obs.update(output=1)


def test_mask_function_scrubs_nested_data():
    mask = _mask_with(RegexRedactor())
    out = mask(data={"q": "S1234567D", "list": ["mail a@b.com", 3]})
    assert out == {"q": "<NRIC_1>", "list": ["mail <EMAIL_1>", 3]}
