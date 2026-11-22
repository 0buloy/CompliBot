"""Input/output guardrails.

1. PII redaction BEFORE anything reaches the LLM, the logs, or the traces.
   Entities: Singapore NRIC/FIN, Singapore phone numbers, email addresses,
   credit card numbers (+ PERSON names if you enable it).
   - PresidioRedactor: Microsoft Presidio AnalyzerEngine with custom Singapore
     recognizers (NRIC/FIN with checksum, +65 phone numbers) plus built-ins.
   - RegexRedactor: same Singapore patterns with no NLP dependency; used when
     Presidio or its spaCy model isn't installed, so redaction never silently
     turns off (fail closed: we degrade to regex, not to nothing).
   Values are replaced with numbered placeholders (<NRIC_1>, <EMAIL_1>, ...);
   the same value gets the same placeholder within one text, so the question
   still reads naturally. Raw values are never stored.
   Policy: recall over precision. A false positive costs a slightly odd prompt;
   a false negative leaks an NRIC to a third-party API.

2. Prompt-injection screen: blocks inputs that try to override the system
   prompt ("ignore previous instructions", "reveal your system prompt"...).
   It's a cheap first layer; the main defences are structural (role filter
   before retrieval, role bound server-side, sources treated as data).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

log = logging.getLogger("complibot.guardrails")

NRIC_RE = re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.I)
SG_PHONE_RE = re.compile(r"(?<![\d+])(?:\+65[\s-]?|65[\s-])?[3689]\d{3}[\s-]?\d{4}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

ENTITY_LABELS = {
    "SG_NRIC_FIN": "NRIC", "SG_PHONE": "PHONE", "EMAIL_ADDRESS": "EMAIL", "CREDIT_CARD": "CARD", "PERSON": "PERSON",
    "PHONE_NUMBER": "PHONE",
}
DEFAULT_ENTITIES = ("SG_NRIC_FIN", "SG_PHONE", "EMAIL_ADDRESS", "CREDIT_CARD")

INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions?|rules?|prompts?)\b",
        r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(system|hidden|initial)\s+(prompt|instructions?)\b",
        r"\byou are now\b.{0,40}\b(unrestricted|jailbroken|dan|developer mode)\b",
        r"\b(developer|god|admin)\s+mode\b",
        r"</?(system|source|sources|question)>",  # trying to forge our prompt structure
    )
]


def nric_checksum_ok(value: str) -> bool:
    """Validate the NRIC/FIN check letter (S/T/F/G series; M series accepted on pattern)."""
    v = value.upper()
    if not NRIC_RE.fullmatch(v):
        return False
    prefix, digits, check = v[0], v[1:8], v[8]
    if prefix == "M":
        return True
    total = sum(int(d) * w for d, w in zip(digits, (2, 7, 6, 5, 4, 3, 2), strict=True))
    if prefix in "TG":
        total += 4
    table = "JZIHGFEDCBA" if prefix in "ST" else "XWUTRQPNMLK"
    return table[total % 11] == check


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


@dataclass
class Span:
    entity: str
    start: int
    end: int
    score: float = 1.0


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)  # entity label -> occurrences (no raw values)

    @property
    def redacted(self) -> bool:
        return bool(self.counts)


def apply_spans(text: str, spans: list[Span]) -> RedactionResult:
    """Replace spans with numbered placeholders, resolving overlaps (longest, then highest score)."""
    chosen: list[Span] = []
    for s in sorted(spans, key=lambda s: (-(s.end - s.start), -s.score, s.start)):
        if all(s.end <= c.start or s.start >= c.end for c in chosen):
            chosen.append(s)
    chosen.sort(key=lambda s: s.start)
    numbering: dict[tuple[str, str], str] = {}
    per_label: dict[str, int] = {}
    counts: dict[str, int] = {}
    out, pos = [], 0
    for s in chosen:
        label = ENTITY_LABELS.get(s.entity, s.entity)
        value = re.sub(r"[\s-]", "", text[s.start : s.end]).upper()
        key = (label, value)
        if key not in numbering:
            per_label[label] = per_label.get(label, 0) + 1
            numbering[key] = f"<{label}_{per_label[label]}>"
        counts[label] = counts.get(label, 0) + 1
        out.append(text[pos : s.start])
        out.append(numbering[key])
        pos = s.end
    out.append(text[pos:])
    return RedactionResult("".join(out), counts)


class RegexRedactor:
    name = "regex"

    def __init__(self, entities: tuple[str, ...] = DEFAULT_ENTITIES):
        self.entities = set(entities)

    def find(self, text: str) -> list[Span]:
        spans: list[Span] = []
        if "SG_NRIC_FIN" in self.entities:
            spans += [Span("SG_NRIC_FIN", m.start(), m.end(), 1.0 if nric_checksum_ok(m.group()) else 0.7)
                      for m in NRIC_RE.finditer(text)]
        if "EMAIL_ADDRESS" in self.entities:
            spans += [Span("EMAIL_ADDRESS", m.start(), m.end()) for m in EMAIL_RE.finditer(text)]
        if "SG_PHONE" in self.entities:
            spans += [Span("SG_PHONE", m.start(), m.end(), 0.8) for m in SG_PHONE_RE.finditer(text)]
        if "CREDIT_CARD" in self.entities:
            spans += [Span("CREDIT_CARD", m.start(), m.end()) for m in CARD_RE.finditer(text) if _luhn_ok(m.group())]
        return spans

    def redact(self, text: str) -> RedactionResult:
        return apply_spans(text, self.find(text))


def _offline_tldextract() -> None:
    """Presidio's email recognizer uses tldextract, which downloads the public suffix list on
    first use. A PII guardrail must not make network calls: use the bundled snapshot."""
    try:
        import tldextract
        import tldextract.tldextract as tld

        tld.TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
    except Exception:  # pragma: no cover - tldextract layout changed; worst case it falls back itself
        pass


class PresidioRedactor:
    """Presidio AnalyzerEngine + Singapore recognizers. Entity detection by Presidio, placeholders by us."""

    name = "presidio"

    def __init__(self, entities: tuple[str, ...] = DEFAULT_ENTITIES, spacy_model: str = "en_core_web_sm",
                 threshold: float = 0.4):
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        _offline_tldextract()

        class SgNricRecognizer(PatternRecognizer):
            def validate_result(self, pattern_text: str) -> bool | None:
                # True raises the score to 1.0; None keeps the pattern score (still redacted).
                return True if nric_checksum_ok(pattern_text) else None

        nlp = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": spacy_model}],
        }).create_engine()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp, languages=["en"])
        registry.add_recognizer(SgNricRecognizer(
            supported_entity="SG_NRIC_FIN", patterns=[Pattern("nric", NRIC_RE.pattern, 0.7)],
            context=["nric", "fin", "ic", "identity", "identification"]))
        registry.add_recognizer(PatternRecognizer(
            supported_entity="SG_PHONE", patterns=[Pattern("sg_phone", SG_PHONE_RE.pattern, 0.6)],
            context=["phone", "mobile", "call", "contact", "hp", "tel", "whatsapp"]))
        self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp, supported_languages=["en"])
        self.entities = list(entities)
        self.threshold = threshold

    def redact(self, text: str) -> RedactionResult:
        results = self.analyzer.analyze(text=text, language="en", entities=self.entities,
                                        score_threshold=self.threshold)
        spans = [Span(r.entity_type, r.start, r.end, r.score) for r in results]
        # Belt and braces: anything our NRIC regex sees is redacted even if NER context lowered its score.
        spans += [Span("SG_NRIC_FIN", m.start(), m.end(), 0.7) for m in NRIC_RE.finditer(text)]
        return apply_spans(text, spans)


@lru_cache(maxsize=4)
def get_redactor(engine: str = "auto", entities: tuple[str, ...] = DEFAULT_ENTITIES,
                 spacy_model: str = "en_core_web_sm"):
    """'presidio' | 'regex' | 'auto' (Presidio if installed, else regex). Cached: Presidio is slow to load."""
    if engine in ("auto", "presidio"):
        try:
            return PresidioRedactor(entities, spacy_model)
        except Exception as e:  # ImportError, missing spaCy model (OSError), etc.
            if engine == "presidio":
                raise
            log.warning("Presidio unavailable (%s); falling back to regex PII redaction", e)
    return RegexRedactor(entities)


def detect_injection(text: str) -> str | None:
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return "possible prompt injection"
    return None


@dataclass
class InputCheck:
    text: str  # redacted text: the ONLY version that flows onward
    pii: dict[str, int]
    blocked: str | None = None


class Guardrails:
    def __init__(self, redactor=None, block_injection: bool = True, redact_output: bool = True):
        self.redactor = redactor
        self.block_injection = block_injection
        self.redact_output_enabled = redact_output

    def check_input(self, text: str) -> InputCheck:
        res = self.redactor.redact(text) if self.redactor else RedactionResult(text)
        blocked = detect_injection(res.text) if self.block_injection else None
        return InputCheck(text=res.text, pii=res.counts, blocked=blocked)

    def redact_output(self, text: str) -> str:
        if not (self.redactor and self.redact_output_enabled):
            return text
        return self.redactor.redact(text).text

    @classmethod
    def from_settings(cls, settings) -> Guardrails:
        if settings.pii_engine == "off":
            return cls(redactor=None, block_injection=settings.block_injection)
        entities = DEFAULT_ENTITIES + (("PERSON",) if settings.pii_person else ())
        return cls(redactor=get_redactor(settings.pii_engine, entities, settings.spacy_model),
                   block_injection=settings.block_injection)
