"""Optional Langfuse tracing behind a tiny wrapper.

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (and the
`langfuse` package is installed); otherwise every call is a no-op, so tracing
can never break a request. The Langfuse client is given our PII redactor as its
`mask` function: even if raw text slipped into a trace, it is scrubbed before
leaving the process.

    with tracer.observe("retrieve", as_type="retriever", input={...}) as obs:
        ...
        obs.update(output={...})
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger("complibot.tracing")


class _NoopObservation:
    def update(self, **kwargs: Any) -> None:
        pass


class Tracer:
    def __init__(self, client=None):
        self.client = client

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @contextmanager
    def observe(self, name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
        if self.client is None:
            yield _NoopObservation()
            return
        try:
            cm = self.client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
            obs = cm.__enter__()
        except Exception as e:  # tracing must never take the request down
            log.warning("tracing disabled for %s: %s", name, e)
            yield _NoopObservation()
            return
        exc_info = (None, None, None)
        try:
            yield obs
        except BaseException as e:
            exc_info = (type(e), e, e.__traceback__)
            raise
        finally:
            try:
                cm.__exit__(*exc_info)
            except Exception as e:  # pragma: no cover
                log.warning("tracing exit failed for %s: %s", name, e)

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception as e:  # pragma: no cover
                log.warning("langfuse flush failed: %s", e)


def _mask_with(redactor):
    def mask(*, data: Any, **_: Any) -> Any:
        if isinstance(data, str):
            return redactor.redact(data).text
        if isinstance(data, dict):
            return {k: mask(data=v) for k, v in data.items()}
        if isinstance(data, list):
            return [mask(data=v) for v in data]
        return data

    return mask


def get_tracer(redactor=None) -> Tracer:
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return Tracer(None)
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning("LANGFUSE_* set but the langfuse package is not installed; tracing off")
        return Tracer(None)
    kwargs: dict[str, Any] = {}
    if redactor is not None:
        kwargs["mask"] = _mask_with(redactor)
    return Tracer(Langfuse(**kwargs))  # reads keys and LANGFUSE_HOST/BASE_URL from env
