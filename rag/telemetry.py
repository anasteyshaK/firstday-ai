"""Privacy-conscious, optional Langfuse v4 tracing for FirstDay AI.

Only operational metadata is sent. User questions, model prompts, generated
answers and private onboarding records are deliberately NOT included.

Tracing errors must never break the application.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("firstday-telemetry")


def _load_environment() -> None:
    """Load the same project-root .env from API and RAG invocations."""
    root = Path(__file__).resolve().parents[1]
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass


@lru_cache(maxsize=1)
def _client():
    _load_environment()

    enabled = os.getenv("LANGFUSE_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None

    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        return None

    try:
        from langfuse import get_client
        return get_client()
    except Exception:
        logger.warning("Langfuse initialization failed; tracing disabled", exc_info=True)
        return None


def is_enabled() -> bool:
    """Return whether a Langfuse client is configured and available."""
    return _client() is not None


class Observation:
    """A safe wrapper around an SDK observation (or a no-op)."""

    def __init__(self, raw: Any = None):
        self._raw = raw
        self._ended = False

    @property
    def child_context(self) -> dict[str, str] | None:
        if self._raw is None:
            return None
        try:
            trace_id = self._raw.trace_id
            span_id = self._raw.id
            if trace_id and span_id:
                return {
                    "trace_id": trace_id,
                    "parent_span_id": span_id,
                }
        except Exception:
            logger.debug("Cannot read Langfuse trace context", exc_info=True)
        return None

    def update(self, **kwargs: Any) -> None:
        if self._raw is None or self._ended:
            return
        try:
            self._raw.update(**kwargs)
        except Exception:
            logger.warning("Langfuse span update failed", exc_info=True)

    def mark_error(self, error: BaseException | str) -> None:
        """Record only the error class, never a potentially sensitive message."""
        error_type = error if isinstance(error, str) else type(error).__name__
        self.update(level="ERROR", status_message=f"{error_type}")

    def end(self) -> None:
        if self._raw is None or self._ended:
            return
        self._ended = True
        try:
            self._raw.end()
        except Exception:
            logger.warning("Langfuse span end failed", exc_info=True)


def start_observation(
    name: str,
    *,
    parent: dict[str, str] | None = None,
    as_type: str = "span",
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> Observation:
    """Create a manual observation, safe across streamed generator yields.

    `parent` explicitly links children to a request trace, avoiding
    OpenTelemetry context manager tokens crossing worker threads.
    """
    client = _client()
    if client is None:
        return Observation()

    kwargs: dict[str, Any] = {"name": name, "as_type": as_type}
    if parent is not None:
        kwargs["trace_context"] = parent
    if metadata is not None:
        kwargs["metadata"] = metadata
    if as_type == "generation" and model is not None:
        kwargs["model"] = model

    try:
        return Observation(client.start_observation(**kwargs))
    except Exception:
        logger.warning("Failed to start Langfuse observation", exc_info=True)
        return Observation()


@contextmanager
def observed(
    name: str,
    *,
    parent: dict[str, str] | None = None,
    as_type: str = "span",
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> Iterator[Observation]:
    """Use only around operations that do not yield across threads."""
    span = start_observation(
        name,
        parent=parent,
        as_type=as_type,
        metadata=metadata,
        model=model,
    )
    try:
        yield span
    except BaseException as error:
        span.mark_error(error)
        raise
    finally:
        span.end()


def flush() -> None:
    """Only needed for diagnostics and short-lived scripts."""
    client = _client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            logger.warning("Langfuse flush failed", exc_info=True)
