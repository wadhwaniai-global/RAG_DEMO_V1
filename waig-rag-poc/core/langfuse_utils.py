"""Langfuse observability helpers.

This module centralizes Langfuse client initialization so the rest of the
codebase can safely interact with the tracing provider without duplicating
configuration logic. The client is optional and only instantiated when
credentials are present in the application settings.
"""

from __future__ import annotations

import atexit
import logging
from functools import lru_cache
from typing import Optional

from langfuse import Langfuse

from config import settings

logger = logging.getLogger(__name__)


def _build_client_kwargs() -> Optional[dict[str, object]]:
    """Prepare keyword arguments for Langfuse client construction."""
    if not settings.langfuse_enabled:
        logger.info("Langfuse tracing disabled: credentials not provided")
        return None

    kwargs: dict[str, object] = {
        "secret_key": settings.langfuse_secret_key,
        "public_key": settings.langfuse_public_key,
        "debug": settings.langfuse_debug,
    }

    if settings.langfuse_host:
        kwargs["host"] = settings.langfuse_host

    if settings.langfuse_sample_rate is not None:
        kwargs["sample_rate"] = settings.langfuse_sample_rate

    if settings.langfuse_tracing_enabled is not None:
        kwargs["tracing_enabled"] = settings.langfuse_tracing_enabled

    return kwargs


@lru_cache(maxsize=1)
def get_langfuse_client() -> Optional[Langfuse]:
    """Return a cached Langfuse client if credentials are available."""
    kwargs = _build_client_kwargs()
    if kwargs is None:
        return None

    client = Langfuse(**kwargs)

    # Ensure queued events are flushed on shutdown for short-lived processes.
    atexit.register(client.flush)
    logger.info("Langfuse tracing is enabled with host=%s", kwargs.get("host", "default"))
    return client


# Initialise client eagerly so that background flushing is registered for long-running apps.
get_langfuse_client()
