"""Shared HTTP client with an honest User-Agent and polite retries."""

from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "ripple-research/0.1 (student research project; contact via GitHub profile)"


def client(timeout: float = 60.0, **kw) -> httpx.Client:
    headers = {"User-Agent": USER_AGENT, **kw.pop("headers", {})}
    return httpx.Client(timeout=timeout, headers=headers, follow_redirects=True, **kw)


def get_with_retry(
    c: httpx.Client, url: str, *, retries: int = 3, backoff: float = 5.0, **kw
) -> httpx.Response:
    """GET with exponential backoff on 5xx / network errors. 4xx are returned to the caller."""
    for attempt in range(retries + 1):
        try:
            r = c.get(url, **kw)
        except httpx.TransportError as e:
            if attempt == retries:
                raise
            log.warning("transport error %s on %s, retrying", e, url)
        else:
            if r.status_code < 500:
                return r
            if attempt == retries:
                return r
            log.warning("HTTP %s on %s, retrying", r.status_code, url)
        time.sleep(backoff * 2**attempt)
    raise RuntimeError("unreachable")
