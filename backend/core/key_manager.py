"""
API key rotation for Google Gemini.

Loads GOOGLE_API_KEY (primary) and GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, ...
(backups) from environment. On 429 (quota) or 503 (overload) errors, the
caller can ask for the next key via rotate_on_error().

Usage in a single call:
    from backend.core.key_manager import call_with_key_rotation
    result = call_with_key_rotation(my_fn, arg1, arg2)

Usage manually:
    key = get_api_key()          # always primary
    key = get_api_key(index=1)   # backup #1
    keys = get_all_keys()        # all keys in order
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {"503", "429", "429 RESOURCE_EXHAUSTED", "503 SERVICE_UNAVAILABLE"}


def get_all_keys() -> list[str]:
    """Return all configured API keys in order (primary first)."""
    keys = []
    primary = os.environ.get("GOOGLE_API_KEY", "")
    if primary:
        keys.append(primary)
    i = 2
    while True:
        k = os.environ.get(f"GOOGLE_API_KEY_{i}", "")
        if not k:
            break
        keys.append(k)
        i += 1
    return keys


def get_api_key(index: int = 0) -> str:
    """Return the Nth key (wraps around if index >= total keys)."""
    keys = get_all_keys()
    if not keys:
        raise RuntimeError("No GOOGLE_API_KEY configured in environment")
    return keys[index % len(keys)]


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc)
    return any(code in msg for code in _RETRYABLE_CODES)


def call_with_key_rotation(fn, *args, retry_delay: float = 5.0, **kwargs):
    """
    Call fn(*args, **kwargs) cycling through all configured API keys on failure.
    fn must accept a `api_key` keyword argument.
    Raises the last exception if all keys are exhausted.
    """
    keys = get_all_keys()
    if not keys:
        raise RuntimeError("No GOOGLE_API_KEY configured")

    last_exc = None
    for i, key in enumerate(keys):
        try:
            return fn(*args, api_key=key, **kwargs)
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and i < len(keys) - 1:
                logger.warning(
                    "Key %d failed (%s), rotating to key %d...", i + 1, exc, i + 2
                )
                time.sleep(retry_delay)
            else:
                raise
    raise last_exc
