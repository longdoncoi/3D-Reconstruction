"""Data governance: PII scrubbing and data privacy policies."""
from __future__ import annotations

import re
from typing import Any

# Basic patterns for sensitive data
_PII_PATTERNS = [
    # API Keys / Tokens (heuristic masking)
    (r"(?i)(?:api_key|apikey|token|secret|password|passwd|pwd)\s*[:=]\s*[\"']?[a-zA-Z0-9_\-\.]{10,}[\"']?", "[REDACTED_SECRET]"),
    # IPv4 Addresses
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED_IP]"),
    # Emails
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "[REDACTED_EMAIL]"),
]


def scrub_pii(text: str) -> str:
    """Mask sensitive information in text before sending to external LLM or logs."""
    if not isinstance(text, str):
        return text

    scrubbed = text
    for pattern, replacement in _PII_PATTERNS:
        scrubbed = re.sub(pattern, replacement, scrubbed)
    return scrubbed


def scrub_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scrub PII from a list of chat messages."""
    scrubbed_messages = []
    for msg in messages:
        new_msg = dict(msg)
        if "content" in new_msg and isinstance(new_msg["content"], str):
            new_msg["content"] = scrub_pii(new_msg["content"])
        scrubbed_messages.append(new_msg)
    return scrubbed_messages


# Keys whose values are masked before a payload is logged or persisted as a
# task event. ``continuation`` carries the internal HITL execution context
# (action id + tool params) and session/request correlation ids are included so
# a log reader cannot join tasks across sessions. These are *redaction* keys,
# not an egress policy: a session id is not a credential and must not block a
# remote delegation.
REDACTION_KEYS = frozenset({
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "authorization", "access_token", "refresh_token", "cookie",
    "continuation", "action_id", "approval_scope",
    "session_id", "request_id", "idempotency_key", "context_id", "contextid",
})

# Backwards-compatible name for the redaction set. New code should use
# ``REDACTION_KEYS`` so the distinction from the egress set stays explicit.
SENSITIVE_KEYS = REDACTION_KEYS

# Keys that are credentials (or capability material) and must never cross the
# trust boundary. Correlation identifiers deliberately do not appear here:
# ``session_id``/``idempotency_key`` travel with a remote delegation so the peer
# can deduplicate and route it, while ``action_id``/``continuation`` would let a
# caller forge an approval and stay egress-restricted.
EGRESS_SECRET_KEYS = frozenset({
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "authorization", "access_token", "refresh_token", "cookie",
    "continuation", "action_id", "approval_scope",
})


def redact_value(value: Any, key: str = "") -> Any:
    """Recursively replace sensitive values with ``[redacted]``.

    Single owner of the transport redaction policy: A2A payloads, durable task
    events and log scrubbing all delegate here so a new sensitive key is added
    in exactly one place. String values are additionally passed through
    :func:`scrub_pii` so a secret pasted under a non-sensitive key
    (``query``/``content``/``message``) is still masked in logs and payloads;
    dispatch refusal (not rewriting) stays the answer for outbound execution
    via :func:`has_sensitive_keys`.
    """
    if key.casefold() in REDACTION_KEYS:
        return "[redacted]"
    if isinstance(value, dict):
        return {str(item_key): redact_value(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return scrub_pii(value)
    return value


def has_sensitive_keys(value: Any, key: str = "") -> bool:
    """True when this payload carries material that must not leave the process.

    Redaction is for material that may still be *useful* once masked; a payload
    being sent somewhere to be executed is not. An outbound dispatch carrying
    a credential is therefore refused instead of rewritten — masking it would
    hand the remote peer a falsified payload to run, and sending it as-is
    would put the secret on the wire.

    Correlation identifiers (``session_id``/``request_id``/``idempotency_key``)
    are redacted in logs and task events but are **not** egress secrets; they
    travel with a delegation so the peer can route and deduplicate it.

    This is the *key* rule. :func:`has_sensitive_content` is the wider one and
    is what the egress boundary uses; both live here so the domain owns both
    rules (ADR 0032: one owner, two named rules).
    """
    if key.casefold() in EGRESS_SECRET_KEYS:
        return True
    if isinstance(value, dict):
        return any(has_sensitive_keys(item_value, str(item_key)) for item_key, item_value in value.items())
    if isinstance(value, (list, tuple)):
        return any(has_sensitive_keys(item) for item in value)
    return False


#: A value that carries a credential *in its own shape*, wherever it appears.
#:
#: These patterns are deliberately unambiguous. A prose pattern ("the password
#: is stored in the vault") cannot be told apart from a real secret, and an
#: egress boundary that refused on prose would refuse ordinary requests — so
#: the value rule only fires on an assignment with a long enough value, or on
#: a token format that carries its own marker.
_EGRESS_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?keys?|apikeys?|secrets?|passwords?|passwd|pwd|"
    r"passphrases?|access[_-]?tokens?|refresh[_-]?tokens?|tokens?|"
    r"credentials?|authorization)\b"
    r"\s*[:=]\s*[\"']?([A-Za-z0-9_\-\.\+/=]{8,})"
)

_EGRESS_TOKEN_SHAPE = re.compile(
    r"(?i)(?:"
    r"\bsk-[A-Za-z0-9_\-]{16,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{16,}"
    r"|\bxox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bAIza[0-9A-Za-z_\-]{35}\b"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
    r")"
)


def carries_credential_shape(value: str) -> bool:
    """True when a free-text value carries a credential in its own shape."""
    return bool(_EGRESS_ASSIGNMENT.search(value) or _EGRESS_TOKEN_SHAPE.search(value))


def scrub_credentials(text: str) -> str:
    """Mask credential-shaped material in free text.

    The *message* of an A2A delegation is the user's own prompt, and it
    crosses the trust boundary in a place no key name describes. Redaction is
    the right tool for free text (the peer never needed the user's password),
    so the shape-based patterns are applied here — and only here, where the
    input is prose that may legitimately mention the word "password".
    """
    if not isinstance(text, str):
        return text
    scrubbed = _EGRESS_TOKEN_SHAPE.sub("[REDACTED_SECRET]", text)
    return _EGRESS_ASSIGNMENT.sub("[REDACTED_SECRET]", scrubbed)


def has_sensitive_content(value: Any, key: str = "") -> bool:
    """True when this payload carries material that must not leave the process.

    Wider than :func:`has_sensitive_keys`, which asks only about the *key*.
    A secret pasted into a free-text value (``{"note": "api_key=sk-…"}``)
    carries no sensitive key at all, so the key-only rule classified it
    ``internal`` and shipped it to the peer in clear — the user's own prompt
    travels as the A2A message, and a credential typed into it took the same
    path. This asks both questions: the key name, and the shape of the value.
    """
    if has_sensitive_keys(value, key):
        return True
    if isinstance(value, str):
        return carries_credential_shape(value)
    if isinstance(value, dict):
        return any(has_sensitive_content(item_value, str(item_key)) for item_key, item_value in value.items())
    if isinstance(value, (list, tuple)):
        return any(has_sensitive_content(item) for item in value)
    return False


__all__ = [
    "EGRESS_SECRET_KEYS",
    "REDACTION_KEYS",
    "SENSITIVE_KEYS",
    "carries_credential_shape",
    "has_sensitive_content",
    "has_sensitive_keys",
    "redact_value",
    "scrub_credentials",
    "scrub_messages",
    "scrub_pii",
]
