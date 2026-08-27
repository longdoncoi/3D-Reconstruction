"""Data Governance: PII Scrubbing and Privacy."""
import re
from typing import Any

# Basic patterns for sensitive data
_PII_PATTERNS = [
    # API Keys / Tokens (very simplistic heuristics)
    (r"(?i)(?:api_key|apikey|token|secret|password|passwd|pwd)\s*[:=]\s*[\"']?[a-zA-Z0-9_\-\.]{10,}[\"']?", "[REDACTED_SECRET]"),
    # IPv4 Addresses (ignoring local/internal for simplicity, masking all)
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
