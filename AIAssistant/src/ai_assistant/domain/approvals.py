"""HITL approval capability: a grant is bound to one exact tool invocation.

The gateway must be able to answer "was *this* tool call, with *these*
parameters, approved by a user?" without consulting free-form booleans that any
caller could set.  The answer is a capability pair:

* ``approval_fingerprint(tool, params)`` — canonical digest of the invocation
  the preview showed to the user;
* an opaque grant token issued by an approval path and validated by
  :class:`~ai_assistant.application.tools.ToolExecutionService`.

Pure stdlib: no transport, no framework.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

__all__ = ["approval_fingerprint"]


def approval_fingerprint(tool_name: str, parameters: Mapping[str, Any] | None = None) -> str:
    """Stable digest of ``(tool, params)`` for grant binding.

    Keys are sorted so an equivalent parameter dictionary produced by the
    engine always maps to the same fingerprint as the one shown in the
    approval preview.
    """
    canonical = json.dumps(
        {"tool": tool_name, "params": dict(parameters or {})},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
