from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from .governance import redact_value
from .security import DataClassification


class SideEffect(StrEnum):
    READ = "read"
    DESKTOP = "desktop"
    WRITE = "write"
    EXECUTE = "execute"


#: The provenance id of the built-in catalogue.  It is always allowed: the
#: third-party ``plugins.toml`` allow-list decides which plugin distributions
#: may register, never whether the product's own tools exist.
BUILTIN_PLUGIN_ID = "builtin.legacy-tools"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    timeout_seconds: int
    side_effect: SideEffect = SideEffect.READ
    requires_approval: bool = False
    required_scope: str | None = None
    maximum_classification: DataClassification = DataClassification.RESTRICTED
    plugin_id: str = BUILTIN_PLUGIN_ID


@dataclass(frozen=True, slots=True)
class ToolRequest:
    tool_name: str
    parameters: Mapping[str, Any]
    correlation_id: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    success: bool
    payload: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.payload)
        result.setdefault("success", self.success)
        if self.error_code:
            result["error_code"] = self.error_code
        if self.message:
            result.setdefault("error", self.message)
        return result


#: Evidence fields per tool. A *successful* result for a listed tool only
#: counts as usable evidence when at least one of these fields carries
#: non-blank content. One table shared by the supervisor's independent
#: verification (``verify_result``) and the coding-workflow classifier so the
#: two views of "did the tool really prove anything" cannot drift apart.
EVIDENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "read_file": ("content",),
    "search_text": ("results",),
    "find_files": ("matches",),
    "list_directory": ("entries",),
    "analyze_code": ("functions", "classes", "includes"),
    "git_diff": ("content",),
}


def tool_result_succeeded(result: object) -> bool:
    """Return whether a tool result represents a successful operation.

    Tool executors share a small result contract. Absence of ``error`` is not
    sufficient: a command can finish with a non-zero exit code and a mutation
    can explicitly report ``success: false``. Keeping the check in the domain
    makes it independent of any particular feature or tool caller.
    """
    if not isinstance(result, dict) or result.get("error"):
        return False
    if result.get("success") is False:
        return False
    return cast(bool, result.get("return_code", 0) == 0)


def tool_result_has_evidence(tool: str, result: object) -> bool:
    """Return whether a successful result contains usable evidence.

    Unlisted tools are trusted on success alone; listed tools must carry at
    least one non-blank evidence field (``" "`` content is not evidence).
    """
    if not isinstance(result, dict) or not tool_result_succeeded(result):
        return False
    fields = EVIDENCE_FIELDS.get(tool)
    if fields is None:
        return True
    return any(_field_has_evidence(result.get(field)) for field in fields)


def _field_has_evidence(value: object) -> bool:
    return bool(value.strip()) if isinstance(value, str) else bool(value)


#: Result keys that carry the tool's *payload* — the document it read, the
#: diff it produced, the rows it found — rather than a fact about the
#: execution. A log record states what the tool did; it does not copy the
#: user's file into a plaintext log file next to the ciphertext the very same
#: content is sealed into (ADR 0032).
PAYLOAD_KEYS = frozenset({
    "content", "results", "matches", "entries", "text", "output",
    "stdout", "stderr", "diff", "body", "data", "patch",
})


def describe_tool_result(result: object) -> str:
    """Summarise a tool result for a log line, without its payload.

    The result itself reaches the caller — and the sealed task event —
    intact; only the *log* receives this summary. A tool that read a private
    document therefore leaves a record of *what* it read and *how much* of it,
    not a copy of the document.

    Scalar facts (a return code, an error code, a path) survive; values are
    passed through the shared redaction policy so a secret that a tool echoed
    back under an innocuous key is masked here too.
    """
    if not isinstance(result, dict):
        return type(result).__name__
    parts: list[str] = []
    for key in sorted(str(item) for item in result):
        value = result[key]
        if key in PAYLOAD_KEYS:
            parts.append(f"{key}=<{_payload_size(value)}>")
        elif isinstance(value, (dict, list, tuple)):
            parts.append(f"{key}=<{len(value)} items>")
        else:
            parts.append(f"{key}={redact_value(value)}")
    return " ".join(parts)


def _payload_size(value: object) -> str:
    if isinstance(value, str):
        return f"{len(value)} chars"
    if isinstance(value, (list, tuple, dict)):
        return f"{len(value)} items"
    return type(value).__name__
