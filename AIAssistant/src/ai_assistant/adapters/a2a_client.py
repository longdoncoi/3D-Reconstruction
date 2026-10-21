"""Trusted A2A client adapter.

The adapter deliberately has no local execution fallback.  A rejected card,
transport failure, or policy violation remains a structured routing result for
the caller to surface or revise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..domain.governance import has_sensitive_content, scrub_credentials
from ..domain.security import DataClassification
from ..observability.tracing import current_traceparent

#: Cards and task envelopes are small JSON documents; a response bigger than
#: this is an attack or a broken peer, and reading it whole is a memory DoS.
_MAX_RESPONSE_BYTES = 1_048_576  # 1 MiB


class _RefuseRedirects(HTTPRedirectHandler):
    """Never follow a redirect on the A2A transport.

    A trusted endpoint that answers 30x points at a destination the allowlist
    never approved: following it would turn one compromised peer into SSRF
    against anything on the host's network. The refusal surfaces as an
    ``HTTPError`` (a ``URLError``), which both call sites already degrade on.
    """

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        # The parameters are the handler contract; none of them are read here.
        return None


_OPENER = build_opener(_RefuseRedirects())
#: The single transport seam. Production always goes through the no-redirect
#: opener above; tests substitute this name.
urlopen = _OPENER.open


def _read_bounded(response: Any) -> bytes:
    """Read at most ``_MAX_RESPONSE_BYTES``; refuse anything larger."""
    body: bytes = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(body) > _MAX_RESPONSE_BYTES:
        msg = f"A2A response exceeds {_MAX_RESPONSE_BYTES} bytes"
        raise ValueError(msg)
    return body


@dataclass(frozen=True, slots=True)
class RemoteAgentCard:
    endpoint: str
    name: str
    version: str
    skills: frozenset[str]
    streaming: bool


class TrustedA2AClient:
    """Discovers and calls only endpoints provided by deployment config."""

    def __init__(self, trusted_endpoints: frozenset[str], timeout_seconds: float = 10.0) -> None:
        self._trusted_endpoints = frozenset(endpoint.rstrip("/") for endpoint in trusted_endpoints)
        self._timeout_seconds = timeout_seconds

    def discover(self) -> dict[str, RemoteAgentCard]:
        cards: dict[str, RemoteAgentCard] = {}
        for endpoint in self._trusted_endpoints:
            try:
                cards[endpoint] = self._read_card(endpoint)
            except (OSError, ValueError, URLError):
                continue
        return cards

    def submit(self, endpoint: str, capability: str, message: str, *, metadata: dict[str, Any] | None = None,
               context_id: str | None = None,
               classification: DataClassification = DataClassification.INTERNAL) -> dict[str, Any]:
        endpoint = endpoint.rstrip("/")
        if endpoint not in self._trusted_endpoints:
            return {"routing": "rejected_policy", "error": "A2A endpoint is not trusted"}
        if classification in {DataClassification.RESTRICTED, DataClassification.REGULATED}:
            return {"routing": "rejected_policy", "error": "Sensitive data cannot leave the local trust boundary"}
        # Defence in depth at the last hop before the wire: the caller is
        # expected to have classified the payload, but *this* is the boundary
        # that owns the rule, so it re-asks the question rather than trusting
        # the caller to have asked it. A payload to be *executed* is refused,
        # never rewritten — masking it would hand the peer a falsified
        # instruction to run.
        if has_sensitive_content(metadata):
            return {"routing": "rejected_policy", "error": "Sensitive data cannot leave the local trust boundary"}
        card = self._read_card(endpoint)
        if capability not in card.skills:
            return {"routing": "rejected_no_eligible_agent", "error": f"Remote agent lacks {capability}"}
        payload = {
            "capability": capability,
            "context_id": context_id,
            "metadata": metadata or {},
            "classification": classification.value,
            # The message is the user's own prompt: free text that no key
            # name describes, so a credential typed into it is masked rather
            # than refused — the peer never needed it.
            "message": {"role": "user", "parts": [{"kind": "text", "text": scrub_credentials(message)}]},
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        # Propagate the active trace so a remote agent's spans can join this
        # trace (outbound half of W3C propagation; inbound extraction is the
        # receiver's job and is not wired on our own inbound A2A routes yet).
        traceparent = current_traceparent()
        if traceparent:
            headers["traceparent"] = traceparent
        request = Request(
            urljoin(endpoint + "/", "a2a/tasks/send"), data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                result = json.loads(_read_bounded(response).decode("utf-8"))
        except (OSError, URLError, ValueError) as error:
            # ValueError covers both the JSON decoder and the response-size
            # refusal: a malformed or oversized answer is a degraded peer,
            # not an exception the caller has to survive.
            return {"routing": "degraded_dependency", "error": str(error), "agent_url": endpoint}
        return {"routing": "remote_selected", "agent_url": endpoint, **result}

    def _read_card(self, endpoint: str) -> RemoteAgentCard:
        request = Request(urljoin(endpoint + "/", ".well-known/agent.json"), headers={"Accept": "application/json"})
        with urlopen(request, timeout=self._timeout_seconds) as response:
            payload = json.loads(_read_bounded(response).decode("utf-8"))
        if not isinstance(payload.get("name"), str) or not isinstance(payload.get("skills"), list):
            # ValueError (not TypeError): discover() skips bad cards by catching it.
            msg = "Invalid A2A Agent Card"
            raise ValueError(msg)  # noqa: TRY004
        skills = frozenset(
            str(skill["id"]) for skill in payload["skills"] if isinstance(skill, dict) and isinstance(skill.get("id"), str)
        )
        if not skills:
            # ValueError (not TypeError): discover() skips bad cards by catching it.
            msg = "A2A Agent Card contains no capabilities"
            raise ValueError(msg)
        return RemoteAgentCard(
            endpoint=endpoint, name=payload["name"], version=str(payload.get("version", "unknown")), skills=skills,
            streaming=bool(payload.get("capabilities", {}).get("streaming", False)),
        )
