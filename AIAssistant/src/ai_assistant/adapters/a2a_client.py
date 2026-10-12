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
from urllib.request import Request, urlopen

from ..domain.security import DataClassification


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
        card = self._read_card(endpoint)
        if capability not in card.skills:
            return {"routing": "rejected_no_eligible_agent", "error": f"Remote agent lacks {capability}"}
        payload = {
            "capability": capability,
            "context_id": context_id,
            "metadata": metadata or {},
            "classification": classification.value,
            "message": {"role": "user", "parts": [{"kind": "text", "text": message}]},
        }
        request = Request(
            urljoin(endpoint + "/", "a2a/tasks/send"), data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as error:
            return {"routing": "degraded_dependency", "error": str(error), "agent_url": endpoint}
        return {"routing": "remote_selected", "agent_url": endpoint, **result}

    def _read_card(self, endpoint: str) -> RemoteAgentCard:
        request = Request(urljoin(endpoint + "/", ".well-known/agent.json"), headers={"Accept": "application/json"})
        with urlopen(request, timeout=self._timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
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
