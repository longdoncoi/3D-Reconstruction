"""A2A (Agent-to-Agent) protocol layer for the unified assistant.

Provides Agent Card discovery, optional remote agent routing, and a standard
JSON-RPC 2.0 transport that falls back to in-process execution when the A2A
SDK is unavailable or remote agents are not configured.

Environment variables
---------------------
A2A_ENABLED         "1" to activate the A2A transport layer (default "0").
A2A_REMOTE_AGENTS   Comma-separated list of remote ``/.well-known/agent.json``
                    URLs. When empty, all delegations stay in-process.
A2A_CARD_NAME       Display name for *this* server's published Agent Card.
A2A_CARD_URL        Public base URL where this server is reachable.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urljoin

from ai_assistant.config.logging import get_agent_logger

logger = get_agent_logger("a2a")

# ── Environment configuration ────────────────────────────────────────────────

A2A_ENABLED = os.getenv("A2A_ENABLED", "0") == "1"
A2A_REMOTE_AGENTS: list[str] = [
    url.strip()
    for url in os.getenv("A2A_REMOTE_AGENTS", "").split(",")
    if url.strip()
]
A2A_CARD_NAME = os.getenv("A2A_CARD_NAME", "3D-Reconstruction AI Assistant")
A2A_CARD_URL = os.getenv("A2A_CARD_URL", "http://127.0.0.1:8080")

# ── Optional SDK import ──────────────────────────────────────────────────────

_sdk_available = False
try:
    if A2A_ENABLED:
        from a2a.types import AgentCard as _SdkAgentCard  # noqa: F401
        _sdk_available = True
        logger.info("a2a-sdk loaded successfully")
except ImportError:
    logger.info("a2a-sdk not installed — A2A layer operates in local-only mode")


def a2a_available() -> bool:
    """Return True when the A2A transport can be used."""
    return A2A_ENABLED and _sdk_available


# ── Agent Card (data model) ──────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentSkill:
    """One capability that an agent advertises in its Agent Card."""
    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentCard:
    """A2A-compatible Agent Card describing this server's capabilities."""
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    protocolVersion: str = "0.3.0"
    skills: list[AgentSkill] = field(default_factory=list)
    capabilities: dict[str, bool] = field(default_factory=lambda: {
        "streaming": True,
        "pushNotifications": False,
    })
    defaultInputModes: list[str] = field(default_factory=lambda: ["text"])
    defaultOutputModes: list[str] = field(default_factory=lambda: ["text"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def build_agent_card(url: str | None = None) -> AgentCard:
    """Build the server's Agent Card from the live Specialist registry."""
    from ai_assistant.orchestration.supervisor import _SPECIALIST_INSTRUCTIONS, Specialist

    skills: list[AgentSkill] = []
    for specialist in Specialist:
        instruction = _SPECIALIST_INSTRUCTIONS.get(specialist, "")
        skills.append(AgentSkill(
            id=specialist.value,
            name=specialist.value.replace("_", " ").title(),
            description=instruction,
            tags=[specialist.value],
        ))

    return AgentCard(
        name=A2A_CARD_NAME,
        description=(
            "Multi-agent AI assistant for the 3D-Reconstruction Qt application. "
            "Supports chatbot, code, research, verification, and desktop workflow specialists."
        ),
        url=url or A2A_CARD_URL,
        skills=skills,
    )


# ── Remote Agent Discovery ───────────────────────────────────────────────────


@dataclass
class RemoteAgent:
    """A discovered remote agent with its capabilities."""
    url: str
    card: dict[str, Any]
    skills: dict[str, dict[str, Any]]  # skill_id → skill dict
    last_discovered: float = 0.0


_remote_registry: dict[str, RemoteAgent] = {}
_DISCOVERY_TIMEOUT = 5.0  # seconds


def discover_remote_agents(urls: list[str] | None = None) -> dict[str, RemoteAgent]:
    """Fetch Agent Cards from remote endpoints and populate the registry."""
    targets = urls if urls is not None else A2A_REMOTE_AGENTS
    if not targets:
        return _remote_registry

    try:
        import urllib.request
    except ImportError:
        logger.warning("urllib.request unavailable — cannot discover remote agents")
        return _remote_registry

    for base_url in targets:
        card_url = (base_url if base_url.rstrip("/").endswith(".well-known/agent.json")
                    else urljoin(base_url.rstrip("/") + "/", ".well-known/agent.json"))
        try:
            req = urllib.request.Request(card_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=_DISCOVERY_TIMEOUT) as resp:
                card_data = json.loads(resp.read().decode("utf-8"))
            skills = {}
            for skill in card_data.get("skills", []):
                skill_id = skill.get("id", "")
                if skill_id:
                    skills[skill_id] = skill
            _remote_registry[base_url] = RemoteAgent(
                url=str(card_data.get("url") or base_url), card=card_data, skills=skills,
                last_discovered=time.time(),
            )
            logger.info("Discovered remote agent: %s (%d skills)", base_url, len(skills))
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to discover agent at %s: %s", card_url, error)

    return _remote_registry


def get_remote_registry() -> dict[str, RemoteAgent]:
    """Return the current remote agent registry (read-only view)."""
    return dict(_remote_registry)


# ── A2A Router ───────────────────────────────────────────────────────────────


class A2ARouter:
    """Route delegations to remote A2A agents without implicit fallback."""

    def find_remote(self, specialist_id: str) -> RemoteAgent | None:
        """Find a remote agent whose skills include ``specialist_id``."""
        for agent in _remote_registry.values():
            if specialist_id in agent.skills:
                return agent
        return None

    def route(self, specialist_id: str, task: str,
              params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Attempt remote A2A dispatch and return its explicit outcome."""
        if not a2a_available():
            return {
                "source": "remote",
                "status": "unavailable",
                "error": "A2A transport is disabled or unavailable",
            }

        remote = self.find_remote(specialist_id)
        if remote is None:
            return {
                "source": "remote",
                "status": "unavailable",
                "error": f"No trusted remote agent supports {specialist_id}",
            }

        try:
            result = self._call_remote(remote, specialist_id, task, params or {})
            logger.info("A2A remote call succeeded: specialist=%s url=%s",
                        specialist_id, remote.url)
            return {"source": "remote", "agent_url": remote.url, **result}
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "A2A remote call failed (specialist=%s url=%s): %s — falling back to local",
                specialist_id, remote.url, error,
            )
            return {
                "source": "remote",
                "status": "failed",
                "agent_url": remote.url,
                "error": str(error),
            }

    @staticmethod
    def _call_remote(agent: RemoteAgent, specialist_id: str,
                     task: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request to a remote A2A agent."""
        import urllib.request

        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "id": f"{specialist_id}-{int(time.time() * 1000)}",
            "params": {
                "id": f"task-{int(time.time() * 1000)}",
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": task}],
                },
                "metadata": params,
            },
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            agent.url.rstrip("/") + "/",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            response = json.loads(resp.read().decode("utf-8"))

        if "error" in response:
            raise RuntimeError(f"A2A error: {response['error']}")
        return response.get("result", {})


__all__ = [
    "A2A_CARD_NAME",
    "A2A_CARD_URL",
    "A2A_ENABLED",
    "A2A_REMOTE_AGENTS",
    "A2ARouter",
    "AgentCard",
    "AgentSkill",
    "RemoteAgent",
    "a2a_available",
    "build_agent_card",
    "discover_remote_agents",
    "get_remote_registry",
]
