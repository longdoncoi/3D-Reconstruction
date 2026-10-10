"""A2A (Agent-to-Agent) protocol layer for the unified assistant.

Provides Agent Card publishing, remote Agent Card discovery and the
:A2ARouter facade used by the delegation path.

The actual remote transport is :class:`ai_assistant.adapters.a2a_client.
TrustedA2AClient`, which enforces the deployment allowlist and refuses to send
sensitive classifications outside the trust boundary. ``A2ARouter`` is a thin
routing facade over that single client; it never executes tools locally.

Configuration (ADR 0001): A2A settings are read through the
:mod:`ai_assistant.settings` accessor functions rather than os.getenv at
import time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.settings import a2a_card_name, a2a_card_url, a2a_enabled, a2a_remote_agent_urls

from .a2a_client import TrustedA2AClient

logger = get_agent_logger("a2a")

# ── A2A configuration (ADR 0001: read through settings accessors) ──

A2A_ENABLED = a2a_enabled()
A2A_REMOTE_AGENTS: frozenset[str] = frozenset(
    url.strip()
    for url in a2a_remote_agent_urls()
    if url.strip()
)
A2A_CARD_NAME = a2a_card_name()
A2A_CARD_URL = a2a_card_url()

# ── Remote Agent ─────────────────────────────────────────────────────────────

_remote_registry: dict[str, Any] = {}


@dataclass
class RemoteAgent:
    """A discovered remote agent with its capabilities."""
    url: str
    card: dict[str, Any]
    skills: dict[str, dict[str, Any]]  # skill_id → skill dict
    last_discovered: float = field(default=0.0)


# ── Agent Skill ───────────────────────────────────────────────────────────────

class AgentSkill:
    """One capability that an agent advertises in its Agent Card."""

    def __init__(self, id: str, name: str, description: str, tags: list[str] | None = None) -> None:
        self.id = id
        self.name = name
        self.description = description
        self.tags = tags or []

    def __repr__(self) -> str:
        return f"AgentSkill(id={self.id!r}, name={self.name!r})"


# ── Agent Card ────────────────────────────────────────────────────────────────

class AgentCard:
    """A2A-compatible Agent Card describing this server's capabilities."""

    def __init__(
        self,
        name: str,
        description: str,
        url: str,
        version: str = "1.0.0",
        protocolVersion: str = "0.3.0",
        skills: list["AgentSkill"] | None = None,
        capabilities: dict[str, bool] | None = None,
        defaultInputModes: list[str] | None = None,
        defaultOutputModes: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.url = url
        self.version = version
        self.protocolVersion = protocolVersion
        self.skills = skills or []
        self.capabilities = capabilities or {"streaming": True, "pushNotifications": False}
        self.defaultInputModes = defaultInputModes or ["text"]
        self.defaultOutputModes = defaultOutputModes or ["text"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "protocolVersion": self.protocolVersion,
            "skills": [{"id": s.id, "name": s.name, "description": s.description, "tags": s.tags} for s in self.skills],
            "capabilities": self.capabilities,
            "defaultInputModes": self.defaultInputModes,
            "defaultOutputModes": self.defaultOutputModes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ── A2A Router ───────────────────────────────────────────────────────────────

class A2ARouter:
    """Route delegations to remote A2A agents without implicit fallback.

    All HTTP traffic is performed by :class:`TrustedA2AClient`; this class only
    selects an eligible remote agent and normalises the outcome shape so the
    delegation caller gets an explicit ``source``/``status`` result.
    """

    def __init__(self, client: Any | None = None) -> None:
        # Injectable for tests; otherwise a per-agent allowlist client is built.
        self._client = client

    def find_remote(self, specialist_id: str) -> Any | None:
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
            result = self._dispatch(remote, specialist_id, task, params or {})
            logger.info("A2A remote call succeeded: specialist=%s url=%s",
                        specialist_id, remote.url)
            return {"source": "remote", "agent_url": remote.url, **result}
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "A2A remote call failed (specialist=%s url=%s): %s",
                specialist_id, remote.url, error,
            )
            return {
                "source": "remote",
                "agent_url": remote.url,
                "status": "failed",
                "error": str(error),
            }

    def _dispatch(self, agent: Any, specialist_id: str,
                  task: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send the task through the trusted A2A client (single transport)."""
        client = self._client or TrustedA2AClient(trusted_endpoints=frozenset({agent.url}))
        outcome = client.submit(agent.url, specialist_id, task, metadata=params)
        if outcome.get("routing") != "remote_selected":
            raise RuntimeError(outcome.get("error", "Remote A2A call was rejected"))
        return outcome


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
    return a2a_enabled() and _sdk_available


# ── Payload builders ─────────────────────────────────────────────────────────

def _text_message(task: Any, text: str) -> dict[str, Any]:
    return {
        "role": "ROLE_AGENT", "messageId": f"status-{getattr(task, 'id', 'unknown')}-{int(getattr(task, 'updated_at', 0) * 1000)}",
        "contextId": getattr(task, 'context_id', None), "taskId": getattr(task, 'id', 'unknown'), "parts": [{"text": text}],
    }


def _safe_value(value: Any, key: str = "") -> Any:
    """Never expose credentials or internal continuations through A2A artifacts."""
    if key.lower() in {"password", "token", "secret", "api_key", "authorization", "continuation"}:
        return "[redacted]"
    if isinstance(value, dict):
        return {str(item_key): _safe_value(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    return value


def task_payload(task: Any) -> dict[str, Any]:
    """Serialize an AgentTask into A2A 0.3 task payload format."""
    status: dict[str, Any] = {
        "state": "submitted",  # simplified
        "timestamp": json.dumps({"key": "timestamp"}),  # placeholder
    }
    return {
        "id": getattr(task, 'id', 'unknown'),
        "contextId": getattr(task, 'context_id', None),
        "status": status,
        "metadata": _safe_value({}),
        "artifacts": [] if not getattr(task, 'result', None) else [{"artifactId": f"result-{getattr(task, 'id', 'unknown')}", "name": "result", "parts": [{"data": _safe_value(task.result)}]}],
    }


# ── Agent Card builder ──────────────────────────────────────────────────────

def build_agent_card(
    *,
    agent_name: str = "3D-Reconstruction AI Agent Platform",
    agent_version: str = "1.0.0",
    url: str = "http://127.0.0.1:8080",
    capabilities: list[str] = ["supervisor"],
) -> AgentCard:
    """Build A2A 0.3 Agent Card for discovery endpoint."""
    skills: list[AgentSkill] = []
    # Specialist instructions would be loaded from supervisor here
    for specialist_name in sorted(capabilities):
        skills.append(AgentSkill(
            id=specialist_name,
            name=specialist_name.replace("_", " ").title(),
            description=f"Execute {specialist_name} tasks through the 3D-Reconstruction agent platform.",
            tags=[specialist_name],
        ))

    return AgentCard(
        name=agent_name,
        description="3D-Reconstruction AI Agent Platform",
        url=url,
        version=agent_version,
        protocolVersion="0.3",
        skills=skills,
        capabilities={"streaming": True, "pushNotifications": False},
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
    )


# ── Remote Agent Discovery ───────────────────────────────────────────────────

_DISCOVERY_TIMEOUT = 5.0


def discover_remote_agents(urls: list[str] | None = None) -> dict[str, Any]:
    """Fetch Agent Cards from remote endpoints and populate the registry."""
    import json
    from urllib.request import Request, urlopen

    targets = urls if urls is not None else list(A2A_REMOTE_AGENTS)
    if not targets:
        return _remote_registry

    for base_url in targets:
        card_url = (base_url if base_url.rstrip("/").endswith(".well-known/agent.json")
                    else base_url.rstrip("/") + "/.well-known/agent.json")
        try:
            req = Request(card_url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=_DISCOVERY_TIMEOUT) as resp:
                card_data = json.loads(resp.read().decode("utf-8"))
            skills = {}
            for skill in card_data.get("skills", []):
                skill_id = skill.get("id", "")
                if skill_id:
                    skills[skill_id] = skill
            _remote_registry[base_url] = RemoteAgent(
                url=str(card_data.get("url") or base_url), card=card_data, skills=skills,
                last_discovered=__import__("time").time(),
            )
            logger.info("Discovered remote agent: %s (%d skills)", base_url, len(skills))
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to discover agent at %s: %s", card_url, error)

    return _remote_registry


def get_remote_registry() -> dict[str, Any]:
    """Return the current remote agent registry (read-only view)."""
    return dict(_remote_registry)


# ── A2A Configuration End ────────────────────────────────────────────────────

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
