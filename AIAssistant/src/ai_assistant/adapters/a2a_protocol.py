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
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.domain.governance import has_sensitive_keys
from ai_assistant.domain.security import DataClassification
from ai_assistant.domain.specialists import Specialist
from ai_assistant.settings import (
    DEFAULT_BASE_URL,
    a2a_enabled,
    a2a_trusted_endpoints,
)

from .a2a_client import TrustedA2AClient
from .a2a_payloads import build_agent_card as _build_agent_card_payload

logger = get_agent_logger("a2a")

# ── A2A configuration (ADR 0001) ─────────────────────────────────────────────
#
# Nothing here is read at import time: ``a2a_enabled()`` and
# ``a2a_trusted_endpoints()`` are consulted when the transport is actually
# used, so importing this adapter never freezes the process environment and
# tests can reconfigure A2A without reloading the module.

# ── Remote Agent ─────────────────────────────────────────────────────────────

#: Discovery cards are small JSON documents; anything larger is a broken or
#: hostile peer. Bounded like the trusted transport (a2a_client, 1 MiB).
_DISCOVERY_TIMEOUT = 5.0
_DISCOVERY_MAX_BYTES = 1_048_576


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


@dataclass
class RemoteAgent:
    """A discovered remote agent with its capabilities."""
    url: str
    card: dict[str, Any]
    skills: dict[str, dict[str, Any]]  # skill_id → skill dict
    last_discovered: float = field(default=0.0)


class RemoteAgentRegistry:
    """Thread-safe index of discovered remote agents.

    The composition root owns exactly one instance and injects it into the
    supervisor's resolver and the A2A router, so there is no process-wide
    registry to patch or reset between runs. All mutation/read paths take the
    instance lock because discovery runs in a startup worker thread while
    request threads resolve specialists.
    """

    def __init__(self) -> None:
        self._agents: dict[str, RemoteAgent] = {}
        self._lock = threading.RLock()

    def find(self, specialist_id: str) -> RemoteAgent | None:
        """Return the first agent advertising ``specialist_id``."""
        with self._lock:
            agents = list(self._agents.values())
        for agent in agents:
            if specialist_id in agent.skills:
                return agent
        return None

    def resolve(self, specialist_id: str) -> str | None:
        """Return the endpoint advertising ``specialist_id``, or ``None``."""
        if not a2a_available():
            return None
        with self._lock:
            items = list(self._agents.items())
        for url, agent in items:
            if specialist_id in agent.skills:
                return url
        return None

    def snapshot(self) -> dict[str, RemoteAgent]:
        """A shallow copy of the current registry (safe to iterate)."""
        with self._lock:
            return dict(self._agents)

    def discover(self, urls: list[str] | None = None) -> dict[str, RemoteAgent]:
        """Fetch Agent Cards from HTTP(S) endpoints and populate the registry.

        Hardened (ADR 0002/0008): only ``http(s)`` URLs are fetched, the body is
        bounded to :data:`_DISCOVERY_MAX_BYTES`, and the registry always stores
        the allowlisted base that discovery was addressed to. A card cannot
        choose the dispatch endpoint: :class:`TrustedA2AClient` compares the
        endpoint against the deployment allowlist by exact equality, so a
        subpath or lookalike host in the card would otherwise either bypass the
        check or make dispatch fail.
        """
        from urllib.request import Request, urlopen

        targets = urls if urls is not None else sorted(a2a_trusted_endpoints())
        if not targets:
            return self.snapshot()

        for base_url in targets:
            if not isinstance(base_url, str) or not _is_http_url(base_url):
                logger.warning("Refusing to discover non-HTTP agent URL: %r", base_url)
                continue
            card_url = (base_url if base_url.rstrip("/").endswith(".well-known/agent.json")
                        else base_url.rstrip("/") + "/.well-known/agent.json")
            try:
                req = Request(card_url, headers={"Accept": "application/json"})
                with urlopen(req, timeout=_DISCOVERY_TIMEOUT) as resp:
                    raw = resp.read(_DISCOVERY_MAX_BYTES + 1)
                if len(raw) > _DISCOVERY_MAX_BYTES:
                    logger.warning("Refusing oversized Agent Card at %s (%d bytes)", card_url, len(raw))
                    continue
                try:
                    card_data = json.loads(raw.decode("utf-8"))
                except ValueError as error:
                    logger.warning("Failed to discover agent at %s: %s", card_url, error)
                    continue
                if not isinstance(card_data, dict):
                    logger.warning("Refusing non-object Agent Card at %s", card_url)
                    continue
                skills = {}
                for skill in card_data.get("skills", []):
                    if not isinstance(skill, dict):
                        continue
                    skill_id = skill.get("id", "")
                    if skill_id:
                        skills[skill_id] = skill
                advertised = card_data.get("url")
                if isinstance(advertised, str) and advertised.rstrip("/") != base_url.rstrip("/"):
                    logger.info(
                        "Agent Card at %s advertises %s; keeping the allowlisted discovery base",
                        card_url, advertised,
                    )
                with self._lock:
                    self._agents[base_url] = RemoteAgent(
                        url=base_url, card=card_data, skills=skills,
                        last_discovered=time.time(),
                    )
                logger.info("Discovered remote agent: %s (%d skills)", base_url, len(skills))
            except Exception as error:
                logger.warning("Failed to discover agent at %s: %s", card_url, error)

        return self.snapshot()


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
        # A2A wire names (camelCase per the protocol schema), not locals.
        protocolVersion: str = "0.3.0",  # noqa: N803
        skills: list["AgentSkill"] | None = None,
        capabilities: dict[str, bool] | None = None,
        defaultInputModes: list[str] | None = None,  # noqa: N803
        defaultOutputModes: list[str] | None = None,  # noqa: N803
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

    def __init__(self, client: Any | None = None,
                 registry: "RemoteAgentRegistry | None" = None) -> None:
        # Injectable for tests; otherwise a per-agent allowlist client is built.
        self._client = client
        # The composition root passes its registry; a bare router stays
        # local-only with an empty one instead of reaching for module state.
        self._registry = registry if registry is not None else RemoteAgentRegistry()

    def find_remote(self, specialist_id: str) -> Any | None:
        """Find a remote agent whose skills include ``specialist_id``."""
        return self._registry.find(specialist_id)

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
        except Exception as error:
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
        else:
            logger.info("A2A remote call succeeded: specialist=%s url=%s",
                        specialist_id, remote.url)
            return {"source": "remote", "agent_url": remote.url, **result}

    def _dispatch(self, agent: Any, specialist_id: str,
                  task: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send the task through the trusted A2A client (single transport).

        The client is built with the deployment allowlist (ADR 0002), not with
        the discovered agent's own URL: otherwise the ``submit`` policy check
        would be vacuously true and any discovered card could be called.
        """
        client = self._client or TrustedA2AClient(trusted_endpoints=a2a_trusted_endpoints())
        # The client's "sensitive data cannot leave the trust boundary" rule is
        # only real if the caller declares what it is sending: a payload holding
        # a credential is classified RESTRICTED here so that check can fire.
        # It is the second, independent refusal behind the orchestrator's own
        # egress rule — this one sits at the transport boundary (ADR 0008/0010).
        classification = (DataClassification.RESTRICTED if has_sensitive_keys(params)
                          else DataClassification.INTERNAL)
        outcome = client.submit(agent.url, specialist_id, task, metadata=params,
                                classification=classification)
        if outcome.get("routing") != "remote_selected":
            raise RuntimeError(outcome.get("error", "Remote A2A call was rejected"))
        return outcome


# ── Optional SDK import ──────────────────────────────────────────────────────

#: ``None`` = not probed yet, ``True``/``False`` = cached result. The SDK cannot
#: appear at runtime, so one probe per process is enough; keeping it lazy means
#: importing this adapter has no configuration-dependent side effects.
_sdk_available: bool | None = None
_sdk_probe_lock = threading.Lock()


def _sdk_present() -> bool:
    """Import-probe the optional ``a2a`` SDK once, on first use."""
    global _sdk_available
    with _sdk_probe_lock:
        if _sdk_available is None:
            if not a2a_enabled():
                _sdk_available = False
            else:
                try:
                    from a2a.types import AgentCard as _SdkAgentCard  # noqa: F401
                except ImportError:
                    logger.info("a2a-sdk not installed — A2A layer operates in local-only mode")
                    _sdk_available = False
                else:
                    logger.info("a2a-sdk loaded successfully")
                    _sdk_available = True
        return _sdk_available


def a2a_available() -> bool:
    """Return True when the A2A transport can be used."""
    return a2a_enabled() and _sdk_present()


# NOTE (ADR 0013): this module used to carry its own ``_text_message`` /
# ``_safe_value`` / ``task_payload`` copies left behind by the ADR 0005 split.
# They had no caller, hardcoded ``"state": "submitted"``, used a placeholder
# timestamp and applied a second redaction key set that bypassed the domain
# policy (ADR 0010). ``ai_assistant.adapters.a2a_payloads`` owns payload
# building and redaction for every A2A surface.


# ── Agent Card builder ──────────────────────────────────────────────────────


def _default_capabilities() -> list[str]:
    """One skill per ``Specialist`` value — the default card advertisement.

    The platform ships one Supervisor plus one specialist per ``Specialist``
    value. Deployments may narrow the advertised set through their agents
    configuration, so the live server card stays config-driven (see
    ``config/agents.toml``).

    ``Specialist`` lives in the domain, so this module-level import is free of
    cycles (the orchestration layer no longer imports this adapter).
    """
    return [specialist.value for specialist in Specialist]


def build_agent_card(
    *,
    agent_name: str = "3D-Reconstruction AI Agent Platform",
    agent_version: str = "1.0.0",
    base_url: str = DEFAULT_BASE_URL,
    capabilities: list[str] | None = None,
) -> AgentCard:
    """Build A2A 0.3 Agent Card for discovery endpoint.

    Payload construction is delegated to
    :mod:`ai_assistant.adapters.a2a_payloads` so the wire card served by the
    router and this dataclass view share a single source of truth; only the
    returned object shape differs.
    """
    if capabilities is None:
        capabilities = _default_capabilities()
    payload = _build_agent_card_payload(
        agent_name=agent_name,
        agent_version=agent_version,
        base_url=base_url,
        capabilities=list(capabilities),
    )
    return AgentCard(
        name=payload["name"],
        description=payload["description"],
        url=payload["url"],
        version=payload["version"],
        protocolVersion=payload["protocolVersion"],
        skills=[
            AgentSkill(
                id=skill["id"],
                name=skill["name"],
                description=skill["description"],
                tags=[skill["id"]],
            )
            for skill in payload["skills"]
        ],
        capabilities=payload["capabilities"],
        defaultInputModes=payload["defaultInputModes"],
        defaultOutputModes=payload["defaultOutputModes"],
    )


# ── A2A Configuration End ────────────────────────────────────────────────────

__all__ = [
    "A2ARouter",
    "AgentCard",
    "AgentSkill",
    "RemoteAgent",
    "RemoteAgentRegistry",
    "a2a_available",
    "build_agent_card",
]
