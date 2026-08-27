"""Shared, versioned desktop-action contract.

The JSON file is deliberately consumable by both the Python server and the Qt
client.  Do not duplicate actions, aliases or intent phrases in code.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "Config" / "agent_action_manifest.json"


def normalise_text(text: str) -> str:
    value = unicodedata.normalize("NFD", text.casefold())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=1)
def manifest() -> dict[str, Any]:
    with _MANIFEST_PATH.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data.get("actions"), list):
        raise ValueError("agent_action_manifest.json requires an actions array")
    return data


@lru_cache(maxsize=1)
def _index() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    actions: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for entry in manifest()["actions"]:
        action = entry["id"]
        actions[action] = entry
        for alias in entry.get("aliases", []):
            aliases[alias] = action
    return actions, aliases


def canonical_action(action: str) -> str | None:
    actions, aliases = _index()
    return action if action in actions else aliases.get(action)


def canonicalise_action_params(params: dict[str, Any]) -> dict[str, Any] | None:
    action = canonical_action(str(params.get("action", "")))
    if action is None:
        return None
    entry = _index()[0][action]
    allowed = {"action", "request_id", *entry.get("parameters", {}).keys()}
    result = {key: value for key, value in params.items() if key in allowed}
    result["action"] = action
    return result


def validate_action_params(params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    result = canonicalise_action_params(params)
    if result is None:
        valid_actions = ", ".join(sorted(action_ids()))
        return None, f"Unsupported desktop action: '{params.get('action', '')}'. Supported actions are: {valid_actions}"
    entry = _index()[0][result["action"]]
    for name, definition in entry.get("parameters", {}).items():
        value = result.get(name)
        if definition.get("required") and (value is None or value == ""):
            return None, f"{result['action']} requires parameter '{name}'"
        if value is not None and definition.get("enum") and value not in definition["enum"]:
            return None, f"{result['action']}.{name} must be one of {definition['enum']}"
    return result, None


def looks_like_ui_action(text: str) -> bool:
    value = normalise_text(text)
    return any(normalise_text(phrase) in value for entry in manifest()["actions"] for phrase in entry.get("phrases", []))


def match_action_intent(text: str) -> dict[str, Any] | None:
    value = normalise_text(text)
    # Language needs an explicit value; this belongs to the manifest action.
    if any(phrase in value for phrase in ("doi sang english", "change to english", "switch to english")):
        return {"action": "language.change", "language": "en"}
    if any(phrase in value for phrase in ("doi sang tieng viet", "change to vietnamese", "switch to vietnamese")):
        return {"action": "language.change", "language": "vi"}
    for entry in manifest()["actions"]:
        if entry["id"] == "language.change":
            continue
        if any(normalise_text(phrase) in value for phrase in entry.get("phrases", [])):
            result: dict[str, Any] = {"action": entry["id"]}
            if entry["id"] == "admin.login":
                result.update({"username": "Admin", "password": "1"})
            return result
    return None


def match_action_sequence(text: str) -> list[dict[str, Any]] | None:
    """Return a manifest-defined ordered UI workflow, if the text matches one."""
    value = normalise_text(text)
    for workflow in manifest().get("workflows", []):
        if not any(normalise_text(phrase) in value for phrase in workflow.get("phrases", [])):
            continue
        actions = workflow.get("actions", [])
        if all(canonical_action(action) is not None for action in actions):
            return [{"action": action} for action in actions]
    return None


def action_ids() -> set[str]:
    return set(_index()[0])
