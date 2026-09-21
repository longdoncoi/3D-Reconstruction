"""Shared, versioned desktop-action contract."""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "Config" / "agent_action_manifest.json"


@lru_cache(maxsize=1)
def manifest() -> dict[str, Any]:
    with _MANIFEST_PATH.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data.get("actions"), list):
        raise ValueError("agent_action_manifest.json requires an actions array")
    return data


@lru_cache(maxsize=1)
def _index() -> dict[str, dict[str, Any]]:
    return {entry["id"]: entry for entry in manifest()["actions"]}


@lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    return {alias: entry["id"] for entry in manifest()["actions"] for alias in entry.get("aliases", [])}


def normalize_text(text: str) -> str:
    """Accent/case-insensitive form so 'Ẩn mô hình 3D' == 'an mo hinh 3d'."""
    text = unicodedata.normalize("NFD", (text or "").casefold().replace("đ", "d"))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


@lru_cache(maxsize=1)
def _intent_index() -> dict[str, tuple[tuple[str, int], ...]]:
    result: dict[str, tuple[tuple[str, int], ...]] = {}
    for action_id, entry in _index().items():
        phrases = {normalize_text(p) for p in entry.get("intents", [])}
        result[action_id] = tuple((f" {p} ", len(p.split())) for p in sorted(phrases) if p)
    return result


def reload_manifest() -> None:
    """Drop caches so an edited manifest is picked up without a server restart."""
    for cached in (manifest, _index, _aliases, _intent_index):
        cached.cache_clear()


def canonical_action(action: str) -> str | None:
    if action in _index():
        return action
    return _aliases().get(action)


def action_ids() -> set[str]:
    return set(_index())


def action_intents(action_id: str) -> tuple[str, ...]:
    return tuple(_index()[action_id].get("intents", []))


def action_catalog() -> str:
    """Compact 'id: meaning' list injected into the tool description."""
    return "\n".join(
        f"- {action_id}: {entry.get('description', '')}".rstrip(": ")
        for action_id, entry in _index().items()
    )


def rank_actions_for_step(step_text: str) -> list[tuple[str, int]]:
    """Actions whose intent phrases occur in the step; score = words in the longest phrase."""
    haystack = f" {normalize_text(step_text)} "
    ranked = []
    for action_id, phrases in _intent_index().items():
        score = max((words for needle, words in phrases if needle in haystack), default=0)
        if score:
            ranked.append((action_id, score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def step_matches_action(step_text: str, action: str) -> bool | None:
    """True/False when the manifest has evidence about the step, None when it has none."""
    canonical = canonical_action(action)
    ranked = rank_actions_for_step(step_text)
    if canonical is None or not ranked:
        return None
    return dict(ranked).get(canonical, 0) >= ranked[0][1]


def canonicalise_action_params(params: dict[str, Any]) -> dict[str, Any] | None:
    action = canonical_action(str(params.get("action", "")))
    if action is None:
        return None
    entry = _index()[action]
    allowed = {"action", "request_id", *entry.get("parameters", {}).keys()}
    result = {key: value for key, value in params.items() if key in allowed}
    result["action"] = action
    return result


def validate_action_params(params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    result = canonicalise_action_params(params)
    if result is None:
        valid_actions = ", ".join(sorted(action_ids()))
        return None, f"Unsupported desktop action: '{params.get('action', '')}'. Supported actions are: {valid_actions}"
    entry = _index()[result["action"]]
    for name, definition in entry.get("parameters", {}).items():
        value = result.get(name)
        if definition.get("required") and (value is None or value == ""):
            return None, f"{result['action']} requires parameter '{name}'"
        if value is not None and definition.get("enum") and value not in definition["enum"]:
            return None, f"{result['action']}.{name} must be one of {definition['enum']}"
    return result, None


def looks_like_ui_action(text: str) -> bool:
    # Deprecated: A2A routing replaces text-based intent matching.
    return False