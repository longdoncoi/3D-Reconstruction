"""Helper utilities shared across orchestration nodes."""
from __future__ import annotations

import re
from typing import Any

from .state import SEMANTIC_REFLECTION_TOOLS

# Execution kinds the Planner assigns to each step. They replace hard-coded
# keyword routing: the LLM decides RAG vs direct answer vs UI tool per step.
_STEP_KINDS = {"rag_search", "direct", "application_action", "model_selected"}


def clean_project_research_answer(answer: str) -> str:
    """Remove RAG transport labels if a model accidentally repeats them."""
    answer = answer or ""
    cleaned = re.sub(r"(?im)^\s*=+\s*(?:TÀI LIỆU THAM KHẢO|MÃ NGUỒN LIÊN QUAN)\s*=+\s*$\n?", "", answer)
    cleaned = re.sub(r"(?im)^\s*(?:TÀI LIỆU THAM KHẢO|MÃ NGUỒN LIÊN QUAN)\s*:?[ \t]*$\n?", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\[\d+\]\s+[^\n]+\.(?:txt|md|pdf|docx?|pptx?|xlsx?|eml|html?)\s*$\n?", "", cleaned)
    cleaned = re.sub(r"\s*\[(?:\d+)(?:\s*,\s*\d+)*\]", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def completed_plan_steps(steps: list[dict[str, Any]], plan: list[str] | None) -> int:
    """Count consecutive accepted plan steps without double-counting retries."""
    if not plan:
        return 0
    reflections = [
        step for step in steps
        if step.get("type") == "reflection" and step.get("result", {}).get("passed") is True
        and (step.get("tool") in SEMANTIC_REFLECTION_TOOLS or step.get("tool") == "step_answer")
    ]
    indexed = [step for step in reflections if isinstance(step.get("plan_step_index"), int)]
    if not indexed:
        return min(len(reflections), len(plan))
    accepted = {
        step["plan_step_index"] for step in indexed
        if 0 <= step["plan_step_index"] < len(plan)
    }
    completed = 0
    while completed in accepted:
        completed += 1
    return completed


def normalise_plan_payload(payload: dict[str, Any]) -> tuple[list[str] | None, dict[str, Any]]:
    """Accept the current structured planner contract and older 'plan' JSON."""
    requires_plan = bool(payload.get("requires_plan"))
    raw_steps = payload.get("steps", payload.get("plan", []))
    plan = [str(item).strip() for item in raw_steps if str(item).strip()] if isinstance(raw_steps, list) else []
    # LLM-classified execution kind per step (and for the single-step case).
    raw_kinds = payload.get("step_kinds", payload.get("kind"))
    if isinstance(raw_kinds, str):
        raw_kinds = [raw_kinds]
    step_kinds: list[str] = []
    if isinstance(raw_kinds, list):
        step_kinds = [str(item).strip().lower() for item in raw_kinds if str(item).strip()]
    step_kinds = [k if k in _STEP_KINDS else "model_selected" for k in step_kinds]
    if plan and step_kinds:
        # Align lengths: truncate extras, pad missing with "model_selected".
        step_kinds = (step_kinds + ["model_selected"] * len(plan))[: len(plan)]
    elif not plan and not step_kinds:
        step_kinds = []
    spec = {
        "requires_plan": requires_plan,
        "goal": str(payload.get("goal", "")).strip(),
        "affected_areas": [str(item) for item in payload.get("affected_areas", [])
                           if str(item).strip()] if isinstance(payload.get("affected_areas", []), list) else [],
        "acceptance_criteria": [str(item) for item in payload.get("acceptance_criteria", [])
                                if str(item).strip()] if isinstance(payload.get("acceptance_criteria", []), list) else [],
        "verification_commands": [str(item) for item in payload.get("verification_commands", [])
                                  if str(item).strip()] if isinstance(payload.get("verification_commands", []), list) else [],
        "steps": plan,
        "step_kinds": step_kinds,
    }
    return (plan or None) if requires_plan else None, spec


_PROJECT_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "devmanager": ("dev manager", "development manager", "quan ly phat trien"),
    "project manager": ("project manager", "pm", "quan ly du an"),
    "teamlead": ("team lead", "truong nhom", "leader"),
    "hrmanager": ("hr manager", "human resources manager", "quan ly nhan su"),
    "ky su": ("engineer", "developer", "kỹ sư"),
}


def project_role_rag_query(text: str, normalize_fn) -> str:
    """Augment role lookup queries with aliases for better RAG coverage."""
    normalized = normalize_fn(text)
    aliases: list[str] = []
    for role, role_aliases in _PROJECT_ROLE_ALIASES.items():
        if role in normalized or any(alias in normalized for alias in role_aliases):
            aliases.extend(alias for alias in role_aliases if alias not in normalized)
    if not aliases:
        return text
    # Append RAG-coverage aliases only; do NOT bias queries toward fixed
    # answer headings like "Nhiệm vụ/Vai trò".
    return f"{text}. {'; '.join(aliases)}."


def is_project_role_lookup(text: str, normalize_fn) -> bool:
    normalized = normalize_fn(text)
    return any(
        role in normalized or any(alias in normalized for alias in aliases)
        for role, aliases in _PROJECT_ROLE_ALIASES.items()
    )


def has_rag_evidence_for_step(steps: list[dict[str, Any]], plan_step_index: int) -> bool:
    """Whether this exact step has a completed RAG observation."""
    return any(
        step.get("type") == "tool_result"
        and step.get("tool") == "rag_search"
        and step.get("plan_step_index") == plan_step_index
        and "error" not in step.get("result", {})
        for step in steps
    )


def has_rag_evidence_without_plan(steps: list[dict[str, Any]]) -> bool:
    """Whether a one-turn project question already has RAG evidence."""
    return any(
        step.get("type") == "tool_result"
        and step.get("tool") == "rag_search"
        and "error" not in step.get("result", {})
        for step in steps
    )


def plan_step_execution_contract(step_text: str, rank_actions_fn, normalize_fn, kind: str | None = None) -> dict[str, Any]:
    """Return the deterministic execution boundary for a plan step.

    Primary source is the LLM-classified ``kind`` from the planner
    (``plan_spec["step_kinds"]``).  Keyword heuristics remain only as a
    fallback when no kind is available (e.g. hand-built tests).
    """
    if kind == "application_action":
        action_matches = rank_actions_fn(step_text)
        if action_matches:
            return {"mode": "application_action", "action": action_matches[0][0]}
        return {"mode": "model_selected"}
    if kind == "rag_search":
        query = project_role_rag_query(step_text, normalize_fn)
        rag_contract: dict[str, Any] = {"mode": "rag_search", "query": query}
        if is_project_role_lookup(step_text, normalize_fn):
            rag_contract["top_k"] = 8
        return rag_contract
    if kind == "direct":
        return {"mode": "direct_answer"}

    action_matches = rank_actions_fn(step_text)
    if action_matches:
        return {"mode": "application_action", "action": action_matches[0][0]}

    normalized = normalize_fn(step_text)
    project_markers = (
        "du an", "project", "ky su", "nhan su", "vai tro", "lich su",
        "tai lieu", "ma nguon", "source code", "code",
    )
    if any(marker in normalized for marker in project_markers):
        query = project_role_rag_query(step_text, normalize_fn)
        contract: dict[str, Any] = {"mode": "rag_search", "query": query}
        if is_project_role_lookup(step_text, normalize_fn):
            contract["top_k"] = 8
        return contract

    general_markers = ("khai niem", "dinh nghia", "la gi", "giai thich", "nguyen ly")
    if any(marker in normalized for marker in general_markers):
        return {"mode": "direct_answer"}
    return {"mode": "model_selected"}
