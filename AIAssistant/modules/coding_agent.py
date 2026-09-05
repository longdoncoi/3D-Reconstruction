"""Isolated policy and context contract for repository engineering tasks."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .agent_logging import get_agent_logger

logger = get_agent_logger("coding")


_CODING_TERMS = (
    "code", "coding", "bug", "fix", "refactor", "test", "unit test", "build",
    "compile", "cmake", "lint", "review", "implement", "implementation", "function",
    "class", "source", "function", "method", "symbol", "snippet", "citation", "cite",
    "feature", "enhance", "enhancement", "improve", "improvement", "export",
    "add feature", "bo sung", "them", "them tinh nang", "tinh nang", "chuc nang",
    "phan tich", "giai thich", "ham", "analyze", "analysis", "explain", "debug", "diagnose",
    "sua", "sua loi", "loi",
    "hàm", "ham", "trích dẫn", "trich dan", "mã nguồn", "ma nguon",
    "mã", "lỗi", "sửa", "kiểm thử", "tái cấu trúc", "biên dịch",
)


@dataclass(frozen=True)
class CodingTaskContext:
    """Minimal, explicit context handed from the supervisor to Coding Agent."""

    task: str
    language: str
    project_root: str


_CHANGE_TERMS = (
    "fix", "bug", "debug", "diagnose", "refactor", "implement",
    "implementation", "feature", "enhance", "enhancement", "improve",
    "improvement", "export", "add", "them", "bo sung", "sua", "sua loi",
    "loi",
)
_ANALYSIS_TERMS = (
    "analyze", "analysis", "explain", "explanation", "phan tich", "giai thich",
    "ham", "function", "method", "symbol", "citation", "cite", "trich dan",
    "ma nguon", "source code",
)
_VERIFY_TERMS = (
    "test", "unit test", "build", "compile", "cmake", "lint", "pytest", "ctest",
    "kiem thu", "kiem tra", "bien dich",
)

_DISCOVERY_TOOLS = frozenset({
    "find_files", "list_directory", "search_text", "read_file", "analyze_code",
    "git_diff", "get_project_status",
})
_SOURCE_TOOLS = frozenset({"read_file", "analyze_code"})
_MUTATION_TOOLS = frozenset({"write_file", "patch_file", "replace_file_content", "multi_replace_file_content", "create_directory"})


def _result_is_successful(result: object) -> bool:
    """Return whether a tool result represents a successful operation.

    Tool executors use a small shared result contract.  Absence of ``error`` is
    not sufficient: a command can finish with a non-zero exit code and a
    mutation can explicitly report ``success: false``.  Keeping this check here
    makes completion evidence independent of a particular feature or tool
    caller.
    """
    if not isinstance(result, dict) or result.get("error"):
        return False
    if result.get("success") is False:
        return False
    return result.get("return_code", 0) == 0


def _result_has_evidence(tool: str, result: object) -> bool:
    """Return whether a successful discovery result contains usable evidence."""
    if not _result_is_successful(result) or not isinstance(result, dict):
        return False
    evidence_fields = {
        "read_file": "content",
        "search_text": "results",
        "find_files": "matches",
        "list_directory": "entries",
        "analyze_code": ("functions", "classes", "includes"),
    }
    fields = evidence_fields.get(tool)
    if fields is None:
        return True
    if isinstance(fields, str):
        value = result.get(fields)
        return bool(value.strip()) if isinstance(value, str) else bool(value)
    return any(bool(result.get(field)) for field in fields)


def _normalise_task(task: str) -> str:
    """Create one accent-insensitive view for intent and workflow checks."""
    return unicodedata.normalize("NFKD", task).encode("ascii", "ignore").decode().casefold()


def _contains_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(
        term in text if " " in term
        else re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None
        for term in terms
    )


def coding_workflow_kind(task: str) -> str:
    """Classify the required engineering workflow, independent of domain words."""
    normalised = _normalise_task(task)
    if _contains_term(normalised, _CHANGE_TERMS):
        return "change"
    if _contains_term(normalised, _ANALYSIS_TERMS):
        return "analysis"
    if _contains_term(normalised, _VERIFY_TERMS):
        return "verify"
    return "investigate"


@dataclass(frozen=True)
class CodingWorkflowStatus:
    """Observable completion state used to prevent premature coding answers."""

    kind: str
    source_evidence: bool
    mutation: bool
    diff_reviewed: bool
    execution_observed: bool
    missing: tuple[str, ...]


def coding_workflow_status(task: str, steps: list[dict]) -> CodingWorkflowStatus:
    """Compute workflow progress from tool evidence, not from feature names.

    A successful tool result is intentionally the only source of progress. A
    pending approval, a model statement, or a planned step cannot masquerade
    as an edit or a verification result.
    """
    kind = coding_workflow_kind(task)
    successful_results = []
    for index, step in enumerate(steps):
        if step.get("type") != "tool_result":
            continue
        result = step.get("result")
        if _result_is_successful(result):
            successful_results.append((index, step))

    source_evidence = any(
        step.get("tool") in _SOURCE_TOOLS and _result_has_evidence(step.get("tool", ""), step.get("result"))
        for _, step in successful_results
    )
    discovery_evidence = any(
        step.get("tool") in _DISCOVERY_TOOLS and _result_has_evidence(step.get("tool", ""), step.get("result"))
        for _, step in successful_results
    )
    mutation_indices = [
        index for index, step in successful_results
        if step.get("tool") in _MUTATION_TOOLS
    ]
    last_mutation = max(mutation_indices, default=None)
    mutation = last_mutation is not None
    diff_reviewed = any(
        (index > last_mutation and step.get("tool") == "git_diff"
         and _result_has_evidence("git_diff", step.get("result")))
        for index, step in successful_results
    ) if last_mutation is not None else False
    execution_observed = any(
        index > last_mutation and step.get("tool") == "run_command"
        for index, step in successful_results
    ) if last_mutation is not None else any(
        step.get("tool") == "run_command" for _, step in successful_results
    )

    missing: list[str] = []
    if kind == "analysis":
        if not source_evidence:
            missing.append("source_evidence")
    elif kind == "change":
        if not source_evidence:
            missing.append("source_evidence")
        if not mutation:
            missing.append("approved_change")
        else:
            if not diff_reviewed:
                missing.append("diff_review")
            if not execution_observed:
                missing.append("verification_command")
    elif kind == "verify":
        if not discovery_evidence:
            missing.append("project_discovery")
        if not execution_observed:
            missing.append("verification_command")
    elif not discovery_evidence:
        missing.append("project_discovery")

    return CodingWorkflowStatus(
        kind=kind,
        source_evidence=source_evidence,
        mutation=mutation,
        diff_reviewed=diff_reviewed,
        execution_observed=execution_observed,
        missing=tuple(missing),
    )


def coding_workflow_guidance(status: CodingWorkflowStatus) -> str:
    """Return the next generic workflow action for the Reasoner."""
    guidance = {
        "source_evidence": "inspect the relevant source with search_text/analyze_code/read_file",
        "project_discovery": "discover the relevant project files and commands",
        "approved_change": "apply the focused patch/write after approval",
        "diff_review": "review the resulting change with git_diff",
        "verification_command": "run the focused test, lint, compile, or build command after the change",
    }
    if not status.missing:
        return "The coding workflow has observable evidence for every required stage."
    return "Next required coding stage: " + guidance[status.missing[0]] + "."


def is_coding_task(task: str) -> bool:
    """Classify engineering requests without intercepting desktop workflows."""
    normalised = task.casefold()
    accent_free = unicodedata.normalize("NFKD", task).encode("ascii", "ignore").decode().casefold()
    searchable_texts = (normalised, accent_free)
    matched = any(
        any(
            term in text if " " in term
            else re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None
            for text in searchable_texts
        )
        for term in _CODING_TERMS
    )
    if matched:
        logger.info("Coding Agent selected | task=%s", task[:160])
    return matched


def instruction(context: CodingTaskContext) -> str:
    """Return the Coding Agent handoff, with state-changing work approval-gated."""
    response_language = "Vietnamese" if context.language == "vi" else "English"
    return f"""
## CODING AGENT HANDOFF

You are the Coding Agent for this request. Work only on the stated engineering
task and files proven relevant by read/search tools. Start with evidence: inspect
the smallest relevant files and, for a bug, reproduce or identify the failing
path before proposing a change. Keep a focused plan and report affected files,
validation commands, and any remaining risk in the final answer.

Repository root: {context.project_root}
Task: {context.task}

Read/search/analyse/status tools are safe to use. Any write, patch, directory
creation, or command execution that changes state requires the existing explicit
approval workflow. Never claim a build or test passed unless its result was
observed. If the request mentions a UI action as part of a bug fix or code
change, do not call application_action; inspect and change the repository code
instead. Use {response_language} for the final response.

Coding playbooks:
- Analyze/explain: locate the symbol, use analyze_code and a focused read_file,
  then explain behavior, inputs/outputs, dependencies, side effects and risks.
  Do not modify files.
- Fix/debug: reproduce or trace the failing path, identify the root cause,
  propose the smallest patch, request approval, then validate and run focused
  tests/build checks.
- Feature/refactor: inspect existing extension points and tests, make a focused
  plan, request approval before edits, then validate, test and review git_diff.

Completion contract for every change task: do not emit a final answer after
discovery alone. Observable evidence must include relevant source reads, an
approved mutation, a post-change git_diff review, and an observed verification
command result. The command may report a failure; report that failure honestly
and continue diagnosis when another focused change can address it.

For all coding playbooks, use repository tools instead of application_action,
even when the code controls a desktop feature.

Available Code Agent toolbox: find_files, list_directory, search_text,
read_file (including symbol/range reads), analyze_code, git_diff,
get_project_status, validate_file, write_file, patch_file, replace_file_content,
multi_replace_file_content, create_directory, and run_command. Prefer the smallest read/search operation
that proves the change; use git_diff and validate_file after an approved edit.
""".strip()
