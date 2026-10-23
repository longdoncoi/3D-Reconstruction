"""LLM completion adapters for the agent loop.

Two modes:
  * constrained  — one JSON envelope per call (tool | final | step_answer)
  * structured   — internal planner/critic JSON with a supplied JSON Schema
"""
from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable
from typing import Any, cast

from ai_assistant.domain.governance import scrub_pii
from ai_assistant.ports import LLMBackend, LLMRuntime
from ai_assistant.settings import load_agent_runtime_settings
from ai_assistant.tools.aliases import TOOL_NAME_ALIASES, canonical_tool_name

logger = logging.getLogger("ai_assistant.agents.completion")

# JSON schemas for planner and critic calls
PLANNER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "requires_plan": {"type": "boolean"},
        "goal": {"type": "string"},
        "affected_areas": {"type": "array", "items": {"type": "string"}},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
        "verification_commands": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
        "plan": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["requires_plan", "steps"],
    "additionalProperties": False,
}

CRITIC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "decision": {"type": "string", "enum": ["continue", "revise"]},
        "reason": {"type": "string"},
    },
    "required": ["passed", "decision", "reason"],
    "additionalProperties": False,
}

# Kept as a module alias for callers/tests that imported the old private
# table; the map itself is owned by ``tools.aliases``.
_TOOL_NAME_ALIASES = TOOL_NAME_ALIASES


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by some reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_tool_call_xml(content: str) -> str | None:
    """Extract tool call from <tool_call>...</tool_call> XML envelope (Qwen models)."""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    tool_name = data.get("name", "") or data.get("tool", "")
    arguments = data.get("arguments")
    if arguments is None:
        arguments = data.get("params", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not tool_name or not isinstance(arguments, dict):
        return None
    envelope = {"kind": "tool", "tool": tool_name, "params": arguments}
    logger.info("Extracted tool call from <tool_call> XML: %s", json.dumps(envelope, ensure_ascii=False))
    return json.dumps(envelope, ensure_ascii=False)


def parse_tool_call(
    response_text: str, tool_models: dict[str, type],
    validate_fn: Callable[..., tuple[dict[str, Any] | None, str | None] | None],
    record_schema_error_fn: Callable[[str], None],
) -> tuple[str | None, dict[str, Any] | None]:
    """Decode the JSON envelope from constrained decoding.

    Returns:
      (tool_name, params)  — where tool_name may be "_step_answer" or "_validation_error"
    """
    def _normalise(name: str) -> str:
        return canonical_tool_name(name)

    try:
        data = json.loads(response_text.strip())
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    if data.get("kind") == "step_answer":
        return "_step_answer", {"content": str(data.get("content", ""))}
    if data.get("kind") != "tool":
        return None, None
    tool_name = _normalise(str(data.get("tool", "")))
    params = data.get("params")
    if not isinstance(params, dict):
        return None, None
    result = validate_fn(tool_name, params, tool_models)
    if result is None:
        # Defensive: if validate_fn returns None, treat as validation error
        logger.warning("validate_tool_call returned None for tool '%s', treating as validation error", tool_name)
        return "_validation_error", {"tool": tool_name, "error": "validate_tool_call returned None"}
    validated, error = result
    if error:
        logger.warning("Rejected invalid constrained tool call %s: %s", tool_name, error)
        record_schema_error_fn(tool_name)
        return "_validation_error", {"tool": tool_name, "error": error}
    return tool_name, validated


def _record_and_decode_content(response: dict[str, Any],
                              record_token_usage_fn: Callable[[int, int], None],
                              label: str) -> str:
    """Record token usage and decode one constrained response to text.

    Both transports (remote OpenAI-compatible and local llama.cpp) answer
    with the same shape, so the usage recording and the tool-call/content
    split live here instead of twice in ``constrained_completion``.
    """
    usage = response.get("usage", {})
    in_tok, out_tok = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    if in_tok or out_tok:
        record_token_usage_fn(in_tok, out_tok)
    message = response.get("choices", [{}])[0].get("message", {})
    if message.get("tool_calls"):
        call = message["tool_calls"][0]["function"]
        content = json.dumps({"kind": "tool", "tool": call["name"],
                              "params": json.loads(call.get("arguments", "{}"))}, ensure_ascii=False)
    else:
        content = message.get("content", "")
    # ADR 0032: the remote leg scrubs exactly like the local leg — raw model
    # output can echo secrets/PII from the context into a plaintext log file,
    # and only the *log call* is scrubbed; the value itself stays intact.
    logger.info("Constrained LLM response%s: %s", f" ({label})" if label else "", scrub_pii(content))
    return content


def _remote_leg_allowed(backend_mode_fn: Callable[[], str]) -> bool:
    """ADR 0018 / ADR 0015: enforce the hybrid policy before any remote call.

    Returns ``True`` when this completion may leave the box. ``backend_mode``
    alone must never be able to leak data out: the gate reads it, but the
    decision is made here, before the transport and before the decode-error
    wrapper, so the refusal is unambiguous.
    """
    if backend_mode_fn() == "llama_cpp":
        return False
    from ai_assistant.llm.inference import cloud_allowed
    if not cloud_allowed("non_sensitive"):
        msg = ("Remote inference blocked by AGENT_HYBRID_POLICY (local_only); "
               "set AGENT_HYBRID_POLICY=cloud_allowed to opt in")
        raise RuntimeError(msg)
    return True


def constrained_completion(messages: list[dict[str, Any]], max_tokens: int, temperature: float,
                           llm_runtime: LLMRuntime, backend_mode_fn: Callable[[], str],
                           openai_compatible_fn: Callable[..., dict[str, Any]],
                           openai_tools: list[dict[str, Any]], grammar_schema: str,
                           record_token_usage_fn: Callable[[int, int], None]) -> str:
    """Generate exactly one final/tool JSON envelope."""
    remote_backend = _remote_leg_allowed(backend_mode_fn)
    try:
        if remote_backend:
            response = openai_compatible_fn(
                messages, max_tokens=max_tokens, temperature=temperature,
                tools=openai_tools, tool_choice="auto",
                response_format={"type": "json_object"},
            )
            return _record_and_decode_content(response, record_token_usage_fn, "remote")

        # Local llama.cpp path. Serialization lives in the backend (it owns
        # the single-threaded llama context), so no caller-side ``llm_lock``
        # is held across the generation.
        _precompiled_grammar = _get_precompiled_grammar(grammar_schema)
        kwargs: dict[str, Any] = {
            "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "repeat_penalty": 1.1, "stream": False,
        }
        if load_agent_runtime_settings().native_tool_calls:
            kwargs.update({"tools": openai_tools, "tool_choice": "auto"})
        elif _precompiled_grammar is not None:
            kwargs.update({"grammar": _precompiled_grammar})

        backend = cast(LLMBackend, llm_runtime.llm)
        response = backend.generate(**kwargs)

    except Exception as error:
        msg = f"Constrained tool decoding failed: {error}"
        raise RuntimeError(msg) from error

    content = _record_and_decode_content(response, record_token_usage_fn, "")

    if not isinstance(content, str):
        # RuntimeError (not TypeError): the decoder contract breach is a runtime
        # failure, and the message is asserted by contract tests.
        msg = "Constrained decoder returned no text content"
        raise RuntimeError(msg)  # noqa: TRY004
    content = _strip_think_tags(content)
    # The redaction chokepoint for log material (governance.scrub_pii): raw
    # model output can echo secrets/PII from context into log files, and only
    # the *log call* is scrubbed — the value itself stays intact for parsing.
    logger.info("Constrained LLM response: %s", scrub_pii(content))

    if "<tool_call>" in content:
        xml_result = _extract_tool_call_xml(content)
        if xml_result:
            return xml_result

    # Safely extract first JSON object in case model appended trailing text
    try:
        content_stripped = content.strip()
        idx = content_stripped.find("{")
        if idx != -1:
            envelope, _ = json.JSONDecoder().raw_decode(content_stripped[idx:])
        else:
            envelope = json.loads(content)
    except json.JSONDecodeError as error:
        logger.warning("Constrained decoder returned non-JSON text: %s", error)
        return content

    if not isinstance(envelope, dict):
        return content
    if envelope.get("kind") == "final" and isinstance(envelope.get("content"), str):
        return cast(str, envelope["content"])
    if envelope.get("kind") == "step_answer":
        return json.dumps(envelope, ensure_ascii=False)
    if envelope.get("kind") == "tool":
        return json.dumps(envelope, ensure_ascii=False)

    logger.warning("Constrained decoder returned unsupported envelope, treating as text.")
    return content


def structured_completion(messages: list[dict[str, Any]], max_tokens: int, temperature: float,
                          schema: dict[str, Any], llm_runtime: LLMRuntime,
                          backend_mode_fn: Callable[[], str],
                          openai_compatible_fn: Callable[..., dict[str, Any]]) -> str:
    """Generate internal planner/critic JSON with a given JSON Schema."""
    # ADR 0018: the same gate ``constrained_completion`` uses, hoisted *above*
    # the decode-error wrapper — inside it, the policy refusal is relabelled as
    # a JSON decoding failure and callers can not tell the two apart (finding 16).
    remote_backend = _remote_leg_allowed(backend_mode_fn)
    try:
        if remote_backend:
            response = openai_compatible_fn(
                messages, max_tokens=max_tokens, temperature=temperature,
                response_format={"type": "json_object"},
            )
        else:
            from llama_cpp.llama_grammar import LlamaGrammar
            backend = cast(LLMBackend, llm_runtime.llm)
            response = backend.generate(
                messages=messages, max_tokens=max_tokens, temperature=temperature,
                repeat_penalty=1.1, stream=False,
                grammar=LlamaGrammar.from_json_schema(json.dumps(schema)),
            )
    except Exception as error:
        msg = f"Structured JSON decoding failed: {error}"
        raise RuntimeError(msg) from error

    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not isinstance(content, str):
        # RuntimeError (not TypeError): same decoder contract as above.
        msg = "Structured decoder returned no text content"
        raise RuntimeError(msg)  # noqa: TRY004
    content = _strip_think_tags(content)
    logger.info("Structured LLM response: %s", content)
    return content


# Module-level grammar cache
_PRECOMPILED_GRAMMAR = None
_GRAMMAR_SCHEMA: str | None = None
_GRAMMAR_LOCK = threading.Lock()


def _get_precompiled_grammar(grammar_schema: str) -> Any:
    """Lazily compile and cache the llama.cpp grammar.

    One dedicated lock (the compile is not safe to run twice for the same
    schema) and the schema *text* as the key: a ``hash()`` collision could
    otherwise serve the grammar of a different schema (finding 26).
    """
    global _PRECOMPILED_GRAMMAR, _GRAMMAR_SCHEMA
    with _GRAMMAR_LOCK:
        if _GRAMMAR_SCHEMA == grammar_schema and _PRECOMPILED_GRAMMAR is not None:
            return _PRECOMPILED_GRAMMAR
        try:
            from llama_cpp.llama_grammar import LlamaGrammar
            _PRECOMPILED_GRAMMAR = LlamaGrammar.from_json_schema(grammar_schema)
            _GRAMMAR_SCHEMA = grammar_schema
        except ImportError:
            _PRECOMPILED_GRAMMAR = None
        return _PRECOMPILED_GRAMMAR
