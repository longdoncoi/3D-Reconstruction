"""LLM completion adapters for the agent loop.

Two modes:
  * constrained  — one JSON envelope per call (tool | final | step_answer)
  * structured   — internal planner/critic JSON with a supplied JSON Schema
"""
from __future__ import annotations

import json
import logging
import re

from ai_assistant.settings import load_agent_runtime_settings

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

# Normalize legacy/alias tool names to canonical
_TOOL_NAME_ALIASES = {
    "app_action_reconstruction": "application_action",
    "app_action_general": "application_action",
    "app_action_ai": "application_action",
    "app_action_viewer": "application_action",
    "application_actions": "application_action",
    "app_action": "application_action",
    "desktop_action": "application_action",
    "ui_action": "application_action",
}


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


def parse_tool_call(response_text: str, tool_models: dict[str, type],
                    validate_fn, record_schema_error_fn) -> tuple[str | None, dict | None]:
    """Decode the JSON envelope from constrained decoding.

    Returns:
      (tool_name, params)  — where tool_name may be "_step_answer" or "_validation_error"
    """
    def _normalise(name: str) -> str:
        return _TOOL_NAME_ALIASES.get(name, name)

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
    validated, error = validate_fn(tool_name, params, tool_models)
    if error:
        logger.warning("Rejected invalid constrained tool call %s: %s", tool_name, error)
        record_schema_error_fn(tool_name)
        return "_validation_error", {"tool": tool_name, "error": error}
    return tool_name, validated


def constrained_completion(messages: list[dict], max_tokens: int, temperature: float,
                           llm_runtime, backend_mode_fn, openai_compatible_fn,
                           openai_tools: list[dict], grammar_schema: str,
                           record_token_usage_fn) -> str:
    """Generate exactly one final/tool JSON envelope."""
    try:
        if backend_mode_fn() != "llama_cpp":
            response = openai_compatible_fn(
                messages, max_tokens=max_tokens, temperature=temperature,
                tools=openai_tools, tool_choice="auto",
                response_format={"type": "json_object"},
            )
            usage = response.get("usage", {})
            in_tok, out_tok = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            if in_tok or out_tok:
                record_token_usage_fn(in_tok, out_tok)
            message = response.get("choices", [{}])[0].get("message", {})
            if message.get("tool_calls"):
                call = message["tool_calls"][0]["function"]
                content = json.dumps({"kind": "tool", "tool": call["name"],
                                      "params": json.loads(call.get("arguments", "{}"))}, ensure_ascii=False)
                logger.info("Constrained LLM response (remote): %s", content)
                return content
            content = message.get("content", "")
            logger.info("Constrained LLM response (remote): %s", content)
            return content

        # Local llama.cpp path
        _precompiled_grammar = _get_precompiled_grammar(grammar_schema)
        kwargs: dict = {
            "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "repeat_penalty": 1.1, "stream": False,
        }
        if load_agent_runtime_settings().native_tool_calls:
            kwargs.update({"tools": openai_tools, "tool_choice": "auto"})
        elif _precompiled_grammar is not None:
            kwargs.update({"grammar": _precompiled_grammar})

        with llm_runtime.llm_lock:
            response = llm_runtime.llm.create_chat_completion(**kwargs)

    except Exception as error:
        raise RuntimeError(f"Constrained tool decoding failed: {error}") from error

    usage = response.get("usage", {})
    in_tok, out_tok = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    if in_tok or out_tok:
        record_token_usage_fn(in_tok, out_tok)

    message = response.get("choices", [{}])[0].get("message", {})
    if message.get("tool_calls"):
        call = message["tool_calls"][0]["function"]
        content = json.dumps({"kind": "tool", "tool": call["name"],
                              "params": json.loads(call.get("arguments", "{}"))}, ensure_ascii=False)
        logger.info("Constrained LLM response: %s", content)
        return content

    content = message.get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("Constrained decoder returned no text content")
    content = _strip_think_tags(content)
    logger.info("Constrained LLM response: %s", content)

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
        return envelope["content"]
    if envelope.get("kind") == "step_answer":
        return json.dumps(envelope, ensure_ascii=False)
    if envelope.get("kind") == "tool":
        return json.dumps(envelope, ensure_ascii=False)

    logger.warning("Constrained decoder returned unsupported envelope, treating as text.")
    return content


def structured_completion(messages: list[dict], max_tokens: int, temperature: float,
                          schema: dict, llm_runtime, backend_mode_fn,
                          openai_compatible_fn) -> str:
    """Generate internal planner/critic JSON with a given JSON Schema."""
    try:
        if backend_mode_fn() != "llama_cpp":
            response = openai_compatible_fn(
                messages, max_tokens=max_tokens, temperature=temperature,
                response_format={"type": "json_object"},
            )
        else:
            from llama_cpp import LlamaGrammar
            with llm_runtime.llm_lock:
                response = llm_runtime.llm.create_chat_completion(
                    messages=messages, max_tokens=max_tokens, temperature=temperature,
                    repeat_penalty=1.1, stream=False,
                    grammar=LlamaGrammar.from_json_schema(json.dumps(schema)),
                )
    except Exception as error:
        raise RuntimeError(f"Structured JSON decoding failed: {error}") from error

    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("Structured decoder returned no text content")
    content = _strip_think_tags(content)
    logger.info("Structured LLM response: %s", content)
    return content


# Module-level grammar cache
_PRECOMPILED_GRAMMAR = None
_GRAMMAR_SCHEMA_HASH: str | None = None


def _get_precompiled_grammar(grammar_schema: str):
    """Lazily compile and cache the llama.cpp grammar."""
    global _PRECOMPILED_GRAMMAR, _GRAMMAR_SCHEMA_HASH
    current_hash = str(hash(grammar_schema))
    if _GRAMMAR_SCHEMA_HASH == current_hash and _PRECOMPILED_GRAMMAR is not None:
        return _PRECOMPILED_GRAMMAR
    try:
        from llama_cpp import LlamaGrammar
        _PRECOMPILED_GRAMMAR = LlamaGrammar.from_json_schema(grammar_schema)
        _GRAMMAR_SCHEMA_HASH = current_hash
    except ImportError:
        _PRECOMPILED_GRAMMAR = None
    return _PRECOMPILED_GRAMMAR
