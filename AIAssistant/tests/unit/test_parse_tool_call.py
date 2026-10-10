# ruff: noqa: I001, S101
"""Regression checks for grammar-constrained tool envelopes."""
import sys
from unittest.mock import patch
sys.path.insert(0, '..')   # f:\PROJECTS\QT\3D-Reconstruction\AIAssistant is cwd
sys.path.insert(0, '.')

from ai_assistant.adapters import legacy_agent as agent_module
from ai_assistant.adapters.legacy_agent import _parse_tool_call

def test(name, text, expected_tool, expected_action=None):
    t, p = _parse_tool_call(text)
    assert t == expected_tool, f"[FAIL] {name}: tool='{t}' expected='{expected_tool}'"
    if expected_action and p:
        assert p.get("action") == expected_action, f"[FAIL] {name}: action='{p.get('action')}' expected='{expected_action}'"
    print(f"[PASS] {name}: tool={t}, action={p.get('action') if p else None}")

# 1. Canonical grammar envelope
test("canonical tool call",
     '{"kind": "tool", "tool": "application_action", "params": {"action": "reconstruction.load_images"}}',
     "application_action", "reconstruction.load_images")

# 2. Backward-compatible tool alias, still inside a constrained envelope
test("app action alias",
     '{"kind": "tool", "tool": "app_action_reconstruction", "params": {"action": "reconstruction.load_images"}}',
     "application_action", "reconstruction.load_images")

# 3. Manifest alias is canonicalised before execution
test("manifest action alias",
     '{"kind": "tool", "tool": "application_action", "params": {"action": "mail.inbox"}}',
     "application_action", "mail.open")

# 4. Final envelope is not a tool call
t, p = _parse_tool_call('{"kind": "final", "content": "Done"}')
assert t is None, f"[FAIL] no tool: got '{t}'"
print(f"[PASS] no tool call: t={t}")

# 5. Invalid parameter values cannot become a tool call
test("invalid desktop params",
     '{"kind": "tool", "tool": "application_action", "params": {"action": "language.change", "language": "fr"}}',
     "_validation_error")

# 6. start_reconstruction
test("start_reconstruction",
     '{"kind": "tool", "tool": "application_action", "params": {"action": "reconstruction.start_reconstruction"}}',
     "application_action", "reconstruction.start_reconstruction")

# 7. The constrained llama.cpp adapter must preserve a step-answer envelope.
# Otherwise LangGraph receives raw text and repeatedly executes the same plan step.
class _StepAnswerModel:
    def create_chat_completion(self, **_kwargs):
        return {"choices": [{"message": {"content":
                '{"kind":"step_answer","content":"Teamlead điều phối dự án."}'}}]}


with (patch.object(agent_module, "backend_mode", return_value="llama_cpp"),
      patch.object(agent_module.llm_runtime, "llm", _StepAnswerModel()),
      patch.dict(agent_module.os.environ, {"AGENT_NATIVE_TOOL_CALLS": "1"})):
    response = agent_module._constrained_agent_completion([], max_tokens=32, temperature=0.0)

assert response == '{"kind": "step_answer", "content": "Teamlead điều phối dự án."}', response
assert _parse_tool_call(response)[0] == "_step_answer"
print("[PASS] constrained step_answer envelope is preserved")

print("\nALL TESTS PASSED")
