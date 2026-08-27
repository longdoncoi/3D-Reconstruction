# ruff: noqa: I001, S101
"""Regression checks for grammar-constrained tool envelopes."""
import sys
sys.path.insert(0, '..')   # f:\PROJECTS\QT\3D-Reconstruction\AIAssistant is cwd
sys.path.insert(0, '.')

from modules.agent_module import _parse_tool_call

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

print("\nALL TESTS PASSED")
