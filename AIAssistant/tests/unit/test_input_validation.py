"""Guard-path coverage for the tool-input schema validator.

``validate_input`` is the minimal JSON-Schema subset enforced at every tool
protocol boundary. Each rejection string is a distinct contract with tool
authors and model callers: a non-object root, an unknown parameter, a wrong
type, a disallowed enum value and numeric bounds on both sides. The
``request_id`` carve-out (framework correlation metadata, not model input)
is pinned so it cannot regress into a false rejection.
"""
from __future__ import annotations

import unittest

from ai_assistant.application.validation import validate_input


class RootShapeTests(unittest.TestCase):
    def test_an_empty_schema_accepts_anything(self):
        self.assertIsNone(validate_input({}, {"whatever": 1}))

    def test_a_non_object_root_is_rejected(self):
        self.assertEqual(
            validate_input({"type": "array"}, []),
            "Tool schema root must be an object",
        )


class RequiredParameterTests(unittest.TestCase):
    def test_a_missing_required_parameter_is_rejected(self):
        schema = {
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
        }
        self.assertEqual(
            validate_input(schema, {}),
            "Missing required parameter: path",
        )
        self.assertIsNone(validate_input(schema, {"path": "a"}))


class UnknownParameterTests(unittest.TestCase):
    _SCHEMA = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "additionalProperties": False,
    }

    def test_an_unknown_parameter_is_rejected(self):
        self.assertEqual(
            validate_input(self._SCHEMA, {"path": "a", "extra": 1}),
            "Unknown parameter(s): extra",
        )

    def test_the_framework_request_id_is_not_an_unknown_parameter(self):
        """``request_id`` is correlation metadata, not model-controlled input."""
        self.assertIsNone(validate_input(self._SCHEMA, {"path": "a", "request_id": "r1"}))

    def test_unknown_parameters_are_allowed_by_default(self):
        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        self.assertIsNone(validate_input(schema, {"path": "a", "extra": 1}))


class TypeCheckTests(unittest.TestCase):
    def test_a_wrong_type_is_rejected(self):
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        self.assertEqual(
            validate_input(schema, {"count": "three"}),
            "Parameter count must be integer",
        )

    def test_a_boolean_is_not_an_integer(self):
        """Python's ``bool < int`` subclassing must not smuggle flags into counts."""
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        self.assertEqual(
            validate_input(schema, {"count": True}),
            "Parameter count must be integer",
        )

    def test_a_float_is_a_number(self):
        schema = {"type": "object", "properties": {"ratio": {"type": "number"}}}
        self.assertIsNone(validate_input(schema, {"ratio": 0.5}))

    def test_an_undeclared_parameter_skips_type_checks(self):
        """No definition means no opinion — the model may pass it through."""
        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        self.assertIsNone(validate_input(schema, {"other": object()}))


class EnumAndRangeTests(unittest.TestCase):
    def test_a_value_outside_the_enum_is_rejected(self):
        schema = {"type": "object", "properties": {"mode": {"enum": ["a", "b"]}}}
        self.assertEqual(
            validate_input(schema, {"mode": "c"}),
            "Parameter mode is not an allowed value",
        )
        self.assertIsNone(validate_input(schema, {"mode": "a"}))

    def test_a_value_below_minimum_is_rejected(self):
        schema = {"type": "object", "properties": {"top_k": {"minimum": 1}}}
        self.assertEqual(
            validate_input(schema, {"top_k": 0}),
            "Parameter top_k is below its minimum",
        )

    def test_a_value_above_maximum_is_rejected(self):
        schema = {"type": "object", "properties": {"top_k": {"maximum": 10}}}
        self.assertEqual(
            validate_input(schema, {"top_k": 11}),
            "Parameter top_k exceeds its maximum",
        )

    def test_a_value_within_bounds_passes(self):
        schema = {"type": "object", "properties": {"top_k": {"minimum": 1, "maximum": 10}}}
        self.assertIsNone(validate_input(schema, {"top_k": 5}))


if __name__ == "__main__":
    unittest.main()
