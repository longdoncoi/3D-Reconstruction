"""Kiểm chứng P0 #5: Pending store không ghi plaintext PDF/password."""
from __future__ import annotations

import unittest

from ai_assistant.domain.governance import redact_value


class TestPendingRedact(unittest.TestCase):
    def test_redact_applies_to_dict_with_password(self) -> None:
        raw = {"action": "login", "params": {"password": "secret123", "user": "admin"}}
        redacted = redact_value(raw)
        self.assertIn("[redacted]", str(redacted))
        self.assertNotIn("secret123", str(redacted))

if __name__ == "__main__":
    unittest.main()
