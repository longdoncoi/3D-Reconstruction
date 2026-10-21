"""Pending approvals are sealed at rest, with legacy plaintext still readable."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_assistant.adapters.persistence import PendingActionStore
from ai_assistant.adapters.persistence.storage_crypto import (
    AesGcmFieldCipher,
    StorageDecryptionError,
    StorageKeyError,
    is_sealed,
)


def _cipher() -> AesGcmFieldCipher:
    return AesGcmFieldCipher(b"p" * 32)


class EncryptedPendingActionStoreTests(unittest.TestCase):
    def test_save_seals_the_payload_and_load_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            store = PendingActionStore(path, cipher=_cipher())
            store["act-1"] = {"tool": "write_file", "params": {"path": "secret.txt"}}
            store.save()

            raw = path.read_text(encoding="utf-8")
            self.assertTrue(is_sealed(raw))
            self.assertNotIn("write_file", raw)
            self.assertNotIn("secret.txt", raw)
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

            reloaded = PendingActionStore(path, cipher=_cipher())
            reloaded.load()
            self.assertEqual(reloaded["act-1"]["tool"], "write_file")

    def test_legacy_plaintext_is_still_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            path.write_text(json.dumps({"act-1": {"tool": "legacy"}}), encoding="utf-8")
            store = PendingActionStore(path, cipher=_cipher())
            store.load()
            self.assertEqual(store["act-1"]["tool"], "legacy")

    def test_a_legacy_plaintext_file_is_resealed_on_load(self):
        """The plaintext copy must stop existing on disk at the first read.

        A file written before encryption was enabled was left in the clear
        until the next save — which may be days away, or may never come, if
        the pending queue simply drains. The load re-seals it and writes it
        back, so the window is one read.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            path.write_text(json.dumps({"act-1": {"tool": "legacy"}}), encoding="utf-8")
            store = PendingActionStore(path, cipher=_cipher())
            store.load()

            raw = path.read_text(encoding="utf-8")
            self.assertTrue(raw.startswith("enc:v1:"), "the legacy file must be re-sealed")
            self.assertNotIn("legacy", raw)

            reloaded = PendingActionStore(path, cipher=_cipher())
            reloaded.load()
            self.assertEqual(reloaded["act-1"]["tool"], "legacy")

    def test_sealed_payload_without_a_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            PendingActionStore(path, cipher=_cipher()).save()
            blind = PendingActionStore(path)
            with self.assertRaises(StorageKeyError):
                blind.load()

    def test_tampered_payload_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            store = PendingActionStore(path, cipher=_cipher())
            store["act-1"] = {"tool": "write_file"}
            store.save()
            raw = path.read_text(encoding="utf-8")
            path.write_text(raw[:-1] + ("A" if raw[-1] != "A" else "B"), encoding="utf-8")

            reloaded = PendingActionStore(path, cipher=_cipher())
            with self.assertRaises(StorageDecryptionError):
                reloaded.load()

    def test_corrupt_plaintext_still_uses_the_legacy_fail_open_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            path.write_text("{not json", encoding="utf-8")
            store = PendingActionStore(path, cipher=_cipher())
            with self.assertLogs("ai_assistant.persistence.pending_store", "WARNING"):
                store.load()
            self.assertEqual(len(store), 0)


if __name__ == "__main__":
    unittest.main()
