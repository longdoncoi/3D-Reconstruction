"""At-rest field encryption and key-loading contracts.

The review's remaining gap was that task/pending state reached disk as
plaintext. These cases pin both halves of the fix: AES-GCM envelopes (including
context binding and tamper rejection) and the fail-closed key sources.
"""
from __future__ import annotations

import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ai_assistant.adapters.persistence.storage_crypto import (
    SEALED_BINARY_PREFIX,
    SEALED_PREFIX,
    AesGcmFieldCipher,
    StorageDecryptionError,
    StorageKeyError,
    _fsync_directory,
    is_sealed,
    load_storage_cipher,
)
from ai_assistant.settings import storage_key, storage_key_location

_KEY = bytes(range(32))
_B64_KEY = base64.urlsafe_b64encode(_KEY).decode("ascii")


class AesGcmFieldCipherTests(unittest.TestCase):
    def test_round_trip_binds_the_context(self):
        cipher = AesGcmFieldCipher(_KEY)
        sealed = cipher.seal("sensitive prompt", context="task.message")
        self.assertTrue(sealed.startswith(SEALED_PREFIX))
        self.assertTrue(cipher.is_sealed(sealed))
        self.assertEqual(cipher.open(sealed, context="task.message"), "sensitive prompt")
        with self.assertRaises(StorageDecryptionError):
            cipher.open(sealed, context="task.metadata")

    def test_legacy_plaintext_is_returned_unchanged(self):
        cipher = AesGcmFieldCipher(_KEY)
        self.assertEqual(cipher.open("old plaintext", context="task.message"), "old plaintext")
        self.assertFalse(cipher.is_sealed("old plaintext"))
        self.assertFalse(is_sealed("old plaintext"))

    def test_tampered_ciphertext_is_rejected(self):
        cipher = AesGcmFieldCipher(_KEY)
        sealed = cipher.seal("value", context="task.message")
        replacement = "A" if sealed[-1] != "A" else "B"
        with self.assertRaises(StorageDecryptionError):
            cipher.open(sealed[:-1] + replacement, context="task.message")

    def test_invalid_base64_envelope_is_rejected(self):
        cipher = AesGcmFieldCipher(_KEY)
        with self.assertRaises(StorageDecryptionError):
            cipher.open(SEALED_PREFIX + "!!!", context="task.message")

    def test_envelope_shorter_than_a_nonce_is_rejected(self):
        cipher = AesGcmFieldCipher(_KEY)
        short = base64.urlsafe_b64encode(b"short").decode("ascii")
        with self.assertRaises(StorageDecryptionError):
            cipher.open(SEALED_PREFIX + short, context="task.message")

    def test_non_utf8_plaintext_is_rejected(self):
        cipher = AesGcmFieldCipher(_KEY)
        nonce = b"\x00" * 12
        encrypted = AESGCM(_KEY).encrypt(nonce, b"\xff", b"ai_assistant.local_state.v1:task.message")
        envelope = SEALED_PREFIX + base64.urlsafe_b64encode(nonce + encrypted).decode("ascii")
        with self.assertRaises(StorageDecryptionError):
            cipher.open(envelope, context="task.message")

    def test_key_length_is_validated(self):
        with self.assertRaises(StorageKeyError):
            AesGcmFieldCipher(b"too short")


class StorageKeyLocationTests(unittest.TestCase):
    def test_key_override_is_stripped(self):
        self.assertEqual(storage_key({"AI_STORAGE_KEY": "  hex:00  "}), "hex:00")
        self.assertEqual(storage_key({}), "")

    def test_explicit_key_file_wins(self):
        path, explicit = storage_key_location({"AI_STORAGE_KEY_FILE": "  C:/secrets/key  "})
        self.assertEqual(path, Path("C:/secrets/key"))
        self.assertTrue(explicit)

    def test_windows_location_uses_local_app_data_then_app_data(self):
        with mock.patch("ai_assistant.settings.sys.platform", "win32"):
            local = storage_key_location({"LOCALAPPDATA": "C:/Local"})
            roaming = storage_key_location({"APPDATA": "C:/Roaming"})
            fallback = storage_key_location({})
        self.assertEqual(local, (Path("C:/Local/3d-reconstruction/ai_assistant.key"), False))
        self.assertEqual(roaming, (Path("C:/Roaming/3d-reconstruction/ai_assistant.key"), False))
        self.assertEqual(fallback[0].name, "ai_assistant.key")
        self.assertEqual(fallback[0].parent.name, "3d-reconstruction")
        self.assertFalse(fallback[1])

    def test_posix_location_uses_xdg_then_home(self):
        with mock.patch("ai_assistant.settings.sys.platform", "linux"):
            xdg = storage_key_location({"XDG_CONFIG_HOME": "/xdg-root/config"})
            fallback = storage_key_location({})
        self.assertEqual(xdg, (Path("/xdg-root/config/3d-reconstruction/ai_assistant.key"), False))
        self.assertEqual(fallback[0].name, "ai_assistant.key")
        self.assertEqual(fallback[0].parent.name, "3d-reconstruction")
        self.assertFalse(fallback[1])


class KeyLoadingTests(unittest.TestCase):
    def test_environment_key_accepts_base64_and_hex(self):
        for value in (_B64_KEY, f"base64:{_B64_KEY}", f"hex:{_KEY.hex()}"):
            with self.subTest(value=value):
                cipher = load_storage_cipher({"AI_STORAGE_KEY": value})
                self.assertEqual(cipher.open(cipher.seal("x", context="test"), context="test"), "x")

    def test_invalid_environment_keys_fail_closed(self):
        for value in ("hex:zz", "!!!not-base64", f"hex:{'00' * 16}"):
            with self.subTest(value=value):
                with self.assertRaises(StorageKeyError):
                    load_storage_cipher({"AI_STORAGE_KEY": value})

    def test_explicit_key_file_is_read_and_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key"
            path.write_text(f"base64:{_B64_KEY}\n", encoding="ascii")
            cipher = load_storage_cipher({"AI_STORAGE_KEY_FILE": str(path)})
            self.assertEqual(cipher.open(cipher.seal("x", context="test"), context="test"), "x")
            path.unlink()
            with self.assertRaises(StorageKeyError):
                load_storage_cipher({"AI_STORAGE_KEY_FILE": str(path)})

    def test_an_unreadable_key_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key"
            path.mkdir()
            with self.assertRaises(StorageKeyError):
                load_storage_cipher({"AI_STORAGE_KEY_FILE": str(path)})

    def test_generated_key_is_created_once_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "key"
            with mock.patch(
                "ai_assistant.adapters.persistence.storage_crypto.storage_key_location",
                return_value=(path, False),
            ):
                first = load_storage_cipher({})
                second = load_storage_cipher({})
            self.assertTrue(path.is_file())
            self.assertEqual(first.open(second.seal("x", context="test"), context="test"), "x")
            # Both loads must use the key that is on disk, not a private one.
            on_disk = AesGcmFieldCipher(
                base64.urlsafe_b64decode(
                    path.read_text(encoding="ascii").strip().removeprefix("base64:"),
                ),
            )
            self.assertEqual(first.open(on_disk.seal("x", context="test"), context="test"), "x")
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_a_racing_creator_loads_the_winning_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key"

            def create_then_lose(_path: str | Path, _flags: int, _mode: int) -> int:
                Path(_path).write_text(f"base64:{_B64_KEY}\n", encoding="ascii")
                raise FileExistsError

            with mock.patch(
                "ai_assistant.adapters.persistence.storage_crypto.storage_key_location",
                return_value=(path, False),
            ), mock.patch(
                "ai_assistant.adapters.persistence.storage_crypto.os.open",
                side_effect=create_then_lose,
            ):
                cipher = load_storage_cipher({})
            # The loser must adopt the winner's key, not keep the one it
            # generated: a self-consistent round trip would not notice the
            # difference, but two processes would then disagree about which
            # key seals the state they share.
            winner = AesGcmFieldCipher(_KEY)
            self.assertEqual(cipher.open(winner.seal("x", context="test"), context="test"), "x")


class BinaryEnvelopeTests(unittest.TestCase):
    """The binary form seals whole artifacts (the RAG cache) without base64."""

    def test_round_trip_binds_the_context(self):
        cipher = AesGcmFieldCipher(_KEY)
        sealed = cipher.seal_bytes(b"artifact payload", context="rag.chunks")

        self.assertTrue(sealed.startswith(SEALED_BINARY_PREFIX))
        self.assertEqual(cipher.open_bytes(sealed, context="rag.chunks"), b"artifact payload")
        with self.assertRaises(StorageDecryptionError):
            cipher.open_bytes(sealed, context="rag.index")

    def test_an_unsealed_value_is_refused(self):
        """Unlike the text form there is no legacy passthrough for binaries."""
        cipher = AesGcmFieldCipher(_KEY)

        with self.assertRaises(StorageDecryptionError):
            cipher.open_bytes(b"a plaintext pickle", context="rag.chunks")

    def test_a_tampered_envelope_is_rejected(self):
        cipher = AesGcmFieldCipher(_KEY)
        sealed = cipher.seal_bytes(b"artifact payload", context="rag.chunks")
        replacement = b"A" if sealed[-1:] != b"A" else b"B"

        with self.assertRaises(StorageDecryptionError):
            cipher.open_bytes(sealed[:-1] + replacement, context="rag.chunks")

    def test_an_envelope_shorter_than_a_nonce_is_rejected(self):
        cipher = AesGcmFieldCipher(_KEY)

        with self.assertRaises(StorageDecryptionError):
            cipher.open_bytes(SEALED_BINARY_PREFIX + b"short", context="rag.chunks")

    def test_a_non_utf8_payload_is_a_valid_artifact(self):
        """Binary artifacts are bytes, not text: no UTF-8 round trip involved."""
        cipher = AesGcmFieldCipher(_KEY)
        payload = b"\xff\xfe\x00binary"

        self.assertEqual(cipher.open_bytes(
            cipher.seal_bytes(payload, context="rag.chunks"), context="rag.chunks"), payload)


class KeyFileDurabilityTests(unittest.TestCase):
    """The key file must survive a crash.

    ``fsync`` on the file only persists the *data*; the directory entry that
    makes the key visible is separate metadata. If it is not flushed, a power
    loss can leave the key file missing — and the next start would generate a
    fresh key, orphaning every row the file was meant to protect.
    """

    def test_the_directory_entry_is_flushed_after_the_key_is_written(self):
        with mock.patch(
            "ai_assistant.adapters.persistence.storage_crypto.os.open",
            return_value=4242,
        ) as opened, mock.patch(
            "ai_assistant.adapters.persistence.storage_crypto.os.fsync",
        ) as fsync, mock.patch(
            "ai_assistant.adapters.persistence.storage_crypto.os.close",
        ) as closed:
            _fsync_directory(Path("C:/keydir"))
        opened.assert_called_once()
        fsync.assert_called_once_with(4242)
        closed.assert_called_once_with(4242)

    def test_a_filesystem_that_refuses_the_directory_handle_is_not_fatal(self):
        with mock.patch(
            "ai_assistant.adapters.persistence.storage_crypto.os.open",
            side_effect=OSError("directory handles are not supported here"),
        ):
            _fsync_directory(Path("C:/keydir"))

    def test_a_filesystem_that_refuses_to_flush_the_directory_is_not_fatal(self):
        with mock.patch(
            "ai_assistant.adapters.persistence.storage_crypto.os.open",
            return_value=4242,
        ), mock.patch(
            "ai_assistant.adapters.persistence.storage_crypto.os.fsync",
            side_effect=OSError("flush refused"),
        ), mock.patch(
            "ai_assistant.adapters.persistence.storage_crypto.os.close",
        ):
            _fsync_directory(Path("C:/keydir"))

    def test_creating_the_key_file_flushes_its_directory_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "key"
            with mock.patch(
                "ai_assistant.adapters.persistence.storage_crypto.storage_key_location",
                return_value=(path, False),
            ), mock.patch(
                "ai_assistant.adapters.persistence.storage_crypto._fsync_directory",
            ) as flushed:
                load_storage_cipher({})
            flushed.assert_called_once_with(path.parent)


if __name__ == "__main__":
    unittest.main()
