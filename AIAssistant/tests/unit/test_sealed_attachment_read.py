"""The AI service must be able to read a chat attachment sealed at rest.

The Qt client stores the attached files encrypted (ADR 0029) with the
platform at-rest key, so the service has to open them before the model can
see them. These cases pin both halves: the kernel helper recognises a sealed
envelope and hands it to the injected unsealer, and the readers that actually
touch an attachment (the chat route and the vision prompt builder) pass the
unsealer through.
"""
from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from ai_assistant.adapters.persistence.storage_crypto import AesGcmFieldCipher
from ai_assistant.image_utils import image_to_data_uri, read_sealed_or_plain
from ai_assistant.llm.prompts import DefaultPromptBuilder
from ai_assistant.ports import CHAT_ATTACHMENT_CONTEXT, SEALED_BINARY_PREFIX

_KEY = b"s" * 32
# A 1x1 PNG — small enough to inline, real enough for PIL.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _cipher() -> AesGcmFieldCipher:
    return AesGcmFieldCipher(_KEY)


def _unseal(raw: bytes) -> bytes:
    return _cipher().open_bytes(raw, context=CHAT_ATTACHMENT_CONTEXT)


class SealedReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_a_sealed_file_is_opened_with_the_injected_unsealer(self):
        path = self.dir / "sealed.bin"
        path.write_bytes(_cipher().seal_bytes(b"payload", context=CHAT_ATTACHMENT_CONTEXT))

        self.assertEqual(read_sealed_or_plain(str(path), _unseal), b"payload")

    def test_a_plaintext_file_is_returned_unchanged(self):
        """An attachment written before sealing existed still loads."""
        path = self.dir / "legacy.bin"
        path.write_bytes(b"plain payload")

        self.assertEqual(read_sealed_or_plain(str(path), _unseal), b"plain payload")

    def test_without_an_unsealer_a_sealed_file_is_left_alone(self):
        """No key means no guessing: the bytes are returned as they are."""
        path = self.dir / "sealed.bin"
        raw = _cipher().seal_bytes(b"payload", context=CHAT_ATTACHMENT_CONTEXT)
        path.write_bytes(raw)

        self.assertEqual(read_sealed_or_plain(str(path)), raw)

    def test_an_image_is_encoded_from_the_sealed_bytes(self):
        path = self.dir / "photo.png"
        path.write_bytes(_cipher().seal_bytes(_PNG, context=CHAT_ATTACHMENT_CONTEXT))

        uri = image_to_data_uri(str(path), unseal=_unseal)

        self.assertTrue(uri.startswith("data:image/png;base64,"))
        self.assertNotIn("enc:v1:", uri)

    def test_a_sealed_image_without_the_key_does_not_leak_bytes(self):
        """A wrong key fails the read instead of handing back ciphertext."""
        path = self.dir / "photo.png"
        path.write_bytes(_cipher().seal_bytes(_PNG, context=CHAT_ATTACHMENT_CONTEXT))

        def wrong(raw: bytes) -> bytes:
            return AesGcmFieldCipher(b"z" * 32).open_bytes(
                raw, context=CHAT_ATTACHMENT_CONTEXT)

        with self.assertRaises(Exception):
            image_to_data_uri(str(path), unseal=wrong)


class VisionPromptSealedAttachmentTests(unittest.TestCase):
    def test_the_vision_builder_reads_a_sealed_attachment(self):
        """The builder must open the attachment, not report it as unreadable."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.png"
            path.write_bytes(_cipher().seal_bytes(_PNG, context=CHAT_ATTACHMENT_CONTEXT))
            builder = DefaultPromptBuilder(unseal=_unseal)

            prepared = builder.build_vision_messages(
                [{"role": "user", "content": "what is this", "attachments": [str(path)]}],
                doc_ctx="", code_ctx="",
            )

        rendered = json.dumps(prepared)
        self.assertIn("data:image/png;base64,", rendered)
        self.assertNotIn("Lỗi đọc ảnh", rendered)


class CompositionRootUnsealerTests(unittest.TestCase):
    """The composition root's unsealer opens what the Qt client sealed."""

    def test_the_unsealer_opens_a_sealed_attachment(self):
        from ai_assistant.bootstrap.modules import attachment_unsealer

        unseal = attachment_unsealer(_cipher())
        sealed = _cipher().seal_bytes(b"payload", context=CHAT_ATTACHMENT_CONTEXT)

        self.assertEqual(unseal(sealed), b"payload")

    def test_the_unsealer_refuses_a_blob_sealed_with_another_key(self):
        from ai_assistant.bootstrap.modules import attachment_unsealer

        unseal = attachment_unsealer(AesGcmFieldCipher(b"z" * 32))
        sealed = _cipher().seal_bytes(b"payload", context=CHAT_ATTACHMENT_CONTEXT)

        with self.assertRaises(Exception):
            unseal(sealed)


# ── Cross-language contract ──────────────────────────────────────────────────
# The Qt client seals the attachment and the AI service opens it: the envelope
# layout, the associated data and the key resolution have to match across the
# two languages. Each side pins the *other* side's output as a fixed vector,
# so a change to one implementation that breaks the other fails here instead
# of surfacing as "the assistant cannot see the image".
#
# Vector: key = 32 x 0x5A, context = "chat.attachment",
#         plaintext = "ai-assistant at-rest interop vector"
_INTEROP_KEY = bytes([0x5A]) * 32
_INTEROP_PLAINTEXT = b"ai-assistant at-rest interop vector"

#: Produced by the Python service (ai_assistant.adapters.persistence.storage_crypto).
_PYTHON_ENVELOPE_HEX = (
    "656e633a76313a77906afb38dfe822faff2836539a6e0aaf5cd33c1d4e767285de382c20d0f64"
    "976c3da8ca9ff527e5f61979b50f6fa9b8bda44fbc00cd1243740c6d5438b43"
)

#: Produced by the Qt client (src/core/AtRestCipher.cpp). The nonce is random,
#: so the vector is a captured sample — regenerate it by running
#: ``TestAtRestCipher::dumpInteropEnvelope`` and pasting the hex it prints.
_CPP_ENVELOPE_HEX = (
    "656e633a76313a39848c95f76743b8b08b3fa70deaad7df08551abbc35475103e9002d7b58a911"
    "a3c94222177148b171c1a5d32ec9e8960124d180bfca6d46d4af4353035e51"
)


class InteropVectorTests(unittest.TestCase):
    def test_the_python_envelope_has_the_documented_shape(self):
        envelope = bytes.fromhex(_PYTHON_ENVELOPE_HEX)

        self.assertTrue(envelope.startswith(SEALED_BINARY_PREFIX))
        self.assertEqual(len(envelope), len(SEALED_BINARY_PREFIX) + 12 + len(_INTEROP_PLAINTEXT) + 16)

    def test_the_python_envelope_opens_with_the_shared_key(self):
        envelope = bytes.fromhex(_PYTHON_ENVELOPE_HEX)

        self.assertEqual(
            AesGcmFieldCipher(_INTEROP_KEY).open_bytes(
                envelope, context=CHAT_ATTACHMENT_CONTEXT),
            _INTEROP_PLAINTEXT,
        )

    def test_the_cpp_envelope_opens_with_the_shared_key(self):
        """The production direction: the Qt client seals, the service opens."""
        envelope = bytes.fromhex(_CPP_ENVELOPE_HEX)

        self.assertTrue(envelope.startswith(SEALED_BINARY_PREFIX))
        self.assertEqual(
            AesGcmFieldCipher(_INTEROP_KEY).open_bytes(
                envelope, context=CHAT_ATTACHMENT_CONTEXT),
            _INTEROP_PLAINTEXT,
        )


if __name__ == "__main__":
    unittest.main()
