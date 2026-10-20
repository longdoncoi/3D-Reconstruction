"""Coverage target for legacy adapter shims (ADR 0014 / 0018)."""
import unittest

from ai_assistant.adapters.persistence.storage_crypto import AesGcmFieldCipher
from ai_assistant.bootstrap.modules import RagModule


class LegacyAdapterShardCoverage(unittest.TestCase):
    def test_rag_adapter_shim_build_retrievers(self) -> None:
        # Covers the no-op legacy bridge: with no index state it hands back
        # the empty pair instead of building anything.
        self.assertEqual(
            RagModule(cipher=AesGcmFieldCipher(b"l" * 32))._build_retrievers(),
            (None, None),
        )

if __name__ == "__main__":
    unittest.main()
