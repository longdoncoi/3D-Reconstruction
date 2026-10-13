"""Coverage target for legacy adapter shims (ADR 0014 / 0018)."""
import unittest

from ai_assistant.bootstrap.server import rag_module


class LegacyAdapterShardCoverage(unittest.TestCase):
    def test_rag_adapter_shim_vector_retriever(self) -> None:
        # Covers _RAGModule.vector_retriever legacy compatibility line.
        self.assertIsNone(rag_module.vector_retriever)

    def test_rag_adapter_shim_build_retrievers(self) -> None:
        # Covers _build_retrievers no-op legacy bridge.
        rag_module._build_retrievers()

if __name__ == "__main__":
    unittest.main()
