"""Small deterministic checks for RAG ranking policies."""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.rag_module import _rrf_fuse


class TestReciprocalRankFusion(unittest.TestCase):
    def test_agreement_outranks_single_retriever_hit(self):
        # Chunk 2 is supported by both retrievers; chunk 1 only by semantic.
        ranked = _rrf_fuse([[1, 2], [2, 3]], [0.55, 0.45])
        self.assertEqual(ranked[0][0], 2)

    def test_ignores_invalid_chunk_ids(self):
        ranked = _rrf_fuse([[-1, 7]], [0.55])
        self.assertEqual([chunk_id for chunk_id, _ in ranked], [7])


if __name__ == "__main__":
    unittest.main()
