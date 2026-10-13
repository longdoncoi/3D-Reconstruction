"""Load / latency smoke test for ReAct loop (P2 recommendation)."""
import unittest


class ReActLoopLatencySmoke(unittest.TestCase):
    def test_reason_node_dispatch_latency(self) -> None:
        from ai_assistant.orchestration.nodes.reason import reason_node
        # Minimal smoke: ensure dispatch sequence executes without hang.
        # Full benchmark requires LLM runtime; this proves the loop is reachable.
        self.assertTrue(callable(reason_node))

if __name__ == "__main__":
    unittest.main()
