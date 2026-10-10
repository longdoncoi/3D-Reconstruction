from ..adapters.persistence import PendingActionStore as _PendingActionStore

__all__ = ["PendingActionStore"]


# Keep the re-export as the only agent-side touchpoint for the persistence
# adapter; direct ``..adapters.persistence`` imports must not exist elsewhere
# in ``ai_assistant.agents`` (enforced by boundary tests).
PendingActionStore = _PendingActionStore
