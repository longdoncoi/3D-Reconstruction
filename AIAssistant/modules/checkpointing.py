"""Legacy re-export of checkpointing."""
from ai_assistant.adapters.persistence.checkpointing import (
    build_checkpointer,
    cleanup_old_checkpoints,
)

__all__ = [
    "build_checkpointer",
    "cleanup_old_checkpoints",
]
