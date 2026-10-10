"""Legacy re-export of checkpointing."""
from ..adapters.persistence.checkpointing import (
    build_checkpointer,
    cleanup_old_checkpoints,
)

__all__ = [
    "build_checkpointer",
    "cleanup_old_checkpoints",
]
