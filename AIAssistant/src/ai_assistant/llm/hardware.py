"""Hardware and memory management utilities."""
from __future__ import annotations

import gc


def release_ml_memory() -> None:
    """Force garbage collection and clear GPU caches."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
