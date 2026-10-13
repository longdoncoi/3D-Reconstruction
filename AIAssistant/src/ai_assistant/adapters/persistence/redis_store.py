"""Sprint 3 expanded: Redis adapter for pending actions / grants (replica-ready)."""
from __future__ import annotations

import json

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None  # type: ignore

class RedisPendingStore:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, prefix: str = "agent:") -> None:
        if redis is None:
            raise ImportError("bad_param")
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.prefix = prefix

    def save(self, action_id: str, payload: dict) -> None:
        self.client.setex(f"{self.prefix}pending:{action_id}", 3600, json.dumps(payload))

    def load(self, action_id: str) -> dict | None:
        raw = self.client.get(f"{self.prefix}pending:{action_id}")
        return json.loads(raw) if raw else None

    def delete(self, action_id: str) -> None:
        self.client.delete(f"{self.prefix}pending:{action_id}")
