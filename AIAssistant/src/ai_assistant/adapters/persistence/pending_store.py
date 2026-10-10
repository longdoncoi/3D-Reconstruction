"""Thread-safe, dict-compatible persistence store for pending approvals and UI actions."""
from __future__ import annotations

import json
import threading
from pathlib import Path


class PendingActionStore:
    """Thread-safe, dict-compatible persistence store for pending approvals and UI actions."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def __getitem__(self, key: str) -> dict:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: dict) -> None:
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self):
        with self._lock:
            return iter(list(self._data.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def pop(self, key: str, default: dict | None = None) -> dict | None:
        with self._lock:
            return self._data.pop(key, default)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def load(self) -> None:
        with self._lock:
            try:
                if not self._path.exists():
                    return
                with open(self._path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    self._data.update(
                        {str(key): value for key, value in payload.items() if isinstance(value, dict)}
                    )
            except (OSError, ValueError):
                pass

    def save(self) -> None:
        with self._lock:
            try:
                if not self._path.name:
                    return
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self._path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as handle:
                    json.dump(dict(self._data), handle, ensure_ascii=False, indent=2, default=str)
                temp_path.replace(self._path)
            except OSError:
                pass

    def cleanup(self, cutoff: float) -> bool:
        with self._lock:
            expired = [
                key for key, value in self._data.items()
                if (created_at := value.get("created_at")) is not None and created_at < cutoff
            ]
            for key in expired:
                del self._data[key]
            return bool(expired)
