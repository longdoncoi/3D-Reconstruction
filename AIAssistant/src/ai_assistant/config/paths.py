"""Path resolution and directory management for the platform."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..settings import app_data_dir


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Resolved absolute paths for all application directories."""
    project_root: Path
    base_dir: Path
    data_dir: Path
    models_dir: Path
    cache_dir: Path
    logs_dir: Path
    docs_dirs: tuple[Path, ...]
    
    @property
    def embed_cache_dir(self) -> Path:
        return self.cache_dir / "embed_model"
    
    @property
    def cache_index(self) -> Path:
        return self.cache_dir / "faiss_index.bin"
    
    @property
    def cache_chunks(self) -> Path:
        return self.cache_dir / "chunks.pkl"
    
    @property
    def cache_bm25(self) -> Path:
        return self.cache_dir / "bm25.pkl"
    
    @property
    def cache_metadata(self) -> Path:
        return self.cache_dir / "metadata.json"


def resolve_paths(base_dir: Path | str, explicit_data_dir: Path | str | None = None) -> AppPaths:
    """Resolve all application paths relative to the base directory.
    
    Creates required directories (logs, cache, models) if they do not exist.
    """
    base = Path(base_dir).resolve()
    project = base.parent
    
    # Allow overriding data dir from settings or environment
    if explicit_data_dir:
        data = Path(explicit_data_dir).resolve()
    else:
        app_data_env = app_data_dir()
        if app_data_env:
            data = Path(app_data_env).resolve() / "AIAssistant"
        else:
            data = project
            
    # Legacy migration: use AITraining if AIAssistant doesn't exist yet but AITraining does
    def _existing_data_path(name: str) -> Path:
        primary = data / name
        legacy = project / "AITraining" / name
        if not primary.exists() and legacy.exists():
            return legacy
        return primary

    models_dir = _existing_data_path("Models")
    cache_dir = _existing_data_path("Cache")
    logs_dir = data / "logs"
    
    docs_dir = project / "Docs"
    ai_docs_dir = base / "Docs"
    
    # Ensure required directories exist
    for d in (models_dir, cache_dir, cache_dir / "embed_model", logs_dir):
        d.mkdir(parents=True, exist_ok=True)
        
    try:
        docs_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass
        
    return AppPaths(
        project_root=project,
        base_dir=base,
        data_dir=data,
        models_dir=models_dir,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        docs_dirs=(docs_dir, ai_docs_dir),
    )

#: The directory the desktop client stores the sealed attachments in, relative
#: to the platform's *data root* (``AppConfig::uploadDir`` on the C++ side).
#: The name is pinned by ``tests/unit/test_upload_dir_contract.py``, which
#: reads the C++ implementation — the two sides have to agree or the AI
#: service refuses the attachment the client just stored.
UPLOAD_DIR_NAME = "Upload"

#: The production install puts the data root in the per-user application data
#: directory, under the product folder. It is the same literal the C++ side
#: uses (``AppConfig.cpp``), and the contract test pins it.
PRODUCT_DATA_DIR_NAME = "3D-Reconstruction"


def attachment_roots() -> tuple[str, ...]:
    """The directories a client-named chat attachment may live in.

    The desktop client seals the attachment it stores and writes it into the
    platform's upload directory (``AppConfig::uploadDir``): ``<data root>/
    Upload``. The data root is the project root for a source checkout and
    ``<APP_DATA_DIR>/3D-Reconstruction`` for an installed deployment, so
    both layouts are listed here — the AI service has to open the file the
    client just stored, and it may not widen the scope to the whole disk.
    """
    paths = get_paths()
    roots = [paths.project_root / UPLOAD_DIR_NAME, paths.data_dir / UPLOAD_DIR_NAME]
    app_data = app_data_dir()
    if app_data:
        roots.append(Path(app_data) / PRODUCT_DATA_DIR_NAME / UPLOAD_DIR_NAME)
    return tuple(str(root) for root in roots)


def safe_relpath(path: str | Path, start: str | Path) -> str:
    """Return a relative path, falling back to absolute if impossible (e.g. different drives)."""
    try:
        return os.path.relpath(str(path), str(start))
    except ValueError:
        return os.path.abspath(str(path))

def get_paths() -> AppPaths:
    """Convenience function to get paths using the default base directory."""
    # This file is at src/ai_assistant/config/paths.py
    base_dir = Path(__file__).resolve().parents[3]
    return resolve_paths(base_dir)

