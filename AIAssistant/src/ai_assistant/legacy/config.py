# ruff: noqa: BLE001, S110, F401
"""
StartChatbotServer.py — 3D-Reconstruction AI Server v2.2
=========================================================
Compatibility shim for legacy config module. New features should use src/ai_assistant/config/*.
"""

import ast
import base64
import ctypes
import gc
import glob
import hashlib
import io
import json
import logging
import logging.handlers
import os
import pickle
import re
import sys
import threading
import time
import unicodedata
import warnings
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --- NEW PLATFORM INTEGRATION ---
# Temporary sys.path modification to ensure ai_assistant is found during migration
# without requiring developers to run `pip install -e .` immediately.
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ai_assistant.config import (
    AppPaths,
    ModelRegistry,
    RAGSettings,
    load_model_registry,
    resolve_paths,
)
from ai_assistant.config import (
    cleanup_old_logs as _new_cleanup,
)
from ai_assistant.config import (
    safe_relpath as _new_safe_relpath,
)
from ai_assistant.config import (
    setup_logging as _new_setup_logging,
)

# LangGraph is imported by ``agent_module`` only after this configuration
# module has finished initialising.
LocalAgentGraph = None
LANGGRAPH_AVAILABLE = False
LANGGRAPH_IMPORT_ERROR = ""

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
USE_LANGGRAPH_AGENT = os.environ.get("USE_LANGGRAPH_AGENT", "1") != "0"
FORCE_LANGGRAPH_AGENT = os.environ.get("FORCE_LANGGRAPH_AGENT", "1") == "1"

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="keras")

# ─── Path Resolution ──────────────────────────────────────────────────────────
MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(MODULES_DIR, "..", "..", ".."))

def _load_local_env() -> None:
    env_path = os.path.join(BASE_DIR, ".env")
    try:
        with open(env_path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key.replace("_", "").isalnum():
                    os.environ.setdefault(key, value)
    except OSError:
        pass

_load_local_env()

# Initialize paths using the new module
_paths = resolve_paths(BASE_DIR)

PROJECT_DIR = str(_paths.project_root)
APP_DATA_DIR = str(_paths.data_dir.parent) if _paths.data_dir.name == "AIAssistant" else str(_paths.data_dir)
DOCS_DIR = str(_paths.docs_dirs[0])
AI_ASSISTANT_DOCS_DIR = str(_paths.docs_dirs[1])
RAG_DOCUMENT_DIRS = (DOCS_DIR, AI_ASSISTANT_DOCS_DIR)

MODELS_DIR = str(_paths.models_dir)
CACHE_DIR = str(_paths.cache_dir)
LOGS_DIR = str(_paths.logs_dir)
EMBED_CACHE = str(_paths.embed_cache_dir)

CACHE_INDEX = str(_paths.cache_index)
CACHE_CHUNKS = str(_paths.cache_chunks)
CACHE_BM25 = str(_paths.cache_bm25)
CACHE_METADATA = str(_paths.cache_metadata)

def _safe_relpath(path: str, start: str) -> str:
    return _new_safe_relpath(path, start)

# ─── RAG Settings ─────────────────────────────────────────────────────────────
_rag_settings = RAGSettings.from_env_and_dict({})

EMBED_MODEL_NAME = _rag_settings.embed_model_name
EMBEDDING_QUERY_PREFIX = _rag_settings.embedding_query_prefix
EMBEDDING_PASSAGE_PREFIX = _rag_settings.embedding_passage_prefix
EMBEDDING_SUPPORTS_IMAGES = _rag_settings.embedding_supports_images
EMBEDDING_DIMENSION = _rag_settings.embedding_dimension
RAG_CACHE_VERSION = _rag_settings.cache_version

USE_RERANKER = _rag_settings.use_reranker
RERANKER_MODEL = _rag_settings.reranker_model
RERANKER_TOP_K = _rag_settings.reranker_top_k

ENABLE_VISION_LLM = os.environ.get("AI_ENABLE_VISION_LLM", "1").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_RAG = _rag_settings.enabled
CHUNK_CHARS = _rag_settings.chunk_chars
OVERLAP_CHARS = _rag_settings.overlap_chars

SIMILARITY_THRESHOLD = _rag_settings.similarity_threshold
MAX_CONTEXT_CHARS = _rag_settings.max_context_chars
CHARS_PER_TOKEN = 2.2
LLM_N_CTX = 8192

# ─── Logging ──────────────────────────────────────────────────────────────────
def cleanup_old_logs(logs_dir: str = LOGS_DIR, max_days: int = 7, max_size_mb: int = 100) -> None:
    _new_cleanup(logs_dir, max_days, max_size_mb)

def setup_logging():
    log, path = _new_setup_logging(LOGS_DIR)
    return log, str(path)

logger, LOG_FILE_PATH = setup_logging()

# ─── Startup timer ────────────────────────────────────────────────────────────
_SERVER_START_TIME = time.monotonic()

@contextmanager
def startup_step(name: str):
    t = time.monotonic()
    try:
        yield
    except Exception as e:
        elapsed = time.monotonic() - t
        logger.error("FAIL step: %s — %.1fs — %s", name, elapsed, e)
        raise
    else:
        elapsed = time.monotonic() - t
        logger.info("DONE step: %-40s %.1fs", name, elapsed)

# ─── Model list ───────────────────────────────────────────────────────────────
_registry = load_model_registry(os.path.join(BASE_DIR, "config"))

# Re-construct legacy lists from the registry
MODELS = [
    {
        "repo_id": m.repo_id,
        "filename": m.filename,
        "desc": m.desc,
        **({"is_vision": m.is_vision} if m.is_vision else {}),
        **({"mmproj_repo_id": m.mmproj_repo_id} if m.mmproj_repo_id else {}),
        **({"mmproj_filename": m.mmproj_filename} if m.mmproj_filename else {}),
    }
    for m in _registry.models
]

FALLBACK_TEXT_MODEL = {
    "repo_id": _registry.fallback.repo_id,
    "filename": _registry.fallback.filename,
    "desc": _registry.fallback.desc,
}

try:
    MODEL_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if MODEL_IDX < 0 or MODEL_IDX >= len(MODELS):
        MODEL_IDX = 0
except (ValueError, IndexError):
    MODEL_IDX = 0

active_model_desc = MODELS[MODEL_IDX]["desc"]
