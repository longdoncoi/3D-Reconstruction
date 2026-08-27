# ruff: noqa: BLE001, S110
"""
StartChatbotServer.py — 3D-Reconstruction AI Server v2.2
=========================================================

[v2.2] Cải tiến chất lượng RAG so với v2.1:
  [FIX-6]  BM25 tokenizer hỗ trợ tiếng Việt — regex cũ bỏ sót TOÀN BỘ từ Việt
  [FIX-7]  Cross-encoder re-ranking (tùy chọn) — tăng precision đáng kể
  [FIX-8]  Embedding model đa ngôn ngữ — paraphrase-multilingual-MiniLM-L12-v2
  [FIX-9]  Chunk nhỏ hơn (1200 chars) — embedding signal tập trung, ít nhiễu hơn
  [FIX-10] Phát hiện finish_reason="length" — cảnh báo và tự thử lại nếu bị cắt
  [FIX-11] Context formatting có số thứ tự + nhãn nguồn — model cite đúng hơn

[v2.1] Giữ nguyên:
  CHARS_PER_TOKEN=2.2, buffer=400 token, MAX_CONTEXT_CHARS=7500
  Sentence-aware chunking, source dedup ≤2/file, rule #8 system prompt

Cấu trúc thư mục:
  AIAssistant/
  ├── StartChatbotServer.py
  ├── requirements.txt
  ├── Cache/
  │   ├── faiss_index.bin
  │   ├── chunks.pkl
  │   ├── bm25.pkl
  │   └── metadata.json
  └── logs/
      └── server_YYYYMMDD_HHMMSS.log

LƯU Ý: v2.2 đổi embedding model và chunk size → xóa Cache/ để rebuild.
"""

# ─── 0. Bootstrap ─────────────────────────────────────────────────────────────
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

try:
    from LangGraphAgent import LocalAgentGraph
    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = ""
except ImportError as error:
    LocalAgentGraph = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = str(error)

# Fix UTF-8 encoding cho Qt Creator (piped stdout — SetConsoleOutputCP không có hiệu lực)
os.environ["PYTHONUTF8"] = "1"
if sys.platform == "win32":
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

def _force_utf8_stream(stream):
    """Bọc stream binary buffer bằng UTF-8 TextIOWrapper (hoạt động cả khi Qt Creator pipe stdout)."""
    try:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
            return stream
        if hasattr(stream, "buffer"):
            return io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
    return stream

sys.stdout = _force_utf8_stream(sys.stdout)
sys.stderr = _force_utf8_stream(sys.stderr)

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

# ─── 1. Stdlib imports ────────────────────────────────────────────────────────
# ─── 2. Đường dẫn ─────────────────────────────────────────────────────────────
MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.abspath(os.path.join(MODULES_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
APP_DATA_DIR = os.environ.get("APP_DATA_DIR", PROJECT_DIR)
DOCS_DIR    = os.path.join(PROJECT_DIR, "Docs")
AI_ASSISTANT_DOCS_DIR = os.path.join(BASE_DIR, "Docs")
# All folders whose user-authored documents are indexed by RAG.
RAG_DOCUMENT_DIRS = (DOCS_DIR, AI_ASSISTANT_DOCS_DIR)


def _existing_data_path(name: str) -> str:
    """Use old duplicate AppData storage only while a user has not migrated it."""
    primary = os.path.join(APP_DATA_DIR, "AIAssistant", name)
    legacy = os.path.join(APP_DATA_DIR, "AITraining", name)
    if not os.path.exists(primary) and os.path.exists(legacy):
        return legacy
    return primary


MODELS_DIR  = _existing_data_path("Models")
CACHE_DIR   = _existing_data_path("Cache")
LOGS_DIR    = os.path.join(APP_DATA_DIR, "AIAssistant", "logs")
EMBED_CACHE = os.path.join(CACHE_DIR, "embed_model")

for _d in (CACHE_DIR, LOGS_DIR, EMBED_CACHE, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)
if not os.path.exists(DOCS_DIR):
    try:
        os.makedirs(DOCS_DIR, exist_ok=True)
    except PermissionError:
        pass

CACHE_INDEX    = os.path.join(CACHE_DIR, "faiss_index.bin")
CACHE_CHUNKS   = os.path.join(CACHE_DIR, "chunks.pkl")
CACHE_BM25     = os.path.join(CACHE_DIR, "bm25.pkl")
CACHE_METADATA = os.path.join(CACHE_DIR, "metadata.json")


def _safe_relpath(path: str, start: str) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return os.path.abspath(path)

# ─── 3. Cấu hình RAG — chỉnh tại đây ─────────────────────────────────────────
# E5 is a text-retrieval model trained for multilingual query/passage matching.
# CLIP was useful for image similarity, but is not a reliable embedding model for
# Vietnamese project documents and source-code questions.  The prefixes below are
# part of the E5 contract and must be applied consistently while indexing/querying.
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "
# Keep image analysis with the vision LLM.  This text model deliberately does
# not accept PIL images, so image attachments fall back to their textual query.
EMBEDDING_SUPPORTS_IMAGES = False
EMBEDDING_DIMENSION = 768
RAG_CACHE_VERSION = 6

# [FIX-7] Cross-encoder re-ranking — bật/tắt tùy tài nguyên
# True  = kết quả chính xác hơn, latency tăng ~100-300ms/request
# False = tắt hoàn toàn, hành vi như v2.1
USE_RERANKER    = True
RERANKER_MODEL  = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_TOP_K  = 8

# [FIX-9] Chunk nhỏ hơn → embedding signal tập trung, ít nhiễu
# 1200 thay vì 1800: mỗi chunk mang một ý chính, không pha trộn nhiều chủ đề
# LƯU Ý: thay đổi giá trị này buộc rebuild cache
ENABLE_VISION_LLM = os.environ.get("AI_ENABLE_VISION_LLM", "1").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_RAG      = os.environ.get("AI_ENABLE_RAG", "1").strip().lower() in {"1", "true", "yes", "on"}
CHUNK_CHARS     = 1200
OVERLAP_CHARS   = 300  # Tăng lên 300 để giữ liên kết tiếng Việt

# v2.1 constants (giữ nguyên)
SIMILARITY_THRESHOLD = 0.30
MAX_CONTEXT_CHARS    = 9000 # Tăng lên 9000 để chứa đủ chi tiết tiếng Việt
CHARS_PER_TOKEN      = 2.2   # Việt+code, tránh underestimate
LLM_N_CTX            = 8192 # Token context

# ─── 4. Logging ───────────────────────────────────────────────────────────────
def setup_logging():
    log_filename = os.path.join(
        LOGS_DIR, f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    fmt_console = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%H:%M:%S")
    fmt_file    = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                    "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt_console)
    ch.setLevel(logging.INFO)
    # Đảm bảo handler emit UTF-8 ngay cả khi stream bị override sau này
    if hasattr(ch, "stream") and hasattr(ch.stream, "reconfigure"):
        try:
            ch.stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    root.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        log_filename, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt_file)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)

    for _n in ("httpx", "httpcore", "urllib3", "sentence_transformers",
               "huggingface_hub", "faiss", "uvicorn.access"):
        logging.getLogger(_n).setLevel(logging.WARNING)

    return logging.getLogger("chatbot_server"), log_filename


def cleanup_old_logs(
    logs_dir: str = LOGS_DIR,
    max_days: int = 7,
    max_size_mb: int = 100,
) -> None:
    """Dọn dẹp file log cũ trong thư mục logs.

    Chính sách xóa (áp dụng tuần tự):
    1. Xóa tất cả file .log / .log.* cũ hơn ``max_days`` ngày (mặc định 7).
    2. Nếu tổng dung lượng vẫn vượt ``max_size_mb`` MB (mặc định 100 MB),
       tiếp tục xóa file cũ nhất cho đến khi dưới ngưỡng.
    Mỗi lỗi IO riêng lẻ bị bỏ qua để không làm gián đoạn quá trình khởi động.
    """
    if not os.path.isdir(logs_dir):
        return

    now = time.time()
    max_age_sec = max_days * 86_400

    # Thu thập tất cả file log (bao gồm .log.1, .log.2 của RotatingFileHandler)
    log_files = sorted(
        (
            f
            for f in (os.path.join(logs_dir, n) for n in os.listdir(logs_dir))
            if os.path.isfile(f)
            and (f.endswith(".log") or ".log." in os.path.basename(f))
        ),
        key=os.path.getmtime,  # cũ nhất trước
    )

    # Bước 1: Xóa file quá cũ
    for fpath in list(log_files):
        try:
            if now - os.path.getmtime(fpath) > max_age_sec:
                os.remove(fpath)
                log_files.remove(fpath)
        except OSError:
            pass

    # Bước 2: Xóa thêm nếu tổng kích thước vẫn vượt ngưỡng
    def _dir_size_mb() -> float:
        return sum(
            os.path.getsize(f) for f in log_files if os.path.isfile(f)
        ) / (1024 * 1024)

    while log_files and _dir_size_mb() > max_size_mb:
        try:
            os.remove(log_files[0])
        except OSError:
            pass
        log_files.pop(0)


# Dọn log cũ trước khi tạo file log mới (file mới không bị tính vào quota)
cleanup_old_logs()
logger, LOG_FILE_PATH = setup_logging()

# ─── 5. Startup timer ─────────────────────────────────────────────────────────
_SERVER_START_TIME = time.monotonic()

@contextmanager
def startup_step(name: str):
    print(f"  ⏳  {name}...", end="", flush=True)
    t = time.monotonic()
    try:
        yield
    except Exception as e:
        elapsed = time.monotonic() - t
        print(f" ✗ ({elapsed:.1f}s) — {e}")
        logger.error("FAIL step: %s — %.1fs — %s", name, elapsed, e)
        raise
    else:
        elapsed = time.monotonic() - t
        print(f" ✓  ({elapsed:.1f}s)")
        logger.info("DONE step: %-40s %.1fs", name, elapsed)

# ─── 6. Model list ────────────────────────────────────────────────────────────
MODELS = [
    {
        "repo_id":  "Qwen/Qwen3-8B-GGUF",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "desc":     "Qwen3-8B (Q4_K_M) — Text / Agent / Coder",
    },
    {
        "repo_id":  "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "desc":     "Qwen2.5-7B (Q4_K_M) — Text",
    },
    {
        "repo_id":  "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "desc":     "Qwen2.5-coder-7B (Q4_K_M) — Coder",
    },
    {
        "repo_id":       "bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF",
        "filename":      "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "desc":          "Qwen2.5-VL-7B (Q4_K_M) — Vision",
        "is_vision":     True,
        "mmproj_repo_id":  "bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF",
        "mmproj_filename": "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf",
    },
]

FALLBACK_TEXT_MODEL = {
    "repo_id":  "bartowski/Qwen2.5-3B-Instruct-GGUF",
    "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "desc":     "Qwen2.5-3B (Q4_K_M) — Text Fallback",
}

try:
    MODEL_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if MODEL_IDX < 0 or MODEL_IDX >= len(MODELS):
        MODEL_IDX = 0
except (ValueError, IndexError):
    MODEL_IDX = 0

active_model_desc = MODELS[MODEL_IDX]["desc"]

