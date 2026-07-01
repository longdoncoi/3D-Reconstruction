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
  AITraining/
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
import sys
import os

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    ctypes.windll.kernel32.SetConsoleCP(65001)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="keras")

# ─── 1. Stdlib imports ────────────────────────────────────────────────────────
import re
import ast
import base64
import glob
import json
import time
import hashlib
import pickle
import logging
import logging.handlers
from abc import ABC, abstractmethod
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

# ─── 2. Đường dẫn ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DOCS_DIR    = os.path.join(PROJECT_DIR, "Docs")
MODELS_DIR  = os.path.join(PROJECT_DIR, "Models")
CACHE_DIR   = os.path.join(BASE_DIR, "Cache")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
EMBED_CACHE = os.path.join(CACHE_DIR, "embed_model")

for _d in (MODELS_DIR, CACHE_DIR, LOGS_DIR, EMBED_CACHE, DOCS_DIR):
    os.makedirs(_d, exist_ok=True)

CACHE_INDEX    = os.path.join(CACHE_DIR, "faiss_index.bin")
CACHE_CHUNKS   = os.path.join(CACHE_DIR, "chunks.pkl")
CACHE_BM25     = os.path.join(CACHE_DIR, "bm25.pkl")
CACHE_METADATA = os.path.join(CACHE_DIR, "metadata.json")

# ─── 3. Cấu hình RAG — chỉnh tại đây ─────────────────────────────────────────
# [FIX-8] Đổi sang model đa ngôn ngữ — hỗ trợ tiếng Việt tốt hơn all-MiniLM-L6-v2
# Kích thước: ~470MB (so với ~80MB), nhưng độ chính xác retrieval tăng rõ rệt.
# Nếu muốn giữ model cũ (tiết kiệm RAM/disk): đổi lại "all-MiniLM-L6-v2" hoặc "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_MODEL_NAME = "clip-ViT-B-32"

# [FIX-7] Cross-encoder re-ranking — bật/tắt tùy tài nguyên
# True  = kết quả chính xác hơn, latency tăng ~100-300ms/request
# False = tắt hoàn toàn, hành vi như v2.1
USE_RERANKER    = True
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~80MB
RERANKER_TOP_K  = 8    # Tăng lên 8 để lấy thêm context

# [FIX-9] Chunk nhỏ hơn → embedding signal tập trung, ít nhiễu
# 1200 thay vì 1800: mỗi chunk mang một ý chính, không pha trộn nhiều chủ đề
# LƯU Ý: thay đổi giá trị này buộc rebuild cache
CHUNK_CHARS     = 1200
OVERLAP_CHARS   = 300  # Tăng lên 300 để giữ liên kết tiếng Việt

# v2.1 constants (giữ nguyên)
SIMILARITY_THRESHOLD = 0.25
MAX_CONTEXT_CHARS    = 9000 # Tăng lên 9000 để chứa đủ chi tiết tiếng Việt
CHARS_PER_TOKEN      = 2.2   # Việt+code, tránh underestimate
LLM_N_CTX            = 8192

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

try:
    MODEL_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if MODEL_IDX < 0 or MODEL_IDX >= len(MODELS):
        MODEL_IDX = 0
except (ValueError, IndexError):
    MODEL_IDX = 0

# ─── 7. BM25 tokenizer hỗ trợ tiếng Việt ─────────────────────────────────────
# [FIX-6] CRITICAL: regex cũ r"[a-z0-9_]+" chỉ bắt Latin → bỏ sót TOÀN BỘ
# từ tiếng Việt trong BM25. Hệ quả: hybrid_retrieve = semantic search thuần,
# BM25 score luôn gần 0 với query/doc tiếng Việt → kết quả không sát nghĩa.
def _tokenize_vn(text: str) -> list:
    """
    Tokenizer BM25 hỗ trợ Việt + English + code.
    - Xử lý một số từ ghép tiếng Việt cơ bản
    - Loại bỏ stopwords cơ bản
    """
    t = text.lower()
    
    # Nối từ ghép cơ bản (có thể mở rộng thêm)
    compounds = {
        "tái tạo": "tái_tạo", "hình ảnh": "hình_ảnh", "mô hình": "mô_hình",
        "dữ liệu": "dữ_liệu", "hệ thống": "hệ_thống", "đầu vào": "đầu_vào",
        "đầu ra": "đầu_ra", "cấu hình": "cấu_hình", "giao diện": "giao_diện"
    }
    for k, v in compounds.items():
        t = t.replace(k, v)

    # Stopwords tiếng Việt cơ bản
    stopwords = {"là", "của", "và", "các", "trong", "được", "có", "cho", "với", "để", "những"}

    latin_tokens = re.findall(r"[a-z0-9][a-z0-9_]*", t)
    viet_tokens  = re.findall(r"[^\x00-\x7f\s.,!?;:()\[\]{}'\"<>/\\|@#$%^&*+=~`]+", t)
    
    tokens = latin_tokens + viet_tokens
    return [tk for tk in tokens if tk not in stopwords]


# ─── 8. Typed chunk ───────────────────────────────────────────────────────────
@dataclass
class ChunkResult:
    text:        str
    source_path: str
    loader_type: str
    is_image:    bool = False
    image_b64:   Optional[str] = None
    metadata:    dict = field(default_factory=dict)

# ─── 8b. Vision helpers ──────────────────────────────────────────────────────
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def _is_image_file(filepath: str) -> bool:
    return os.path.splitext(filepath)[1].lower() in _IMAGE_EXTS

def _image_to_data_uri(filepath: str, max_dim: int = 512) -> str:
    from PIL import Image
    import io
    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
                ".bmp": "bmp", ".gif": "gif", ".webp": "webp"}
    mime = mime_map.get(ext, "jpeg")
    try:
        img = Image.open(filepath)
        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        if mime == "jpeg" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        save_fmt = "JPEG" if mime == "jpeg" else mime.upper()
        img.save(buf, format=save_fmt, quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"

# ─── 9. Document Loaders ─────────────────────────────────────────────────────
class BaseDocumentLoader(ABC):
    # [FIX-9] Chunk nhỏ hơn: 1200 thay vì 1800
    MAX_CHUNK_CHARS = CHUNK_CHARS
    MIN_CHUNK_CHARS = 80
    OVERLAP_CHARS   = OVERLAP_CHARS

    _SENT_ENDINGS = (".\n", ". ", "!\n", "! ", "?\n", "? ", ";\n", "\n\n")

    @abstractmethod
    def can_handle(self, filepath: str) -> bool: ...

    @abstractmethod
    def load(self, filepath: str) -> list: ...

    def _is_quality_chunk(self, block: str) -> bool:
        stripped = block.strip()
        if len(stripped) < self.MIN_CHUNK_CHARS:
            return False
        non_comment = re.sub(
            r"^\s*(//[^\n]*|/\*.*?\*/)", "", stripped,
            flags=re.DOTALL | re.MULTILINE,
        ).strip()
        return len(non_comment) >= self.MIN_CHUNK_CHARS // 2

    def _snap_to_sentence(self, text: str) -> str:
        """Snap về ranh giới câu gần cuối nhất trong nửa sau của text."""
        min_pos = len(text) // 2
        best    = -1
        for ending in self._SENT_ENDINGS:
            pos = text.rfind(ending)
            if pos > min_pos and pos > best:
                best = pos
        return text[:best + 1] if best > min_pos else text

    def _sliding_window_chunks(self, content: str, filepath: str, label: str = "Source") -> list:
        """Sentence-aware sliding window chunking (v2.1+)."""
        rel     = os.path.relpath(filepath, PROJECT_DIR)
        results = []
        pos     = 0

        while pos < len(content):
            end   = pos + self.MAX_CHUNK_CHARS
            block = content[pos:end]

            if end < len(content) and len(block) > self.OVERLAP_CHARS * 2:
                snapped = self._snap_to_sentence(block)
                if len(snapped.strip()) >= self.MIN_CHUNK_CHARS:
                    block = snapped

            block = block.strip()
            if self._is_quality_chunk(block):
                results.append(ChunkResult(
                    text        = f"[{label}: {rel}]\n{block}",
                    source_path = filepath,
                    loader_type = label.lower().replace(" ", "_"),
                ))

            advance = max(len(block) - self.OVERLAP_CHARS,
                         self.MAX_CHUNK_CHARS - self.OVERLAP_CHARS)
            pos    += advance

        return results

    def _read_text_file(self, filepath: str) -> Optional[str]:
        for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, ValueError):
                continue
        return None


class DocxLoader(BaseDocumentLoader):
    def can_handle(self, fp: str) -> bool: return fp.lower().endswith(".docx")
    def load(self, fp: str) -> list:
        from docx import Document
        doc  = Document(fp)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return self._sliding_window_chunks(text, fp, label="Tai lieu")


class PdfLoader(BaseDocumentLoader):
    def can_handle(self, fp: str) -> bool: return fp.lower().endswith(".pdf")
    def load(self, fp: str) -> list:
        text = self._extract_pdf_text(fp)
        if not text:
            raise ValueError(f"Không đọc được PDF: {fp}")
        return self._sliding_window_chunks(text, fp, label="Tai lieu PDF")

    def _extract_pdf_text(self, fp: str) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(fp) as pdf:
                parts = [pg.extract_text(x_tolerance=2, y_tolerance=2) for pg in pdf.pages]
            text = "\n\n".join(p for p in parts if p).strip()
            if len(text) > 100:
                return text
        except Exception as e:
            logger.warning("pdfplumber failed %s: %s", fp, e)
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(fp)
            if text and len(text.strip()) > 100:
                return text.strip()
        except Exception as e:
            logger.warning("pdfminer failed %s: %s", fp, e)
        return ""


class TxtLoader(BaseDocumentLoader):
    def can_handle(self, fp: str) -> bool: return fp.lower().endswith(".txt")
    def load(self, fp: str) -> list:
        content = self._read_text_file(fp)
        if not content or len(content.strip()) < self.MIN_CHUNK_CHARS:
            raise ValueError(f"File rỗng: {fp}")
        return self._sliding_window_chunks(content, fp, label="Tai lieu TXT")


class MarkdownLoader(BaseDocumentLoader):
    HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def can_handle(self, fp: str) -> bool: return fp.lower().endswith(".md")

    def load(self, fp: str) -> list:
        content = self._read_text_file(fp)
        if not content or len(content.strip()) < self.MIN_CHUNK_CHARS:
            raise ValueError(f"File rỗng: {fp}")

        rel     = os.path.relpath(fp, PROJECT_DIR)
        matches = list(self.HEADING_RE.finditer(content))
        if not matches:
            return self._sliding_window_chunks(content, fp, label="Source MD")

        results    = []
        boundaries = [m.start() for m in matches] + [len(content)]

        for i, match in enumerate(matches):
            level   = len(match.group(1))
            heading = match.group(2).strip()
            section = content[boundaries[i]:boundaries[i+1]].strip()
            section = self._strip_large_code_blocks(section)
            if not self._is_quality_chunk(section):
                continue

            prefix = f"[Source MD: {rel}] {'#'*level} {heading}\n"

            if len(section) <= self.MAX_CHUNK_CHARS:
                results.append(ChunkResult(
                    text        = prefix + section,
                    source_path = fp,
                    loader_type = "md",
                    metadata    = {"heading": heading, "level": level},
                ))
            else:
                for sc in self._sliding_window_chunks(section, fp, label="Source MD"):
                    sc.text     = prefix + sc.text.split("\n", 1)[-1]
                    sc.metadata = {"heading": heading, "level": level}
                    results.append(sc)

        return results

    def _strip_large_code_blocks(self, text: str) -> str:
        def maybe_strip(m):
            lines = m.group(0).count("\n")
            return m.group(0) if lines <= 30 else f"[code block omitted – {lines} lines]"
        return re.sub(r"```[\s\S]*?```", maybe_strip, text)


class CppHeaderLoader(BaseDocumentLoader):
    SOURCE_EXTS = {".cpp", ".h", ".py", ".cmake"}
    FUNC_RE     = re.compile(
        r"(?:^|\n)(?:"
        r"(?:class|struct|namespace)\s+\w+.*?\{"
        r"|(?:[\w:*&<>\[\]~]+\s+)+(?:\w+::)*\w+\s*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{"
        r")",
        re.MULTILINE,
    )

    def can_handle(self, fp: str) -> bool:
        return os.path.splitext(fp)[1].lower() in self.SOURCE_EXTS

    def load(self, fp: str) -> list:
        content = self._read_text_file(fp)
        if not content or len(content.strip()) < self.MIN_CHUNK_CHARS:
            raise ValueError(f"File rỗng: {fp}")
        ext = os.path.splitext(fp)[1].lower()
        return self._load_python(content, fp) if ext == ".py" else self._load_cpp(content, fp)

    def _load_cpp(self, content: str, fp: str) -> list:
        rel       = os.path.relpath(fp, PROJECT_DIR)
        positions = [m.start() for m in self.FUNC_RE.finditer(content)]
        results   = []
        if positions:
            positions.append(len(content))
            for i, start in enumerate(positions[:-1]):
                block = content[start:positions[i+1]].strip()
                if not self._is_quality_chunk(block):
                    continue
                header = block.split("\n")[0].strip().rstrip("{").strip()
                results.append(ChunkResult(
                    text        = f"[Source: {rel}] {header}\n{block[:self.MAX_CHUNK_CHARS]}",
                    source_path = fp,
                    loader_type = "source",
                    metadata    = {"symbol": header},
                ))
        else:
            results = self._sliding_window_chunks(content, fp, label="Source")
        return results

    def _load_python(self, content: str, fp: str) -> list:
        rel     = os.path.relpath(fp, PROJECT_DIR)
        results = []
        lines   = content.splitlines()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._sliding_window_chunks(content, fp, label="Source")

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = node.lineno - 1
            end   = getattr(node, "end_lineno", start + 50)
            block = "\n".join(lines[start:end]).strip()
            if not self._is_quality_chunk(block):
                continue
            scope = "Class" if isinstance(node, ast.ClassDef) else "Function"
            results.append(ChunkResult(
                text        = f"[Source: {rel}] {scope}: {node.name}\n{block[:self.MAX_CHUNK_CHARS]}",
                source_path = fp,
                loader_type = "source",
                metadata    = {"symbol": node.name, "scope": scope},
            ))

        return results or self._sliding_window_chunks(content, fp, label="Source")


class ImageLoader(BaseDocumentLoader):
    def can_handle(self, fp: str) -> bool:
        return _is_image_file(fp)

    def load(self, fp: str) -> list:
        try:
            b64 = _image_to_data_uri(fp)
            rel = os.path.relpath(fp, PROJECT_DIR)
            return [ChunkResult(
                text=f"[Image: {rel}]",
                source_path=fp,
                loader_type="image",
                is_image=True,
                image_b64=b64
            )]
        except Exception as e:
            logger.error("Error loading image %s: %s", fp, e)
            return []


# ─── 10. Loader Registry ──────────────────────────────────────────────────────
class DocumentLoaderRegistry:
    def __init__(self): self._loaders: list = []

    def register(self, loader) -> "DocumentLoaderRegistry":
        self._loaders.append(loader)
        return self

    def get_loader(self, fp: str) -> Optional[object]:
        for loader in self._loaders:
            if loader.can_handle(fp):
                return loader
        return None

    def load_file(self, fp: str) -> list:
        loader = self.get_loader(fp)
        if loader is None:
            logger.debug("No loader for: %s", fp)
            return []
        try:
            return loader.load(fp)
        except Exception as e:
            logger.error("Error loading %s: %s", fp, e)
            return []


def build_registry() -> DocumentLoaderRegistry:
    return (
        DocumentLoaderRegistry()
        .register(DocxLoader())
        .register(PdfLoader())
        .register(TxtLoader())
        .register(MarkdownLoader())
        .register(CppHeaderLoader())
        .register(ImageLoader())
    )


# ─── 11. Document scanning ────────────────────────────────────────────────────
EXCLUDED_DIRS  = {
    ".git", "build", "__pycache__", ".qtcreator", ".cache", "Cache",
    "runs", "Dicom", "Predict", "3DModels", "Dataset", "logs",
    ".github", ".prompts", ".review", ".tasks", "scripts"
}
SCANNABLE_EXTS = {".cpp", ".h", ".py", ".md", ".cmake", ".jpg", ".jpeg", ".png", ".webp"}
DOC_EXTS_GLOB  = ("*.docx", "*.pdf", "*.txt", "*.jpg", "*.jpeg", "*.png", "*.webp")


def load_documents() -> list:
    registry   = build_registry()
    all_chunks: list = []
    stats = {"docx":0,"pdf":0,"txt":0,"md":0,"source":0,"errors":0,"files":0}

    if os.path.isdir(DOCS_DIR):
        for pattern in DOC_EXTS_GLOB:
            for fp in sorted(glob.glob(os.path.join(DOCS_DIR, pattern))):
                stats["files"] += 1
                results = registry.load_file(fp)
                if not results:
                    stats["errors"] += 1
                    continue
                for r in results:
                    all_chunks.append(r)
                    ext = os.path.splitext(fp)[1].lower().lstrip(".")
                    if ext in stats:
                        stats[ext] += 1

    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for filename in sorted(files):
            if filename.startswith("~") or filename.endswith(".user"):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SCANNABLE_EXTS:
                continue
            fp = os.path.join(root, filename)
            stats["files"] += 1
            for r in registry.load_file(fp):
                all_chunks.append(r)
                stats["md" if r.loader_type == "md" else "source"] += 1

    logger.info(
        "Scanned: %d files → %d chunks (docx=%d pdf=%d txt=%d md=%d src=%d err=%d)",
        stats["files"], len(all_chunks),
        stats["docx"], stats["pdf"], stats["txt"],
        stats["md"], stats["source"], stats["errors"]
    )
    print(f"       files={stats['files']}  chunks={len(all_chunks)}"
          f"  (docx={stats['docx']} pdf={stats['pdf']} txt={stats['txt']}"
          f" md={stats['md']} src={stats['source']} err={stats['errors']})")
    return all_chunks


# ─── 12. Cache management ─────────────────────────────────────────────────────
def get_file_system_hash() -> str:
    entries = []
    if os.path.isdir(DOCS_DIR):
        for root, _, files in os.walk(DOCS_DIR):
            for f in sorted(files):
                path = os.path.join(root, f)
                try:
                    st = os.stat(path)
                    entries.append(f"{path}:{st.st_mtime:.3f}:{st.st_size}")
                except OSError:
                    pass

    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in SCANNABLE_EXTS:
                continue
            path = os.path.join(root, f)
            try:
                st = os.stat(path)
                entries.append(f"{path}:{st.st_mtime:.3f}:{st.st_size}")
            except OSError:
                pass

    # [v2.2] Thêm config vào hash: đổi EMBED_MODEL hoặc CHUNK_CHARS → tự rebuild
    entries.append(f"embed_model={EMBED_MODEL_NAME}")
    entries.append(f"chunk_chars={CHUNK_CHARS}")

    combined = "\n".join(entries).encode("utf-8")
    return hashlib.md5(combined).hexdigest()


def is_cache_valid() -> bool:
    required = [CACHE_INDEX, CACHE_CHUNKS, CACHE_BM25, CACHE_METADATA]
    if not all(os.path.exists(p) for p in required):
        logger.debug("Cache miss: files missing")
        return False
    try:
        with open(CACHE_METADATA, "r", encoding="utf-8") as f:
            meta = json.load(f)
        current_hash = get_file_system_hash()
        valid = meta.get("fs_hash") == current_hash
        if not valid:
            logger.info("Cache stale (built_at=%s)", meta.get("built_at", "?"))
        else:
            logger.info("Cache valid: built_at=%s, chunks=%d",
                        meta.get("built_at", "?"), meta.get("chunk_count", 0))
        return valid
    except Exception as e:
        logger.warning("Cache read error: %s", e)
        return False


def save_cache(index, chunks: list, bm25) -> None:
    import faiss as _faiss
    t = time.monotonic()
    _faiss.write_index(index, CACHE_INDEX)
    with open(CACHE_CHUNKS, "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(CACHE_BM25, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "fs_hash":     get_file_system_hash(),
        "chunk_count": len(chunks),
        "built_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_idx":   MODEL_IDX,
        "embed_model": EMBED_MODEL_NAME,
        "chunk_chars": CHUNK_CHARS,
    }
    with open(CACHE_METADATA, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    total_mb = sum(os.path.getsize(p) for p in [CACHE_INDEX, CACHE_CHUNKS, CACHE_BM25]) / 1024**2
    logger.info("Cache saved: %.1fs | %.1fMB | %d chunks", time.monotonic()-t, total_mb, len(chunks))


def load_cache(embed_model):
    import faiss as _faiss
    t = time.monotonic()
    try:
        index = _faiss.read_index(CACHE_INDEX)
        with open(CACHE_CHUNKS, "rb") as f:
            chunks = pickle.load(f)
        with open(CACHE_BM25, "rb") as f:
            bm25 = pickle.load(f)
        logger.info("Cache loaded: %.1fs | chunks=%d", time.monotonic()-t, len(chunks))
        print(f"       chunks={len(chunks)}  (from Cache/)")
        return index, chunks, bm25
    except Exception as e:
        logger.error("Cache load failed: %s", e)
        return None, None, None


def build_index_from_scratch(chunks: list, embed_model):
    import faiss as _faiss
    import numpy as np
    from rank_bm25 import BM25Okapi
    from PIL import Image

    t = time.monotonic()
    logger.info("Building index: %d chunks", len(chunks))

    t_enc = time.monotonic()
    text_indices = []
    texts = []
    image_indices = []
    images = []

    for i, c in enumerate(chunks):
        if getattr(c, "is_image", False):
            try:
                images.append(Image.open(c.source_path).convert("RGB"))
                image_indices.append(i)
            except Exception:
                texts.append(getattr(c, "text", str(c)))
                text_indices.append(i)
        else:
            texts.append(getattr(c, "text", str(c)))
            text_indices.append(i)

    final_embeddings = [None] * len(chunks)
    
    if texts:
        print(f"\n       Encoding {len(texts)} text chunks", end="", flush=True)
        text_embs = embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=64)
        for idx, emb in zip(text_indices, text_embs):
            final_embeddings[idx] = emb
            
    if images:
        print(f"\n       Encoding {len(images)} image chunks", end="", flush=True)
        img_embs = embed_model.encode(images, show_progress_bar=False, normalize_embeddings=True, batch_size=64)
        for idx, emb in zip(image_indices, img_embs):
            final_embeddings[idx] = emb

    embeddings = final_embeddings
    print(f"\n ✓ Encoding complete ({time.monotonic()-t_enc:.1f}s)")

    emb  = np.array(embeddings, dtype="float32")
    dim, n = emb.shape[1], len(emb)

    if n < 1000:
        index = _faiss.IndexFlatIP(dim)
        _faiss.normalize_L2(emb)
        index.add(emb)
    else:
        nlist     = min(int(n**0.5), 256)
        quantizer = _faiss.IndexFlatIP(dim)
        index     = _faiss.IndexIVFFlat(quantizer, dim, nlist, _faiss.METRIC_INNER_PRODUCT)
        normed    = emb.copy()
        _faiss.normalize_L2(normed)
        index.train(normed)
        index.add(normed)

    logger.info("FAISS built: ntotal=%d dim=%d", index.ntotal, dim)

    # [FIX-6] Dùng _tokenize_vn thay vì r"[a-z0-9_]+" để BM25 xử lý được tiếng Việt
    tokenized = [_tokenize_vn(getattr(c, "text", str(c))) for c in chunks]
    bm25      = BM25Okapi(tokenized)

    save_cache(index, chunks, bm25)
    logger.info("Index built: %.1fs total", time.monotonic()-t)
    return index, chunks, bm25


def load_or_build_index(chunks: list, embed_model):
    if is_cache_valid():
        print("       [CACHE HIT]", end="")
        index, chunks_loaded, bm25 = load_cache(embed_model)
        if index is not None:
            return index, chunks_loaded, bm25
        print(" load failed, rebuilding...")
    print("       [CACHE MISS] Building index...")
    logger.info("Cache miss — rebuilding")
    return build_index_from_scratch(chunks, embed_model)


# ─── 13. RAG retrieval ────────────────────────────────────────────────────────
knowledge_index  = None
knowledge_chunks = []
bm25_index       = None
embed_model_ref  = None
_reranker        = None   # Cross-encoder, load lúc startup nếu USE_RERANKER=True
is_vision_model  = False  # True khi chạy Qwen2.5-VL (vision model)


def hybrid_retrieve(query: str, query_image_b64: str = None, k: int = 14, final_k: int = 10) -> list:
    """
    Hybrid semantic (text/image) + BM25 (text only) retrieval.
    """
    import numpy as np

    if knowledge_index is None or not knowledge_chunks:
        return []

    n = min(k, len(knowledge_chunks))

    # Semantic search with query text or query image
    if query_image_b64:
        from PIL import Image
        import io
        img_data = base64.b64decode(query_image_b64.split(",")[1])
        qv_input = [Image.open(io.BytesIO(img_data)).convert("RGB")]
    else:
        qv_input = [query]

    qv = embed_model_ref.encode(qv_input, normalize_embeddings=True)
    sem_raw, sem_idx = knowledge_index.search(np.array(qv, dtype="float32"), n)
    sem_raw = sem_raw[0]
    sem_idx = sem_idx[0]

    sem_min, sem_max = sem_raw.min(), sem_raw.max()
    sem_range = sem_max - sem_min if sem_max != sem_min else 1.0
    sem_norm  = {int(i): (s-sem_min)/sem_range for i, s in zip(sem_idx, sem_raw) if i >= 0}

    # [FIX-6] BM25 với Vietnamese tokenizer
    if query_image_b64:
        # BM25 is not used for image queries
        combined = [(cid, float(sem_raw[list(sem_idx).index(cid)])) for cid in sem_idx if cid >= 0]
    else:
        bm25_raw     = np.array(bm25_index.get_scores(_tokenize_vn(query)))
        bm25_top_idx = np.argsort(bm25_raw)[::-1][:n]
        b_scores     = bm25_raw[bm25_top_idx]
        b_min, b_max = b_scores.min(), b_scores.max()
        b_range      = b_max - b_min if b_max != b_min else 1.0
        bm25_norm    = {int(i): (s-b_min)/b_range for i, s in zip(bm25_top_idx, b_scores)}

        sem_idx_list = list(sem_idx)
        combined = []
        for cid in set(sem_norm) | set(bm25_norm):
            s = sem_norm.get(cid, 0.0)
            b = bm25_norm.get(cid, 0.0)
            raw_cos = float(sem_raw[sem_idx_list.index(cid)]) if cid in sem_norm else 0.0
            if raw_cos < SIMILARITY_THRESHOLD and b < 0.3:
                continue
            combined.append((cid, 0.55*s + 0.45*b))

    combined.sort(key=lambda x: x[1], reverse=True)
    return [knowledge_chunks[cid] for cid, _ in combined[:final_k]]


# [FIX-7] Cross-encoder re-ranking
def _rerank(query: str, chunks: list) -> list:
    if _reranker is None or not chunks:
        return chunks
    try:
        # Giới hạn độ dài để reranker nhanh hơn. Reranker chỉ chạy trên text.
        pairs  = [(query, getattr(c, "text", str(c))[:600]) for c in chunks]
        scores = _reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        logger.debug("Rerank scores: %s", [f"{s:.3f}" for s, _ in ranked])
        return [c for _, c in ranked]
    except Exception as e:
        logger.warning("Reranker failed, fallback: %s", e)
        return chunks


# Source deduplication (v2.1)
def _dedup_by_source(chunks: list, max_per_source: int = 2) -> list:
    seen: dict = {}
    result: list = []
    for chunk in chunks:
        src = getattr(chunk, "source_path", str(chunk)[:80])
        if seen.get(src, 0) < max_per_source:
            result.append(chunk)
            seen[src] = seen.get(src, 0) + 1
    logger.debug("Dedup: %d → %d chunks (%d sources)", len(chunks), len(result), len(seen))
    return result


# [FIX-11] Context formatting với số thứ tự — model cite nguồn chính xác hơn
def _format_context_block(chunks: list, section_title: str) -> str:
    if not chunks:
        return ""
    lines = [f"=== {section_title} ==="]
    for i, chunk in enumerate(chunks, 1):
        src  = os.path.basename(getattr(chunk, "source_path", "nguồn"))
        text = getattr(chunk, "text", str(chunk))
        body = text.split("\n", 1)[-1].strip()  # Bỏ dòng prefix [...]
        lines.append(f"\n[{i}] {src}\n{body}")
    return "\n".join(lines)


def get_context(query: str, query_image_b64: str = None) -> tuple:
    # Bước 1: Hybrid retrieve
    candidates = hybrid_retrieve(query, query_image_b64=query_image_b64, k=14, final_k=10)

    # Bước 2: Cross-encoder re-rank
    if USE_RERANKER and query:
        candidates = _rerank(query, candidates)
        candidates = candidates[:RERANKER_TOP_K]

    # Bước 3: Source dedup
    candidates = _dedup_by_source(candidates, max_per_source=2)

    # Bước 4: Phân loại + cắt theo ngân sách context
    doc_chunks   = []
    code_chunks  = []
    image_chunks = []
    total        = 0

    for chunk in candidates:
        if getattr(chunk, "is_image", False):
            if chunk.image_b64:
                image_chunks.append(chunk.image_b64)
            continue

        text_content = getattr(chunk, "text", str(chunk))
        remaining = MAX_CONTEXT_CHARS - total
        if remaining < 150:
            break
        trimmed_text = text_content[:remaining] if len(text_content) > remaining else text_content
        
        # Tạo bản sao chunk với text đã cắt gọn
        from copy import copy
        trimmed_chunk = copy(chunk) if hasattr(chunk, "text") else chunk
        if hasattr(trimmed_chunk, "text"):
            trimmed_chunk.text = trimmed_text

        if text_content.startswith("[Tai lieu"):
            doc_chunks.append(trimmed_chunk)
        else:
            code_chunks.append(trimmed_chunk)
        total += len(trimmed_text)

    doc_ctx  = _format_context_block(doc_chunks,  "TÀI LIỆU THAM KHẢO")
    code_ctx = _format_context_block(code_chunks, "MÃ NGUỒN LIÊN QUAN")
    return doc_ctx, code_ctx, image_chunks


# ─── 14. FastAPI + LLM ────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

llm = None
_chat_handler = None  # Qwen25VLChatHandler, chỉ dùng cho vision model


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def trim_history(messages: list, max_tokens: int = 2000) -> list:
    messages = list(messages)
    total    = sum(estimate_tokens(m.get("content", "")) for m in messages)
    while total > max_tokens and len(messages) > 1:
        removed = messages.pop(0)
        total  -= estimate_tokens(removed.get("content", ""))
    return messages


def build_text_messages(messages: list, doc_ctx: str, code_ctx: str) -> list:
    system_prompt = _build_system_prompt(doc_ctx, code_ctx)
    result = [{"role": "system", "content": system_prompt}]
    
    history = trim_history(list(messages[:-1]), max_tokens=2000)
    for msg in history:
        role = "user" if msg.get("role") == "user" else "assistant"
        content = msg.get("content", "")
        if msg.get("attachments"):
            content += f"\n\n[Hệ thống: Người dùng tải lên: {', '.join(msg['attachments'])}. Model text-only, không thể xem ảnh.]"
        result.append({"role": role, "content": content})
        
    last = messages[-1]
    last_content = last.get("content", "")
    if last.get("attachments"):
        last_content += f"\n\n[Hệ thống: Người dùng tải lên: {', '.join(last['attachments'])}. Model text-only, không thể xem ảnh.]"
    result.append({"role": "user", "content": last_content})
    return result


def _build_system_prompt(doc_ctx: str, code_ctx: str) -> str:
    """Build system prompt chung cho cả text-only và vision model."""
    system_prompt = (
        "Bạn là trợ lý AI chuyên nghiệp cho dự án 3D-Reconstruction.\n"
        "NGUYÊN TẮC TRẢ LỜI:\n"
        "1. Dựa chủ yếu vào tài liệu và mã nguồn được cung cấp bên dưới để trả lời.\n"
        "2. Nếu câu hỏi không liên quan đến bất kỳ nội dung nào trong ngữ cảnh, "
        "   hãy nói ngắn gọn: 'Câu hỏi này nằm ngoài phạm vi tài liệu dự án.' rồi dừng.\n"
        "3. Không bịa đặt hoặc suy đoán thông tin kỹ thuật không có trong tài liệu.\n"
        "4. Khi nhắc đến code hoặc tài liệu, hãy ghi rõ số thứ tự nguồn [1], [2]... "
        "   tương ứng với danh sách ngữ cảnh bên dưới.\n"
        "5. Ưu tiên trả lời ĐẦY ĐỦ và CHI TIẾT — giải thích từng bước, nêu lý do "
        "   kỹ thuật, trích dẫn trực tiếp từ tài liệu khi có thể.\n"
        "6. Cấu trúc câu trả lời: tóm tắt ngắn → giải thích chi tiết → ví dụ/code.\n"
        "7. Trả lời bằng tiếng Việt trừ khi người dùng hỏi bằng tiếng Anh. Nếu tài liệu "
        "   nguồn là tiếng Anh, hãy DỊCH và GIẢI THÍCH sang tiếng Việt.\n"
        "8. LUÔN sử dụng định dạng Markdown (tiêu đề in đậm, bullet points, code blocks "
        "   có highlight syntax) để trình bày đẹp và dễ đọc.\n"
        "9. QUAN TRỌNG: Luôn hoàn thành câu cuối cùng trước khi kết thúc. "
        "   Không bao giờ dừng giữa câu, giữa đoạn code, hoặc giữa danh sách.\n"
        "10. Nếu người dùng gửi ảnh, hãy phân tích nội dung ảnh chi tiết "
        "    và liên hệ với tài liệu dự án nếu có thể.\n"
        "11. Nếu câu hỏi liên quan đến nhân vật trong dự án (như thành viên, tác giả, người tham gia), hãy trả lời trực tiếp mà KHÔNG trích dẫn tài liệu tham khảo.\n"
        "12. KHÔNG liệt kê hay in lại log 'TÀI LIỆU THAM KHẢO' hoặc 'MÃ NGUỒN LIÊN QUAN' trong câu trả lời.\n"
    )
    if doc_ctx:
        system_prompt += f"\n\n{doc_ctx}"
    if code_ctx:
        system_prompt += f"\n\n{code_ctx}"
    if not doc_ctx and not code_ctx:
        system_prompt += "\n\n[Không tìm thấy ngữ cảnh liên quan. Từ chối theo nguyên tắc số 2.]"
    return system_prompt


def build_vision_messages(messages: list, doc_ctx: str, code_ctx: str, image_chunks: list = None) -> list:
    """
    Build messages format cho create_chat_completion() — vision model.
    Ảnh đính kèm được encode thành base64 data URI theo OpenAI multimodal format.
    Chỉ ảnh ở message cuối cùng được gửi — ảnh cũ trong history bị bỏ qua
    để tiết kiệm context.
    """
    system_prompt = _build_system_prompt(doc_ctx, code_ctx)
    result = [{"role": "system", "content": system_prompt}]

    # History: chỉ gửi text, bỏ ảnh cũ
    history = trim_history(list(messages[:-1]), max_tokens=2000)
    for msg in history:
        role = "user" if msg.get("role") == "user" else "assistant"
        content = msg.get("content", "")
        result.append({"role": role, "content": content})

    # Message cuối: xử lý cả text + ảnh
    last = messages[-1]
    content_parts = []

    text_content = last.get("content", "")
    if text_content:
        content_parts.append({"type": "text", "text": text_content})

    # Encode ảnh đính kèm thành base64 data URI
    attachments = last.get("attachments") or []
    has_images = False
    for att in attachments:
        if os.path.isfile(att) and _is_image_file(att):
            try:
                data_uri = _image_to_data_uri(att)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                })
                has_images = True
                logger.info("Vision: encoded image %s (%d bytes)",
                           os.path.basename(att), os.path.getsize(att))
            except Exception as e:
                logger.error("Failed to encode image %s: %s", att, e)
                content_parts.append(
                    {"type": "text", "text": f"[Lỗi đọc ảnh: {os.path.basename(att)}]"}
                )
        else:
            content_parts.append(
                {"type": "text", "text": f"[File đính kèm: {os.path.basename(att)}]"}
            )

    if image_chunks:
        for b64 in image_chunks:
            content_parts.append(
                {"type": "image_url", "image_url": {"url": b64}}
            )
            has_images = True
            logger.info("Vision: added retrieved image chunk to context")

    # Nếu không có ảnh, thêm hint cho model
    if attachments and not has_images:
        content_parts.append(
            {"type": "text", "text": "[Không có ảnh hợp lệ trong file đính kèm.]"}
        )

    if not content_parts:
        content_parts.append({"type": "text", "text": "(trống)"})

    result.append({"role": "user", "content": content_parts})
    logger.debug("Vision messages: %d parts (has_images=%s)", len(content_parts), has_images)
    return result


# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role:        str
    content:     str = Field(..., max_length=32000)
    attachments: Optional[List[str]] = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "system"):
            raise ValueError("role phải là user/assistant/system")
        return v

class ChatRequest(BaseModel):
    messages:    List[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens:  int   = Field(2048, ge=1, le=4096)


# ── FastAPI app ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    total = time.monotonic() - _SERVER_START_TIME
    logger.info("Server ready in %.1fs — http://127.0.0.1:8080", total)
    print(f"\n{'═'*54}")
    print(f"  ✅  Server ready in {total:.1f}s total")
    print(f"  🌐  http://127.0.0.1:8080")
    print(f"  📋  Log: {os.path.relpath(LOG_FILE_PATH, BASE_DIR)}")
    print(f"  🔍  Reranker: {'ON' if _reranker else 'OFF'}")
    print(f"  👁️  Vision: {'YES' if is_vision_model else 'no'}")
    print(f"{'═'*54}\n")
    print("[SUCCESS] AI Server started successfully")
    sys.stdout.flush()
    yield
    logger.info("Server shutdown after %.1fs", time.monotonic() - _SERVER_START_TIME)
    print("\n👋 Server stopped.")


app = FastAPI(title="3D-Reconstruction AI Server", version="2.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["POST","GET"], allow_headers=["*"]
)


import threading
import asyncio

llm_lock = threading.Lock()
current_task_id = None

@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest, http_req: Request):
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM chưa khởi tạo")

    req_start  = time.monotonic()
    user_query = request.messages[-1].content
    attachments = request.messages[-1].attachments or []
    query_image_b64 = None
    for att in attachments:
        if _is_image_file(att):
            try:
                query_image_b64 = _image_to_data_uri(att)
                break
            except Exception as e:
                logger.error("Failed to read attachment for retrieval: %s", e)

    logger.info("Query from %s: %s…", http_req.client.host,
                user_query[:60].replace("\n", " "))

    t_rag             = time.monotonic()
    doc_ctx, code_ctx, image_chunks = get_context(user_query, query_image_b64=query_image_b64)
    rag_ms            = (time.monotonic() - t_rag) * 1000

    messages_raw     = [m.model_dump() for m in request.messages]

    if is_vision_model:
        msgs = build_vision_messages(messages_raw, doc_ctx, code_ctx, image_chunks)
        estimated_tokens = sum(estimate_tokens(m.get("text", "")) for p in msgs for m in (p.get("content") if isinstance(p.get("content"), list) else [{"text": p.get("content", "")}]))
    else:
        msgs = build_text_messages(messages_raw, doc_ctx, code_ctx)
        estimated_tokens = sum(estimate_tokens(m.get("content", "")) for m in msgs)

    if estimated_tokens >= LLM_N_CTX - 512:
        raise HTTPException(
            status_code=400,
            detail=f"Hội thoại quá dài (~{estimated_tokens} tokens). Vui lòng bắt đầu phiên mới."
        )

    available_tokens = LLM_N_CTX - estimated_tokens - 400
    max_tokens       = min(request.max_tokens, max(512, available_tokens))

    def run_llm():
        with llm_lock:
            answer = ""
            finish_reason = "stop"
            start_time = time.monotonic()
            
            try:
                response_iter = llm.create_chat_completion(
                    messages       = msgs,
                    max_tokens     = max_tokens,
                    temperature    = request.temperature,
                    repeat_penalty = 1.1,
                    stream         = True,
                )
                
                for chunk in response_iter:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        answer += delta["content"]
                        
                    fr = chunk["choices"][0].get("finish_reason")
                    if fr is not None:
                        finish_reason = fr

            except Exception as e:
                logger.error("LLM error: %s", e)
                raise
                
            return answer.strip(), finish_reason, (time.monotonic() - start_time) * 1000

    try:
        answer, finish_reason, llm_ms = run_llm()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi inference: {e}")

    total_ms = (time.monotonic() - req_start) * 1000

    if finish_reason == "length":
        logger.warning(
            "Answer truncated (finish_reason=length): max_tokens=%d, estimated_prompt=%d",
            max_tokens, estimated_tokens
        )
        answer += "\n\n⚠️ *(Câu trả lời có thể chưa hoàn chỉnh — ngữ cảnh quá dài. "
        answer += "Hãy hỏi cụ thể hơn hoặc bắt đầu hội thoại mới.)*"

    logger.info(
        "Done | rag=%.0fms llm=%.0fms total=%.0fms | "
        "ctx=%d+%d ans=%d tokens≈%d/%d finish=%s vision=%s",
        rag_ms, llm_ms, total_ms,
        len(doc_ctx), len(code_ctx), len(answer),
        estimated_tokens, LLM_N_CTX, finish_reason, is_vision_model,
    )

    return {
        "id":      f"chatcmpl-{int(req_start*1000)}",
        "object":  "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": answer},
                     "finish_reason": finish_reason}],
        "usage":   {},
        "x_meta":  {
            "rag_ms":           round(rag_ms),
            "llm_ms":           round(llm_ms),
            "total_ms":         round(total_ms),
            "estimated_tokens": estimated_tokens,
            "max_tokens_used":  max_tokens,
            "finish_reason":    finish_reason,
            "reranker_active":  _reranker is not None,
            "vision_model":     is_vision_model,
        },
    }


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "uptime_sec":      round(time.monotonic() - _SERVER_START_TIME, 1),
        "llm_loaded":      llm is not None,
        "rag_chunks":      len(knowledge_chunks),
        "reranker":        _reranker is not None,
        "embed_model":     EMBED_MODEL_NAME,
        "chunk_chars":     CHUNK_CHARS,
        "chars_per_token": CHARS_PER_TOKEN,
        "max_context":     MAX_CONTEXT_CHARS,
        "model":           MODELS[MODEL_IDX]["desc"],
        "is_vision":       is_vision_model,
    }


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": MODELS[MODEL_IDX]["filename"],
                      "object": "model",
                      "desc": MODELS[MODEL_IDX]["desc"]}]}


# ─── 15. Main ─────────────────────────────────────────────────────────────────
def _print_banner():
    desc = MODELS[MODEL_IDX]["desc"]
    vision_str = "YES" if MODELS[MODEL_IDX].get("is_vision", False) else "no"
    print(f"""
╔══════════════════════════════════════════════════════╗
║         3D-Reconstruction AI Server v2.2             ║
╠══════════════════════════════════════════════════════╣
║  Model   : {desc:<42}║
║  Vision  : {vision_str:<42}║
║  Embed   : {EMBED_MODEL_NAME:<42}║
║  Reranker: {str(USE_RERANKER)+" ("+RERANKER_MODEL+")" if USE_RERANKER else "disabled":<42}║
║  Cache   : {os.path.relpath(CACHE_DIR, BASE_DIR):<42}║
╚══════════════════════════════════════════════════════╝""")


if __name__ == "__main__":
    import uvicorn
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    _print_banner()

    # Step 1: Embedding model (đa ngôn ngữ)
    with startup_step(f"Loading embedding model ({EMBED_MODEL_NAME})"):
        _embed = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=EMBED_CACHE)
        embed_model_ref = _embed

    # Step 2: Cross-encoder reranker (optional)
    if USE_RERANKER:
        with startup_step(f"Loading reranker ({RERANKER_MODEL})"):
            try:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(
                    RERANKER_MODEL,
                    max_length   = 512,
                    cache_folder = EMBED_CACHE,
                )
                logger.info("Reranker loaded: %s", RERANKER_MODEL)
            except Exception as e:
                logger.warning("Reranker load failed (%s) — continuing without", e)
                _reranker = None
    else:
        print("  ⏭   Reranker disabled (USE_RERANKER=False)")

    # Step 3: Scan documents
    with startup_step("Scanning documents"):
        raw_chunks = load_documents()
        if not raw_chunks:
            logger.warning("No documents found — RAG context will be empty")

    # Step 4: RAG index
    with startup_step("Loading/building RAG index"):
        if raw_chunks:
            knowledge_index, knowledge_chunks, bm25_index = \
                load_or_build_index(raw_chunks, _embed)
        else:
            knowledge_index  = None
            knowledge_chunks = []
            bm25_index       = None
            print("       (skipped — no documents)")

    # Step 5: Download LLM if needed
    selected   = MODELS[MODEL_IDX]
    model_path = os.path.join(MODELS_DIR, selected["filename"])

    if not os.path.exists(model_path):
        with startup_step(f"Downloading {selected['desc']}"):
            hf_hub_download(
                repo_id=selected["repo_id"],
                filename=selected["filename"],
                local_dir=MODELS_DIR,
            )
            logger.info("Downloaded: %s (%.1fGB)",
                        selected["filename"],
                        os.path.getsize(model_path)/1024**3)
    else:
        size_gb = os.path.getsize(model_path)/1024**3
        logger.info("Model cached: %s (%.1fGB)", model_path, size_gb)
        print(f"  ✓  Model found locally: {selected['filename']} ({size_gb:.1f}GB)")

    # Step 5b: Download mmproj (vision model only)
    if selected.get("is_vision", False):
        mmproj_path = os.path.join(MODELS_DIR, selected["mmproj_filename"])
        if not os.path.exists(mmproj_path):
            with startup_step(f"Downloading mmproj ({selected['mmproj_filename']})"):
                hf_hub_download(
                    repo_id=selected["mmproj_repo_id"],
                    filename=selected["mmproj_filename"],
                    local_dir=MODELS_DIR,
                )
                logger.info("Downloaded mmproj: %s (%.1fMB)",
                            selected["mmproj_filename"],
                            os.path.getsize(mmproj_path)/1024**2)
        else:
            size_mb = os.path.getsize(mmproj_path)/1024**2
            logger.info("mmproj cached: %s (%.1fMB)", mmproj_path, size_mb)
            print(f"  ✓  mmproj found locally: {selected['mmproj_filename']} ({size_mb:.0f}MB)")

    # Step 6: Load LLM
    if selected.get("is_vision", False):
        # ── Vision model: cần chat handler + mmproj ──
        with startup_step(f"Loading VL chat handler"):
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            _chat_handler = Qwen25VLChatHandler(
                clip_model_path=os.path.join(MODELS_DIR, selected["mmproj_filename"])
            )
            logger.info("Qwen25VLChatHandler loaded with %s", selected["mmproj_filename"])

        with startup_step(f"Loading LLM Vision ({selected['desc']})"):
            llm = Llama(
                model_path   = model_path,
                chat_handler = _chat_handler,
                chat_format  = "qwen2.5-vl",
                n_gpu_layers = 99,
                n_ctx        = LLM_N_CTX,
                n_batch      = 512,
                verbose      = False,
                use_mmap     = True,
                use_mlock    = False,
            )
        is_vision_model = True
        logger.info("Vision model activated: %s", selected["desc"])
    else:
        # ── Text-only model: giữ nguyên flow cũ ──
        with startup_step(f"Loading LLM ({selected['desc']})"):
            llm = Llama(
                model_path   = model_path,
                n_gpu_layers = 99,
                n_ctx        = LLM_N_CTX,
                n_batch      = 512,
                verbose      = False,
                use_mmap     = True,
                use_mlock    = False,
            )

    # Start server
    logger.info("Starting uvicorn on 127.0.0.1:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning", access_log=False)