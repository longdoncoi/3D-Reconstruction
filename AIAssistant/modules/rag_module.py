from .config import *  # noqa: I001
from .config import _safe_relpath


class _E5LlamaEmbedding:
    """Adapter that lets LlamaIndex use the application's E5 encoder/prefixes."""

    def __init__(self, model):
        from llama_index.core.embeddings import BaseEmbedding

        class E5Embedding(BaseEmbedding):
            model: object

            def _get_query_embedding(self, query: str):
                return self.model.encode(EMBEDDING_QUERY_PREFIX + query,
                                         normalize_embeddings=True).tolist()

            async def _aget_query_embedding(self, query: str):
                return self._get_query_embedding(query)

            def _get_text_embedding(self, text: str):
                return self.model.encode(EMBEDDING_PASSAGE_PREFIX + text,
                                         normalize_embeddings=True).tolist()

            def _get_text_embeddings(self, texts: list[str]):
                return self.model.encode([EMBEDDING_PASSAGE_PREFIX + text for text in texts],
                                         normalize_embeddings=True).tolist()

        self.value = E5Embedding(model=model)

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
    image_b64:   str | None = None
    metadata:    dict = field(default_factory=dict)

# ─── 8b. Vision helpers ──────────────────────────────────────────────────────
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def _is_image_file(filepath: str) -> bool:
    return os.path.splitext(filepath)[1].lower() in _IMAGE_EXTS

def _image_to_data_uri(filepath: str, max_dim: int = 512) -> str:
    import io

    from PIL import Image
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
    except Exception:
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"

def _load_image_for_embedding(filepath: str, max_dim: int = 256):
    from PIL import Image, ImageOps
    with Image.open(filepath) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return img.copy()

def _release_ml_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

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
        rel     = _safe_relpath(filepath, PROJECT_DIR)
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

    def _read_text_file(self, filepath: str) -> str | None:
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


class EmailLoader(BaseDocumentLoader):
    """Extract readable text from saved .eml project correspondence."""
    def can_handle(self, fp: str) -> bool: return fp.lower().endswith(".eml")

    def load(self, fp: str) -> list:
        from email import policy
        from email.parser import BytesParser

        with open(fp, "rb") as handle:
            message = BytesParser(policy=policy.default).parse(handle)
        parts = []
        subject = str(message.get("subject", "")).strip()
        if subject:
            parts.append(f"Subject: {subject}")
        for part in message.walk():
            if part.get_content_disposition() == "attachment" or part.get_content_type() != "text/plain":
                continue
            try:
                body = part.get_content().strip()
            except (LookupError, UnicodeError):
                body = ""
            if body:
                parts.append(body)
        content = "\n\n".join(parts)
        if len(content) < self.MIN_CHUNK_CHARS:
            raise ValueError(f"Email rong hoac khong co text: {fp}")
        return self._sliding_window_chunks(content, fp, label="Email")


class MarkdownLoader(BaseDocumentLoader):
    HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def can_handle(self, fp: str) -> bool: return fp.lower().endswith(".md")

    def load(self, fp: str) -> list:
        content = self._read_text_file(fp)
        if not content or len(content.strip()) < self.MIN_CHUNK_CHARS:
            raise ValueError(f"File rỗng: {fp}")

        rel     = _safe_relpath(fp, PROJECT_DIR)
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
        rel       = _safe_relpath(fp, PROJECT_DIR)
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
        rel     = _safe_relpath(fp, PROJECT_DIR)
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
            rel = _safe_relpath(fp, PROJECT_DIR)
            return [ChunkResult(
                text=f"[Image: {rel}]",
                source_path=fp,
                loader_type="image",
                is_image=True,
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

    def get_loader(self, fp: str) -> object | None:
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
        .register(EmailLoader())
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
DOC_EXTS = {".docx", ".pdf", ".txt", ".eml", ".jpg", ".jpeg", ".png", ".webp"}


def load_documents() -> list:
    registry   = build_registry()
    all_chunks: list = []
    stats = {"docx":0,"pdf":0,"txt":0,"eml":0,"md":0,"source":0,"errors":0,"files":0}

    for docs_dir in dict.fromkeys(RAG_DOCUMENT_DIRS):
        if os.path.isdir(docs_dir):
            for root, _, files in os.walk(docs_dir):
                for filename in sorted(files):
                    if os.path.splitext(filename)[1].lower() not in DOC_EXTS:
                        continue
                    fp = os.path.join(root, filename)
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
          f"  (docx={stats['docx']} pdf={stats['pdf']} txt={stats['txt']} eml={stats['eml']}"
          f" md={stats['md']} src={stats['source']} err={stats['errors']})")
    return all_chunks


# ─── 12. Cache management ─────────────────────────────────────────────────────
def get_file_system_hash() -> str:
    entries = []
    for docs_dir in dict.fromkeys(RAG_DOCUMENT_DIRS):
        if not os.path.isdir(docs_dir):
            continue
        for root, _, files in os.walk(docs_dir):
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
    entries.append(f"cache_version={RAG_CACHE_VERSION}")

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
        valid = (meta.get("fs_hash") == current_hash
                 and meta.get("embedding_dimension") == EMBEDDING_DIMENSION)
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
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_chars": CHUNK_CHARS,
        "cache_version": RAG_CACHE_VERSION,
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
        if index.d != EMBEDDING_DIMENSION:
            raise ValueError(f"FAISS dimension {index.d} does not match {EMBEDDING_DIMENSION}")
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

    t = time.monotonic()
    logger.info("Building index: %d chunks", len(chunks))

    t_enc = time.monotonic()
    text_indices = []
    texts = []
    image_items = []

    for i, c in enumerate(chunks):
        if getattr(c, "is_image", False) and EMBEDDING_SUPPORTS_IMAGES:
            image_items.append((i, c.source_path))
        else:
            # multilingual-e5 requires passage/query prefixes.  Omitting them
            # noticeably reduces retrieval quality, especially for Vietnamese.
            texts.append(EMBEDDING_PASSAGE_PREFIX + getattr(c, "text", str(c)))
            text_indices.append(i)

    final_embeddings = [None] * len(chunks)
    
    if texts:
        print(f"Encoding {len(texts)} text chunks", end="", flush=True)
        text_embs = embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=64)
        for idx, emb in zip(text_indices, text_embs):
            final_embeddings[idx] = emb
            
    if image_items:
        image_batch_size = 4
        print(f"Encoding {len(image_items)} image chunks", end="", flush=True)
        for start in range(0, len(image_items), image_batch_size):
            batch_items = image_items[start:start + image_batch_size]
            batch_indices = []
            batch_images = []
            for idx, image_path in batch_items:
                try:
                    batch_images.append(_load_image_for_embedding(image_path))
                    batch_indices.append(idx)
                except Exception as e:
                    logger.warning("Image load skipped for embedding: %s | %s", image_path, e)

            if not batch_images:
                continue

            try:
                img_embs = embed_model.encode(
                    batch_images,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=len(batch_images)
                )
                for idx, emb in zip(batch_indices, img_embs):
                    final_embeddings[idx] = emb
            except Exception as e:
                logger.warning(
                    "Image embedding batch failed (%d-%d): %s; retrying one by one",
                    start + 1, start + len(batch_items), e
                )
                for idx, img in zip(batch_indices, batch_images):
                    try:
                        emb = embed_model.encode(
                            [img],
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            batch_size=1
                        )[0]
                        final_embeddings[idx] = emb
                    except Exception as single_e:
                        logger.warning(
                            "Image embedding skipped for %s: %s",
                            getattr(chunks[idx], "source_path", idx), single_e
                        )
            finally:
                for img in batch_images:
                    try:
                        img.close()
                    except Exception:
                        pass
                _release_ml_memory()

    missing_indices = [i for i, emb in enumerate(final_embeddings) if emb is None]
    if missing_indices:
        logger.warning("Falling back to text embeddings for %d chunks", len(missing_indices))
        fallback_texts = [getattr(chunks[i], "text", str(chunks[i])) for i in missing_indices]
        fallback_embs = embed_model.encode(
            fallback_texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32
        )
        for idx, emb in zip(missing_indices, fallback_embs):
            final_embeddings[idx] = emb

    embeddings = final_embeddings
    print(f"\n ✓ Encoding complete ({time.monotonic()-t_enc:.1f}s)")

    emb  = np.array(embeddings, dtype="float32")
    dim, n = emb.shape[1], len(emb)
    if dim != EMBEDDING_DIMENSION:
        raise ValueError(f"Embedding model returned {dim} dimensions; expected {EMBEDDING_DIMENSION}")

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
        # IVF defaults to probing one cluster, which trades away recall.  RAG
        # answers value recall at this stage; reranking handles precision later.
        index.nprobe = min(16, nlist)

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
rag_lock = threading.RLock()
vector_retriever = None
bm25_retriever = None


def build_llamaindex_runtime(chunks: list, encoder):
    """Build the persistent dense FAISS + sparse BM25 retrievers through LlamaIndex."""
    import faiss
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.vector_stores.faiss import FaissVectorStore

    nodes = [TextNode(text=chunk.text, metadata={"chunk_index": index})
             for index, chunk in enumerate(chunks) if not chunk.is_image]
    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(EMBEDDING_DIMENSION))
    storage = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage,
                             embed_model=_E5LlamaEmbedding(encoder).value,
                             show_progress=False)
    storage.persist(persist_dir=os.path.join(CACHE_DIR, "llamaindex"))
    return index, index.as_retriever(similarity_top_k=30), BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=30, tokenizer=_tokenize_vn,
        skip_stemming=True)


def initialize_rag(force_rebuild: bool = False, enable_reranker: bool = True) -> int:
    """Load embeddings and atomically publish a complete RAG runtime."""
    global knowledge_index, knowledge_chunks, bm25_index, embed_model_ref, _reranker, vector_retriever, bm25_retriever

    if not ENABLE_RAG:
        with rag_lock:
            knowledge_index, knowledge_chunks, bm25_index = None, [], None
            vector_retriever, bm25_retriever = None, None
            embed_model_ref, _reranker = None, None
        return 0

    from sentence_transformers import CrossEncoder, SentenceTransformer
    with startup_step(f"Loading embedding model ({EMBED_MODEL_NAME})"):
        embed_model = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=EMBED_CACHE)
    reranker = None
    if USE_RERANKER and enable_reranker:
        try:
            with startup_step(f"Loading reranker ({RERANKER_MODEL})"):
                reranker = CrossEncoder(RERANKER_MODEL, max_length=512, cache_folder=EMBED_CACHE)
        except Exception as error:
            logger.warning("Reranker load failed (%s); continuing without it", error)

    with startup_step("Scanning documents"):
        raw_chunks = load_documents()
    if raw_chunks:
        with startup_step("Rebuilding RAG index" if force_rebuild else "Loading/building RAG index"):
            index, dense, sparse = build_llamaindex_runtime(raw_chunks, embed_model)
            chunks, bm25 = raw_chunks, sparse
    else:
        index, chunks, bm25, dense, sparse = None, [], None, None, None
        logger.warning("No documents found; RAG context will be empty")

    with rag_lock:
        knowledge_index, knowledge_chunks, bm25_index = index, chunks, bm25
        vector_retriever, bm25_retriever = dense, sparse
        embed_model_ref, _reranker = embed_model, reranker
    return len(chunks)


def release_embedding_for_vision() -> None:
    """Keep the FAISS/BM25 index but free embedding-model memory for a VL model."""
    global embed_model_ref
    with rag_lock:
        embed_model_ref = None
    _release_ml_memory()
knowledge_chunks = []
bm25_index       = None
vector_retriever = None
bm25_retriever   = None
embed_model_ref  = None
_reranker        = None   # Cross-encoder, load lúc startup nếu USE_RERANKER=True
is_vision_model  = False  # True khi chạy Qwen2.5-VL (vision model)


def _rrf_fuse(rankings: list[list[int]], weights: list[float], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked lists without query-dependent score normalization.

    Min/max normalization makes a weak result look strong whenever every
    candidate has a similar score. Reciprocal-rank fusion preserves agreement
    between lexical and semantic retrievers and is robust across queries.
    """
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranking, 1):
            if chunk_id >= 0:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def hybrid_retrieve(query: str, query_image_b64: str | None = None, k: int = 30, final_k: int = 12) -> list:
    """
    Hybrid semantic (text/image) + BM25 (text only) retrieval.
    """
    if vector_retriever is None or bm25_retriever is None or not knowledge_chunks or (not query.strip() and not query_image_b64):
        return []

    dense = vector_retriever.retrieve(query)
    sparse = bm25_retriever.retrieve(query)
    dense_ids = [item.node.metadata["chunk_index"] for item in dense
                 if item.score is None or item.score >= SIMILARITY_THRESHOLD]
    sparse_ids = [item.node.metadata["chunk_index"] for item in sparse]
    combined = _rrf_fuse([dense_ids, sparse_ids], [0.55, 0.45])   # Reciprocal Rank Fusion: 0.55 semantic + 0.45 bm25

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


def get_context(query: str, query_image_b64: str | None = None, result_k: int = RERANKER_TOP_K) -> tuple:
    if not ENABLE_RAG:
        return "", "", []

    # Bước 1: Hybrid retrieve
    candidates = hybrid_retrieve(query, query_image_b64=query_image_b64, k=30, final_k=12)

    # Bước 2: Cross-encoder re-rank
    if USE_RERANKER and query:
        candidates = _rerank(query, candidates)
        candidates = candidates[:RERANKER_TOP_K]
    else:
        candidates = candidates[:12]

    # Bước 3: Source dedup
    candidates = _dedup_by_source(candidates, max_per_source=2)
    candidates = candidates[:max(1, min(result_k, RERANKER_TOP_K))]

    # Bước 4: Phân loại + cắt theo ngân sách context
    doc_chunks   = []
    code_chunks  = []
    image_chunks = []
    total        = 0

    for chunk in candidates:
        if getattr(chunk, "is_image", False):
            try:
                image_chunks.append(chunk.image_b64 or _image_to_data_uri(chunk.source_path))
            except Exception as e:
                logger.warning("Failed to prepare image context %s: %s",
                               getattr(chunk, "source_path", "?"), e)
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


