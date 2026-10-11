"""Document loaders and semantic chunking."""
from __future__ import annotations

import ast
import logging
import os
import re
from abc import ABC, abstractmethod

from ai_assistant.config.paths import safe_relpath

from .domain import ChunkResult
from .image_utils import is_image_file

logger = logging.getLogger("ai_assistant.rag.loaders")


class BaseDocumentLoader(ABC):
    def __init__(self, chunk_chars: int = 1200, overlap_chars: int = 300, min_chunk_chars: int = 80):
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars
        self._sent_endings = (".\n", ". ", "!\n", "! ", "?\n", "? ", ";\n", "\n\n")

    @abstractmethod
    def can_handle(self, filepath: str) -> bool: ...

    @abstractmethod
    def load(self, filepath: str, project_dir: str) -> list[ChunkResult]: ...

    def _is_quality_chunk(self, block: str) -> bool:
        stripped = block.strip()
        if len(stripped) < self.min_chunk_chars:
            return False
        non_comment = re.sub(
            r"^\s*(//[^\n]*|/\*.*?\*/)", "", stripped,
            flags=re.DOTALL | re.MULTILINE,
        ).strip()
        return len(non_comment) >= self.min_chunk_chars // 2

    def _snap_to_sentence(self, text: str) -> str:
        """Snap về ranh giới câu gần cuối nhất trong nửa sau của text."""
        min_pos = len(text) // 2
        best = -1
        for ending in self._sent_endings:
            pos = text.rfind(ending)
            if pos > min_pos and pos > best:
                best = pos
        return text[:best + 1] if best > min_pos else text

    def _sliding_window_chunks(self, content: str, filepath: str, project_dir: str, label: str = "Source") -> list[ChunkResult]:
        """Sentence-aware sliding window chunking."""
        rel = safe_relpath(filepath, project_dir)
        results = []
        pos = 0

        while pos < len(content):
            end = pos + self.chunk_chars
            block = content[pos:end]

            if end < len(content) and len(block) > self.overlap_chars * 2:
                snapped = self._snap_to_sentence(block)
                if len(snapped.strip()) >= self.min_chunk_chars:
                    block = snapped

            block = block.strip()
            if self._is_quality_chunk(block):
                results.append(ChunkResult(
                    text=f"[{label}: {rel}]\n{block}",
                    source_path=filepath,
                    loader_type=label.lower().replace(" ", "_"),
                ))

            advance = max(len(block) - self.overlap_chars, self.chunk_chars - self.overlap_chars)
            if advance <= 0:
                # Guard against misconfigured loaders (chunk_chars <= overlap_chars):
                # both candidates can go non-positive, which would make ``pos``
                # move backwards and spin forever.
                advance = max(self.chunk_chars, 1)
            pos += advance

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
    def load(self, fp: str, project_dir: str) -> list[ChunkResult]:
        from docx import Document
        doc = Document(fp)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return self._sliding_window_chunks(text, fp, project_dir, label="Tai lieu")


class PdfLoader(BaseDocumentLoader):
    def can_handle(self, fp: str) -> bool: return fp.lower().endswith(".pdf")
    def load(self, fp: str, project_dir: str) -> list[ChunkResult]:
        text = self._extract_pdf_text(fp)
        if not text:
            raise ValueError(f"Không đọc được PDF: {fp}")
        return self._sliding_window_chunks(text, fp, project_dir, label="Tai lieu PDF")

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
    def load(self, fp: str, project_dir: str) -> list[ChunkResult]:
        content = self._read_text_file(fp)
        if not content or len(content.strip()) < self.min_chunk_chars:
            raise ValueError(f"File rỗng: {fp}")
        return self._sliding_window_chunks(content, fp, project_dir, label="Tai lieu TXT")


class EmailLoader(BaseDocumentLoader):
    def can_handle(self, fp: str) -> bool: return fp.lower().endswith(".eml")
    def load(self, fp: str, project_dir: str) -> list[ChunkResult]:
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
        if len(content) < self.min_chunk_chars:
            raise ValueError(f"Email rong hoac khong co text: {fp}")
        return self._sliding_window_chunks(content, fp, project_dir, label="Email")


class MarkdownLoader(BaseDocumentLoader):
    HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def can_handle(self, fp: str) -> bool: return fp.lower().endswith(".md")
    def load(self, fp: str, project_dir: str) -> list[ChunkResult]:
        content = self._read_text_file(fp)
        if not content or len(content.strip()) < self.min_chunk_chars:
            raise ValueError(f"File rỗng: {fp}")

        rel = safe_relpath(fp, project_dir)
        matches = list(self.HEADING_RE.finditer(content))
        if not matches:
            return self._sliding_window_chunks(content, fp, project_dir, label="Source MD")

        results = []
        boundaries = [m.start() for m in matches] + [len(content)]
        h1_parent_text: str | None = None

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(2).strip()
            section = content[boundaries[i]:boundaries[i+1]].strip()
            section = self._strip_large_code_blocks(section)
            if not self._is_quality_chunk(section):
                continue

            prefix = f"[Source MD: {rel}] {'#'*level} {heading}\n"

            if level == 1:
                h1_parent_text = prefix + section[:self.chunk_chars]
            parent_text_for_chunk = h1_parent_text if level > 1 else None
            h_level = level

            if len(section) <= self.chunk_chars:
                results.append(ChunkResult(
                    text=prefix + section,
                    source_path=fp,
                    loader_type="md",
                    metadata={"heading": heading, "level": level},
                    parent_text=parent_text_for_chunk,
                    hierarchy_level=h_level,
                ))
            else:
                for sc in self._sliding_window_chunks(section, fp, project_dir, label="Source MD"):
                    sc.text = prefix + sc.text.split("\n", 1)[-1]
                    # Keep the loader type consistent with short sections: a
                    # section headed with ``#`` is markdown, not a generic source.
                    sc.loader_type = "md"
                    sc.metadata = {"heading": heading, "level": level}
                    sc.parent_text = parent_text_for_chunk
                    sc.hierarchy_level = h_level
                    results.append(sc)

        return results

    def _strip_large_code_blocks(self, text: str) -> str:
        def maybe_strip(m):
            lines = m.group(0).count("\n")
            return m.group(0) if lines <= 30 else f"[code block omitted – {lines} lines]"
        return re.sub(r"```[\s\S]*?```", maybe_strip, text)


class CppHeaderLoader(BaseDocumentLoader):
    SOURCE_EXTS = {".cpp", ".h", ".py", ".cmake"}
    # The leading return-type/qualifier tokens use a flat quantifier
    # (``T(?:\s+T)*\s+``) instead of the nested ``(?:T\s+)+``. The two accept
    # the same language, but the flat form cannot become ambiguous if the token
    # class is later extended to include whitespace, so it is defensive without
    # changing which spans are matched.
    FUNC_RE = re.compile(
        r"(?:^|\n)(?:"
        r"(?:class|struct|namespace)\s+\w+.*?\{"
        r"|[\w:*&<>\[\]~]+(?:\s+[\w:*&<>\[\]~]+)*\s+(?:\w+::)*\w+\s*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{"
        r")",
        re.MULTILINE,
    )

    def can_handle(self, fp: str) -> bool:
        return os.path.splitext(fp)[1].lower() in self.SOURCE_EXTS

    def load(self, fp: str, project_dir: str) -> list[ChunkResult]:
        content = self._read_text_file(fp)
        if not content or len(content.strip()) < self.min_chunk_chars:
            raise ValueError(f"File rỗng: {fp}")
        ext = os.path.splitext(fp)[1].lower()
        return self._load_python(content, fp, project_dir) if ext == ".py" else self._load_cpp(content, fp, project_dir)

    def _load_cpp(self, content: str, fp: str, project_dir: str) -> list[ChunkResult]:
        rel = safe_relpath(fp, project_dir)
        positions = [m.start() for m in self.FUNC_RE.finditer(content)]
        results = []
        if positions:
            positions.append(len(content))
            for i, start in enumerate(positions[:-1]):
                block = content[start:positions[i+1]].strip()
                if not self._is_quality_chunk(block):
                    continue
                header = block.split("\n")[0].strip().rstrip("{").strip()
                results.append(ChunkResult(
                    text=f"[Source: {rel}] {header}\n{block[:self.chunk_chars]}",
                    source_path=fp,
                    loader_type="source",
                    metadata={"symbol": header},
                ))
        else:
            results = self._sliding_window_chunks(content, fp, project_dir, label="Source")
        return results

    def _load_python(self, content: str, fp: str, project_dir: str) -> list[ChunkResult]:
        rel = safe_relpath(fp, project_dir)
        results = []
        lines = content.splitlines()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._sliding_window_chunks(content, fp, project_dir, label="Source")

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 50)
            block = "\n".join(lines[start:end]).strip()
            if not self._is_quality_chunk(block):
                continue

            is_class = isinstance(node, ast.ClassDef)
            scope = "Class" if is_class else "Function"

            if is_class:
                class_text = f"[Source: {rel}] Class: {node.name}\n{block[:self.chunk_chars]}"
                class_chunk = ChunkResult(
                    text=class_text,
                    source_path=fp,
                    loader_type="source",
                    metadata={"symbol": node.name, "scope": "Class"},
                    hierarchy_level=1,
                )
                results.append(class_chunk)

                for method in ast.iter_child_nodes(node):
                    if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    ms = method.lineno - 1
                    me = getattr(method, "end_lineno", ms + 20)
                    mblock = "\n".join(lines[ms:me]).strip()
                    if not self._is_quality_chunk(mblock):
                        continue
                    results.append(ChunkResult(
                        text=f"[Source: {rel}] Method: {node.name}.{method.name}\n{mblock[:self.chunk_chars]}",
                        source_path=fp,
                        loader_type="source",
                        metadata={"symbol": method.name, "scope": "Method", "class": node.name},
                        parent_text=class_text,
                        hierarchy_level=2,
                    ))
            else:
                results.append(ChunkResult(
                    text=f"[Source: {rel}] {scope}: {node.name}\n{block[:self.chunk_chars]}",
                    source_path=fp,
                    loader_type="source",
                    metadata={"symbol": node.name, "scope": scope},
                ))

        return results or self._sliding_window_chunks(content, fp, project_dir, label="Source")


class ImageLoader(BaseDocumentLoader):
    def can_handle(self, fp: str) -> bool:
        return is_image_file(fp)

    def load(self, fp: str, project_dir: str) -> list[ChunkResult]:
        try:
            rel = safe_relpath(fp, project_dir)
            return [ChunkResult(
                text=f"[Image: {rel}]",
                source_path=fp,
                loader_type="image",
                is_image=True,
            )]
        except Exception as e:
            logger.error("Error loading image %s: %s", fp, e)
            return []


class DocumentLoaderRegistry:
    def __init__(self, chunk_chars: int = 1200, overlap_chars: int = 300):
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self._loaders: list[BaseDocumentLoader] = []

    def register(self, loader: BaseDocumentLoader) -> "DocumentLoaderRegistry":
        self._loaders.append(loader)
        return self

    def get_loader(self, fp: str) -> BaseDocumentLoader | None:
        for loader in self._loaders:
            if loader.can_handle(fp):
                return loader
        return None

    def load_file(self, fp: str, project_dir: str) -> list[ChunkResult]:
        loader = self.get_loader(fp)
        if loader is None:
            logger.debug("No loader for: %s", fp)
            return []
        try:
            return loader.load(fp, project_dir)
        except Exception as e:
            logger.error("Error loading %s: %s", fp, e)
            return []

    @classmethod
    def create_default(cls, chunk_chars: int = 1200, overlap_chars: int = 300) -> "DocumentLoaderRegistry":
        """Create a registry pre-populated with all standard loaders."""
        reg = cls(chunk_chars, overlap_chars)
        reg.register(DocxLoader(chunk_chars, overlap_chars))
        reg.register(PdfLoader(chunk_chars, overlap_chars))
        reg.register(TxtLoader(chunk_chars, overlap_chars))
        reg.register(EmailLoader(chunk_chars, overlap_chars))
        reg.register(MarkdownLoader(chunk_chars, overlap_chars))
        reg.register(CppHeaderLoader(chunk_chars, overlap_chars))
        reg.register(ImageLoader(chunk_chars, overlap_chars))
        return reg
