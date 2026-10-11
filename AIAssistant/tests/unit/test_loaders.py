"""Unit tests for ``ai_assistant.rag.loaders``.

Only the offline loaders are exercised end-to-end (text, markdown, source and
image); the docx/pdf/email adapters are only checked for extension routing so
the suite never imports python-docx, pdfplumber or email parsers.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.rag.loaders import (
    BaseDocumentLoader,
    CppHeaderLoader,
    DocumentLoaderRegistry,
    DocxLoader,
    EmailLoader,
    ImageLoader,
    MarkdownLoader,
    PdfLoader,
    TxtLoader,
)

_SRC = Path(__file__).resolve().parents[2] / "src"


class _ConcreteLoader(BaseDocumentLoader):
    """Minimal concrete loader to exercise the shared chunking helpers."""

    def can_handle(self, filepath: str) -> bool:
        return True

    def load(self, filepath: str, project_dir: str):
        return self._sliding_window_chunks("x" * 5000, filepath, project_dir)


class BaseLoaderHelpersTests(unittest.TestCase):
    def setUp(self):
        self.loader = _ConcreteLoader(chunk_chars=500, overlap_chars=100, min_chunk_chars=10)

    def test_quality_chunk_length_gate(self):
        self.assertFalse(self.loader._is_quality_chunk("short"))
        self.assertTrue(self.loader._is_quality_chunk("a" * 20))

    def test_quality_chunk_rejects_comment_only_blocks(self):
        block = "// just a comment\n// and another one"
        self.assertFalse(self.loader._is_quality_chunk(block))

    def test_snap_to_sentence_keeps_text_without_late_ending(self):
        text = "Hello. " + "word " * 40
        self.assertEqual(self.loader._snap_to_sentence(text), text)

    def test_snap_to_sentence_trims_at_late_boundary(self):
        text = "This is a reasonably long sentence here. And more trailing text"
        snapped = self.loader._snap_to_sentence(text)
        self.assertTrue(snapped.endswith("here."))
        self.assertLess(len(snapped), len(text))

    def test_sliding_window_produces_multiple_prefixed_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "big.txt")
            results = self.loader._sliding_window_chunks("word " * 2000, fp, tmp)
        self.assertGreater(len(results), 1)
        for chunk in results:
            self.assertTrue(chunk.text.startswith("[Source: "))
            self.assertEqual(chunk.loader_type, "source")
            self.assertEqual(chunk.source_path, fp)

    def test_sliding_window_skips_content_below_minimum(self):
        self.assertEqual(self.loader._sliding_window_chunks("hi", "a.txt", "."), [])

    def test_misconfigured_overlap_never_spins(self):
        """chunk_chars <= overlap_chars must not move the cursor backwards."""
        odd = _ConcreteLoader(chunk_chars=50, overlap_chars=300, min_chunk_chars=10)
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "big.txt")
            results = odd._sliding_window_chunks("word " * 5000, fp, tmp)
        self.assertTrue(results)

    def test_read_text_file_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "note.txt")
            Path(fp).write_text("café ☕", encoding="utf-8")
            self.assertEqual(self.loader._read_text_file(fp), "café ☕")

    def test_read_text_file_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.loader._read_text_file("does-not-exist.txt")


class TxtLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = TxtLoader(min_chunk_chars=20)

    def test_can_handle(self):
        self.assertTrue(self.loader.can_handle("a.txt"))
        self.assertFalse(self.loader.can_handle("a.md"))

    def test_load_chunks_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "doc.txt")
            Path(fp).write_text("Sentence one. " * 40, encoding="utf-8")
            chunks = self.loader.load(fp, tmp)
        self.assertTrue(chunks)
        self.assertTrue(chunks[0].text.startswith("[Tai lieu TXT: "))

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "empty.txt")
            Path(fp).write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.loader.load(fp, tmp)


class MarkdownLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = MarkdownLoader(min_chunk_chars=20)

    def test_can_handle(self):
        self.assertTrue(self.loader.can_handle("a.md"))
        self.assertFalse(self.loader.can_handle("a.txt"))

    def test_headings_produce_hierarchical_chunks(self):
        body = "Lots of words here. " * 10
        content = f"# Heading One\n\n{body}\n\n## Heading Two\n\n{body}\n"
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "doc.md")
            Path(fp).write_text(content, encoding="utf-8")
            chunks = self.loader.load(fp, tmp)
        h1 = [c for c in chunks if c.metadata.get("level") == 1]
        h2 = [c for c in chunks if c.metadata.get("level") == 2]
        self.assertEqual(len(h1), 1)
        self.assertEqual(len(h2), 1)
        self.assertIsNone(h1[0].parent_text)
        self.assertIsNotNone(h2[0].parent_text)
        self.assertEqual(h2[0].hierarchy_level, 2)
        self.assertTrue(h1[0].text.startswith("[Source MD: "))

    def test_without_headings_uses_sliding_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "flat.md")
            Path(fp).write_text("paragraph text. " * 60, encoding="utf-8")
            chunks = self.loader.load(fp, tmp)
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].loader_type, "source_md")

    def test_tiny_section_is_skipped(self):
        content = "# A\n\nx\n\n# B\n\n" + "real content here. " * 10
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "t.md")
            Path(fp).write_text(content, encoding="utf-8")
            chunks = self.loader.load(fp, tmp)
        headings = {c.metadata.get("heading") for c in chunks}
        self.assertNotIn("A", headings)

    def test_large_code_block_is_stripped(self):
        big_fence = "```\n" + "\n" * 40 + "```"
        result = self.loader._strip_large_code_blocks(big_fence)
        self.assertIn("code block omitted", result)
        self.assertNotIn("```", result)

    def test_small_code_block_is_kept(self):
        small_fence = "```python\nprint(1)\n```"
        self.assertEqual(self.loader._strip_large_code_blocks(small_fence), small_fence)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "empty.md")
            Path(fp).write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.loader.load(fp, tmp)


class CppHeaderLoaderTests(unittest.TestCase):
    # The pre-hardening form kept only as an oracle: ``FUNC_RE`` must match
    # exactly the same spans after the nested quantifier was flattened.
    _PRE_HARDENING_FUNC_RE = re.compile(
        r"(?:^|\n)(?:"
        r"(?:class|struct|namespace)\s+\w+.*?\{"
        r"|(?:[\w:*&<>\[\]~]+\s+)+(?:\w+::)*\w+\s*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{"
        r")",
        re.MULTILINE,
    )

    def setUp(self):
        self.loader = CppHeaderLoader(min_chunk_chars=10)

    def test_func_re_matches_same_spans_as_pre_hardening_form(self):
        samples = [
            "class Baz {\npublic:\n    void run() {}\n};\n",
            "void free_function() {\n    int x = 1;\n}\n",
            "namespace app {\nclass Widget {\n    void paint() const noexcept {\n    }\n};\n}\n",
            "std::vector<int> get_items(int a, int b) const {\n    return {};\n}\n",
            "template<class T> T identity(T a) noexcept {\n    return a;\n}\n",
            "int Foo::bar( int x ) {\n    return x;\n}\n",
            "struct Point { int x; int y; };\n",
            "int add(int a, int b);\nint sub(int a, int b);\n",
            "unsigned long long compute(const char* name, std::size_t n) {\n    return n;\n}\n",
            "static inline bool ready() { return true; }\n",
        ]
        for sample in samples:
            with self.subTest(sample=sample.splitlines()[0]):
                self.assertEqual(
                    [m.span() for m in self.loader.FUNC_RE.finditer(sample)],
                    [m.span() for m in self._PRE_HARDENING_FUNC_RE.finditer(sample)],
                )

    def test_func_re_stays_bounded_on_pathological_input(self):
        # A long run of space-separated tokens without a trailing brace is the
        # shape that would hang a genuinely ambiguous ``(?:X+\s+)+`` pattern.
        # Run it in a subprocess so a future regression fails fast instead of
        # hanging the whole suite.
        script = (
            "import time\n"
            f"import sys; sys.path.insert(0, {str(_SRC)!r})\n"
            "from ai_assistant.rag.loaders import CppHeaderLoader\n"
            "payload = 'word ' * 4000 + 'end'\n"
            "start = time.monotonic()\n"
            "CppHeaderLoader.FUNC_RE.findall(payload)\n"
            "print(time.monotonic() - start)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(float(result.stdout.strip()), 5.0)

    def test_python_class_non_method_members_are_skipped(self):
        # Class bodies also contain docstrings and class-level assignments;
        # those must be ignored so only real methods become chunks.
        content = (
            "class Widget:\n"
            '    """Docstring and data members are not methods."""\n'
            "    LIMIT = 5\n"
            "\n"
            "    def render(self, value):\n"
            "        return value * self.LIMIT\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "widget.py")
            Path(fp).write_text(content, encoding="utf-8")
            chunks = self.loader.load(fp, tmp)
        symbols = [c.metadata.get("symbol") for c in chunks]
        self.assertIn("render", symbols)
        self.assertNotIn("LIMIT", symbols)

    def test_can_handle_source_extensions(self):
        for ext in (".cpp", ".h", ".py", ".cmake"):
            self.assertTrue(self.loader.can_handle(f"x{ext}"), ext)
        self.assertFalse(self.loader.can_handle("x.txt"))

    def test_python_ast_chunks_classes_methods_functions(self):
        content = (
            "def foo():\n"
            "    return 1\n"
            "\n"
            "\n"
            "class Bar:\n"
            "    def method(self):\n"
            "        return 2\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "mod.py")
            Path(fp).write_text(content, encoding="utf-8")
            chunks = self.loader.load(fp, tmp)
        scopes = {c.metadata.get("scope") for c in chunks}
        self.assertIn("Function", scopes)
        self.assertIn("Class", scopes)
        self.assertIn("Method", scopes)
        method = next(c for c in chunks if c.metadata.get("scope") == "Method")
        self.assertEqual(method.hierarchy_level, 2)
        self.assertEqual(method.metadata["class"], "Bar")
        self.assertIsNotNone(method.parent_text)

    def test_python_syntax_error_falls_back_to_sliding_window(self):
        content = "def broken(:\n    pass\n" + "x = 1\n" * 40
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "broken.py")
            Path(fp).write_text(content, encoding="utf-8")
            chunks = self.loader.load(fp, tmp)
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].loader_type, "source")

    def test_cpp_functions_are_chunked(self):
        content = (
            "class Baz {\n"
            "public:\n"
            "    void run() {}\n"
            "};\n"
            "\n"
            "void free_function() {\n"
            "    int x = 1;\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "code.cpp")
            Path(fp).write_text(content, encoding="utf-8")
            chunks = self.loader.load(fp, tmp)
        self.assertTrue(chunks)
        self.assertTrue(any("symbol" in c.metadata for c in chunks))

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "empty.py")
            Path(fp).write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.loader.load(fp, tmp)


class ImageLoaderTests(unittest.TestCase):
    def test_can_handle_images(self):
        loader = ImageLoader()
        self.assertTrue(loader.can_handle("a.png"))
        self.assertTrue(loader.can_handle("a.JPG"))
        self.assertFalse(loader.can_handle("a.txt"))

    def test_load_returns_image_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "pic.png")
            Path(fp).write_bytes(b"\x89PNG")
            chunks = ImageLoader().load(fp, tmp)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].is_image)
        self.assertEqual(chunks[0].loader_type, "image")


class DocumentLoaderRegistryTests(unittest.TestCase):
    def test_register_returns_self_and_routes(self):
        registry = DocumentLoaderRegistry()
        loader = TxtLoader()
        self.assertIs(registry.register(loader), registry)
        self.assertIs(registry.get_loader("a.txt"), loader)
        self.assertIsNone(registry.get_loader("a.unknown"))

    def test_load_file_returns_empty_without_loader(self):
        self.assertEqual(DocumentLoaderRegistry().load_file("a.unknown", "."), [])

    def test_load_file_swallows_loader_errors(self):
        class BoomLoader(BaseDocumentLoader):
            def can_handle(self, filepath: str) -> bool:
                return True

            def load(self, filepath: str, project_dir: str):
                raise RuntimeError("boom")

        registry = DocumentLoaderRegistry().register(BoomLoader())
        self.assertEqual(registry.load_file("a.txt", "."), [])

    def test_create_default_registers_standard_loaders(self):
        registry = DocumentLoaderRegistry.create_default()
        self.assertIsInstance(registry.get_loader("a.txt"), TxtLoader)
        self.assertIsInstance(registry.get_loader("a.md"), MarkdownLoader)
        self.assertIsInstance(registry.get_loader("a.py"), CppHeaderLoader)
        self.assertIsInstance(registry.get_loader("a.png"), ImageLoader)

    def test_optional_loader_extension_routing(self):
        self.assertTrue(DocxLoader().can_handle("a.docx"))
        self.assertTrue(PdfLoader().can_handle("a.pdf"))
        self.assertTrue(EmailLoader().can_handle("a.eml"))


class ReadTextFallbackTests(unittest.TestCase):
    def test_read_text_file_returns_none_when_every_encoding_fails(self):
        # Every attempt can raise before decoding (e.g. an embedded NUL byte in
        # the path surfaces as ValueError); the contract is then ``None``.
        loader = TxtLoader()
        failure = UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
        with mock.patch("builtins.open", side_effect=failure):
            self.assertIsNone(loader._read_text_file("unreadable.txt"))

    def test_unknown_encoding_falls_through_to_latin_1(self):
        """Bytes that break utf-8/utf-16/cp1252 are still readable via latin-1."""
        loader = TxtLoader()
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "odd.bin")
            Path(fp).write_bytes(b"\x81\xfe\xff\x41")
            content = loader._read_text_file(fp)
        self.assertIsNotNone(content)
        self.assertIn("\x81", content)


class DocxLoaderTests(unittest.TestCase):
    def test_load_joins_paragraph_text(self):
        fake_doc = SimpleNamespace(paragraphs=[
            SimpleNamespace(text="First paragraph content with words. " * 10),
            SimpleNamespace(text="   "),
            SimpleNamespace(text="Second paragraph content with words. " * 10),
        ])
        fake_docx = SimpleNamespace(Document=lambda fp: fake_doc)
        with mock.patch.dict(sys.modules, {"docx": fake_docx}):
            loader = DocxLoader(chunk_chars=100, min_chunk_chars=20)
            chunks = loader.load("document.docx", ".")
        self.assertTrue(chunks)
        self.assertTrue(chunks[0].text.startswith("[Tai lieu: "))
        self.assertEqual(chunks[0].loader_type, "tai_lieu")


def _fake_pdf_modules(pdfplumber_text=None, pdfplumber_raises=False, pdfminer_text="", pdfminer_raises=False):
    class _Page:
        def extract_text(self, **kwargs):
            return pdfplumber_text

    class _Pdf:
        pages = [_Page(), _Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    if pdfplumber_raises:
        pdfplumber = SimpleNamespace(open=mock.Mock(side_effect=RuntimeError("boom")))
    else:
        pdfplumber = SimpleNamespace(open=lambda fp: _Pdf())

    def _extract(fp):
        if pdfminer_raises:
            raise RuntimeError("pdfminer boom")
        return pdfminer_text

    high_level = SimpleNamespace(extract_text=_extract)
    pdfminer = SimpleNamespace(high_level=high_level)
    return {
        "pdfplumber": pdfplumber,
        "pdfminer": pdfminer,
        "pdfminer.high_level": high_level,
    }


class PdfLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = PdfLoader(chunk_chars=200, min_chunk_chars=20)

    def _load(self, tmp, **kwargs):
        fp = os.path.join(tmp, "report.pdf")
        Path(fp).write_bytes(b"%PDF-1.4")
        fake_modules = _fake_pdf_modules(**kwargs)
        with mock.patch.dict(sys.modules, fake_modules):
            return self.loader.load(fp, tmp)

    def test_uses_pdfplumber_when_it_returns_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks = self._load(tmp, pdfplumber_text="Paragraph sentence text. " * 40)
        self.assertTrue(chunks)
        self.assertTrue(chunks[0].text.startswith("[Tai lieu PDF: "))

    def test_pdfplumber_short_text_falls_through_to_pdfminer(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks = self._load(
                tmp, pdfplumber_text="short.", pdfminer_text="Fully readable pdfminer body. " * 40)
        self.assertTrue(chunks)

    def test_pdfplumber_failure_falls_back_to_pdfminer(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks = self._load(
                tmp, pdfplumber_raises=True, pdfminer_text="Fallback pdfminer body text. " * 40)
        self.assertTrue(chunks)

    def test_unreadable_pdf_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._load(tmp, pdfplumber_raises=True, pdfminer_text="")

    def test_pdfminer_failure_logs_and_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._load(tmp, pdfplumber_raises=True, pdfminer_raises=True)


class EmailLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = EmailLoader(chunk_chars=200, min_chunk_chars=20)

    def _write_eml(self, tmp, body, subject="Re: Chat", extra_parts=True, broken_charset=False):
        parts = []
        if broken_charset:
            parts.append(
                '--b\r\nContent-Type: text/plain; charset="utf-99"\r\n\r\n'
                "unknowable body\r\n"  # unknown charset → LookupError on get_content()
            )
        if extra_parts:
            parts.append(
                "--b\r\nContent-Type: text/plain\r\n\r\n" + body + "\r\n"
            )
        parts.append(
            "--b\r\nContent-Type: application/octet-stream\r\n"
            'Content-Disposition: attachment; filename="data.bin"\r\n\r\n'
            "binary\x00\r\n"
        )
        content = (
            "From: a@example.com\r\n"
            "To: b@example.com\r\n"
            f"Subject: {subject}\r\n"
            'Content-Type: multipart/mixed; boundary="b"\r\n'
            "\r\n" + "".join(parts) + "--b--\r\n"
        )
        fp = os.path.join(tmp, "mail.eml")
        Path(fp).write_bytes(content.encode("latin-1"))
        return fp

    def test_load_extracts_subject_and_text_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = self._write_eml(tmp, "Plain text body with plenty of words. " * 30)
            chunks = self.loader.load(fp, tmp)
        self.assertTrue(chunks)
        self.assertTrue(chunks[0].text.startswith("[Email: "))
        self.assertIn("Subject: Re: Chat", chunks[0].text)

    def test_load_tolerates_undecodable_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = self._write_eml(tmp, "Good readable body with words. " * 20, broken_charset=True)
            chunks = self.loader.load(fp, tmp)
        self.assertTrue(chunks)

    def test_empty_email_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = self._write_eml(tmp, "tiny", extra_parts=False)
            with self.assertRaises(ValueError):
                self.loader.load(fp, tmp)


class MarkdownLongSectionTests(unittest.TestCase):
    def test_long_section_slides_into_multiple_hierarchical_chunks(self):
        loader = MarkdownLoader(chunk_chars=60, min_chunk_chars=20)
        body = "A reasonably long paragraph with plenty of sentence text here. " * 20
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "long.md")
            Path(fp).write_text(f"# Big Section\n\n{body}\n", encoding="utf-8")
            chunks = loader.load(fp, tmp)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["heading"], "Big Section")
            self.assertEqual(chunk.loader_type, "md")
            self.assertTrue(chunk.text.startswith("[Source MD: "))


class CppEdgeCasesTests(unittest.TestCase):
    _no_body = "int add(int a, int b);\nint sub(int a, int b);\n"

    def test_cpp_without_matching_functions_uses_sliding_window(self):
        loader = CppHeaderLoader(chunk_chars=200, min_chunk_chars=20)
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "prose.cpp")
            Path(fp).write_text(self._no_body * 5, encoding="utf-8")
            chunks = loader.load(fp, tmp)
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].loader_type, "source")

    def test_cpp_short_block_is_skipped_but_next_chunked(self):
        loader = CppHeaderLoader(min_chunk_chars=20)
        content = (
            "void tiny() { }\n"
            "void full() {\n"
            "    int x = 1;\n"
            "    return x;\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "mix.cpp")
            Path(fp).write_text(content, encoding="utf-8")
            chunks = loader.load(fp, tmp)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].metadata["symbol"].startswith("void full"))

    def test_python_comment_only_blocks_are_skipped(self):
        loader = CppHeaderLoader()  # default min_chunk_chars=80
        content = (
            "def stub():\n"
            "    return 42\n"
            "\n"
            "\n"
            "class Klass:\n"
            "    def real_method(self):\n"
            "        items = ['alpha', 'beta', 'gamma']\n"
            "        return ' '.join(items) * 40\n"
            "\n"
            "    def ghost_method(self):\n"
            "        pass\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "mod.py")
            Path(fp).write_text(content, encoding="utf-8")
            chunks = loader.load(fp, tmp)
        symbols = {c.metadata.get("symbol") for c in chunks}
        scopes = {c.metadata.get("scope") for c in chunks}
        self.assertIn("Klass", symbols)
        self.assertIn("real_method", symbols)
        self.assertNotIn("stub", symbols)
        self.assertNotIn("ghost_method", symbols)
        self.assertIn("Class", scopes)
        self.assertIn("Method", scopes)


class ImageLoaderErrorTests(unittest.TestCase):
    def test_load_swallows_path_errors(self):
        with mock.patch("ai_assistant.rag.loaders.safe_relpath",
                        side_effect=ValueError("boom")):
            self.assertEqual(ImageLoader().load("a.png", "."), [])


if __name__ == "__main__":
    unittest.main()
