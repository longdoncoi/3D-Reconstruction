"""Unit tests for ``ai_assistant.rag.loaders``.

Only the offline loaders are exercised end-to-end (text, markdown, source and
image); the docx/pdf/email adapters are only checked for extension routing so
the suite never imports python-docx, pdfplumber or email parsers.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

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
    def setUp(self):
        self.loader = CppHeaderLoader(min_chunk_chars=10)

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


if __name__ == "__main__":
    unittest.main()
