"""The documented architecture, turned into a contract CI can break.

The README draws a one-way dependency graph and states three structural rules
— the orchestrator never names a concrete backend, ``ultralytics`` is reached
through exactly one adapter, and every module owns one reason to change. Those
claims used to be prose: nothing stopped the next refactor from adding the
import that turns ``cvtrain`` into a cycle or quietly gives ``trainer`` a
second job. This module is the machine-checked half of that document.

Everything here reads the *source* (``ast``), never the running package, so a
violation is reported as a fact about a file instead of as a mysterious
failure somewhere else. Adding a module or a new import is meant to be a
conscious change: the tables below have to be edited in the same commit.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from cvtrain.cli import build_parser

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "cvtrain"
README = Path(__file__).resolve().parents[1] / "README.md"
ENTRY_POINT = Path(__file__).resolve().parents[1] / "TrainModel.py"

MODULES: tuple[str, ...] = tuple(sorted(path.stem for path in PACKAGE_DIR.glob("*.py")))
"""Every module in ``cvtrain``; the tables below must name all of them."""

ALLOWED_IMPORTS: dict[str, frozenset[str]] = {
    "__init__": frozenset(),
    "backend": frozenset({"config", "errors"}),
    "checks": frozenset(),
    "cli": frozenset({"backend", "config", "errors", "fsio", "logging_setup", "manifest", "trainer"}),
    "config": frozenset(),
    "dataset": frozenset({"checks"}),
    "errors": frozenset(),
    "fsio": frozenset({"errors"}),
    "gate": frozenset({"config", "errors", "fsio", "manifest"}),
    "logging_setup": frozenset(),
    "manifest": frozenset({"config", "errors", "fsio"}),
    "preflight": frozenset({"checks", "config", "dataset"}),
    "progress": frozenset(),
    "trainer": frozenset(
        {"backend", "config", "errors", "fsio", "gate", "manifest", "preflight", "progress"}
    ),
}
"""The dependency graph of the package: module -> the modules it may import.

One direction only, and every edge is named here rather than inferred: a new
edge is a change to this table, reviewed like any other. The shape encodes the
README diagram — leaves (``config``, ``checks``, ``errors``, ``progress``,
``logging_setup``) know nothing, ``fsio`` sits on ``errors`` (it raises a typed
``ConcurrentRunError`` from the advisory lock), ``manifest``/``dataset``/
``backend`` sit on them, ``gate``/``preflight`` hold the policies, ``trainer``
orchestrates, and ``cli`` alone is the composition root.
"""

HEAVY_DEPENDENCIES: dict[str, frozenset[str]] = {
    "ultralytics": frozenset({"backend"}),
    "tensorboard": frozenset({"backend"}),
    "yaml": frozenset({"dataset"}),
}
"""Third-party modules that may only be imported by the listed files.

The ML stack (and even PyYAML) is deliberately unreachable from the rest of
the package so that ``--help``, ``--check``, ``--dry-run`` and the whole unit
suite run without it installed. ``ultralytics`` belongs to the one adapter;
``yaml`` belongs to the one module that parses ``data.yaml``; ``tensorboard``
is the optional sink behind ``--tensorboard``, probed by the same adapter so
that feature fails fast rather than logging a success it did not have.
"""


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _internal_imports(tree: ast.Module) -> set[str]:
    """Names of ``cvtrain`` modules this file imports (``from .x import ...``)."""

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module in MODULES:
                found.add(node.module)
            for alias in node.names:
                # ``from . import __version__, trainer`` mixes a package
                # attribute with a submodule: only the submodule is an edge.
                if alias.name in MODULES:
                    found.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.partition(".")[0]
                if top_level in MODULES:
                    found.add(top_level)
    return found


def _heavy_imports(tree: ast.Module) -> dict[str, list[bool]]:
    """``{heavy module: [was each import lazy, ...]}`` for one file.

    An import is *lazy* when it sits inside a function body: that is what keeps
    the heavy stack out of ``--help``/``--check`` and out of the unit suite.
    """

    found: dict[str, list[bool]] = {}

    def visit(node: ast.AST, inside_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            nested = inside_function or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            if isinstance(child, ast.Import):
                names = [alias.name.partition(".")[0] for alias in child.names]
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                names = [child.module.partition(".")[0]]
            else:
                names = []
            for name in names:
                if name in HEAVY_DEPENDENCIES:
                    found.setdefault(name, []).append(nested)
            visit(child, nested)

    visit(tree, False)
    return found


class ImportGraphTests(unittest.TestCase):
    """The README's dependency diagram, enforced edge by edge."""

    def test_every_module_declares_its_dependencies(self) -> None:
        self.assertEqual(set(ALLOWED_IMPORTS), set(MODULES), "ALLOWED_IMPORTS is out of date")

    def test_no_module_imports_outside_its_neighbours(self) -> None:
        for name in MODULES:
            with self.subTest(module=name):
                tree = _tree(PACKAGE_DIR / f"{name}.py")
                unexpected = _internal_imports(tree) - ALLOWED_IMPORTS[name]
                self.assertFalse(unexpected, f"{name}.py imports {sorted(unexpected)}, which the map forbids")

    def test_the_declared_graph_has_no_cycle(self) -> None:
        # Edges point one way, so nothing can ever depend on itself. Cycles
        # are what turn "leaf modules" into import-order roulette.
        visiting: set[str] = set()
        visited: set[str] = set()

        def walk(name: str) -> None:
            self.assertNotIn(name, visiting, f"cycle through {name}")
            if name in visited:
                return
            visiting.add(name)
            for neighbour in ALLOWED_IMPORTS.get(name, frozenset()):
                walk(neighbour)
            visiting.discard(name)
            visited.add(name)

        for name in MODULES:
            walk(name)

    def test_the_concrete_backend_is_named_only_by_the_composition_root(self) -> None:
        # ``trainer`` imports ``backend`` for the *protocol* and the opaque
        # handle types, but naming the adapter — constructing it, choosing it —
        # is ``cli``'s alone. A ``Name`` node is exactly that: a mention of the
        # class, not the module that defines it.
        for name in MODULES:
            if name == "cli":
                continue
            with self.subTest(module=name):
                tree = _tree(PACKAGE_DIR / f"{name}.py")
                mentions = {
                    node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "UltralyticsBackend"
                }
                self.assertFalse(mentions, f"{name}.py names the concrete backend; only cli.py may")


class LazyDependencyTests(unittest.TestCase):
    """The heavy stack is reachable through one seam, and only when training."""

    def test_heavy_dependencies_are_imported_only_by_their_owner(self) -> None:
        for name in MODULES:
            tree = _tree(PACKAGE_DIR / f"{name}.py")
            for heavy, importers in _heavy_imports(tree).items():
                with self.subTest(module=name, dependency=heavy):
                    self.assertIn(
                        name,
                        HEAVY_DEPENDENCIES[heavy],
                        f"{name}.py imports {heavy}; only {sorted(HEAVY_DEPENDENCIES[heavy])} may",
                    )

    def test_every_heavy_import_is_lazy(self) -> None:
        for name in MODULES:
            tree = _tree(PACKAGE_DIR / f"{name}.py")
            for heavy, lazy_flags in _heavy_imports(tree).items():
                with self.subTest(module=name, dependency=heavy):
                    self.assertTrue(
                        all(lazy_flags),
                        f"{name}.py imports {heavy} at module level; import it inside the "
                        "function that needs it so --help/--check work without it",
                    )


class DocumentationTests(unittest.TestCase):
    """Every module explains itself; the README layout mirrors the package."""

    _TOOLING_FLAGS = frozenset({"--help", "--version", "-h", "-v"})
    """Argparse-level switches that are not part of the documented option list.

    ``--help``/``-h`` and ``--version`` are parser conventions rather than
    training options, and ``-v`` is the short alias of the documented
    ``--verbose``.
    """

    def test_every_module_has_a_docstring(self) -> None:
        for name in MODULES:
            with self.subTest(module=name):
                self.assertTrue(ast.get_docstring(_tree(PACKAGE_DIR / f"{name}.py")), f"{name}.py has no docstring")
        self.assertTrue(ast.get_docstring(_tree(ENTRY_POINT)), "TrainModel.py has no docstring")

    def test_the_readme_layout_lists_exactly_the_package_modules(self) -> None:
        text = README.read_text(encoding="utf-8")
        layout = text.split("## Layout", 1)[1].split("```", 2)[1]
        documented = set(re.findall(r"\b[\w-]+\.py\b", layout))
        actual = {f"{name}.py" for name in MODULES} | {ENTRY_POINT.name}
        self.assertEqual(documented, actual, "README Layout section is out of date")

    def test_every_flag_the_readme_shows_is_accepted_by_the_cli(self) -> None:
        # A renamed flag would otherwise ship as documentation for a command
        # that rejects it — the same drift the Qt contract test guards from
        # the other side.
        text = README.read_text(encoding="utf-8")
        documented = set(re.findall(r"--[a-z][a-z0-9-]*", text))
        accepted = set(build_parser()._option_string_actions)
        self.assertEqual(documented - accepted, set(), "README documents flags the CLI does not accept")

    def test_every_flag_the_cli_accepts_is_documented(self) -> None:
        # The reverse direction: a flag the parser accepts but the README never
        # mentions is an undocumented feature. ``--help``/``--version`` are
        # tooling switches rather than run options, so they are exempt.
        text = README.read_text(encoding="utf-8")
        documented = set(re.findall(r"--[a-z][a-z0-9-]*", text))
        accepted = set(build_parser()._option_string_actions)
        self.assertEqual(
            accepted - documented - self._TOOLING_FLAGS,
            set(),
            "CLI accepts flags the README never mentions",
        )


if __name__ == "__main__":
    unittest.main()
