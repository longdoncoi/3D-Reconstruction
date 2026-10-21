"""Static guard: one rule for resolving a program, and one way to break it.

The platform resolves a program in three places — the command sandbox, the
git tools and the LSP client — and each of them used to do it its own way.
The sandbox resolved it correctly; the other two did not, and one of them
threw the verified absolute hit away with a fallback that handed the search
back to the operating system (which searches the *current directory* first
on Windows).

This guard is the reason the third instance cannot appear: **no module in
the package may resolve a program with the platform lookup**. It is a static
check because the defect it prevents is invisible at runtime — the mock in
the old test returned a *relative* name, the very shape a current-directory
hit produces, so the suite stayed green while the control did not work.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "ai_assistant"

#: The module that owns the rule. Everything else resolves through it.
OWNER = "ai_assistant/exec_policy.py"


def _modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _calls_to(tree: ast.AST, dotted: str) -> list[ast.Call]:
    """Every call to ``dotted`` in the tree (e.g. ``shutil.which``)."""
    parts = dotted.split(".")
    found: list[ast.Call] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == parts[-1]:
                owner = func.value
                if isinstance(owner, ast.Name) and owner.id == parts[-2]:
                    found.append(node)
                elif isinstance(owner, ast.Attribute) and owner.attr == parts[-3:-2]:
                    found.append(node)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return found


class ExecutableResolutionGuardTests(unittest.TestCase):
    def test_the_package_has_modules_to_check(self) -> None:
        self.assertGreater(len(_modules()), 50)

    def test_no_module_resolves_a_program_with_the_platform_lookup(self) -> None:
        """``shutil.which`` prepends the current directory on Windows.

        That is the hijack: a binary planted next to the process wins over
        the operator's own tool, and the value it returns is *relative*, so
        the "not on PATH" refusal cannot fire.
        """
        offenders: list[str] = []
        for module in _modules():
            tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
            for call in _calls_to(tree, "shutil.which"):
                offenders.append(f"{module.relative_to(PACKAGE)}:{call.lineno}")
            # The other spelling of the same call: ``from shutil import which``.
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "shutil":
                    for alias in node.names:
                        if alias.name == "which":
                            offenders.append(f"{module.relative_to(PACKAGE)}:{node.lineno}")
        self.assertEqual(
            offenders, [],
            "shutil.which resolves the current directory first on Windows — "
            "use ai_assistant.exec_policy.resolve_executable instead",
        )

    def test_the_owner_of_the_rule_exists_and_exports_the_resolver(self) -> None:
        owner = PACKAGE.parent / OWNER
        self.assertTrue(owner.exists(), f"{OWNER} is the owner of the rule")
        source = owner.read_text(encoding="utf-8")
        self.assertIn("def resolve_executable(", source)
        self.assertIn("def executable_candidates(", source)

    def test_the_three_callers_resolve_through_the_owner(self) -> None:
        """Every place that runs a program must go through the owner.

        The three are the command sandbox, the git tools and the LSP client.
        A fourth one would be a fourth copy of the rule.
        """
        callers = {
            "tools/sandbox.py": "resolve_executable",
            "tools/builtin/vcs_tools.py": "resolve_executable",
            "tools/lsp_client.py": "resolve_tool_binary",
        }
        for relative, symbol in callers.items():
            with self.subTest(module=relative):
                source = (PACKAGE / relative).read_text(encoding="utf-8")
                self.assertIn(symbol, source)
                # The AST walk above is the real check; a mention in a
                # docstring (which explains *why* the platform lookup is
                # forbidden) is not a call.
                tree = ast.parse(source)
                self.assertEqual(_calls_to(tree, "shutil.which"), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
