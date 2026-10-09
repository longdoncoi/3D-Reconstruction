"""Architecture boundary guards for the modular AI Assistant platform.

These tests fail when a lower layer starts importing an outer layer or a
framework, which is how Clean Architecture boundaries rot over time. They are
pure AST scans (no application code is imported), so they are fast and safe to
run in CI. See ``Docs/adr/0001-platform-boundaries.md``,
`0002-mcp-a2a-trust-and-execution.md` and `0003-single-execution-core.md`.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE = SRC / "ai_assistant"

FRAMEWORKS = frozenset(
    {
        "fastapi",
        "starlette",
        "pydantic",
        "langgraph",
        "langchain",
        "langchain_core",
        "langsmith",
        "llama_cpp",
        "mcp",
        "a2a",
        "httpx",
        "uvicorn",
    }
)

# Top-level ai_assistant subpackages the application layer may import.
APPLICATION_ALLOWED = {"application", "domain", "plugins", "ports"}


def _iter_python(layer: str | None = None):
    base = PACKAGE if layer is None else PACKAGE / layer
    yield from sorted(base.rglob("*.py"))


def _imports(path: Path) -> list[str]:
    """Return the absolute dotted module name for every import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parts = path.relative_to(PACKAGE).with_suffix("").parts
    # Works for both ``pkg/module.py`` and ``pkg/__init__.py``.
    package_parts = parts[:-1]
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                trim = (
                    package_parts[: len(package_parts) - (node.level - 1)]
                    if node.level > 1
                    else package_parts
                )
                base = "ai_assistant" + ("." + ".".join(trim) if trim else "")
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if module:
                modules.append(module)
    return modules


def _attribute_calls(path: Path, attribute: str) -> list[int]:
    """Return line numbers of ``something.attribute(...)`` calls in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attribute
        ):
            lines.append(node.lineno)
    return lines


def _legacy(module: str) -> bool:
    return module == "modules" or module.startswith("modules.")


def _format(violations: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"{path}: {module} ({reason})" for path, module, reason in violations)


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_domain_is_pure(self):
        """Domain may import only the standard library and other domain code."""
        violations: list[tuple[str, str, str]] = []
        for path in _iter_python("domain"):
            for module in _imports(path):
                if module == "modules" or module.startswith("modules."):
                    reason = "imports top-level legacy package"
                elif module.split(".")[0] in FRAMEWORKS:
                    reason = "imports forbidden framework"
                elif module.startswith("ai_assistant"):
                    if module.startswith("ai_assistant.domain"):
                        continue
                    reason = "imports outer layer"
                elif module.split(".")[0] in sys.stdlib_module_names:
                    continue
                else:
                    reason = "imports non-stdlib module"
                violations.append((str(path.relative_to(PACKAGE)), module, reason))
        self.assertEqual(violations, [], _format(violations))

    def test_application_does_not_depend_on_adapters_or_frameworks(self):
        """Application depends inward only (domain/ports/plugins)."""
        violations: list[tuple[str, str, str]] = []
        for path in _iter_python("application"):
            for module in _imports(path):
                top_level = module.split(".")[0]
                if module == "modules" or module.startswith("modules."):
                    reason = "imports top-level legacy package"
                elif module.split(".")[0] in FRAMEWORKS:
                    reason = "imports forbidden framework"
                elif module.startswith("ai_assistant"):
                    sub = module.split(".")[1] if "." in module else ""
                    if sub in APPLICATION_ALLOWED:
                        continue
                    reason = f"imports outer layer 'ai_assistant.{sub}'"
                elif top_level in sys.stdlib_module_names:
                    continue
                else:
                    reason = "imports non-stdlib module"
                violations.append((str(path.relative_to(PACKAGE)), module, reason))
        self.assertEqual(violations, [], _format(violations))

    def test_no_new_code_imports_the_legacy_modules_package(self):
        violations: list[tuple[str, str, str]] = []
        for path in _iter_python():
            for module in _imports(path):
                if module == "modules" or module.startswith("modules."):
                    violations.append(
                        (str(path.relative_to(PACKAGE)), module, "imports top-level legacy package")
                    )
        self.assertEqual(violations, [], _format(violations))

    def test_agents_and_orchestration_do_not_import_the_http_adapter(self):
        """Guards the fix for the service -> HTTP adapter coupling."""
        violations: list[tuple[str, str, str]] = []
        for layer in ("agents", "orchestration"):
            for path in _iter_python(layer):
                for module in _imports(path):
                    if module.startswith("ai_assistant.adapters.http"):
                        violations.append(
                            (str(path.relative_to(PACKAGE)), module, "imports HTTP adapter")
                        )
        self.assertEqual(violations, [], _format(violations))

    def test_tool_execution_has_a_single_choke_point(self):
        """ADR 0003: nothing may invoke ``ToolSpec.handler`` directly."""
        violations: list[tuple[str, str, str]] = []
        for path in _iter_python():
            for line in _attribute_calls(path, "handler"):
                violations.append(
                    (str(path.relative_to(PACKAGE)), f"line {line}", "direct handler() call")
                )
        self.assertEqual(violations, [], _format(violations))

    def test_tool_models_are_not_duplicated(self):
        """ADR 0003: exactly one type named ``ToolSpec`` — the domain policy model."""
        definitions: list[str] = []
        for path in _iter_python():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == "ToolSpec":
                    definitions.append(path.relative_to(PACKAGE).as_posix())
        self.assertEqual(definitions, ["domain/tools.py"], definitions)

    def test_no_code_imports_the_removed_core_package(self):
        """The old ``ai_assistant.core`` tool models were unified into ``tools``."""
        violations: list[tuple[str, str, str]] = []
        for path in _iter_python():
            for module in _imports(path):
                if module == "ai_assistant.core" or module.startswith("ai_assistant.core."):
                    violations.append(
                        (str(path.relative_to(PACKAGE)), module, "imports removed core package")
                    )
        self.assertEqual(violations, [], _format(violations))

    def test_agent_engines_use_the_shared_execution_gateway(self):
        """Both orchestration strategies execute tools via one gateway (ADR 0003)."""
        runner = (PACKAGE / "agents" / "runner.py").read_text(encoding="utf-8")
        agent_runs = (PACKAGE / "application" / "agent_runs.py").read_text(encoding="utf-8")
        deterministic = (PACKAGE / "adapters" / "orchestration" / "deterministic.py").read_text(encoding="utf-8")
        registry = (PACKAGE / "tools" / "registry.py").read_text(encoding="utf-8")

        self.assertIn("execute_tool", runner)
        self.assertIn("execute_approved_tool", runner)
        self.assertIn("ToolExecutionService", (PACKAGE / "application" / "tools.py").read_text(encoding="utf-8"))
        # The only in-code decision loop lives in an adapter; application holds none.
        self.assertIn("self._tools.execute", deterministic)
        self.assertNotIn("_run_deterministic", agent_runs)
        self.assertNotIn("def execute(", registry, "ToolRegistry.execute is a bypass")

    def test_a2a_reuses_the_canonical_langgraph_engine_through_a_port(self):
        """ADR 0003: the A2A lifecycle delegates to the shared engine via a port."""
        agent_runs = (PACKAGE / "application" / "agent_runs.py").read_text(encoding="utf-8")
        self.assertIn("AgentOrchestrator", agent_runs)
        self.assertIn("self._orchestrator.run", agent_runs)

        root = (SRC.parent / "StartChatbotServer.py").read_text(encoding="utf-8")
        self.assertIn("LangGraphAgentOrchestrator", root)
        self.assertIn("orchestrator=LangGraphAgentOrchestrator", root)
        self.assertNotIn("LegacyConstrainedCompletion", root)


if __name__ == "__main__":
    unittest.main()
