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


def _call_targets(path: Path) -> set[str]:
    """Dotted targets of the calls in ``path`` (``obj.attr``/``name``).

    Structural replacement for source substring matching: ``"gateway.execute"``
    no longer accidentally matches ``gateway.execute_approved``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}


def _defined_names(path: Path) -> set[str]:
    """Names of every class/function/method defined in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _identifiers(path: Path) -> set[str]:
    """Every bare name and attribute name referenced in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _keyword_values(path: Path) -> dict[str, set[str]]:
    """Keyword-argument name -> set of unparsed values (call targets unwrapped).

    Lets a test assert ``orchestrator=LangGraphAgentOrchestrator`` structurally
    instead of grepping for that exact source text.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if not keyword.arg:
                continue
            value = keyword.value
            rendered = ast.unparse(value.func if isinstance(value, ast.Call) else value)
            values.setdefault(keyword.arg, set()).add(rendered)
    return values


def _format(violations: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"{path}: {module} ({reason})" for path, module, reason in violations)


def _is_env_read(node: ast.AST) -> bool:
    """True for ``os.getenv(...)`` and ``os.environ[...]`` expressions."""
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "getenv"
        )
    if isinstance(node, ast.Subscript):
        return (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
        )
    return False


class _ImportTimeEnvReader(ast.NodeVisitor):
    """Detect environment reads that execute when a module is imported.

    Covers module/class scope, function default arguments and decorators, and
    lambda bodies. Function *bodies* are skipped on purpose: reading the
    environment on demand inside a function is the approved pattern.
    """

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def _scan(self, nodes: list[ast.AST | None]) -> None:
        for node in nodes:
            if node is None:
                continue
            for nested in ast.walk(node):
                if _is_env_read(nested):
                    self.hits.append((nested.lineno, ast.unparse(nested)))

    def _visit_callable_defaults(self, node: ast.AST) -> None:
        args = getattr(node, "args", None)  # type: ignore[attr-defined]
        if args is not None:
            self._scan(list(args.defaults) + list(args.kw_defaults))
        for decorator in getattr(node, "decorator_list", []):
            self._scan([decorator])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable_defaults(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_callable_defaults(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._scan(list(node.args.defaults) + list(node.args.kw_defaults))

    def visit_Call(self, node: ast.Call) -> None:
        if _is_env_read(node):
            self.hits.append((node.lineno, ast.unparse(node)[:80]))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _is_env_read(node):
            self.hits.append((node.lineno, ast.unparse(node)[:80]))
        self.generic_visit(node)


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

    def test_agents_and_application_do_not_import_the_legacy_package(self):
        """ADR 0001: legacy shims are adapters; agent/application code stays clean."""
        violations: list[tuple[str, str, str]] = []
        for layer in ("agents", "application"):
            for path in _iter_python(layer):
                for module in _imports(path):
                    if module == "ai_assistant.legacy" or module.startswith("ai_assistant.legacy."):
                        violations.append(
                            (str(path.relative_to(PACKAGE)), module, "imports ai_assistant.legacy")
                        )
        self.assertEqual(violations, [], _format(violations))

    def test_no_framework_free_layer_imports_the_compatibility_adapter(self):
        """The ``legacy_agent`` adapter is for adapters and the composition root only."""
        violations: list[tuple[str, str, str]] = []
        for layer in ("agents", "application", "domain", "orchestration", "ports", "tools", "llm"):
            for path in _iter_python(layer):
                for module in _imports(path):
                    if module == "ai_assistant.adapters.legacy_agent":
                        violations.append(
                            (str(path.relative_to(PACKAGE)), module, "imports compatibility adapter")
                        )
        self.assertEqual(violations, [], _format(violations))

    def test_no_import_time_environment_reads_outside_settings_and_bootstrap(self):
        """ADR 0001: environment reads happen only in settings/bootstrap.

        ``os.getenv``/``os.environ[key]`` at module/class scope or in function
        defaults run at import time and are forbidden outside the two approved
        owners (``legacy`` remains a compatibility shim). Reads inside function
        bodies are allowed — that is the on-demand pattern.
        """
        excluded = {"settings.py"}
        violations: list[str] = []
        for path in _iter_python():
            rel = path.relative_to(PACKAGE).as_posix()
            if rel in excluded or path.parts[-2] in {"bootstrap", "legacy"}:
                continue
            reader = _ImportTimeEnvReader()
            reader.visit(ast.parse(path.read_text(encoding="utf-8")))
            for lineno, snippet in reader.hits:
                violations.append(f"{rel}:{lineno}: {snippet}")
        self.assertEqual(violations, [], "\n".join(violations))

    def test_agent_engines_use_the_shared_execution_gateway(self):
        """Both orchestration strategies execute tools via one gateway (ADR 0003)."""
        runner_calls = _call_targets(PACKAGE / "agents" / "runner.py")
        deterministic_calls = _call_targets(PACKAGE / "adapters" / "orchestration" / "deterministic.py")
        agent_runs = (PACKAGE / "application" / "agent_runs.py").read_text(encoding="utf-8")

        self.assertIn("tool_gateway.execute", runner_calls)
        self.assertIn("tool_gateway.execute_approved", runner_calls)
        self.assertIn("tool_gateway.approval_covers", runner_calls)
        self.assertIsNotNone(
            _defined_names(PACKAGE / "application" / "tools.py") & {"ToolExecutionService"}
        )
        # The only in-code decision loop lives in an adapter; application holds none.
        self.assertIn("self._tools.execute", deterministic_calls)
        self.assertIn("self._tools.issue_approval_grant", deterministic_calls)
        self.assertNotIn("_run_deterministic", agent_runs)
        # ToolRegistry.execute is a bypass: the registry defines no execute method.
        self.assertNotIn("execute", _defined_names(PACKAGE / "tools" / "registry.py"))

    def test_agents_do_not_import_the_composition_root(self):
        """The tool gateway is injected; agent code never reaches bootstrap (ADR 0003)."""
        violations: list[str] = []
        for path in _iter_python("agents"):
            for module in _imports(path):
                if module == "ai_assistant.bootstrap" or module.startswith("ai_assistant.bootstrap."):
                    violations.append(str(path.relative_to(PACKAGE)))
        self.assertEqual(violations, [], "\n".join(violations))

    def test_no_global_tool_service_locator_remains(self):
        """The mutable ``_tool_service`` locator was replaced by injected gateways."""
        offenders = [
            str(path.relative_to(PACKAGE))
            for path in _iter_python()
            if "configure_tool_service" in _identifiers(path)
        ]
        self.assertEqual(offenders, [])

    def test_a2a_reuses_the_canonical_langgraph_engine_through_a_port(self):
        """ADR 0003: the A2A lifecycle delegates to the shared engine via a port."""
        agent_runs_path = PACKAGE / "application" / "agent_runs.py"
        self.assertIn("AgentOrchestrator", _identifiers(agent_runs_path))
        self.assertIn("self._orchestrator.run", _call_targets(agent_runs_path))

        root_path = SRC.parent / "StartChatbotServer.py"
        root_kwargs = _keyword_values(root_path)
        self.assertIn("LangGraphAgentOrchestrator", root_kwargs.get("orchestrator", set()))
        self.assertNotIn("LegacyConstrainedCompletion", root_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
