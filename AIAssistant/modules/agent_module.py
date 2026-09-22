# ruff: noqa: I001
import ast
import fnmatch
import queue
import re
import subprocess
from collections.abc import Callable

from .config import *
from .config import _safe_relpath
from . import llm_module as llm_runtime
from . import rag_module as rag_runtime
from .action_manifest import (
    action_catalog,
    canonicalise_action_params,
    validate_action_params,
)
from .tool_contract import (
    build_tool_models,
    enrich_tool_definitions,
    grammar_schema,
    openai_tools,
    validate_tool_call,
)
from .sandbox import (
    create_directory as create_sandboxed_directory,
    run as run_sandboxed_command,
    write_file as write_sandboxed_file,
)
from .observability import langsmith_trace, record_approval, record_schema_error, record_tool, span
from .inference import backend_mode, openai_compatible_completion, strip_think_tags
from ai_assistant.bootstrap.runtime import (
    execute_approved_tool as execute_approved_platform_tool,
    execute_tool as execute_platform_tool,
)
from .coding_agent import CodingTaskContext, instruction as coding_instruction, is_coding_task
from .agent_tools import ToolRegistry
from . import lsp_client as _lsp_client
from .approval_manager import PendingActionStore
from .task_coordinator import coordinator as task_coordinator
from .multi_agent import (
    Specialist,
    audit as audit_agent,
    authorise as authorise_delegation,
    delegate,
    reflect_result,
    specialist_instruction,
    verify_result,
)

try:
    from .a2a_protocol import A2ARouter
except ImportError:  # pragma: no cover - retained for minimal deployments
    A2ARouter = None  # type: ignore[assignment,misc]

try:
    from LangGraphAgent import LocalAgentGraph
    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = ""
except ImportError as error:
    LocalAgentGraph = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = str(error)

# ── Tool Definitions (mô tả cho LLM) ──────────────────────────────────────────
AGENT_TOOLS = [
    {
        "name": "transfer_to_code_agent",
        "description": "Transfer the user request to the Code Agent, which specializes in writing, modifying, debugging, or analyzing software source code.",
        "parameters": {
            "intent": {"type": "string", "description": "Summary of what needs to be coded", "required": True}
        },
    },
    {
        "name": "transfer_to_toolapp_agent",
        "description": "Transfer the user request to the ToolApp Agent, which specializes in executing desktop UI actions (loading images, 3D models, AI detection, etc).",
        "parameters": {
            "intent": {"type": "string", "description": "Summary of the UI actions needed", "required": True}
        },
    },
    {
        "name": "transfer_to_chatbot_agent",
        "description": "Transfer the user request to the Chatbot Agent, which specializes in answering general questions and conversations.",
        "parameters": {
            "intent": {"type": "string", "description": "Summary of the user's question", "required": True}
        },
    },
    {
        "name": "read_file",
        "description": "Read a focused range from a file. Returns file content as text. "
                       "For source files, first use search_text/analyze_code to locate a symbol, "
                       "then provide start_line/end_line (or symbol) instead of reading a large file in full.",
        "parameters": {
            "path": {"type": "string", "description": "Relative path from project root (e.g. 'src/main.cpp')", "required": True},
            "start_line": {"type": "integer", "description": "Start line (1-indexed, inclusive). Required for large source files unless symbol is provided.", "required": False},
            "end_line": {"type": "integer", "description": "End line (1-indexed, inclusive). Required for large source files unless symbol is provided.", "required": False},
            "symbol": {"type": "string", "description": "Optional C/C++/Python function or method name; returns only its source range.", "required": False},
        },
    },
    {
        "name": "list_directory",
        "description": "List files and subdirectories in a directory. "
                       "Returns a structured listing with file sizes and types.",
        "parameters": {
            "path": {"type": "string", "description": "Relative path from project root (e.g. 'src/modules'). Use '.' for project root.", "required": True},
            "recursive": {"type": "boolean", "description": "If true, list recursively. Default false.", "required": False},
            "max_depth": {"type": "integer", "description": "Max depth for recursive listing. Default 3.", "required": False},
        },
    },
    {
        "name": "search_text",
        "description": "Search for text/pattern in project files. Returns matching lines with file paths and line numbers. "
                       "Similar to grep. Use this to find usages, definitions, or occurrences of text.",
        "parameters": {
            "query": {"type": "string", "description": "Text or pattern to search for", "required": True},
            "path": {"type": "string", "description": "Relative path to search in. Default: entire project.", "required": False},
            "file_pattern": {"type": "string", "description": "Glob pattern to filter files, e.g. '*.cpp' or '*.py'", "required": False},
            "case_sensitive": {"type": "boolean", "description": "Case sensitive search. Default true.", "required": False},
            "max_results": {"type": "integer", "description": "Max results to return. Default 50.", "required": False},
        },
    },
    {
        "name": "analyze_code",
        "description": "Analyze the structure of a source code file. Returns classes, functions, imports, "
                       "and a structural summary. Supports Python and C/C++ files.",
        "parameters": {
            "path": {"type": "string", "description": "Relative path to the source file", "required": True},
        },
    },
    {
        "name": "find_files",
        "description": "Find source and project files by glob pattern without reading their contents.",
        "parameters": {
            "pattern": {"type": "string", "description": "Glob pattern such as '*.cpp' or 'test_*.py'", "required": True},
            "path": {"type": "string", "description": "Relative directory to search. Default: project root.", "required": False},
            "max_results": {"type": "integer", "description": "Maximum number of paths to return. Default 100.", "required": False},
        },
    },
    {
        "name": "git_diff",
        "description": "Read the current Git diff for a file or the project without changing state.",
        "parameters": {
            "path": {"type": "string", "description": "Optional relative file or directory to inspect.", "required": False},
            "staged": {"type": "boolean", "description": "Read staged changes instead of working-tree changes.", "required": False},
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. THIS REQUIRES USER APPROVAL before execution. "
                       "Use this to fix bugs, add code, modify configs, etc.",
        "parameters": {
            "path": {"type": "string", "description": "Relative path from project root", "required": True},
            "content": {"type": "string", "description": "Full content to write to the file", "required": True},
            "description": {"type": "string", "description": "Brief description of what this change does", "required": True},
        },
    },
    {
        "name": "run_command",
        "description": "Execute a shell command. THIS REQUIRES USER APPROVAL before execution. "
                       "Use for building, testing, or other project tasks.",
        "parameters": {
            "command": {"type": "string", "description": "The shell command to run", "required": True},
            "cwd": {"type": "string", "description": "Working directory (relative to project root). Default: project root.", "required": False},
            "timeout": {"type": "integer", "description": "Timeout in seconds. Default 30.", "minimum": 1, "maximum": 120, "required": False},
        },
    },
    {
        "name": "get_project_status",
        "description": "Inspect the project safely: current Git branch, changed files and high-level source file counts. Does not change files.",
        "parameters": {},
    },
    {
        "name": "validate_file",
        "description": "Validate a Python or JSON file without modifying it. Python files are syntax-checked; JSON files are parsed.",
        "parameters": {
            "path": {"type": "string", "description": "Relative Python or JSON file path", "required": True},
        },
    },
    {
        "name": "patch_file",
        "description": "Replace an exact text fragment in an existing file. THIS REQUIRES USER APPROVAL. Read the file first and use a unique fragment.",
        "parameters": {
            "path": {"type": "string", "description": "Relative path from project root", "required": True},
            "find": {"type": "string", "description": "Exact existing text to replace", "required": True},
            "replace": {"type": "string", "description": "Replacement text", "required": True},
            "description": {"type": "string", "description": "Brief description of the change", "required": True},
        },
    },
    {
        "name": "replace_file_content",
        "description": "Replace a single contiguous block of code in an existing file. THIS REQUIRES USER APPROVAL. Read the file first and provide EXACT original text.",
        "parameters": {
            "path": {"type": "string", "description": "Relative path from project root", "required": True},
            "targetContent": {"type": "string", "description": "Exact existing text to replace, including whitespaces", "required": True},
            "replacementContent": {"type": "string", "description": "Replacement text", "required": True},
            "description": {"type": "string", "description": "Brief description of the change", "required": True},
        },
    },
    {
        "name": "multi_replace_file_content",
        "description": "Edit an existing file by replacing multiple non-contiguous blocks of code. THIS REQUIRES USER APPROVAL. Read the file first.",
        "parameters": {
            "path": {"type": "string", "description": "Relative path from project root", "required": True},
            "replacements": {
                "type": "array", 
                "items": {
                    "type": "object",
                    "properties": {
                        "targetContent": {"type": "string", "description": "Exact text to replace", "required": True},
                        "replacementContent": {"type": "string", "description": "Replacement text", "required": True}
                    }
                },
                "description": "List of replacements", 
                "required": True
            },
            "description": {"type": "string", "description": "Brief description of the change", "required": True},
        },
    },
    {
        "name": "create_directory",
        "description": "Create a directory inside the project. THIS REQUIRES USER APPROVAL.",
        "parameters": {
            "path": {"type": "string", "description": "Relative directory path", "required": True},
            "description": {"type": "string", "description": "Why this directory is needed", "required": True},
        },
    },
    {
        "name": "app_action_viewer",
        "description": "Run a Viewer desktop action. Use this EXACTLY when the user wants to OPEN or LOAD a pre-existing 3D model, 2D image, or DICOM series from disk. "
                       "Supported actions: viewer.load_2d (use this for loading a single 2D image), viewer.load_3d, viewer.load_dicom. "
                       "DO NOT use this tool for viewing reconstructed models. Use it for loading/opening external static files.",
        "parameters": {
            "action": {"type": "string", "description": "One of: viewer.load_2d, viewer.load_3d, viewer.load_dicom", "required": True},
        },
    },
    {
        "name": "app_action_reconstruction",
        "description": "Run a 3D Reconstruction desktop action. Use this when the user wants to perform 3D reconstruction tasks: load a dataset of source images for 3D reconstruction, start the reconstruction process, or view/close the RESULTING 3D point cloud model. "
                       "Supported actions: reconstruction.load_images (use ONLY for loading multiple images to build a 3D model, NOT for a single 2D image), reconstruction.start_reconstruction, reconstruction.view_3d_model, reconstruction.close_3d_model. "
                       "NOTE: reconstruction.view_3d_model is ONLY for showing the generated reconstruction result, NOT for loading a 3D file from disk.",
        "parameters": {
            "action": {"type": "string", "description": "One of: reconstruction.load_images, reconstruction.start_reconstruction, reconstruction.view_3d_model, reconstruction.close_3d_model", "required": True},
        },
    },
    {
        "name": "app_action_ai",
        "description": "Run an AI / Deep Learning desktop action. Use this for AI detection, segmentation, video tracking, hiding results, training a model, or viewing training charts. "
                       "Supported actions: ai.run_detection, ai.run_segmentation, ai.video_tracking, ai.hide_results, ai.training_model, ai.view_training_charts.",
        "parameters": {
            "action": {"type": "string", "description": "One of the supported ai.* actions", "required": True},
        },
    },
    {
        "name": "app_action_general",
        "description": "Run general application actions: opening/closing AI assistant, reloading assistant components, mail, settings, language, and authentication. "
                       "Supported actions: assistant.open, assistant.close, assistant.reload_model, assistant.reload_rag, assistant.reload_agent, assistant.reload_server, "
                       "mail.open, mail.close, mail.settings, help.about, language.change, admin.settings, admin.change_avatar, admin.change_password, admin.logout, admin.login. "
                       "Use language.change with language='vi' or language='en'.",
        "parameters": {
            "action": {"type": "string", "description": "One of the supported general actions", "required": True},
            "language": {"type": "string", "description": "Only for language.change: 'vi' or 'en'", "required": False},
            "username": {"type": "string", "description": "Only for admin.login", "required": False},
            "password": {"type": "string", "description": "Only for admin.login", "required": False},
        },
    },
    {
        "name": "rag_search",
        "description": "Search PROJECT-SPECIFIC documentation and source code using semantic similarity. "
                       "Use ONLY when the question requires internal project information such as project members, "
                       "roles, project history, internal APIs, code patterns, or project documentation. "
                       "Do NOT use for general knowledge questions (e.g. 'NLP là gì?', 'RAG là gì?', 'OOP là gì?') "
                       "— answer those directly with final_answer without calling any tool.",
        "parameters": {
            "query": {"type": "string", "description": "The question or topic to search for", "required": True},
            "top_k": {"type": "integer", "description": "Max number of results to return (default 5, max 10)", "required": False},
        },
    },
]

# ── LSP tools (go_to_definition, find_references) ─────────────────────────────
AGENT_TOOLS.extend([
    {
        "name": "go_to_definition",
        "description": "Jump to the definition of a C++ or Python symbol using the language server (clangd/pylsp). "
                       "Use when you need to find where a function, class, or variable is defined. "
                       "Requires clangd to be installed for C++ files, pylsp for Python files.",
        "parameters": {
            "path":      {"type": "string",  "description": "Relative path to source file", "required": True},
            "line":      {"type": "integer", "description": "1-based line number of the symbol", "required": True},
            "character": {"type": "integer", "description": "0-based character offset in the line", "required": True},
        },
    },
    {
        "name": "find_references",
        "description": "Find all usages/references of a C++ or Python symbol using the language server (clangd/pylsp). "
                       "Use when you need to understand the full impact of a change before editing. "
                       "Requires clangd to be installed for C++ files, pylsp for Python files.",
        "parameters": {
            "path":      {"type": "string",  "description": "Relative path to source file", "required": True},
            "line":      {"type": "integer", "description": "1-based line number of the symbol", "required": True},
            "character": {"type": "integer", "description": "0-based character offset in the line", "required": True},
        },
    },
])

# ``application_action`` is the sole public UI tool.  The former category
# names remain parser aliases for old conversations, but are not model tools.
app_action_tools = [tool for tool in AGENT_TOOLS if tool["name"].startswith("app_action_")]
AGENT_TOOLS = [tool for tool in AGENT_TOOLS if not tool["name"].startswith("app_action_")]

AGENT_TOOLS.append({
    "name": "application_action",
    "description": ("Execute exactly ONE canonical desktop action. Choose the id whose meaning matches the "
                    "CURRENT plan step (show/view and hide/close are different actions). Catalog:\n"
                    + action_catalog()),
    "parameters": {
        "action": {"type": "string", "description": "Canonical desktop action id (e.g., viewer.load_2d, ai.run_detection)", "required": True},
        "language": {"type": "string", "description": "Required only for language.change: vi or en", "required": False},
        "username": {"type": "string", "description": "Optional login username", "required": False},
        "password": {"type": "string", "description": "Optional login password", "required": False},
    },
})
AGENT_TOOLS = enrich_tool_definitions(AGENT_TOOLS)
_TOOL_PARAM_MODELS = build_tool_models(AGENT_TOOLS)
_LLAMA_CPP_TOOLS = openai_tools(AGENT_TOOLS)
_TOOL_GRAMMAR_SCHEMA = grammar_schema(AGENT_TOOLS)
_TOOL_DEFINITIONS = {tool["name"]: tool for tool in AGENT_TOOLS}

# ── Safety: các thư mục/file cấm truy cập ────────────────────────────────────
_AGENT_BLOCKED_DIRS = {".git", "build", "__pycache__", ".vs", "node_modules"}
_AGENT_BLOCKED_EXTS = {".exe", ".dll", ".so", ".bin", ".dat", ".pkl", ".gguf", ".onnx", ".pt"}
_AGENT_MAX_FILE_READ_CHARS = 24000  # bounded source context
_AGENT_MAX_UNSCOPED_SOURCE_LINES = 240
_AGENT_MAX_ITERATIONS = 12
_CODE_CITATION_PATTERN = re.compile(
    r"(?:trích\s*dẫn|trich\s*dan|cite|quote|show\s+code|source\s+code)"
    r"[\s\S]{0,80}?([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\s*\(",
    re.IGNORECASE,
)
_PYTHON_DEF_PATTERN = re.compile(
    r"(?:async\s+)?def\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(",
    re.IGNORECASE,
)
_PYTHON_SYMBOL_LABEL_PATTERN = re.compile(
    r"(?:h.m|ham|function|method|symbol)\s*[:：]?\s*"
    r"(?:async\s+)?(?:def\s+)?([A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)*)",
    re.IGNORECASE,
)
_PYTHON_PATH_PATTERN = re.compile(
    r"(?P<path>[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)*\.py)\b",
    re.IGNORECASE,
)
_EXPLICIT_CITATION_PATTERN = re.compile(
    r"(?:tr.{0,3}ch\s*d.{0,3}n|cite|quote|show\s+code|source\s+code)",
    re.IGNORECASE,
)


def _agent_safe_path(rel_path: str) -> str | None:
    """Validate and resolve a relative path within PROJECT_DIR. Returns None if unsafe."""
    if not rel_path:
        return None
    # Normalize separators
    rel_path = rel_path.replace("\\", "/").strip("/")
    # Block traversal
    if ".." in rel_path.split("/"):
        return None
    abs_path = os.path.normpath(os.path.join(PROJECT_DIR, rel_path))
    # Ensure within project (commonpath avoids sibling-prefix escapes).
    try:
        if os.path.commonpath([abs_path, os.path.normpath(PROJECT_DIR)]) != os.path.normpath(PROJECT_DIR):
            return None
    except ValueError:
        return None
    # Check blocked dirs
    parts = rel_path.split("/")
    for part in parts:
        if part in _AGENT_BLOCKED_DIRS:
            return None
    return abs_path


# ── Tool Executor Functions ───────────────────────────────────────────────────

def tool_read_file(params: dict) -> dict:
    """Read file content with optional line range."""
    path = params.get("path", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None:
        return {"error": f"Đường dẫn không hợp lệ hoặc bị chặn: {path}"}
    if not os.path.isfile(abs_path):
        return {"error": f"File không tồn tại: {path}"}
    ext = os.path.splitext(abs_path)[1].lower()
    if ext in _AGENT_BLOCKED_EXTS:
        return {"error": f"Không thể đọc file binary: {path}"}

    try:
        for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
            try:
                with open(abs_path, "r", encoding=enc) as f:
                    lines = f.readlines()
                break
            except (UnicodeDecodeError, ValueError):
                continue
        else:
            return {"error": f"Không đọc được encoding của file: {path}"}

        total_lines = len(lines)
        requested_start = params.get("start_line")
        requested_end = params.get("end_line")
        symbol = str(params.get("symbol", "")).strip()
        if symbol and requested_start is None and requested_end is None:
            symbol_range = (
                _find_python_symbol_range(lines, symbol)
                if ext == ".py" else _find_symbol_range(lines, symbol)
            )
            if symbol_range is None:
                return {"error": f"Không tìm thấy symbol '{symbol}' trong file: {path}",
                        "path": path, "total_lines": total_lines}
            requested_start, requested_end = symbol_range

        if (requested_start is None and requested_end is None
                and ext in {".cpp", ".h", ".c", ".hpp", ".cc", ".cxx"}
                and total_lines > _AGENT_MAX_UNSCOPED_SOURCE_LINES):
            return {
                "error": (
                    f"File nguồn có {total_lines} dòng; cần đọc theo phạm vi. "
                    "Hãy dùng search_text/analyze_code để tìm symbol rồi gọi "
                    "read_file với symbol hoặc start_line/end_line."
                ),
                "path": path,
                "total_lines": total_lines,
            }

        start = max(1, requested_start or 1) - 1  # 0-indexed
        end = min(total_lines, requested_end or total_lines)
        if ext in {".cpp", ".h", ".c", ".hpp", ".cc", ".cxx"}:
            end = _expand_function_end(lines, start, end)

        selected = lines[start:end]
        content = "".join(selected)

        if len(content) > _AGENT_MAX_FILE_READ_CHARS:
            content = content[:_AGENT_MAX_FILE_READ_CHARS] + f"\n... [truncated at {_AGENT_MAX_FILE_READ_CHARS} chars]"

        return {
            "path": path,
            "total_lines": total_lines,
            "showing": f"lines {start+1}-{end}",
            "content": content,
        }
    except Exception as e:
        return {"error": f"Lỗi đọc file {path}: {e}"}


def _find_symbol_range(lines: list[str], symbol: str) -> tuple[int, int] | None:
    """Find a bounded source block for a named function/method."""
    short_name = symbol.rsplit("::", 1)[-1].strip()
    candidates = []
    fallback = []
    qualified = symbol.strip()
    for i, line in enumerate(lines):
        if not short_name or not re.search(rf"\b{re.escape(short_name)}\s*\(", line):
            continue
        if qualified and qualified in line:
            fallback.append(i)
        # Prefer a definition (opening brace on this line or immediately
        # after it) over a call/declaration such as ``onRunReconstruction();``.
        lookahead = " ".join(lines[i:min(len(lines), i + 4)])
        if "{" in line or (qualified and qualified in line and "{" in lookahead):
            candidates.append(i)
    candidates = candidates or fallback
    if not candidates:
        return None
    start = candidates[0]
    depth = 0
    opened = False
    end = min(len(lines), start + 80)
    for idx in range(start, min(len(lines), start + 240)):
        depth += lines[idx].count("{") - lines[idx].count("}")
        if "{" in lines[idx]:
            opened = True
        if opened and depth <= 0:
            end = idx + 1
            break
    return start + 1, end


def _find_python_symbol_range(lines: list[str], symbol: str) -> tuple[int, int] | None:
    """Find the exact Python function/method range using the AST."""
    source = "".join(lines)
    short_name = symbol.rsplit("::", 1)[-1].rsplit(".", 1)[-1].strip()
    qualified = symbol.replace("::", ".").strip()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    if tree is not None:
        matches = []

        def visit(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node_path = ".".join((*parents, node.name))
                if node_path == qualified or node.name == short_name:
                    decorator_lines = [decorator.lineno for decorator in node.decorator_list]
                    start = min([node.lineno, *decorator_lines])
                    end = getattr(node, "end_lineno", node.lineno)
                    matches.append((start, end, node_path == qualified))
                next_parents = (*parents, node.name)
            elif isinstance(node, ast.ClassDef):
                next_parents = (*parents, node.name)
            else:
                next_parents = parents
            for child in ast.iter_child_nodes(node):
                visit(child, next_parents)

        visit(tree)
        if matches:
            matches.sort(key=lambda item: (not item[2], item[0]))
            return matches[0][0], matches[0][1]

    # Syntax-error fallback: use Python indentation to keep the citation bounded.
    definition = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(short_name)}\s*\(")
    for index, line in enumerate(lines):
        if not definition.search(line):
            continue
        indent = len(line) - len(line.lstrip())
        end = len(lines)
        for candidate in range(index + 1, len(lines)):
            next_line = lines[candidate]
            if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= indent:
                end = candidate
                break
        return index + 1, end
    return None


def _expand_function_end(lines: list[str], start: int, end: int) -> int:
    """Extend a range when it begins at a C/C++ function definition."""
    header = " ".join(lines[start:min(len(lines), start + 8)])
    if not re.search(r"\b[\w:~]+\s*\([^;{}]*\)\s*(?:const\s*)?(?:override\s*)?\{", header):
        return end
    depth = 0
    opened = False
    for idx in range(start, min(len(lines), start + 240)):
        depth += lines[idx].count("{") - lines[idx].count("}")
        if "{" in lines[idx]:
            opened = True
        if opened and depth <= 0:
            return max(end, idx + 1)
    return end


def _requested_code_symbol(task: str) -> str | None:
    """Extract a C/C++ or Python symbol from an explicit citation request."""
    match = _CODE_CITATION_PATTERN.search(task)
    if match:
        return match.group(1)
    match = _PYTHON_DEF_PATTERN.search(task)
    if match:
        return match.group(1)
    match = _PYTHON_SYMBOL_LABEL_PATTERN.search(task)
    return match.group(1) if match else None


def _citation_is_python(task: str, symbol: str) -> bool:
    return bool(
        _PYTHON_PATH_PATTERN.search(task)
        or re.search(r"\b(?:python|def|async\s+def)\b", task, re.IGNORECASE)
        or "." in symbol
    )


def _citation_path_hint(task: str) -> str | None:
    match = _PYTHON_PATH_PATTERN.search(task)
    if not match:
        return None
    return match.group("path").replace("\\", "/")


def _code_language(path: str) -> str:
    return {
        ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".h": "cpp", ".hpp": "cpp",
        ".c": "c", ".py": "python", ".js": "javascript", ".ts": "typescript",
    }.get(os.path.splitext(path)[1].lower(), "text")


def _build_code_citation_result(task: str, session_id: str, request_started: float) -> dict | None:
    """Return an exact source citation without asking the LLM to reproduce code.

    This is a generic Coding Agent read-only workflow for explicit, qualified
    symbol citations. It prevents a context-limited model from emitting an
    unterminated or partial code fence after the exact source is already known.
    """
    symbol = _requested_code_symbol(task)
    if not symbol or not _EXPLICIT_CITATION_PATTERN.search(task):
        return None
    is_python = _citation_is_python(task, symbol)
    path_hint = _citation_path_hint(task) if is_python else None
    search_symbol = symbol.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    if path_hint:
        search_path = os.path.dirname(path_hint) or "."
        search_pattern = os.path.basename(path_hint)
    elif is_python:
        search_path = "AIAssistant"
        search_pattern = "*.py"
    else:
        search_path = "src"
        search_pattern = "*.cpp"
    logger.info(
        "[CODING CITATION] Resolve symbol=%s path=%s pattern=%s",
        symbol, search_path, search_pattern,
    )
    search_params = {
        "query": search_symbol if is_python else symbol,
        "path": search_path,
        "file_pattern": search_pattern,
        "max_results": 50 if is_python else 20,
    }
    search_result = tool_search_text(search_params)
    matches = search_result.get("results", []) if isinstance(search_result, dict) else []
    if path_hint:
        matches = [
            item for item in matches
            if str(item.get("file", "")).replace("\\", "/").endswith(path_hint)
        ]
    if is_python:
        definition = next(
            (
                item for item in matches
                if re.search(
                    rf"\b(?:async\s+)?def\s+{re.escape(search_symbol)}\s*\(",
                    str(item.get("content", "")),
                )
            ),
            None,
        )
    else:
        definition = next((item for item in matches if "{" in str(item.get("content", ""))), None)
    if definition is None and not is_python:
        search_params = {"query": symbol, "path": "src", "file_pattern": "*.h", "max_results": 20}
        search_result = tool_search_text(search_params)
        matches = search_result.get("results", []) if isinstance(search_result, dict) else []
        definition = next((item for item in matches if "{" in str(item.get("content", ""))), None)
    if definition is None:
        return None

    path = str(definition["file"])
    read_params = {"path": path, "symbol": search_symbol if is_python else symbol}
    read_result = tool_read_file(read_params)
    if read_result.get("error"):
        return None
    content = str(read_result.get("content", "")).rstrip()
    if not content:
        return None

    search_verification = {"passed": True, "reason": "Found a source definition for the requested symbol."}
    read_verification = {"passed": True, "reason": "Read the complete source range for the requested symbol."}
    showing = str(read_result.get("showing", ""))
    final_answer = (
        f"### Trích dẫn mã nguồn\n\n"
        f"`{path}` — {showing}\n\n"
        f"```{_code_language(path)}\n{content}\n```"
    )
    display_read_result = dict(read_result)
    display_read_result["presentation"] = "citation_source"
    steps = [
        {"type": "delegation", "agent": "code", "tool": "search_text", "iteration": 0},
        {"type": "tool_call", "tool": "search_text", "params": search_params, "iteration": 0},
        {"type": "tool_result", "tool": "search_text", "result": search_result, "iteration": 0},
        {"type": "verification", "tool": "search_text", "result": search_verification, "iteration": 0},
        {"type": "delegation", "agent": "code", "tool": "read_file", "iteration": 0},
        {"type": "tool_call", "tool": "read_file", "params": read_params, "iteration": 0},
        {"type": "tool_result", "tool": "read_file", "result": display_read_result, "iteration": 0},
        {"type": "verification", "tool": "read_file", "result": read_verification, "iteration": 0},
        {"type": "final_answer", "content": final_answer, "iteration": 0},
    ]
    logger.info("[CODING CITATION] Resolved %s | %s | %s", symbol, path, showing)
    return {
        "status": "completed", "session_id": session_id, "steps": steps,
        "prior_step_count": 0, "iterations": 0,
        "total_ms": round((time.monotonic() - request_started) * 1000),
    }


def tool_list_directory(params: dict) -> dict:
    """List directory contents."""
    path = params.get("path", ".")
    if path == ".":
        abs_path = PROJECT_DIR
    else:
        abs_path = _agent_safe_path(path)
    if abs_path is None:
        return {"error": f"Đường dẫn không hợp lệ: {path}"}
    if not os.path.isdir(abs_path):
        return {"error": f"Thư mục không tồn tại: {path}"}

    recursive = params.get("recursive", False)
    max_depth = params.get("max_depth", 3)
    entries = []
    count = 0
    # Keep recursive repository discovery small enough for the model context;
    # search_text/find_files are the precise tools for locating code.
    max_entries = 150

    try:
        if recursive:
            for root, dirs, files in os.walk(abs_path):
                dirs[:] = sorted(d for d in dirs if d not in _AGENT_BLOCKED_DIRS)
                depth = root.replace(abs_path, "").count(os.sep)
                if depth >= max_depth:
                    dirs.clear()
                    continue
                rel_root = os.path.relpath(root, PROJECT_DIR)
                for d in sorted(dirs):
                    if count >= max_entries:
                        break
                    entries.append({"name": os.path.join(rel_root, d), "type": "directory"})
                    count += 1
                for f in sorted(files):
                    if count >= max_entries:
                        break
                    fp = os.path.join(root, f)
                    try:
                        size = os.path.getsize(fp)
                    except OSError:
                        size = 0
                    entries.append({
                        "name": os.path.join(rel_root, f),
                        "type": "file",
                        "size_bytes": size,
                    })
                    count += 1
                if count >= max_entries:
                    break
        else:
            for item in sorted(os.listdir(abs_path)):
                if item in _AGENT_BLOCKED_DIRS:
                    continue
                if count >= max_entries:
                    break
                fp = os.path.join(abs_path, item)
                rel = os.path.relpath(fp, PROJECT_DIR)
                if os.path.isdir(fp):
                    entries.append({"name": rel, "type": "directory"})
                else:
                    try:
                        size = os.path.getsize(fp)
                    except OSError:
                        size = 0
                    entries.append({"name": rel, "type": "file", "size_bytes": size})
                count += 1

        return {"path": path, "count": len(entries), "entries": entries}
    except Exception as e:
        return {"error": f"Lỗi liệt kê thư mục {path}: {e}"}


def tool_find_files(params: dict) -> dict:
    """Find files by glob without loading file contents into model context."""
    pattern = str(params.get("pattern", "")).strip()
    if not pattern or pattern in {".", ".."}:
        return {"error": "File pattern must not be empty."}
    path = params.get("path", ".")
    abs_root = PROJECT_DIR if path == "." else _agent_safe_path(path)
    if abs_root is None or not os.path.isdir(abs_root):
        return {"error": f"Invalid or missing directory: {path}"}
    max_results = min(max(int(params.get("max_results", 100)), 1), 500)
    matches = []
    try:
        for root, dirs, files in os.walk(abs_root):
            dirs[:] = sorted(d for d in dirs if d not in _AGENT_BLOCKED_DIRS)
            for filename in sorted(files):
                relative_to_root = os.path.relpath(os.path.join(root, filename), abs_root)
                if not fnmatch.fnmatch(filename, pattern) and not fnmatch.fnmatch(relative_to_root, pattern):
                    continue
                matches.append(os.path.relpath(os.path.join(root, filename), PROJECT_DIR))
                if len(matches) >= max_results:
                    return {"pattern": pattern, "path": path, "count": len(matches),
                            "truncated": True, "matches": matches}
        return {"pattern": pattern, "path": path, "count": len(matches),
                "truncated": False, "matches": matches}
    except OSError as error:
        return {"error": f"Unable to find files: {error}"}


def tool_search_text(params: dict) -> dict:
    """Search for text in project files."""
    query = params.get("query", "")
    if not query:
        return {"error": "Query rỗng"}

    search_path = params.get("path", ".")
    if search_path == ".":
        abs_search = PROJECT_DIR
    else:
        abs_search = _agent_safe_path(search_path)
    if abs_search is None:
        return {"error": f"Đường dẫn không hợp lệ: {search_path}"}
    if not os.path.isdir(abs_search) and not os.path.isfile(abs_search):
        return {"error": f"Đường dẫn không tồn tại: {search_path}"}

    file_pattern = params.get("file_pattern", "*")
    file_patterns = [pattern.strip() for pattern in re.split(r"[;,]", str(file_pattern)) if pattern.strip()]
    if not file_patterns:
        file_patterns = ["*"]
    file_patterns = [f"*{pattern}" if pattern.startswith(".") else pattern for pattern in file_patterns]
    case_sensitive = params.get("case_sensitive", True)
    max_results = min(params.get("max_results", 50), 100)

    results = []
    search_query = query if case_sensitive else query.lower()
    text_exts = {".cpp", ".h", ".py", ".md", ".txt", ".cmake", ".json", ".xml", ".html", ".css", ".js", ".ts", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".bat", ".sh"}

    try:
        walk_roots = (
            [(os.path.dirname(abs_search), [], [os.path.basename(abs_search)])]
            if os.path.isfile(abs_search) else os.walk(abs_search)
        )
        search_root = os.path.dirname(abs_search) if os.path.isfile(abs_search) else abs_search
        for root, dirs, files in walk_roots:
            dirs[:] = sorted(d for d in dirs if d not in _AGENT_BLOCKED_DIRS)
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in text_exts:
                    continue
                relative_to_search = os.path.relpath(os.path.join(root, filename), search_root)
                relative_to_project = os.path.relpath(os.path.join(root, filename), PROJECT_DIR)
                if file_patterns != ["*"] and not any(
                    fnmatch.fnmatch(filename, pattern)
                    or fnmatch.fnmatch(relative_to_search, pattern)
                    or fnmatch.fnmatch(relative_to_project, pattern)
                    for pattern in file_patterns
                ):
                    continue

                fp = os.path.join(root, filename)
                rel = os.path.relpath(fp, PROJECT_DIR)
                try:
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        for line_no, line in enumerate(f, 1):
                            check_line = line if case_sensitive else line.lower()
                            if search_query in check_line:
                                results.append({
                                    "file": rel,
                                    "line": line_no,
                                    "content": line.rstrip()[:200],
                                })
                                if len(results) >= max_results:
                                    return {"query": query, "count": len(results), "truncated": True, "results": results}
                except (OSError, UnicodeDecodeError):
                    continue

        return {"query": query, "count": len(results), "truncated": False, "results": results}
    except Exception as e:
        return {"error": f"Lỗi tìm kiếm: {e}"}


def tool_analyze_code(params: dict) -> dict:
    """Analyze code structure of a file."""
    path = params.get("path", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None:
        return {"error": f"Đường dẫn không hợp lệ: {path}"}
    if not os.path.isfile(abs_path):
        return {"error": f"File không tồn tại: {path}"}

    ext = os.path.splitext(abs_path)[1].lower()
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return {"error": f"Lỗi đọc file: {e}"}

    total_lines = content.count("\n") + 1
    result = {
        "path": path,
        "extension": ext,
        "total_lines": total_lines,
        "size_bytes": len(content.encode("utf-8")),
    }

    if ext == ".py":
        return _analyze_python(content, result)
    elif ext in (".cpp", ".h", ".c", ".hpp"):
        return _analyze_cpp(content, result)
    else:
        # Generic analysis
        result["analysis"] = "File type không hỗ trợ phân tích chi tiết. Dùng read_file để xem nội dung."
        return result


def _analyze_python(content: str, result: dict) -> dict:
    """Python AST-based analysis."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        result["syntax_error"] = str(e)
        return result

    classes = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            bases = [ast.dump(b) if not hasattr(b, "id") else b.id for b in node.bases]
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
                "bases": bases,
            })
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            else:
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

    # Re-parse for top-level functions only
    top_functions = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            docstring = ast.get_docstring(node)
            top_functions.append({
                "name": node.name,
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "args": args,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "docstring": (docstring[:100] + "...") if docstring and len(docstring) > 100 else docstring,
            })

    result["classes"] = classes
    result["functions"] = top_functions
    result["imports"] = imports[:30]  # Limit
    return result


def _analyze_cpp(content: str, result: dict) -> dict:
    """Regex-based C++ analysis."""
    classes = []
    functions = []
    includes = []

    # Find #include
    for m in re.finditer(r'^#include\s+[<"]([^>"]+)[>"]', content, re.MULTILINE):
        includes.append(m.group(1))

    # Find class/struct declarations
    for m in re.finditer(r'^(?:class|struct)\s+(?:\w+\s+)?(\w+)\s*(?::\s*(?:public|private|protected)\s+(\w+))?\s*\{',
                          content, re.MULTILINE):
        classes.append({
            "name": m.group(1),
            "base": m.group(2),
            "line": content[:m.start()].count("\n") + 1,
        })

    # Find function definitions (simplified)
    func_re = re.compile(
        r'^(?:[\w:*&<>\[\]~]+\s+)+(?:(\w+)::)?(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:noexcept\s*)?(?:\{|;)',
        re.MULTILINE,
    )
    for m in func_re.finditer(content):
        scope = m.group(1) or ""
        name = m.group(2)
        if name in ("if", "for", "while", "switch", "return", "catch"):
            continue
        functions.append({
            "name": f"{scope}::{name}" if scope else name,
            "line": content[:m.start()].count("\n") + 1,
        })

    result["classes"] = classes[:50]
    result["functions"] = functions[:100]
    result["includes"] = includes[:30]
    return result


# ── Tool dispatch ──────────────────────────────────────────────────────────────

def tool_get_project_status(params: dict) -> dict:
    """Return a lightweight, read-only project status."""
    source_counts = {}
    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in _AGENT_BLOCKED_DIRS]
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in {".cpp", ".h", ".py", ".json", ".cmake"}:
                source_counts[ext] = source_counts.get(ext, 0) + 1
    try:
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_DIR,
                                capture_output=True, text=True, timeout=5, check=False).stdout.strip()
        changed = subprocess.run(["git", "status", "--short"], cwd=PROJECT_DIR,
                                 capture_output=True, text=True, timeout=5, check=False).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        branch, changed = "", []
    return {"project_root": PROJECT_DIR, "git_branch": branch, "changed_files": changed[:100],
            "source_file_counts": source_counts}


def tool_git_diff(params: dict) -> dict:
    """Return a bounded Git diff for Code Agent review."""
    path = params.get("path", ".")
    if path == ".":
        diff_path = "."
    else:
        abs_path = _agent_safe_path(path)
        if abs_path is None or not os.path.exists(abs_path):
            return {"error": f"Invalid or missing diff path: {path}"}
        diff_path = os.path.relpath(abs_path, PROJECT_DIR)
    command = ["git", "diff", "--no-ext-diff"]
    if params.get("staged", False):
        command.append("--cached")
    command.extend(["--", diff_path])
    try:
        completed = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=10, check=False)
        if completed.returncode != 0:
            return {"error": completed.stderr.strip() or "git diff failed", "return_code": completed.returncode}
        content = completed.stdout
        limit = 16000
        return {"path": path, "staged": bool(params.get("staged", False)),
                "content": content[:limit], "truncated": len(content) > limit}
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"Unable to read Git diff: {error}"}


def tool_validate_file(params: dict) -> dict:
    """Validate Python or JSON syntax without executing project code."""
    path = params.get("path", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None or not os.path.isfile(abs_path):
        return {"error": f"Invalid or missing file: {path}"}
    ext = os.path.splitext(abs_path)[1].lower()
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        if ext == ".py":
            compile(content, path, "exec")
        elif ext == ".json":
            json.loads(content)
        else:
            return {"error": "Only .py and .json files can be validated."}
        return {"success": True, "path": path, "validation": "syntax_valid"}
    except (SyntaxError, ValueError) as error:
        return {"success": False, "path": path, "error": str(error)}


def _canonical_desktop_action(params: dict) -> dict | None:
    """Compatibility wrapper around the shared action manifest."""
    return canonicalise_action_params(params)


def _looks_like_ui_action(text: str) -> bool:
    return False


def _match_desktop_action(task: str) -> dict | None:
    return None


def _match_desktop_action_sequence(task: str) -> list[dict] | None:
    return None





def tool_application_action(params: dict) -> dict:
    """Create a UI-action request; success is only reported after Qt ACKs it."""
    canonical_params, error = validate_action_params(params)
    if error:
        return {"error": error}
    return {
        "pending_ui_ack": True,
        "action": canonical_params["action"],
        "request_id": canonical_params.get("request_id", ""),
        "message": "Action is awaiting acknowledgement from the Qt desktop client.",
    }


def _execute_approved_patch_file(params: dict) -> dict:
    path = params.get("path", "")
    find_text = params.get("find", "")
    replacement = params.get("replace", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None or not os.path.isfile(abs_path):
        return {"error": f"Invalid or missing file: {path}"}
    if not find_text:
        return {"error": "Patch text must not be empty."}
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        matches = content.count(find_text)
        if matches != 1:
            return {"error": f"Patch requires exactly one matching fragment; found {matches}.", "path": path}
        updated = content.replace(find_text, replacement, 1)
        result = write_sandboxed_file(abs_path, updated, PROJECT_DIR)
        if result.get("error"):
            return result
        return {"success": True, "path": path, "replacements": 1,
                "bytes_written": result.get("bytes_written", 0), "sandbox": result.get("sandbox")}
    except OSError as error:
        return {"error": f"Unable to patch file: {error}"}


def _execute_approved_replace_file_content(params: dict) -> dict:
    path = params.get("path", "")
    target = params.get("targetContent", "")
    replacement = params.get("replacementContent", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None or not os.path.isfile(abs_path):
        return {"error": f"Invalid or missing file: {path}"}
    if not target:
        return {"error": "Target content must not be empty."}
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        matches = content.count(target)
        if matches != 1:
            return {"error": f"Replace requires exactly one matching fragment; found {matches}.", "path": path}
        updated = content.replace(target, replacement, 1)
        result = write_sandboxed_file(abs_path, updated, PROJECT_DIR)
        if result.get("error"):
            return result
        return {"success": True, "path": path, "replacements": 1,
                "bytes_written": result.get("bytes_written", 0), "sandbox": result.get("sandbox")}
    except OSError as error:
        return {"error": f"Unable to replace file content: {error}"}


def _execute_approved_multi_replace_file_content(params: dict) -> dict:
    path = params.get("path", "")
    replacements = params.get("replacements", [])
    abs_path = _agent_safe_path(path)
    if abs_path is None or not os.path.isfile(abs_path):
        return {"error": f"Invalid or missing file: {path}"}
    if not replacements:
        return {"error": "Replacements list must not be empty."}
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        
        for idx, rep in enumerate(replacements):
            target = rep.get("targetContent", "")
            replacement = rep.get("replacementContent", "")
            if not target:
                return {"error": f"Target content at index {idx} must not be empty."}
            matches = content.count(target)
            if matches != 1:
                return {"error": f"Replace at index {idx} requires exactly one matching fragment; found {matches}.", "path": path}
            content = content.replace(target, replacement, 1)

        result = write_sandboxed_file(abs_path, content, PROJECT_DIR)
        if result.get("error"):
            return result
        return {"success": True, "path": path, "replacements": len(replacements),
                "bytes_written": result.get("bytes_written", 0), "sandbox": result.get("sandbox")}
    except OSError as error:
        return {"error": f"Unable to multi-replace file content: {error}"}


def _execute_approved_create_directory(params: dict) -> dict:
    path = params.get("path", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None:
        return {"error": f"Invalid directory path: {path}"}
    return create_sandboxed_directory(abs_path, PROJECT_DIR)


def tool_rag_search(params: dict) -> dict:
    """Dynamic RAG search tool — called by the agent at runtime."""
    query = params.get("query", "").strip()
    if not query:
        return {"error": "Query không được để trống."}
    top_k = min(int(params.get("top_k", 5)), 10)
    if not rag_runtime.knowledge_chunks:
        return {"error": "RAG index chưa sẵn sàng."}
    try:
        doc_ctx, code_ctx, _ = rag_runtime.get_context(query, result_k=top_k)
        results = []
        if doc_ctx:
            results.append({"source": "documentation", "content": doc_ctx[:3000]})
        if code_ctx:
            results.append({"source": "source_code", "content": code_ctx[:3000]})
        if not results:
            return {"query": query, "found": False, "message": "Không tìm thấy kết quả phù hợp."}
        return {"query": query, "found": True, "top_k": top_k, "results": results}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Lỗi RAG search: {e}"}


# Implementations hosted by modules.mcp_server. Keep this map private to the
# MCP boundary: the agent invokes the public MCP endpoint rather than these
# functions directly.
MCP_LOCAL_EXECUTORS = {
    "read_file":                tool_read_file,
    "list_directory":    tool_list_directory,
    "find_files":        tool_find_files,
    "search_text":       tool_search_text,
    "analyze_code":      tool_analyze_code,
    "get_project_status": tool_get_project_status,
    "git_diff":          tool_git_diff,
"validate_file":                tool_validate_file,
"application_action":             tool_application_action,
    "app_action_viewer":         tool_application_action,
    "app_action_reconstruction": tool_application_action,
    "app_action_ai":             tool_application_action,
    "app_action_general":        tool_application_action,
    "rag_search":                tool_rag_search,
    # LSP tools (read-only, require clangd or pylsp on PATH)
    "go_to_definition":          _lsp_client.tool_go_to_definition,
    "find_references":           _lsp_client.tool_find_references,
    # write_file, run_command, patch_file, replace_file_content,
    # multi_replace_file_content, create_directory require approval
}

# Safe tools invoke the platform directly. MCP is an external protocol adapter
# and is deliberately not used as an HTTP loopback transport.
def tool_transfer_to_code_agent(params: dict) -> dict:
    return {"status": "transferred_to_code", "intent": params.get("intent")}

def tool_transfer_to_toolapp_agent(params: dict) -> dict:
    return {"status": "transferred_to_toolapp", "intent": params.get("intent")}

def tool_transfer_to_chatbot_agent(params: dict) -> dict:
    return {"status": "transferred_to_chatbot", "intent": params.get("intent")}

_TOOL_EXECUTORS = {
    name: (lambda params, tool_name=name: execute_platform_tool(tool_name, params))
    for name in MCP_LOCAL_EXECUTORS
}
_TOOL_EXECUTORS.update({
    "transfer_to_code_agent": tool_transfer_to_code_agent,
    "transfer_to_toolapp_agent": tool_transfer_to_toolapp_agent,
    "transfer_to_chatbot_agent": tool_transfer_to_chatbot_agent,
})
TOOL_REGISTRY = ToolRegistry(_TOOL_EXECUTORS)

_TOOLS_REQUIRING_APPROVAL = {
    tool["name"] for tool in AGENT_TOOLS if tool.get("requires_approval")
}

# ── Pending actions storage (in-memory, per session) ──────────────────────────
_pending_lock = threading.Lock()
_PENDING_ACTIONS_FILE = os.path.join(APP_DATA_DIR, "AIAssistant", "pending_agent_actions.json")
_pending_actions = PendingActionStore(_PENDING_ACTIONS_FILE)


def _save_pending_actions() -> None:
    """Persist pending approvals so a server restart does not invalidate the UI action."""
    _pending_actions.save()


def _load_pending_actions() -> None:
    _pending_actions.load()
    if _pending_actions.cleanup(time.time() - 600):
        _save_pending_actions()


def _generate_action_id() -> str:
    return hashlib.md5(f"{time.time()}-{threading.current_thread().ident}".encode()).hexdigest()[:12]


def _execute_approved_write_file(params: dict) -> dict:
    """Execute write_file after user approval."""
    path = params.get("path", "")
    content = params.get("content", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None:
        return {"error": f"Đường dẫn không hợp lệ: {path}"}

    return write_sandboxed_file(abs_path, content, PROJECT_DIR)


def _execute_approved_run_command(params: dict) -> dict:
    """Execute shell command after user approval."""
    command = params.get("command", "")
    cwd = params.get("cwd", ".")
    timeout = min(params.get("timeout", 30), 120)  # Max 2 minutes

    if cwd == ".":
        abs_cwd = PROJECT_DIR
    else:
        abs_cwd = _agent_safe_path(cwd)
    if abs_cwd is None or not os.path.isdir(abs_cwd):
        return {"error": f"Invalid command working directory: {cwd}"}

    return run_sandboxed_command(command, abs_cwd, timeout)


def platform_executors() -> dict[str, Callable[[dict], dict]]:
    """Return all legacy implementations for the new platform adapter.

    This function is the migration seam: registration happens at composition
    time, rather than through module import side effects.
    """
    return {
        **MCP_LOCAL_EXECUTORS,
        "transfer_to_code_agent": tool_transfer_to_code_agent,
        "transfer_to_toolapp_agent": tool_transfer_to_toolapp_agent,
        "transfer_to_chatbot_agent": tool_transfer_to_chatbot_agent,
        "write_file": _execute_approved_write_file,
        "run_command": _execute_approved_run_command,
        "patch_file": _execute_approved_patch_file,
        "replace_file_content": _execute_approved_replace_file_content,
        "multi_replace_file_content": _execute_approved_multi_replace_file_content,
        "create_directory": _execute_approved_create_directory,
    }


# ── Agent System Prompt ───────────────────────────────────────────────────────

def _build_agent_system_prompt(language: str = "vi") -> str:
    tool_desc_parts = []
    for tool in AGENT_TOOLS:
        params_desc = []
        for pname, pinfo in tool["parameters"].items():
            req = " (required)" if pinfo.get("required") else " (optional)"
            params_desc.append(f"    - {pname}: {pinfo['type']}{req} — {pinfo['description']}")
        params_str = "\n".join(params_desc)
        tool_desc_parts.append(f"  {tool['name']}: {tool['description']}\n    Parameters:\n{params_str}")

    tools_block = "\n\n".join(tool_desc_parts)

    return f"""Bạn là AI Agent chuyên nghiệp cho dự án 3D-Reconstruction.
Bạn có khả năng THỰC THI các tác vụ trên project bằng cách gọi các tools.

## AVAILABLE TOOLS:

{tools_block}

## RESPONSE CONTRACT (STRICT):

The server uses llama.cpp grammar-constrained decoding. Return EXACTLY one JSON
object and no Markdown or reasoning.  To call a tool use
{{"kind":"tool","tool":"tool_name","params":{{...}}}}.  For the final
answer use {{"kind":"final","content":"your concise answer"}}.

## RULES:

1. Quan sát yêu cầu người dùng, xác định mục tiêu rõ ràng trước khi hành động.
2. Gọi tool từng bước một. Sau mỗi kết quả tool, phân tích và quyết định bước tiếp.
3. Khi đã có đủ thông tin, trả lời người dùng bằng text bình thường (KHÔNG gọi tool).
4. Luôn đọc file trước khi sửa — KHÔNG viết file mà chưa đọc nội dung gốc.
5. Mỗi lần chỉ gọi MỘT tool duy nhất.
6. Khi gọi tool write_file hoặc run_command, hệ thống sẽ yêu cầu người dùng phê duyệt.
7. Trả lời bằng tiếng Việt trừ khi được hỏi bằng tiếng Anh.
8. Sử dụng Markdown formatting cho câu trả lời cuối cùng.
9. Nếu task quá lớn hoặc nguy hiểm, hãy giải thích và hỏi lại trước khi thực hiện.
10. Scope: chỉ làm việc trong thư mục project — không truy cập file ngoài project.
11. For application UI requests (loading data, reconstruction, AI tools, mail, language, help, or account actions), you MUST call application_action directly. Do NOT call rag_search, search_text, read_file, or another research tool to discover a UI action. If your plan has multiple steps, call the canonical UI actions sequentially. Provide a short confirmation message only when all steps are completed.
12. Dùng rag_search TRƯỚC khi dùng read_file khi chưa biết file nào chứa thông tin cần tìm.
13. DỪNG NGAY khi task đã hoàn thành — KHÔNG gọi thêm tool nếu kết quả đã rõ ràng.
14. Với câu trả lời về project, chỉ khẳng định điều có bằng chứng từ RAG/tool. Nếu kết quả không đủ bằng chứng, nói rõ không tìm thấy thay vì suy đoán. Khi cần chi tiết chính xác, dùng read_file để xác minh đoạn nguồn trước khi kết luận.

15. Khi yêu cầu trích dẫn hoặc giải thích một hàm/symbol, không đọc toàn bộ file nguồn. Hãy dùng search_text để lấy file và dòng, sau đó gọi read_file với symbol hoặc start_line/end_line để lấy đúng đoạn code.

16. Với task coding có thay đổi repository, phải đi đủ chuỗi bằng chứng: đọc
source liên quan, gọi patch_file/write_file sau khi đã được duyệt, gọi git_diff
để review thay đổi và gọi run_command để kiểm chứng. Không được coi việc tìm
thấy file hoặc đề xuất patch là đã hoàn thành. Không suy ra file/command từ
tên tính năng; hãy lấy chúng từ source, CMake, test và kết quả tool thực tế.

17. QUAN TRỌNG: Với câu hỏi kiến thức chung (định nghĩa khái niệm, giải thích thuật ngữ công nghệ,
lý thuyết khoa học, v.v.), trả lời trực tiếp bằng final_answer mà KHÔNG gọi rag_search hoặc
bất kỳ tool nào. Chỉ dùng rag_search khi câu hỏi CẦN thông tin nội bộ dự án (nhân sự,
vai trò, lịch sử dự án, tài liệu nội bộ, code cụ thể trong project).

## EXAMPLE:

User: đổi project sang tiếng việt giúp tôi
Assistant: {{"kind":"tool","tool":"application_action","params":{{"action":"language.change","language":"vi"}}}}

User: mở hộp thư
Assistant: {{"kind":"tool","tool":"application_action","params":{{"action":"mail.open"}}}}

User: bắt đầu tái tạo 3D
Assistant: {{"kind":"tool","tool":"application_action","params":{{"action":"reconstruction.start_reconstruction"}}}}

User: chạy nhận diện đối tượng
Assistant: {{"kind":"tool","tool":"application_action","params":{{"action":"ai.run_detection"}}}}

## PROJECT INFORMATION:
- Project root: {_safe_relpath(PROJECT_DIR, PROJECT_DIR)} (thư mục gốc)
- Ngôn ngữ chính: C++ (Qt), Python
- Build system: CMake

## RESPONSE LANGUAGE:
Respond to the user in {"Vietnamese" if language == "vi" else "English"}. Keep tool names and JSON keys unchanged.
"""


def _extract_tool_call_xml(content: str) -> str | None:
    """Extract tool call from ``<tool_call>...</tool_call>`` XML envelope.

    Qwen text models using their native GGUF chat template emit tool calls
    as XML-wrapped JSON in the *content* field instead of populating the
    structured ``tool_calls`` response field.  This helper converts the XML
    envelope into the canonical ``{"kind": "tool", ...}`` JSON string that
    the rest of the agent pipeline expects.
    """
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    # Qwen format: {"name": "tool_name", "arguments": {...}}
    # Fallback format: {"tool": "tool_name", "params": {...}}
    tool_name = data.get("name", "") or data.get("tool", "")
    arguments = data.get("arguments")
    if arguments is None:
        arguments = data.get("params", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not tool_name or not isinstance(arguments, dict):
        return None
    envelope = {"kind": "tool", "tool": tool_name, "params": arguments}
    logger.info("Extracted tool call from <tool_call> XML envelope: %s",
                json.dumps(envelope, ensure_ascii=False))
    return json.dumps(envelope, ensure_ascii=False)


def _parse_tool_call(response_text: str) -> tuple:
    """Decode the JSON envelope emitted by llama.cpp constrained decoding.

    This intentionally no longer searches prose with regex.  The fallback
    accepts only a whole JSON object, so invalid/partial model output cannot
    accidentally execute a tool.
    """
    _TOOL_NAME_ALIASES = {
        "app_action_reconstruction": "application_action",
        "app_action_general":        "application_action",
        "app_action_ai":             "application_action",
        "app_action_viewer":         "application_action",
        "application_actions":       "application_action",
        "app_action":                "application_action",
        "desktop_action":            "application_action",
        "ui_action":                 "application_action",
    }

    def _normalise_tool_name(name: str) -> str:
        return _TOOL_NAME_ALIASES.get(name, name)

    try:
        data = json.loads(response_text.strip())
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(data, dict) or data.get("kind") != "tool":
        return None, None
    tool_name = _normalise_tool_name(str(data.get("tool", "")))
    params = data.get("params")
    if not isinstance(params, dict):
        return None, None
    validated, error = validate_tool_call(tool_name, params, _TOOL_PARAM_MODELS)
    if error:
        logger.warning("Rejected invalid constrained tool call %s: %s", tool_name, error)
        record_schema_error(tool_name)
        return "_validation_error", {"tool": tool_name, "error": error}
    return tool_name, validated


_PRECOMPILED_TOOL_GRAMMAR = None

def _get_tool_grammar():
    global _PRECOMPILED_TOOL_GRAMMAR
    if _PRECOMPILED_TOOL_GRAMMAR is None:
        try:
            from llama_cpp import LlamaGrammar
            _PRECOMPILED_TOOL_GRAMMAR = LlamaGrammar.from_json_schema(_TOOL_GRAMMAR_SCHEMA)
        except ImportError:
            _PRECOMPILED_TOOL_GRAMMAR = None
    return _PRECOMPILED_TOOL_GRAMMAR


def _constrained_agent_completion(messages: list[dict], max_tokens: int, temperature: float) -> str:
    """Generate exactly one final/tool envelope with llama.cpp grammar.

    Grammar is the default because it works with local models that do not
    implement a native function-calling chat template. Set
    ``AGENT_NATIVE_TOOL_CALLS=1`` to use llama-cpp-python's OpenAI ``tools``
    interface instead; both paths pass through the same Pydantic validation.
    """
    try:
        # logger.info("Constrained LLM messages: %s", json.dumps(messages, ensure_ascii=False, default=str))
        if backend_mode() != "llama_cpp":
            response = openai_compatible_completion(
                messages, max_tokens=max_tokens, temperature=temperature,
                tools=_LLAMA_CPP_TOOLS, tool_choice="auto",
                response_format={"type": "json_object"},
            )
            
            usage = response.get("usage", {})
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            if in_tok or out_tok:
                from .observability import record_token_usage
                record_token_usage(in_tok, out_tok)

            message = response.get("choices", [{}])[0].get("message", {})
            if message.get("tool_calls"):
                call = message["tool_calls"][0]["function"]
                content = json.dumps({"kind": "tool", "tool": call["name"],
                                      "params": json.loads(call.get("arguments", "{}"))}, ensure_ascii=False)
                logger.info("Constrained LLM response: %s", content)
                return content
            content = message.get("content", "")
            # Remote deployments are expected to return the same envelope.
            logger.info("Constrained LLM response: %s", content)
            return content
        kwargs = {
            "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "repeat_penalty": 1.1, "stream": False,
        }
        use_native = os.environ.get("AGENT_NATIVE_TOOL_CALLS", "0") == "1"
        if use_native:
            kwargs.update({"tools": _LLAMA_CPP_TOOLS, "tool_choice": "auto"})
        else:
            grammar = _get_tool_grammar()
            if grammar is not None:
                kwargs.update({"grammar": grammar})
        with llm_runtime.llm_lock:
            response = llm_runtime.llm.create_chat_completion(**kwargs)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"Constrained tool decoding failed: {error}") from error

    usage = response.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    if in_tok or out_tok:
        from .observability import record_token_usage
        record_token_usage(in_tok, out_tok)

    message = response.get("choices", [{}])[0].get("message", {})
    if message.get("tool_calls"):
        call = message["tool_calls"][0]["function"]
        content = json.dumps({"kind": "tool", "tool": call["name"],
                              "params": json.loads(call.get("arguments", "{}"))}, ensure_ascii=False)
        logger.info("Constrained LLM response: %s", content)
        return content
    content = message.get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("Constrained decoder returned no text content")
    content = strip_think_tags(content)
    logger.info("Constrained LLM response: %s", content)
    # Detect <tool_call> XML envelope emitted by Qwen text models
    if "<tool_call>" in content:
        xml_result = _extract_tool_call_xml(content)
        if xml_result:
            return xml_result
    try:
        # Xử lý trường hợp model sinh ra thêm văn bản rác sau chuỗi JSON
        content_stripped = content.strip()
        idx = content_stripped.find('{')
        if idx != -1:
            json_str = content_stripped[idx:]
            envelope, _ = json.JSONDecoder().raw_decode(json_str)
        else:
            envelope = json.loads(content)
    except json.JSONDecodeError as error:
        logger.warning("Constrained decoder returned non-JSON text, treating as final response: %s", error)
        return content
        
    if envelope.get("kind") == "final" and isinstance(envelope.get("content"), str):
        return envelope["content"]
    if envelope.get("kind") == "tool":
        return json.dumps(envelope, ensure_ascii=False)
        
    logger.warning("Constrained decoder returned an unsupported envelope, treating as raw text.")
    return content


_PLANNER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "requires_plan": {"type": "boolean"},
        "goal": {"type": "string"},
        "affected_areas": {"type": "array", "items": {"type": "string"}},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
        "verification_commands": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
        # Kept for persisted/older planner clients; new prompts emit steps.
        "plan": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["requires_plan", "steps"],
    "additionalProperties": False,
}
_CRITIC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "decision": {"type": "string", "enum": ["continue", "revise"]},
        "reason": {"type": "string"},
    },
    "required": ["passed", "decision", "reason"],
    "additionalProperties": False,
}


def _structured_agent_completion(messages: list[dict], max_tokens: int,
                                 temperature: float, schema: dict) -> str:
    """Generate internal planner/critic JSON without the ReAct tool schema."""
    try:
        if backend_mode() != "llama_cpp":
            response = openai_compatible_completion(
                messages, max_tokens=max_tokens, temperature=temperature,
                response_format={"type": "json_object"},
            )
        else:
            from llama_cpp import LlamaGrammar  # imported lazily for testability
            with llm_runtime.llm_lock:
                response = llm_runtime.llm.create_chat_completion(
                    messages=messages, max_tokens=max_tokens, temperature=temperature,
                    repeat_penalty=1.1, stream=False,
                    # llama-cpp-python expects a serialized JSON Schema here;
                    # passing the Python dict raises ``JSON object must be str``.
                    grammar=LlamaGrammar.from_json_schema(json.dumps(schema)),
                )
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"Structured JSON decoding failed: {error}") from error

    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("Structured decoder returned no text content")
    content = strip_think_tags(content)
    logger.info("Structured LLM response: %s", content)
    return content


def _run_langgraph_agent(system_prompt: str, task: str, session_id: str,
                         temperature: float, language: str, request_started: float,
                         initial_messages: list[dict[str, str]] | None = None,
                          initial_steps: list[dict] | None = None,
                          initial_iteration: int = 0,
                          resume_with_reflection: bool = False,
                          event_sink: Callable[[dict], None] | None = None,
                          prior_step_count: int = 0,
                          approval_granted: bool = False,
                          approval_scope: str = "",
                          supervisor_route: Specialist | None = None) -> dict:
    """Run the tool loop through LangGraph while preserving the Qt API response."""
    if not LANGGRAPH_AVAILABLE or LocalAgentGraph is None:
        raise HTTPException(
            status_code=503,
            detail="LangGraph is required for Agent mode. Run: pip install -r AIAssistant/requirements.txt",
        )
    if supervisor_route is None:
        supervisor_route = Specialist.SUPERVISOR

    def complete(messages: list[dict[str, str]], current_temperature: float) -> str:
        total_chars = sum(len(message.get("content", "")) for message in messages)
        estimated_tokens = int(total_chars / CHARS_PER_TOKEN)
        if estimated_tokens >= LLM_N_CTX - 512:
            return "Context quá dài, dừng Agent."
        max_tokens = min(2048, max(512, LLM_N_CTX - estimated_tokens - 400))

        def call(msgs: list[dict[str, str]]) -> str:
            return _constrained_agent_completion(msgs, max_tokens, current_temperature)

        print(f"[AGENT TRACE] ── LangGraph: Gọi LLM ({len(messages)} msgs, ~{estimated_tokens} tokens)", flush=True)
        logger.info("LangGraph gọi Model (messages: %d, estimated_tokens: %d)", len(messages), estimated_tokens)
        answer = call(messages)
        print(f"[AGENT TRACE] ── LangGraph: LLM output ({len(answer)} chars): {answer[:120].replace(chr(10), ' ')}", flush=True)
        logger.info("LangGraph nhận phản hồi từ Model (length: %d chars)", len(answer))

        # [FIX-13] Self-correction NGAY TRONG vòng lặp LangGraph: ở lượt suy
        # luận đầu tiên (messages chỉ gồm system+user, chưa có tool nào chạy),
        # nếu model trả lời bằng văn bản thường (không phát ```tool_call```)
        # trong khi câu hỏi của người dùng mang dáng dấp một lệnh điều khiển
        # UI (rule #11 trong system prompt), cho model MỘT cơ hội tự sửa bằng
        # một system reminder nhấn mạnh rule #11, trước khi chấp nhận đó là
        return answer

    def execute(tool_name: str, params: dict) -> dict:
        delegation = delegate(task, session_id, tool_name, params, prefer_code=is_coding_task(task))
        allowed, reason = authorise_delegation(delegation, tool_name in _TOOLS_REQUIRING_APPROVAL)
        if not allowed:
            audit_agent("tool_denied", delegation, reason=reason)
            return {"error": reason or "Tool call denied by supervisor policy."}
        if tool_name == "application_action":
            canonical_params, error = validate_action_params(params)
            if error:
                return {"error": error}
            canonical_params["request_id"] = _generate_action_id()
            params.clear()
            params.update(canonical_params)
        if tool_name == "_validation_error":
            return {"error": f"Lỗi xác thực tham số tool '{params.get('tool')}': {params.get('error')}"}
        executor = TOOL_REGISTRY.get(tool_name)  # get tool executor
        if executor is None:
            return {"error": f"Tool không tồn tại: {tool_name}"}
        tool_started = time.monotonic()

        def execute_local(_: str, __: str, ___: dict | None) -> dict:
            with span("agent.tool", tool=tool_name, session_id=session_id):
                return executor(params)

        if delegation.remote_endpoint and A2ARouter is not None:
            remote_payload = {
                "tool": tool_name,
                "parameters": params,
                "session_id": session_id,
                "idempotency_key": delegation.idempotency_key,
            }
            with span("agent.a2a_delegate", tool=tool_name,
                      specialist=delegation.specialist.value,
                      remote_endpoint=delegation.remote_endpoint):
                result = A2ARouter().route(
                    delegation.specialist.value, task, remote_payload,
                )
            audit_agent("tool_transport", delegation, source=result.get("source", "local"))
        else:
            result = execute_local(delegation.specialist.value, task, None)
        audit_agent("tool_completed", delegation, success="error" not in result)
        record_tool(tool_name, "error" not in result, time.monotonic() - tool_started)
        return result

    def select_specialist(tool_name: str, params: dict) -> dict:
        if tool_name == "_validation_error":
            return {}
        delegation = delegate(task, session_id, tool_name, params, prefer_code=is_coding_task(task))
        audit_agent("tool_delegated", delegation)
        return {
            "specialist": str(delegation.specialist),
            "idempotency_key": delegation.idempotency_key,
            "instruction": specialist_instruction(delegation),
            "remote_endpoint": delegation.remote_endpoint,
        }

    def verify_tool_result(tool_name: str, params: dict, result: dict) -> dict:
        if tool_name == "_validation_error":
            return {"passed": False, "reason": result.get("error", "Validation error")}
        delegation = delegate(task, session_id, tool_name, params, prefer_code=is_coding_task(task))
        verification = verify_result(delegation, result)
        audit_agent("tool_verified", delegation, **verification)
        return verification

    def deterministic_reflection(tool_name: str, params: dict, result: dict,
                                 verification: dict) -> dict:
        delegation = delegate(task, session_id, tool_name, params, prefer_code=is_coding_task(task))
        reflection = reflect_result(delegation, result, verification)
        audit_agent("tool_reflected", delegation, **reflection)
        return reflection

    logger.info("Khởi động LangGraph vòng lặp thực thi tool (session: %s)", session_id)
    print(f"[AGENT TRACE] ▶ LangGraph session={session_id} task={task[:80].replace(chr(10), ' ')}", flush=True)

    # LangSmith: wrap entire agent session as a top-level traced run.
    _ls_ctx_mgr = langsmith_trace(
        "agent.session",
        run_type="chain",
        inputs={"task": task[:200], "session_id": session_id,
                "supervisor_route": supervisor_route.value if supervisor_route else "none"},
        metadata={"session_id": session_id, "temperature": temperature,
                  "language": language},
    )
    _ls_ctx = _ls_ctx_mgr.__enter__()
    graph = LocalAgentGraph(
        complete=complete,      # Gọi model để sinh ra câu trả lời
        parse=_parse_tool_call, # Parse tool_call ra khỏi câu trả lời
        execute=execute,        # Thực thi tool
        needs_approval=lambda tool_name: tool_name in _TOOLS_REQUIRING_APPROVAL, # Kiểm tra xem có cần approval không
        max_iterations=_AGENT_MAX_ITERATIONS, # Số lần lặp tối đa
        emit=event_sink,
        select_specialist=select_specialist,
        verify_result=verify_tool_result,
        reflect_result=deterministic_reflection,
        plan_complete=lambda messages, temp: _structured_agent_completion(
            messages, 512, temp, _PLANNER_JSON_SCHEMA),
        reflect_complete=lambda messages, temp: _structured_agent_completion(
            messages, 512, temp, _CRITIC_JSON_SCHEMA),
        plan_reflect_complete=lambda messages, temp: _structured_agent_completion(
            messages, 512, temp, _CRITIC_JSON_SCHEMA),
        cancel_checker=lambda: task_coordinator.is_cancelled(session_id),
    )
    messages = initial_messages or [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    # An explicit manifest workflow is optional context only.  A single action
    # match is not promoted to a sequence; every plan step remains an LLM
    # tool-calling decision and is reviewed by Reflect.
    matched_ui_actions = []
    try:
        state = graph.run(messages, session_id, temperature, initial_steps, initial_iteration,
                          resume_with_reflection=resume_with_reflection,
                          required_ui_actions=matched_ui_actions,
                          supervisor_route=supervisor_route.value,
                          enforce_plan_completion=supervisor_route.value == "code",
                          approval_granted=approval_granted or bool((initial_steps or []) and
                                                any(step.get("type") == "approval_granted"
                                                    for step in (initial_steps or []))),
                          approval_scope=approval_scope)
    except BaseException as error:
        _ls_ctx["outputs"] = {"status": "error", "error": str(error)}
        _ls_ctx_mgr.__exit__(type(error), error, error.__traceback__)
        raise
    pending_state = state.get("pending_tool") or {}
    pending_status = ("pending_ui_action" if pending_state.get("ui_ack")
                      else "pending_approval" if pending_state else "completed")
    if state.get("cancelled"):
        pending_status = "cancelled"
    if pending_state:
        task_coordinator.update(session_id, status="waiting_approval" if not pending_state.get("ui_ack")
                                else "waiting_ui_ack", iterations=state.get("iteration", 0))
    else:
        task_coordinator.finish(session_id, success=not state.get("cancelled"),
                                iterations=state.get("iteration", 0))
    total_lg_ms = round((time.monotonic() - request_started) * 1000)
    print(f"[AGENT TRACE] ✓ Done (LangGraph) | iter={state.get('iteration', 0)} status={pending_status} | {total_lg_ms}ms", flush=True)
    logger.info("LangGraph hoàn thành vòng lặp execution (iteration: %d, status: %s)", state.get("iteration", 0), pending_status)

    # Close LangSmith session trace with final outputs.
    _ls_ctx["outputs"] = {
        "status": pending_status,
        "iterations": state.get("iteration", 0),
        "duration_ms": total_lg_ms,
        "step_count": len(state.get("steps", [])),
    }
    _ls_ctx_mgr.__exit__(None, None, None)
    steps = state["steps"]
    pending = state.get("pending_tool")
    if pending:
        action_id = _generate_action_id()
        is_ui_ack = bool(pending.get("ui_ack"))
        request_id = pending["params"].get("request_id", action_id)
        with _pending_lock:
            _pending_actions[request_id if is_ui_ack else action_id] = {
                "tool": pending["tool"],
                "params": pending["params"],
                "session_id": session_id,
                "task": task,
                "messages": state["messages"],
                "steps": steps,
                "iteration": state["iteration"],
                "temperature": temperature,
                "language": language,
                "created_at": time.time(),
                "ui_ack": is_ui_ack,
                "approval_scope": pending.get("approval_scope", ""),
                "approval_preview": pending.get("approval_preview"),
            }
            _save_pending_actions()
        if is_ui_ack:
            return {
                "status": "pending_ui_action", "session_id": session_id,
                "steps": steps, "request_id": request_id,
                "ui_action": {"request_id": request_id, "action": pending["params"]["action"],
                              "params": pending["params"]},
                "total_ms": round((time.monotonic() - request_started) * 1000),
            }
        steps.append({
            "type": "pending_approval",
            "action_id": action_id,
            "tool": pending["tool"],
            "params": pending["params"],
            "description": pending["params"].get("description", f"Thực thi {pending['tool']}"),
            "approval_scope": pending.get("approval_scope", ""),
            "preview": pending.get("approval_preview"),
        })
        return {
            "status": "pending_approval", "session_id": session_id,
            "steps": steps, "action_id": action_id,
            "prior_step_count": prior_step_count,
            "total_ms": round((time.monotonic() - request_started) * 1000),
        }

    if not any(step["type"] == "final_answer" for step in steps):
        steps.append({"type": "final_answer", "content": "Agent đã kết thúc mà chưa có kết luận."})
    return {
        "status": "completed", "session_id": session_id, "steps": steps,
        "prior_step_count": prior_step_count,
        "iterations": state["iteration"],
        "total_ms": round((time.monotonic() - request_started) * 1000),
    }


# ── Pydantic models for Agent ─────────────────────────────────────────────────

class AgentExecuteRequest(BaseModel):
    task:                str   = Field(..., min_length=1, max_length=4000)
    session_id:          str   = Field(default="")
    temperature:         float = Field(0.3, ge=0.0, le=1.5)
    language:            str   = Field(default="vi", pattern="^(vi|en)$")
    history:             list[dict] = Field(default_factory=list, max_length=20)
    attachments:         list[str]  = Field(default_factory=list, max_length=10)
    # force_langgraph=True buộc chạy qua LangGraph bất kể USE_LANGGRAPH_AGENT.
    # force_langgraph=False buộc chạy legacy loop. None = theo global setting.
    force_langgraph:     bool | None = Field(default=None)
    # retry_message_index: index của message trong lịch sử Qt đang được Retry.
    # Server echo lại để Qt biết insert/replace đúng vị trí.
    retry_message_index: int | None = Field(default=None)

class AgentApproveRequest(BaseModel):
    action_id:  str = Field(..., min_length=1)
    approved:   bool = Field(...)
    session_id: str = Field(default="")


class AgentUiActionResultRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    success: bool
    result: dict = Field(default_factory=dict)


class AgentCancelRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    request_id: str = Field(default="")


# ── Agent Endpoints ───────────────────────────────────────────────────────────

_load_pending_actions()


from fastapi import APIRouter  # noqa: E402,I001
from fastapi.responses import StreamingResponse  # noqa: E402,I001
agent_router = APIRouter()


def _agent_response(payload: dict, request: Request):
    """Negotiate JSON (Qt compatibility) or SSE on the same execute URL."""
    if "text/event-stream" not in request.headers.get("accept", ""):
        return payload

    def events():
        yield f"event: status\ndata: {json.dumps({'status': payload.get('status'), 'session_id': payload.get('session_id')})}\n\n"
        for step in payload.get("steps", []):
            yield f"event: step\ndata: {json.dumps(step, ensure_ascii=False)}\n\n"
        yield f"event: done\ndata: {json.dumps({key: value for key, value in payload.items() if key != 'steps'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _stream_langgraph_execution(run: Callable[[Callable[[dict], None]], dict]):
    """Stream LangGraph node/tool steps as they happen over SSE."""
    events: queue.Queue[tuple[str, object]] = queue.Queue()

    def worker() -> None:
        try:
            events.put(("result", run(lambda step: events.put(("step", step)))))
        except Exception as error:  # noqa: BLE001
            events.put(("error", str(error)))

    threading.Thread(target=worker, daemon=True, name="agent-sse").start()

    def stream():
        yield "event: status\ndata: {\"status\": \"running\"}\n\n"
        while True:
            kind, value = events.get()
            if kind == "step":
                yield f"event: step\ndata: {json.dumps(value, ensure_ascii=False)}\n\n"
            elif kind == "result":
                payload = value
                yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            else:
                yield f"event: error\ndata: {json.dumps({'detail': value}, ensure_ascii=False)}\n\n"
                return

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@agent_router.post("/v1/agent/execute")
def agent_execute(request: AgentExecuteRequest, http_req: Request):
    """
    Execute an agentic task with tool-calling loop.
    Returns a list of steps (tool_call, tool_result, thinking, final_answer, pending_approval).
    """
    _cleanup_pending_actions()
    if llm_runtime.llm is None:
        raise HTTPException(status_code=503, detail="LLM chưa khởi tạo")

    req_start = time.monotonic()
    task = request.task
    session_id = request.session_id or "agent_default"
    task_coordinator.start(session_id, task=request.task)
    
    retry_idx = request.retry_message_index

    print(f"[AGENT TRACE] ▶ Session={session_id} | LangGraph=ON | Task={task[:80].replace(chr(10), ' ')}", flush=True)
    logger.info("[MODE: AGENT] Task from %s: %s…", http_req.client.host, task[:80].replace("\n", " "))

    system_prompt = _build_agent_system_prompt(request.language)
    if is_coding_task(task):
        system_prompt += "\n\n" + coding_instruction(CodingTaskContext(
            task=task, language=request.language, project_root=_safe_relpath(PROJECT_DIR, PROJECT_DIR),
        ))

    # Agent mode obtains project evidence through the explicit ``rag_search``
    # tool. Do not eagerly append the same retrieval to the system prompt: it
    # duplicates the subsequent tool result and can exhaust the 8k context
    # window before the agent has a chance to answer from the retrieved data.

    history_messages: list[dict[str, str]] = []
    for entry in request.history:
        role = entry.get("role")
        content_msg = entry.get("content")
        if role in {"user", "assistant"} and isinstance(content_msg, str) and content_msg.strip():
            history_messages.append({"role": role, "content": content_msg[:32000]})

    task_with_attachments = task
    if request.attachments:
        names = [os.path.basename(path) for path in request.attachments]
        task_with_attachments += "\n\n[Attached files: " + ", ".join(names) + "]"

    if "text/event-stream" in http_req.headers.get("accept", ""):
        return _stream_langgraph_execution(
            lambda sink: _run_langgraph_agent(system_prompt, task_with_attachments, session_id,
                                               request.temperature, request.language, req_start,
                                               initial_messages=[{"role": "system", "content": system_prompt},
                                                                 *history_messages,
                                                                 {"role": "user", "content": task_with_attachments}],
                                               event_sink=sink, supervisor_route=Specialist.SUPERVISOR))
    result = _run_langgraph_agent(
        system_prompt, task_with_attachments, session_id, request.temperature, request.language, req_start,
        initial_messages=[{"role": "system", "content": system_prompt}, *history_messages,
                          {"role": "user", "content": task_with_attachments}], supervisor_route=Specialist.SUPERVISOR)
    
    if retry_idx is not None:
        result["retry_message_index"] = retry_idx
    return _agent_response(result, http_req)

@agent_router.post("/v1/agent/cancel")
def agent_cancel(request: AgentCancelRequest):
    """Request cooperative cancellation for a running session/request."""
    cancelled = task_coordinator.cancel(request.session_id, request.request_id)
    if cancelled is None:
        raise HTTPException(status_code=404, detail="Unknown or already finished agent task")
    return {"status": "cancelled", **cancelled}


@agent_router.post("/v1/agent/ui-action-result")
def agent_ui_action_result(request: AgentUiActionResultRequest):
    """Close the desktop-action loop after the Qt slot has run."""
    _cleanup_pending_actions()
    with _pending_lock:
        action = _pending_actions.pop(request.request_id, None)
        _save_pending_actions()
    if action is None or not action.get("ui_ack"):
        raise HTTPException(status_code=404, detail="Unknown or expired UI action request")

    params = action["params"]
    result = {"success": request.success, "action": params["action"], **request.result}
    steps = []
    steps.append({"type": "tool_result", "tool": "application_action", "request_id": request.request_id,
                  "result": result, "iteration": action.get("iteration", 0)})
    # The initial dispatch only proves that Qt received the request.  Verify the
    # actual ACK separately so reflect evaluates the desktop outcome, including
    # a failure reported by the client.
    delegation = delegate(action["task"], action["session_id"], "application_action", params)
    verification = verify_result(delegation, result)
    audit_agent("tool_verified", delegation, **verification)
    steps.append({"type": "verification", "tool": "application_action", "result": verification,
                  "iteration": action.get("iteration", 0)})



    # Nếu LLM (LangGraph) đang chạy, tiếp tục graph để thực hiện bước tiếp theo trong kế hoạch
    if action.get("messages") and USE_LANGGRAPH_AGENT and LANGGRAPH_AVAILABLE:
        messages = action["messages"]
        tool_call_text = json.dumps({"tool": "application_action", "params": params}, ensure_ascii=False)
        messages.append({"role": "assistant", "content": f"```tool_call\n{tool_call_text}\n```"})
        
        result_text = json.dumps(result, ensure_ascii=False, indent=2)
        if len(result_text) > 8000:
            result_text = result_text[:8000] + "\n... [truncated]"
        messages.append({
            "role": "user",
            "content": f"Tool `application_action` returned:\n```json\n{result_text}\n```\n\nPhân tích kết quả. NẾU kế hoạch của bạn CÒN bước tiếp theo, hãy bắt buộc GỌI TOOL cho bước đó ngay lập tức (KHÔNG HỎI LẠI NGƯỜI DÙNG). Nếu đã hoàn thành toàn bộ, đưa ra thông báo kết thúc."
        })
        
        system_prompt = messages[0]["content"] if messages and messages[0].get("role") == "system" else _build_agent_system_prompt(action.get("language", "vi"))
        return _run_langgraph_agent(
            system_prompt=system_prompt,
            task=action["task"],
            session_id=action["session_id"],
            temperature=action["temperature"],
            language=action.get("language", "vi"),
            request_started=time.monotonic(),
            initial_messages=messages,
            initial_steps=action.get("steps", []) + steps,
            initial_iteration=action.get("iteration", 0),
            resume_with_reflection=True,
            prior_step_count=len(action.get("steps", [])),
        )

    content = (f"Đã thực thi {params['action']}." if request.success
               else f"Không thể thực thi {params['action']}: {result.get('error', 'unknown error')}")
    steps.append({"type": "final_answer", "content": content})
    all_steps = action.get("steps", []) + steps
    task_coordinator.finish(action["session_id"], success=request.success)
    retry_idx_stored = action.get("retry_message_index")
    return {
        "status": "completed" if request.success else "failed",
        "session_id": action["session_id"],
        "request_id": request.request_id,
        "prior_step_count": len(action.get("steps", [])),
        "steps": all_steps,
        **({"retry_message_index": retry_idx_stored} if retry_idx_stored is not None else {}),
    }


@agent_router.post("/v1/agent/approve")
def agent_approve(request: AgentApproveRequest, http_req: Request):
    """
    Approve or reject a pending agent action (write_file, run_command).
    If approved, executes the action and resumes the agent loop.
    """
    _cleanup_pending_actions()
    if llm_runtime.llm is None:
        raise HTTPException(status_code=503, detail="LLM chưa khởi tạo")

    action_id = request.action_id
    with _pending_lock:
        action = _pending_actions.pop(action_id, None)
        _save_pending_actions()

    if action is None:
        record_approval("missing")
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")

    if task_coordinator.is_cancelled(action.get("session_id", "")):
        task_coordinator.finish(action.get("session_id", ""), success=False)
        return {"status": "cancelled", "action_id": action_id,
                "steps": [*action.get("steps", []), {
                    "type": "cancelled", "content": "Task cancelled before approval."}]}

    if request.session_id and request.session_id != action.get("session_id"):
        with _pending_lock:
            _pending_actions[action_id] = action
            _save_pending_actions()
        record_approval("unauthorized")
        raise HTTPException(status_code=403, detail="Action does not belong to this session")
    if not request.approved:
        record_approval("rejected")
        # User rejected
        return {
            "status": "rejected",
            "action_id": action_id,
            "approval_scope": action.get("approval_scope", ""),
            "approval_preview": action.get("approval_preview"),
            "prior_step_count": len(action["steps"]),
            "steps": action["steps"] + [{
                "type": "tool_result",
                "tool": action["tool"],
                "action_id": action_id,
                "result": {"rejected": True, "message": "Người dùng từ chối thực thi action này."},
                "iteration": action["iteration"],
            }],
        }

    # Execute the approved action
    record_approval("approved")
    tool_name = action["tool"]
    tool_params = action["params"]

    approved_tool_started = time.monotonic()
    tool_result = execute_approved_platform_tool(tool_name, tool_params)
    record_tool(tool_name, "error" not in tool_result, time.monotonic() - approved_tool_started)

    prior_step_count = len(action["steps"])
    steps = action["steps"]
    steps.append({
        "type": "tool_result",
        "tool": tool_name,
        "action_id": action_id,
        "result": tool_result,
        "iteration": action["iteration"],
    })
    delegation = delegate(action["task"], action["session_id"], tool_name, tool_params)
    verification = verify_result(delegation, tool_result)
    audit_agent("tool_verified", delegation, **verification)
    steps.append({
        "type": "verification", "tool": tool_name, "result": verification,
        "iteration": action["iteration"],
    })

    # Resume agent loop with remaining context
    messages = action["messages"]
    # Add the tool call and result to messages
    tool_call_text = json.dumps({"tool": tool_name, "params": tool_params}, ensure_ascii=False)
    messages.append({"role": "assistant", "content": f"```tool_call\n{tool_call_text}\n```"})

    result_text = json.dumps(tool_result, ensure_ascii=False, indent=2)
    if len(result_text) > 8000:
        result_text = result_text[:8000] + "\n... [truncated]"
    messages.append({
        "role": "user",
        "content": f"Tool `{tool_name}` was approved and executed. Result:\n```json\n{result_text}\n```\n\nContinue with your analysis or provide final answer.",
    })

    if USE_LANGGRAPH_AGENT and LANGGRAPH_AVAILABLE:
        system_prompt = messages[0]["content"] if messages and messages[0].get("role") == "system" else _build_agent_system_prompt(action.get("language", "vi"))
        steps.append({"type": "approval_granted", "scope_id": action.get("approval_scope", ""),
                      "tool": tool_name, "action_id": action_id})
        return _run_langgraph_agent(
            system_prompt=system_prompt,
            task=action["task"],
            session_id=action["session_id"],
            temperature=action["temperature"],
            language=action.get("language", "vi"),
            request_started=time.monotonic(),
            initial_messages=messages,
            initial_steps=steps,
            initial_iteration=action["iteration"],
            resume_with_reflection=True,
            prior_step_count=prior_step_count,
            approval_granted=True,
            approval_scope=action.get("approval_scope", ""),
        )

    # Continue the agent loop
    iteration = action["iteration"]
    temperature = action["temperature"]
    req_start = time.monotonic()
    approval_already_granted = True

    while iteration < _AGENT_MAX_ITERATIONS:
        iteration += 1

        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = int(total_chars / CHARS_PER_TOKEN)
        available_tokens = LLM_N_CTX - estimated_tokens - 400
        max_tokens = min(2048, max(512, available_tokens))

        if estimated_tokens >= LLM_N_CTX - 512:
            steps.append({"type": "error", "content": "Context quá dài."})
            break

        try:
            answer = _constrained_agent_completion(messages, max_tokens, temperature)
        except Exception as e:
            steps.append({"type": "error", "content": f"Lỗi LLM: {e}"})
            break

        answer = answer.strip()
        if not answer:
            break

        tool_name_next, tool_params_next = _parse_tool_call(answer)

        if tool_name_next == "application_action":
            canonical_params_next = _canonical_desktop_action(tool_params_next)
            if canonical_params_next is not None:
                tool_params_next = canonical_params_next

        if tool_name_next is None:
            steps.append({"type": "final_answer", "content": answer})
            break

        clean_answer = strip_think_tags(answer)
        think_text = clean_answer
        if "```tool_call" in clean_answer:
            think_text = clean_answer.split("```tool_call")[0].strip()
        elif "{" in clean_answer:
            think_text = clean_answer.split("{")[0].strip()
            
        if think_text:
            steps.append({"type": "thinking", "content": think_text, "iteration": iteration})

        steps.append({
            "type": "tool_call",
            "tool": tool_name_next,
            "params": tool_params_next,
            "iteration": iteration,
        })

        if tool_name_next in _TOOLS_REQUIRING_APPROVAL and not approval_already_granted:
            new_action_id = _generate_action_id()
            with _pending_lock:
                _pending_actions[new_action_id] = {
                    "tool": tool_name_next,
                    "params": tool_params_next,
                    "session_id": action["session_id"],
                    "task": action["task"],
                    "messages": messages.copy(),
                    "steps": steps.copy(),
                    "iteration": iteration,
                    "temperature": temperature,
                    "created_at": time.time(),
                    "approval_scope": action.get("approval_scope", ""),
                }
                _save_pending_actions()
            task_coordinator.update(action["session_id"], new_action_id, status="waiting_approval")

            steps.append({
                "type": "pending_approval",
                "action_id": new_action_id,
                "tool": tool_name_next,
                "params": tool_params_next,
                "description": tool_params_next.get("description", f"Thực thi {tool_name_next}"),
            })

            total_ms = (time.monotonic() - req_start) * 1000
            return {
                "status": "pending_approval",
                "session_id": action["session_id"],
                "prior_step_count": prior_step_count,
                "steps": steps,
                "action_id": new_action_id,
                "total_ms": round(total_ms),
            }

        if tool_name_next == "_validation_error":
            tool_result_next = {"error": f"Lỗi xác thực tham số tool '{tool_params_next.get('tool')}': {tool_params_next.get('error')}"}
        elif tool_name_next in TOOL_REGISTRY.names():
            try:
                tool_result_next = TOOL_REGISTRY.execute(tool_name_next, tool_params_next)
            except Exception as e:
                tool_result_next = {"error": f"Tool exception: {e}"}
        else:
            tool_result_next = {"error": f"Tool không tồn tại: {tool_name_next}"}

        steps.append({
            "type": "tool_result",
            "tool": tool_name_next,
            "result": tool_result_next,
            "iteration": iteration,
        })

        messages.append({"role": "assistant", "content": answer})
        result_text_next = json.dumps(tool_result_next, ensure_ascii=False, indent=2)
        if len(result_text_next) > 8000:
            result_text_next = result_text_next[:8000] + "\n... [truncated]"
        messages.append({
            "role": "user",
            "content": f"Tool `{tool_name_next}` returned:\n```json\n{result_text_next}\n```\n\nContinue.",
        })

    if not any(s["type"] == "final_answer" for s in steps):
        steps.append({
            "type": "final_answer",
            "content": "⚠️ Agent đã đạt giới hạn iterations.",
        })

    total_ms = (time.monotonic() - req_start) * 1000
    return {
        "status": "completed",
        "session_id": action["session_id"],
        "prior_step_count": prior_step_count,
        "steps": steps,
        "iterations": iteration,
        "total_ms": round(total_ms),
    }


# Cleanup expired pending actions (older than 10 minutes)
def _cleanup_pending_actions():
    cutoff = time.time() - 600
    with _pending_lock:
        had_expired = _pending_actions.cleanup(cutoff)
        if had_expired:
            _save_pending_actions()
            logger.info("Cleaned up expired pending agent actions")


def reset_agent_state() -> None:
    """Forget all pending approvals and their persisted state."""
    with _pending_lock:
        _pending_actions.clear()
        try:
            if os.path.exists(_PENDING_ACTIONS_FILE):
                os.remove(_PENDING_ACTIONS_FILE)
        except OSError as error:
            logger.warning("Unable to remove pending action state: %s", error)


