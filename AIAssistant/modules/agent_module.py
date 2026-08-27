# ruff: noqa: I001
import fnmatch
import queue
import subprocess
from collections.abc import Callable

from .config import *
from .config import _safe_relpath
from . import llm_module as llm_runtime
from . import rag_module as rag_runtime
from .action_manifest import (
    canonicalise_action_params,
    looks_like_ui_action as _manifest_looks_like_ui_action,
    match_action_intent as _manifest_match_action_intent,
    match_action_sequence as _manifest_match_action_sequence,
    normalise_text as _normalise_agent_task,
    validate_action_params,
)
from .tool_contract import build_tool_models, grammar_schema, openai_tools, validate_tool_call
from .sandbox import run as run_sandboxed_command, write_file as write_sandboxed_file
from .observability import record_tool, span
from .inference import backend_mode, openai_compatible_completion
from .mcp_client import call_tool as call_mcp_tool
from .multi_agent import (
    audit as audit_agent,
    authorise as authorise_delegation,
    delegate,
    reflect_result,
    specialist_instruction,
    verify_result,
)

# ── Tool Definitions (mô tả cho LLM) ──────────────────────────────────────────
AGENT_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the content of a file. Returns file content as text. "
                       "Use this to examine source code, configs, documentation, etc.",
        "parameters": {
            "path": {"type": "string", "description": "Relative path from project root (e.g. 'src/main.cpp')", "required": True},
            "start_line": {"type": "integer", "description": "Start line (1-indexed, inclusive). Omit to read from beginning.", "required": False},
            "end_line": {"type": "integer", "description": "End line (1-indexed, inclusive). Omit to read to end.", "required": False},
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
            "timeout": {"type": "integer", "description": "Timeout in seconds. Default 30.", "required": False},
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
                       "Supported actions: viewer.load_2d, viewer.load_3d, viewer.load_dicom. "
                       "DO NOT use this tool for viewing reconstructed models. Use it for loading/opening external static files.",
        "parameters": {
            "action": {"type": "string", "description": "One of: viewer.load_2d, viewer.load_3d, viewer.load_dicom", "required": True},
        },
    },
    {
        "name": "app_action_reconstruction",
        "description": "Run a 3D Reconstruction desktop action. Use this when the user wants to perform 3D reconstruction tasks: load source images for reconstruction, start the reconstruction process, or view/close the RESULTING 3D point cloud model. "
                       "Supported actions: reconstruction.load_images, reconstruction.start_reconstruction, reconstruction.view_3d_model, reconstruction.close_3d_model. "
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
        "description": "Search project documentation and source code using semantic similarity. "
                       "Use this to find relevant technical information, APIs, patterns, or code examples "
                       "before reading specific files. More efficient than read_file when you don't know exactly which file to look at.",
        "parameters": {
            "query": {"type": "string", "description": "The question or topic to search for", "required": True},
            "top_k": {"type": "integer", "description": "Max number of results to return (default 5, max 10)", "required": False},
        },
    },
]

# ``application_action`` is the sole public UI tool.  The former category
# names remain parser aliases for old conversations, but are not model tools.
AGENT_TOOLS = [tool for tool in AGENT_TOOLS if not tool["name"].startswith("app_action_")]
AGENT_TOOLS.append({
    "name": "application_action",
    "description": "Execute a canonical desktop action from the shared action manifest.",
    "parameters": {
        "action": {"type": "string", "description": "Canonical desktop action id", "required": True},
        "language": {"type": "string", "description": "Required only for language.change: vi or en", "required": False},
        "username": {"type": "string", "description": "Optional login username", "required": False},
        "password": {"type": "string", "description": "Optional login password", "required": False},
    },
})
_TOOL_PARAM_MODELS = build_tool_models(AGENT_TOOLS)
_LLAMA_CPP_TOOLS = openai_tools(AGENT_TOOLS)
_TOOL_GRAMMAR_SCHEMA = grammar_schema(AGENT_TOOLS)

# ── Safety: các thư mục/file cấm truy cập ────────────────────────────────────
_AGENT_BLOCKED_DIRS = {".git", "build", "__pycache__", ".vs", "node_modules"}
_AGENT_BLOCKED_EXTS = {".exe", ".dll", ".so", ".bin", ".dat", ".pkl", ".gguf", ".onnx", ".pt"}
_AGENT_MAX_FILE_READ_CHARS = 50000  # ~25K tokens
_AGENT_MAX_ITERATIONS = 12


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
    # Ensure within project
    if not abs_path.startswith(os.path.normpath(PROJECT_DIR)):
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
        start = max(1, params.get("start_line", 1)) - 1  # 0-indexed
        end = min(total_lines, params.get("end_line", total_lines))

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
    max_entries = 500

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

    file_pattern = params.get("file_pattern", "*")
    case_sensitive = params.get("case_sensitive", True)
    max_results = min(params.get("max_results", 50), 100)

    results = []
    search_query = query if case_sensitive else query.lower()
    text_exts = {".cpp", ".h", ".py", ".md", ".txt", ".cmake", ".json", ".xml", ".html", ".css", ".js", ".ts", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".bat", ".sh"}

    try:
        for root, dirs, files in os.walk(abs_search):
            dirs[:] = sorted(d for d in dirs if d not in _AGENT_BLOCKED_DIRS)
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in text_exts:
                    continue
                if file_pattern != "*" and not fnmatch.fnmatch(filename, file_pattern):
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
    functions = []
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
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only top-level functions (not methods)
            if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree)):
                args = [a.arg for a in node.args.args]
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": args,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
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
    return _manifest_looks_like_ui_action(text)


def _match_desktop_action(task: str) -> dict | None:
    return _manifest_match_action_intent(task)


def _match_desktop_action_sequence(task: str) -> list[dict] | None:
    return _manifest_match_action_sequence(task)


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
        with open(abs_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content.replace(find_text, replacement, 1))
        return {"success": True, "path": path, "replacements": 1}
    except OSError as error:
        return {"error": f"Unable to patch file: {error}"}


def _execute_approved_create_directory(params: dict) -> dict:
    path = params.get("path", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None:
        return {"error": f"Invalid directory path: {path}"}
    try:
        existed = os.path.isdir(abs_path)
        os.makedirs(abs_path, exist_ok=True)
        return {"success": True, "path": path, "created": not existed}
    except OSError as error:
        return {"error": f"Unable to create directory: {error}"}


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
    "search_text":       tool_search_text,
    "analyze_code":      tool_analyze_code,
    "get_project_status": tool_get_project_status,
"validate_file":                tool_validate_file,
"application_action":             tool_application_action,
    "app_action_viewer":         tool_application_action,
    "app_action_reconstruction": tool_application_action,
    "app_action_ai":             tool_application_action,
    "app_action_general":        tool_application_action,
    "rag_search":                tool_rag_search,
    # write_file, run_command, patch_file, create_directory require approval
}

# Safe tools use the local Streamable HTTP MCP endpoint. Approval-gated write
# tools do not enter this map and remain behind the existing approval flow.
_TOOL_EXECUTORS = {
    name: (lambda params, tool_name=name: call_mcp_tool(tool_name, params))
    for name in MCP_LOCAL_EXECUTORS
}

_TOOLS_REQUIRING_APPROVAL = {"write_file", "run_command", "patch_file", "create_directory"}

# ── Pending actions storage (in-memory, per session) ──────────────────────────
_pending_actions: dict = {}  # action_id -> {tool, params, session_id}
_pending_lock = threading.Lock()
_PENDING_ACTIONS_FILE = os.path.join(APP_DATA_DIR, "AIAssistant", "pending_agent_actions.json")


def _save_pending_actions() -> None:
    """Persist pending approvals so a server restart does not invalidate the UI action."""
    os.makedirs(os.path.dirname(_PENDING_ACTIONS_FILE), exist_ok=True)
    temp_file = _PENDING_ACTIONS_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as handle:
        json.dump(_pending_actions, handle, ensure_ascii=False)
    os.replace(temp_file, _PENDING_ACTIONS_FILE)


def _load_pending_actions() -> None:
    if not os.path.exists(_PENDING_ACTIONS_FILE):
        return
    try:
        with open(_PENDING_ACTIONS_FILE, "r", encoding="utf-8") as handle:
            saved_actions = json.load(handle)
        cutoff = time.time() - 600
        _pending_actions.update({
            action_id: action for action_id, action in saved_actions.items()
            if action.get("created_at", 0) >= cutoff
        })
    except (OSError, ValueError, TypeError) as error:
        logger.warning("Unable to restore pending agent actions: %s", error)


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
    if abs_cwd is None:
        abs_cwd = PROJECT_DIR

    return run_sandboxed_command(command, abs_cwd, timeout)


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

## EXAMPLE:

User: đổi project sang tiếng việt giúp tôi
Assistant: {{"kind":"tool","tool":"application_action","params":{{"action":"language.change","language":"vi"}}}}

User: tải ảnh chụp và tái tạo 3d
Assistant: {{"kind":"tool","tool":"application_action","params":{{"action":"reconstruction.load_images"}}}}

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
        return "_validation_error", {"tool": tool_name, "error": error}
    return tool_name, validated


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
        from llama_cpp import LlamaGrammar  # imported lazily for testability
        kwargs = {
            "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "repeat_penalty": 1.1, "stream": False,
        }
        native_tools = os.getenv("AGENT_NATIVE_TOOL_CALLS", "0") == "1"
        if native_tools:
            kwargs.update({"tools": _LLAMA_CPP_TOOLS, "tool_choice": "auto"})
        else:
            kwargs["grammar"] = LlamaGrammar.from_json_schema(_TOOL_GRAMMAR_SCHEMA)
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
    logger.info("Constrained LLM response: %s", content)
    try:
        envelope = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("Constrained decoder returned invalid JSON") from error
    if envelope.get("kind") == "final" and isinstance(envelope.get("content"), str):
        return envelope["content"]
    if envelope.get("kind") == "tool":
        return json.dumps(envelope, ensure_ascii=False)
    raise RuntimeError("Constrained decoder returned an unsupported envelope")


_PLANNER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "requires_plan": {"type": "boolean"},
        "plan": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["requires_plan", "plan"],
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
    logger.info("Structured LLM response: %s", content)
    return content


def _run_langgraph_agent(system_prompt: str, task: str, session_id: str,
                         temperature: float, language: str, request_started: float,
                         initial_messages: list[dict[str, str]] | None = None,
                          initial_steps: list[dict] | None = None,
                          initial_iteration: int = 0,
                          resume_with_reflection: bool = False,
                          event_sink: Callable[[dict], None] | None = None) -> dict:
    """Run the tool loop through LangGraph while preserving the Qt API response."""
    if not LANGGRAPH_AVAILABLE or LocalAgentGraph is None:
        raise HTTPException(
            status_code=503,
            detail="LangGraph is required for Agent mode. Run: pip install -r AIAssistant/requirements.txt",
        )

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
        # final_answer. Quyết định gọi tool cuối cùng vẫn hoàn toàn do model
        # đưa ra qua đúng cơ chế parse/execute của LangGraph — không bypass.
        if len(messages) == 2:
            tool_name, _ = _parse_tool_call(answer)
            user_task = messages[-1].get("content", "")
            if tool_name is None and _looks_like_ui_action(_normalise_agent_task(user_task)):
                logger.info("LangGraph: chưa thấy tool_call ở lượt đầu nhưng task giống lệnh UI — nhắc lại rule #11 và thử lại")
                reminder = {
                    "role": "system",
                    "content": ("Nhắc lại RULE #11: đây là một yêu cầu điều khiển ứng dụng (application UI "
                                "request). Bạn PHẢI trả lời CHÍNH XÁC bằng một khối ```tool_call``` gọi "
                                "application_action. KHÔNG được trả lời bằng văn bản thường, KHÔNG được dịch "
                                "hay diễn giải bất kỳ nội dung nào — chỉ trả về đúng JSON tool_call theo format "
                                "đã hướng dẫn."),
                }
                retry_answer = call([*messages, reminder])
                logger.info("LangGraph nhận phản hồi từ Model sau khi nhắc lại (length: %d chars)", len(retry_answer))
                retry_tool_name, _ = _parse_tool_call(retry_answer)
                if retry_tool_name is not None:
                    logger.info("LangGraph: model đã tự sửa và phát tool_call ở lần thử lại")
                    return retry_answer
                logger.info("LangGraph: model vẫn không phát tool_call sau khi nhắc — giữ nguyên câu trả lời gốc")

        return answer

    def execute(tool_name: str, params: dict) -> dict:
        delegation = delegate(task, session_id, tool_name, params)
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
        executor = _TOOL_EXECUTORS.get(tool_name)
        if executor is None:
            return {"error": f"Tool không tồn tại: {tool_name}"}
        with span("agent.tool", tool=tool_name, session_id=session_id):
            result = executor(params)
        audit_agent("tool_completed", delegation, success="error" not in result)
        record_tool(tool_name, "error" not in result)
        return result

    def select_specialist(tool_name: str, params: dict) -> dict:
        if tool_name == "_validation_error":
            return {}
        delegation = delegate(task, session_id, tool_name, params)
        audit_agent("tool_delegated", delegation)
        return {
            "specialist": str(delegation.specialist),
            "idempotency_key": delegation.idempotency_key,
            "instruction": specialist_instruction(delegation),
        }

    def verify_tool_result(tool_name: str, params: dict, result: dict) -> dict:
        if tool_name == "_validation_error":
            return {"passed": False, "reason": result.get("error", "Validation error")}
        delegation = delegate(task, session_id, tool_name, params)
        verification = verify_result(delegation, result)
        audit_agent("tool_verified", delegation, **verification)
        return verification

    def deterministic_reflection(tool_name: str, params: dict, result: dict,
                                 verification: dict) -> dict:
        delegation = delegate(task, session_id, tool_name, params)
        reflection = reflect_result(delegation, result, verification)
        audit_agent("tool_reflected", delegation, **reflection)
        return reflection

    logger.info("Khởi động LangGraph vòng lặp thực thi tool (session: %s)", session_id)
    print(f"[AGENT TRACE] ▶ LangGraph session={session_id} task={task[:80].replace(chr(10), ' ')}", flush=True)
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
    )
    messages = initial_messages or [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    # Normal UI requests use the deterministic fast-path. When LangGraph is
    # explicitly forced, preserve the same canonical routing constraint so the
    # LLM cannot substitute repository research for a desktop command.
    matched_ui_actions = _match_desktop_action_sequence(_normalise_agent_task(task))
    if not matched_ui_actions:
        matched_ui_action = _match_desktop_action(_normalise_agent_task(task))
        matched_ui_actions = [matched_ui_action] if matched_ui_action else []
    state = graph.run(messages, session_id, temperature, initial_steps, initial_iteration,
                      resume_with_reflection=resume_with_reflection,
                      required_ui_actions=matched_ui_actions)
    pending_state = state.get("pending_tool") or {}
    pending_status = ("pending_ui_action" if pending_state.get("ui_ack")
                      else "pending_approval" if pending_state else "completed")
    total_lg_ms = round((time.monotonic() - request_started) * 1000)
    print(f"[AGENT TRACE] ✓ Done (LangGraph) | iter={state.get('iteration', 0)} status={pending_status} | {total_lg_ms}ms", flush=True)
    logger.info("LangGraph hoàn thành vòng lặp execution (iteration: %d, status: %s)", state.get("iteration", 0), pending_status)
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
        })
        return {
            "status": "pending_approval", "session_id": session_id,
            "steps": steps, "action_id": action_id,
            "total_ms": round((time.monotonic() - request_started) * 1000),
        }

    if not any(step["type"] == "final_answer" for step in steps):
        steps.append({"type": "final_answer", "content": "Agent đã kết thúc mà chưa có kết luận."})
    return {
        "status": "completed", "session_id": session_id, "steps": steps,
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
    retry_idx  = request.retry_message_index  # None ở request thường

    # Đưa ra quyết định dùng LangGraph hay legacy loop
    force_langgraph = FORCE_LANGGRAPH_AGENT
    if request.force_langgraph is not None:
        force_langgraph = request.force_langgraph

    if force_langgraph:
        use_langgraph = LANGGRAPH_AVAILABLE
    else:
        use_langgraph = USE_LANGGRAPH_AGENT and LANGGRAPH_AVAILABLE

    print(f"[AGENT TRACE] ▶ Session={session_id} | LangGraph={'ON' if use_langgraph else 'OFF'} | Task={task[:80].replace(chr(10), ' ')}", flush=True)
    logger.info("[MODE: AGENT] Task from %s: %s…", http_req.client.host, task[:80].replace("\n", " "))

    # Fast-path key matching: deterministic, khong can LLM.
    # Bỏ qua khi force_langgraph=True để ép LLM tự suy luận (dùng để test độ chính xác).
    desktop_sequence = _match_desktop_action_sequence(task)
    if desktop_sequence:
        workflow_id = _generate_action_id()
        for workflow_params in desktop_sequence:
            workflow_params["workflow_id"] = workflow_id
    desktop_params = (desktop_sequence or [_match_desktop_action(task)])[0]
    if desktop_params and not force_langgraph:
        print(f"[AGENT TRACE] ⚡ Fast-path match: {desktop_params.get('action', '?')} (bypass LLM)", flush=True)
        desktop_params, error = validate_action_params(desktop_params)
        if error:
            raise HTTPException(status_code=422, detail=error)
        request_id = _generate_action_id()
        desktop_params["request_id"] = request_id
        with _pending_lock:
            _pending_actions[request_id] = {
                "tool": "application_action", "params": desktop_params,
                "session_id": session_id, "task": task, "messages": [], "steps": [],
                "iteration": 0, "temperature": request.temperature, "language": request.language,
                "created_at": time.time(), "ui_ack": True,
                "next_actions": (desktop_sequence or [])[1:],
                "retry_message_index": retry_idx,
            }
            _save_pending_actions()
        return _agent_response({
            "status": "pending_ui_action",
            "session_id": session_id,
            "steps": [
                {"type": "tool_call", "tool": "application_action", "params": desktop_params, "iteration": 0},
            ],
            "request_id": request_id,
            "ui_action": {"request_id": request_id, "action": desktop_params["action"], "params": desktop_params},
            "total_ms": round((time.monotonic() - req_start) * 1000),
            **({"retry_message_index": retry_idx} if retry_idx is not None else {}),
        }, http_req)
    elif desktop_params and force_langgraph:
        print(f"[AGENT TRACE] ⚡ Fast-path match ignored (force_langgraph=True): {desktop_params.get('action', '?')} → going to LangGraph", flush=True)

    # Build initial messages. The desktop owns persisted history; retain only
    # conversational roles here so JSON Agent-step snapshots never reach LLM.
    system_prompt = _build_agent_system_prompt(request.language)

    # RAG nay duoc cung cap nhu mot tool dong (rag_search) thay vi inject vao system prompt.
    # Chi inject mot luong nho context khi task KHONG phai UI action va RAG san sang,
    # de tranh lam day context window voi thong tin khong lien quan.
    if ENABLE_RAG and rag_runtime.knowledge_chunks and not _looks_like_ui_action(_normalise_agent_task(task)):
        try:
            doc_ctx, code_ctx, _ = rag_runtime.get_context(task)
            if doc_ctx:
                system_prompt += f"\n\n## RELEVANT DOCUMENTATION (tu khoa tim kiem: {task[:60]}):\n{doc_ctx[:1500]}"
            if code_ctx:
                system_prompt += f"\n\n## RELEVANT CODE (tu khoa tim kiem: {task[:60]}):\n{code_ctx[:1500]}"
        except Exception:  # noqa: BLE001
            pass

    history_messages: list[dict[str, str]] = []
    for entry in request.history:
        role = entry.get("role")
        content = entry.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            history_messages.append({"role": role, "content": content[:32000]})

    task_with_attachments = task
    if request.attachments:
        names = [os.path.basename(path) for path in request.attachments]
        task_with_attachments += "\n\n[Attached files: " + ", ".join(names) + "]"

    if use_langgraph:
        if "text/event-stream" in http_req.headers.get("accept", ""):
            return _stream_langgraph_execution(
                lambda sink: _run_langgraph_agent(system_prompt, task_with_attachments, session_id,
                                                   request.temperature, request.language, req_start,
                                                   initial_messages=[{"role": "system", "content": system_prompt},
                                                                     *history_messages,
                                                                     {"role": "user", "content": task_with_attachments}],
                                                   event_sink=sink))
        result = _run_langgraph_agent(
            system_prompt, task_with_attachments, session_id, request.temperature, request.language, req_start,
            initial_messages=[{"role": "system", "content": system_prompt}, *history_messages,
                              {"role": "user", "content": task_with_attachments}])
        if retry_idx is not None:
            result["retry_message_index"] = retry_idx
        return _agent_response(result, http_req)

    messages = [
        {"role": "system", "content": system_prompt},
        *history_messages,
        {"role": "user", "content": task_with_attachments},
    ]

    steps = []
    iteration = 0

    while iteration < _AGENT_MAX_ITERATIONS:
        iteration += 1

        # Estimate tokens and calculate budget
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = int(total_chars / CHARS_PER_TOKEN)
        available_tokens = LLM_N_CTX - estimated_tokens - 400
        max_tokens = min(2048, max(512, available_tokens))

        if estimated_tokens >= LLM_N_CTX - 512:
            steps.append({
                "type": "error",
                "content": "Context quá dài, dừng agent loop.",
            })
            break

        # Call LLM
        print(f"[AGENT TRACE] ── Iter {iteration}/{_AGENT_MAX_ITERATIONS}: Gọi LLM (~{estimated_tokens} tokens)", flush=True)
        logger.info("Agent iter %d/%d | tokens≈%d", iteration, _AGENT_MAX_ITERATIONS, estimated_tokens)

        try:
            answer = _constrained_agent_completion(messages, max_tokens, request.temperature)
        except Exception as e:
            logger.error("Agent LLM error at iter %d: %s", iteration, e)
            print(f"[AGENT TRACE] ── Iter {iteration}: Lỗi LLM: {e}", flush=True)
            steps.append({"type": "error", "content": f"Lỗi LLM: {e}"})
            break

        answer = answer.strip()
        print(f"[AGENT TRACE] ── Iter {iteration}: LLM output ({len(answer)} chars): {answer[:120].replace(chr(10), ' ')}", flush=True)
        if not answer:
            steps.append({"type": "error", "content": "LLM trả về response rỗng."})
            break

        # Parse: is it a tool call or final answer?
        tool_name, tool_params = _parse_tool_call(answer)

        if tool_name == "application_action":
            canonical_params = _canonical_desktop_action(tool_params)
            if canonical_params is not None:
                tool_params = canonical_params

        if tool_name is None:
            # Final answer — no more tool calls
            print(f"[AGENT TRACE] ── Iter {iteration}: Final answer", flush=True)
            steps.append({
                "type": "final_answer",
                "content": answer,
            })
            break

        think_text = answer
        if "```tool_call" in answer:
            think_text = answer.split("```tool_call")[0].strip()
        elif "{" in answer:
            think_text = answer.split("{")[0].strip()
            
        if think_text:
            steps.append({"type": "thinking", "content": think_text, "iteration": iteration})

        print(f"[AGENT TRACE] ── Iter {iteration}: Tool call → {tool_name}({str(tool_params)[:80]})", flush=True)
        # It's a tool call
        steps.append({
            "type": "tool_call",
            "tool": tool_name,
            "params": tool_params,
            "iteration": iteration,
        })

        if tool_name == "application_action":
            tool_params["request_id"] = _generate_action_id()
            request_id = tool_params["request_id"]
            with _pending_lock:
                _pending_actions[request_id] = {
                    "tool": tool_name, "params": tool_params, "session_id": session_id,
                    "task": task, "messages": messages.copy(), "steps": steps.copy(),
                    "iteration": iteration, "temperature": request.temperature,
                    "language": request.language, "created_at": time.time(), "ui_ack": True,
                    "retry_message_index": retry_idx,
                }
                _save_pending_actions()
            return _agent_response({"status": "pending_ui_action", "session_id": session_id,
                "request_id": request_id, "steps": steps,
                "ui_action": {"request_id": request_id, "action": tool_params["action"], "params": tool_params},
                **({"retry_message_index": retry_idx} if retry_idx is not None else {}),
            }, http_req)

        # Check if tool requires approval
        if tool_name in _TOOLS_REQUIRING_APPROVAL:
            action_id = _generate_action_id()
            with _pending_lock:
                _pending_actions[action_id] = {
                    "tool": tool_name,
                    "params": tool_params,
                    "session_id": session_id,
                    "task": task,
                    "messages": messages.copy(),
                    "steps": steps.copy(),
                    "iteration": iteration,
                    "temperature": request.temperature,
                    "language": request.language,
                    "created_at": time.time(),
                    "retry_message_index": retry_idx,
                }
                _save_pending_actions()

            steps.append({
                "type": "pending_approval",
                "action_id": action_id,
                "tool": tool_name,
                "params": tool_params,
                "description": tool_params.get("description", f"Thực thi {tool_name}"),
            })

            # Return immediately — client must approve/reject
            total_ms = (time.monotonic() - req_start) * 1000
            logger.info("Agent paused for approval | action=%s tool=%s | %.0fms", action_id, tool_name, total_ms)
            return _agent_response({
                "status": "pending_approval",
                "session_id": session_id,
                "steps": steps,
                "action_id": action_id,
                "total_ms": round(total_ms),
                **({"retry_message_index": retry_idx} if retry_idx is not None else {}),
            }, http_req)

        # Execute safe tool
        if tool_name == "_validation_error":
            tool_result = {"error": f"Lỗi xác thực tham số tool '{tool_params.get('tool')}': {tool_params.get('error')}"}
        elif tool_name in _TOOL_EXECUTORS:
            try:
                tool_result = _TOOL_EXECUTORS[tool_name](tool_params)
            except Exception as e:
                tool_result = {"error": f"Tool exception: {e}"}
        else:
            tool_result = {"error": f"Tool không tồn tại: {tool_name}"}

        result_preview = str(tool_result)[:120].replace("\n", " ")
        print(f"[AGENT TRACE] ── Iter {iteration}: Tool result ← {result_preview}", flush=True)
        steps.append({
            "type": "tool_result",
            "tool": tool_name,
            "result": tool_result,
            "iteration": iteration,
        })

        # Append to conversation for next iteration
        messages.append({"role": "assistant", "content": answer})

        # Format tool result for LLM
        result_text = json.dumps(tool_result, ensure_ascii=False, indent=2)
        if len(result_text) > 8000:
            result_text = result_text[:8000] + "\n... [truncated]"
        messages.append({
            "role": "user",
            "content": f"Tool `{tool_name}` returned:\n```json\n{result_text}\n```\n\nContinue with your analysis or call another tool if needed.",
        })

    # If we exhausted iterations without final answer
    if not any(s["type"] == "final_answer" for s in steps):
        steps.append({
            "type": "final_answer",
            "content": "⚠️ Agent đã đạt giới hạn iterations mà chưa hoàn thành. "
                       "Vui lòng chia nhỏ task hoặc hỏi cụ thể hơn.",
        })

    total_ms = (time.monotonic() - req_start) * 1000
    print(f"[AGENT TRACE] ✓ Done (legacy) | {iteration} iters, {len(steps)} steps, {round(total_ms)}ms", flush=True)
    logger.info(
        "Agent done | iterations=%d steps=%d | %.0fms",
        iteration, len(steps), total_ms,
    )

    return _agent_response({
        "status": "completed",
        "session_id": session_id,
        "steps": steps,
        "iterations": iteration,
        "total_ms": round(total_ms),
        **({"retry_message_index": retry_idx} if retry_idx is not None else {}),
    }, http_req)


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

    # A workflow advances only after the desktop has positively acknowledged
    # the preceding action.  Each continuation has a fresh request_id, so the
    # Qt client can dispatch it once without replaying the earlier step.
    next_actions = action.get("next_actions", [])
    if request.success and next_actions:
        next_params, error = validate_action_params(next_actions[0])
        if error:
            raise HTTPException(status_code=422, detail=error)
        next_request_id = _generate_action_id()
        next_params["request_id"] = next_request_id
        next_action = {**action, "params": next_params,
                       "next_actions": next_actions[1:], "created_at": time.time()}
        with _pending_lock:
            _pending_actions[next_request_id] = next_action
            _save_pending_actions()
        steps.append({"type": "tool_call", "tool": "application_action",
                      "params": next_params, "iteration": action.get("iteration", 0) + 1})
        return {"status": "pending_ui_action", "session_id": action["session_id"],
                "request_id": next_request_id, "steps": action.get("steps", []) + steps,
                "ui_action": {"request_id": next_request_id, "action": next_params["action"],
                              "params": next_params}}

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
        )

    content = (f"Đã thực thi {params['action']}." if request.success
               else f"Không thể thực thi {params['action']}: {result.get('error', 'unknown error')}")
    steps.append({"type": "final_answer", "content": content})
    retry_idx_stored = action.get("retry_message_index")
    return {
        "status": "completed" if request.success else "failed",
        "session_id": action["session_id"],
        "request_id": request.request_id,
        "steps": steps,
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
        raise HTTPException(status_code=404, detail=f"Action không tồn tại hoặc đã hết hạn: {action_id}")

    if not request.approved:
        # User rejected
        return {
            "status": "rejected",
            "action_id": action_id,
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
    tool_name = action["tool"]
    tool_params = action["params"]

    if tool_name == "write_file":
        tool_result = _execute_approved_write_file(tool_params)
    elif tool_name == "run_command":
        tool_result = _execute_approved_run_command(tool_params)
    elif tool_name == "patch_file":
        tool_result = _execute_approved_patch_file(tool_params)
    elif tool_name == "create_directory":
        tool_result = _execute_approved_create_directory(tool_params)
    else:
        tool_result = {"error": f"Unknown approval tool: {tool_name}"}

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
        )

    # Continue the agent loop
    iteration = action["iteration"]
    temperature = action["temperature"]
    req_start = time.monotonic()

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

        think_text = answer
        if "```tool_call" in answer:
            think_text = answer.split("```tool_call")[0].strip()
        elif "{" in answer:
            think_text = answer.split("{")[0].strip()
            
        if think_text:
            steps.append({"type": "thinking", "content": think_text, "iteration": iteration})

        steps.append({
            "type": "tool_call",
            "tool": tool_name_next,
            "params": tool_params_next,
            "iteration": iteration,
        })

        if tool_name_next in _TOOLS_REQUIRING_APPROVAL:
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
                }
                _save_pending_actions()

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
                "prior_step_count": len(action["steps"]),
                "steps": steps,
                "action_id": new_action_id,
                "total_ms": round(total_ms),
            }

        if tool_name_next == "_validation_error":
            tool_result_next = {"error": f"Lỗi xác thực tham số tool '{tool_params_next.get('tool')}': {tool_params_next.get('error')}"}
        elif tool_name_next in _TOOL_EXECUTORS:
            try:
                tool_result_next = _TOOL_EXECUTORS[tool_name_next](tool_params_next)
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
        "prior_step_count": len(action["steps"]),
        "steps": steps,
        "iterations": iteration,
        "total_ms": round(total_ms),
    }


# Cleanup expired pending actions (older than 10 minutes)
def _cleanup_pending_actions():
    cutoff = time.time() - 600
    with _pending_lock:
        expired = [k for k, v in _pending_actions.items() if v.get("created_at", 0) < cutoff]
        for k in expired:
            del _pending_actions[k]
        if expired:
            _save_pending_actions()
        if expired:
            logger.info("Cleaned up %d expired pending agent actions", len(expired))


def reset_agent_state() -> None:
    """Forget all pending approvals and their persisted state."""
    with _pending_lock:
        _pending_actions.clear()
        try:
            if os.path.exists(_PENDING_ACTIONS_FILE):
                os.remove(_PENDING_ACTIONS_FILE)
        except OSError as error:
            logger.warning("Unable to remove pending action state: %s", error)


