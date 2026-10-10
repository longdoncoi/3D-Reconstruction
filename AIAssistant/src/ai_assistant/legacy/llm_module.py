# ruff: noqa: F401
"""
Compatibility shim for the legacy llm_module.
New code should use ai_assistant.llm package directly.
"""

import threading

# --- NEW PLATFORM INTEGRATION ---
from ai_assistant.llm import (
    CHARACTER_QUERY_PATTERNS as _CHARACTER_QUERY_PATTERNS,
)
from ai_assistant.llm import (
    PROJECT_CHARACTER_NAMES as _PROJECT_CHARACTER_NAMES,
)
from ai_assistant.llm import (
    ROLE_QUERY_PATTERN as _ROLE_QUERY_PATTERN,
)
from ai_assistant.llm import (
    DefaultPromptBuilder,
    estimate_tokens,
    trim_history,
)
from ai_assistant.llm import (
    download_if_missing as _new_download,
)
from ai_assistant.llm import (
    load_model as _new_load,
)
from ai_assistant.llm import (
    release_ml_memory as _release_ml_memory,
)
from ai_assistant.llm import (
    reload_model as _new_reload,
)
from ai_assistant.ports import LockLike
from ai_assistant.rag.image_utils import image_to_data_uri as _image_to_data_uri
from ai_assistant.rag.image_utils import is_image_file as _is_image_file

# Configuration is imported from the shim
from .config import (
    ENABLE_VISION_LLM,
    LLM_N_CTX,
    MODEL_IDX,
    MODELS_DIR,
    _registry,
    active_model_desc,
    logger,
    startup_step,
)

# Global state mirroring for backward compatibility
llm = None
# Annotated with the structural port so this module keeps satisfying
# ``ports.LLMRuntime`` regardless of whether it owns a Lock or an RLock.
llm_lock: LockLike = threading.RLock()
is_vision_model = False
_chat_handler = None
_prompt_builder = DefaultPromptBuilder()

_RAG_SYSTEM_PROMPT = _prompt_builder._build_system_prompt("", "", False, "vi")

def load_model(model_idx: int | None = None):
    global llm, is_vision_model, active_model_desc
    
    selected_idx = MODEL_IDX if model_idx is None else model_idx
    with startup_step("Loading LLM Backend"):
        backend = _new_load(
            registry=_registry,
            models_dir=MODELS_DIR,
            model_idx=selected_idx,
            enable_vision=ENABLE_VISION_LLM,
            n_ctx=LLM_N_CTX
        )
    
    with llm_lock:
        llm = backend
        is_vision_model = backend.is_vision_supported
        active_model_desc = backend.model_description
        return llm

def reload_model():
    global llm, is_vision_model, active_model_desc
    backend = _new_reload(
        registry=_registry,
        models_dir=MODELS_DIR,
        model_idx=MODEL_IDX,
        enable_vision=ENABLE_VISION_LLM,
        n_ctx=LLM_N_CTX
    )
    with llm_lock:
        llm = backend
        is_vision_model = backend.is_vision_supported
        active_model_desc = backend.model_description
        return llm

def _normalize_for_intent(text: str) -> str:
    # Handled inside the prompt builder internally, but exposed if still imported
    from ai_assistant.llm.prompts import _normalize_for_intent as norm
    return norm(text)

def _is_character_query(query: str) -> bool:
    return _prompt_builder.is_character_query(query)

def _strip_reference_citations_for_character_answer(answer: str) -> str:
    return _prompt_builder.strip_citations(answer)

def build_text_messages(messages: list, doc_ctx: str, code_ctx: str, suppress_citations: bool = False, language: str = "vi") -> list:
    return _prompt_builder.build_messages(messages, doc_ctx, code_ctx, language, suppress_citations)

def _build_system_prompt(doc_ctx: str, code_ctx: str, suppress_citations: bool = False, language: str = "vi") -> str:
    return _prompt_builder._build_system_prompt(doc_ctx, code_ctx, suppress_citations, language)

def build_vision_messages(messages: list, doc_ctx: str, code_ctx: str, image_chunks: list | None = None, suppress_citations: bool = False, language: str = "vi") -> list:
    return _prompt_builder.build_vision_messages(messages, doc_ctx, code_ctx, image_chunks, language, suppress_citations)
