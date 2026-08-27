from .config import *
from .rag_module import _image_to_data_uri, _is_image_file, _release_ml_memory

_RAG_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên nghiệp cho dự án 3D-Reconstruction.
NGUYÊN TẮC TRẢ LỜI:
1. Dựa chủ yếu vào tài liệu và mã nguồn được cung cấp bên dưới để trả lời.
2. Nếu câu hỏi không liên quan đến bất kỳ nội dung nào trong ngữ cảnh, hãy nói ngắn gọn: 'Câu hỏi này nằm ngoài phạm vi tài liệu dự án.' rồi dừng.
3. Không bịa đặt hoặc suy đoán thông tin kỹ thuật không có trong tài liệu.
4. Khi nhắc đến code hoặc tài liệu, hãy ghi rõ số thứ tự nguồn [1], [2]... tương ứng với danh sách ngữ cảnh bên dưới.
5. Ưu tiên trả lời ĐẦY ĐỦ và CHI TIẾT — giải thích từng bước, nêu lý do kỹ thuật, trích dẫn trực tiếp từ tài liệu khi có thể.
6. Cấu trúc câu trả lời: tóm tắt ngắn → giải thích chi tiết → ví dụ/code.
7. Trả lời bằng tiếng Việt trừ khi người dùng hỏi bằng tiếng Anh. Nếu tài liệu nguồn là tiếng Anh, hãy DỊCH và GIẢI THÍCH sang tiếng Việt.
8. LUÔN sử dụng định dạng Markdown (tiêu đề in đậm, bullet points, code blocks có highlight syntax) để trình bày đẹp và dễ đọc.
9. QUAN TRỌNG: Luôn hoàn thành câu cuối cùng trước khi kết thúc. Không bao giờ dừng giữa câu, giữa đoạn code, hoặc giữa danh sách.
10. Nếu người dùng gửi ảnh, hãy phân tích nội dung ảnh chi tiết và liên hệ với tài liệu dự án nếu có thể.
11. Nếu câu hỏi liên quan đến nhân vật trong dự án (như thành viên, tác giả, người tham gia), hãy trả lời trực tiếp mà KHÔNG trích dẫn tài liệu tham khảo.
12. KHÔNG liệt kê hay in lại log 'TÀI LIỆU THAM KHẢO' hoặc 'MÃ NGUỒN LIÊN QUAN' trong câu trả lời.
"""

# ─── 14. FastAPI + LLM ────────────────────────────────────────────────────────
llm = None
llm_lock = threading.RLock()
is_vision_model = False
active_model_desc = MODELS[MODEL_IDX]["desc"]


def _download_if_missing(model: dict, key: str = "filename", repo_key: str = "repo_id") -> str:
    """Return a local model path, downloading the selected artifact if needed."""
    from huggingface_hub import hf_hub_download

    filename = model[key]
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        with startup_step(f"Downloading {model.get('desc', filename)}"):
            hf_hub_download(repo_id=model[repo_key], filename=filename, local_dir=MODELS_DIR)
    return path


def load_model(model_idx: int | None = None):
    """Load the configured Llama model and publish it as the shared runtime."""
    global llm, _chat_handler, is_vision_model, active_model_desc

    selected_idx = MODEL_IDX if model_idx is None else model_idx
    if selected_idx < 0 or selected_idx >= len(MODELS):
        selected_idx = 0
    selected = MODELS[selected_idx]
    if selected.get("is_vision") and not ENABLE_VISION_LLM:
        logger.warning("Vision LLM disabled; using text fallback")
        selected = FALLBACK_TEXT_MODEL

    with llm_lock:
        _release_ml_memory()
        model_path = _download_if_missing(selected)
        _chat_handler = None
        is_vision_model = bool(selected.get("is_vision"))
        used_cpu = False
        try:
            if is_vision_model:
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                mmproj_path = _download_if_missing(selected, "mmproj_filename", "mmproj_repo_id")
                _chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
                llm = Llama(model_path=model_path, chat_handler=_chat_handler,
                            chat_format="qwen2.5-vl", n_gpu_layers=99,
                            n_ctx=LLM_N_CTX, n_batch=256, verbose=False,
                            use_mmap=True, use_mlock=False)
            else:
                from llama_cpp import Llama
                llm = Llama(model_path=model_path, n_gpu_layers=99,
                            n_ctx=LLM_N_CTX, n_batch=512, verbose=False,
                            use_mmap=True, use_mlock=False)
        except Exception as gpu_error:
            logger.warning("GPU model load failed (%s); retrying on CPU", gpu_error)
            _release_ml_memory()
            used_cpu = True
            from llama_cpp import Llama
            if is_vision_model:
                llm = Llama(model_path=model_path, chat_handler=_chat_handler,
                            chat_format="qwen2.5-vl", n_gpu_layers=0,
                            n_ctx=LLM_N_CTX, n_batch=128, verbose=False,
                            use_mmap=True, use_mlock=False)
            else:
                llm = Llama(model_path=model_path, n_gpu_layers=0,
                            n_ctx=LLM_N_CTX, n_batch=128, verbose=False,
                            use_mmap=True, use_mlock=False)

        active_model_desc = selected["desc"] + (" — CPU" if used_cpu else "")
        logger.info("Model loaded: %s", active_model_desc)
        return llm


def reload_model():
    """Release the active model before reloading it, retaining the server process."""
    global llm
    with llm_lock:
        if llm is None:
            raise RuntimeError("LLM has not been initialized")
        old_llm, llm = llm, None
        del old_llm
        _release_ml_memory()
        return load_model()
_chat_handler = None  # Qwen25VLChatHandler, chỉ dùng cho vision model


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def trim_history(messages: list, max_tokens: int = 2000) -> list:
    messages = list(messages)
    total    = sum(estimate_tokens(m.get("content", "")) for m in messages)
    while total > max_tokens and len(messages) > 1:
        removed = messages.pop(0)
        total  -= estimate_tokens(removed.get("content", ""))
    return messages


_CHARACTER_QUERY_PATTERNS = (
    r"\b(nhan\s*vat|thanh\s*vien|tac\s*gia|nguoi\s*tham\s*gia|nhan\s*su|doi\s*ngu|member|author|participant|character|person|people)\b",
    r"\b(team\s*lead|teamlead|leader|project\s*manager|dev\s*manager|devmanager|hr\s*manager|hrmanager|ky\s*su|engineer|developer)\b",
    r"\b(la\s+ai|who\s+is|who'?s|nguoi\s+nao|ai\s+phu\s+trach|ai\s+quan\s+ly)\b",
)
_ROLE_QUERY_PATTERN = r"\b(vai\s*tro|role)\b"
_PROJECT_CHARACTER_NAMES = (
    "john",
    "carpenter",
    "chris",
    "hoang",
    "nancy",
    "snow",
    "lavrov",
)


def _normalize_for_intent(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def _is_character_query(query: str) -> bool:
    """True khi câu hỏi đang hỏi về nhân vật/thành viên/vai trò trong dự án."""
    normalized = _normalize_for_intent(query)
    if not normalized:
        return False

    if any(re.search(pattern, normalized) for pattern in _CHARACTER_QUERY_PATTERNS):
        return True

    has_project_cue = re.search(r"\b(du\s*an|project|e2|e3|ewoosoft|story)\b", normalized)
    has_known_character = any(re.search(rf"\b{re.escape(name)}\b", normalized) for name in _PROJECT_CHARACTER_NAMES)
    if has_project_cue and has_known_character:
        return True

    return bool((has_project_cue or has_known_character) and re.search(_ROLE_QUERY_PATTERN, normalized))


def _strip_reference_citations_for_character_answer(answer: str) -> str:
    if not answer:
        return answer

    cleaned = re.sub(r"\s*\[(?:\d+)(?:\s*,\s*\d+)*\]", "", answer)
    source_exts = r"txt|md|pdf|docx?|pptx?|xlsx?|eml|html?"
    source_intro = rf"(?:Theo|Dựa trên)\s+(?:tài liệu|nguồn)\s*(?:tham khảo)?\s*(?:[^,\n]{{1,160}}\.(?:{source_exts})[,.:;]?\s*)?"
    cleaned = re.sub(rf"(?im)^\s*{source_intro}", "", cleaned)
    cleaned = re.sub(rf"(?i)(:\s*(?:\*\*)?\s*){source_intro}", r"\1", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:Tài liệu|Nguồn)\s*(?:tham khảo)?\s*[:\-–]\s*", "", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_text_messages(messages: list, doc_ctx: str, code_ctx: str, suppress_citations: bool = False,
                        language: str = "vi") -> list:
    system_prompt = _build_system_prompt(doc_ctx, code_ctx, suppress_citations=suppress_citations,
                                         language=language)
    result = [{"role": "system", "content": system_prompt}]
    
    history = trim_history(list(messages[:-1]), max_tokens=2000)
    for msg in history:
        role = "user" if msg.get("role") == "user" else "assistant"
        content = msg.get("content", "")
        if msg.get("attachments"):
            content += f"\n\n[Hệ thống: Người dùng tải lên: {', '.join(msg['attachments'])}. Model text-only, không thể xem ảnh.]"
        result.append({"role": role, "content": content})
        
    last = messages[-1]
    last_content = last.get("content", "")
    if last.get("attachments"):
        last_content += f"\n\n[Hệ thống: Người dùng tải lên: {', '.join(last['attachments'])}. Model text-only, không thể xem ảnh.]"
    result.append({"role": "user", "content": last_content})
    return result


def _build_system_prompt(doc_ctx: str, code_ctx: str, suppress_citations: bool = False,
                         language: str = "vi") -> str:
    """Build system prompt chung cho cả text-only và vision model."""
    system_prompt = _RAG_SYSTEM_PROMPT
    response_language = "Vietnamese" if language == "vi" else "English"
    system_prompt += (
        f"\n13. Respond exclusively in {response_language}, matching the current application language. "
        "Do not choose the response language from the user's message or source documents.\n"
    )
    if suppress_citations:
        system_prompt += (
            "\nCHẾ ĐỘ CÂU HỎI NHÂN VẬT/VAI TRÒ ĐANG BẬT:\n"
            "- Câu hỏi hiện tại liên quan đến nhân vật, thành viên hoặc vai trò trong dự án.\n"
            "- Trả lời trực tiếp, tự nhiên; TUYỆT ĐỐI KHÔNG dùng ký hiệu nguồn như [1], [2].\n"
            "- KHÔNG viết các cụm mở đầu như 'Theo tài liệu', 'Theo nguồn', 'Tài liệu tham khảo'.\n"
            "- Vẫn dùng thông tin trong ngữ cảnh, nhưng không để lộ citation hoặc tên file nguồn trong câu trả lời.\n"
        )
    if doc_ctx:
        system_prompt += f"\n\n{doc_ctx}"
    if code_ctx:
        system_prompt += f"\n\n{code_ctx}"
    if not doc_ctx and not code_ctx:
        system_prompt += "\n\n[Không tìm thấy ngữ cảnh liên quan. Từ chối theo nguyên tắc số 2.]"
    return system_prompt


def build_vision_messages(messages: list, doc_ctx: str, code_ctx: str, image_chunks: list | None = None,
                          suppress_citations: bool = False, language: str = "vi") -> list:
    """
    Build messages format cho create_chat_completion() — vision model.
    Ảnh đính kèm được encode thành base64 data URI theo OpenAI multimodal format.
    Chỉ ảnh ở message cuối cùng được gửi — ảnh cũ trong history bị bỏ qua
    để tiết kiệm context.
    """
    system_prompt = _build_system_prompt(doc_ctx, code_ctx, suppress_citations=suppress_citations,
                                         language=language)
    result = [{"role": "system", "content": system_prompt}]

    # History: chỉ gửi text, bỏ ảnh cũ
    history = trim_history(list(messages[:-1]), max_tokens=2000)
    for msg in history:
        role = "user" if msg.get("role") == "user" else "assistant"
        content = msg.get("content", "")
        result.append({"role": role, "content": content})

    # Message cuối: xử lý cả text + ảnh
    last = messages[-1]
    content_parts = []

    text_content = last.get("content", "")
    if text_content:
        content_parts.append({"type": "text", "text": text_content})

    # Encode ảnh đính kèm thành base64 data URI
    attachments = last.get("attachments") or []
    has_images = False
    for att in attachments:
        if os.path.isfile(att) and _is_image_file(att):
            try:
                data_uri = _image_to_data_uri(att)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                })
                has_images = True
                logger.info("Vision: encoded image %s (%d bytes)",
                           os.path.basename(att), os.path.getsize(att))
            except Exception as e:
                logger.error("Failed to encode image %s: %s", att, e)
                content_parts.append(
                    {"type": "text", "text": f"[Lỗi đọc ảnh: {os.path.basename(att)}]"}
                )
        else:
            content_parts.append(
                {"type": "text", "text": f"[File đính kèm: {os.path.basename(att)}]"}
            )

    if image_chunks:
        for b64 in image_chunks:
            content_parts.append(
                {"type": "image_url", "image_url": {"url": b64}}
            )
            has_images = True
            logger.info("Vision: added retrieved image chunk to context")

    # Nếu không có ảnh, thêm hint cho model
    if attachments and not has_images:
        content_parts.append(
            {"type": "text", "text": "[Không có ảnh hợp lệ trong file đính kèm.]"}
        )

    if not content_parts:
        content_parts.append({"type": "text", "text": "(trống)"})

    result.append({"role": "user", "content": content_parts})
    logger.debug("Vision messages: %d parts (has_images=%s)", len(content_parts), has_images)
    return result


