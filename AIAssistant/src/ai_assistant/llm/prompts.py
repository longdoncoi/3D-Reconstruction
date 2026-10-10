"""Prompt construction and history management."""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Any

logger = logging.getLogger("ai_assistant.llm.prompts")

RAG_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên nghiệp cho dự án 3D-Reconstruction.
NGUYÊN TẮC TRẢ LỜI:
1. Dựa chủ yếu vào tài liệu và mã nguồn được cung cấp bên dưới để trả lời.
2. Nếu câu hỏi hoặc hình ảnh không nằm trong ngữ cảnh dự án, hãy sử dụng kiến thức chung để trả lời và hỗ trợ người dùng một cách tốt nhất có thể.
3. Không bịa đặt hoặc suy đoán thông tin kỹ thuật dự án nếu không có trong tài liệu.
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

CHARACTER_QUERY_PATTERNS = (
    r"\b(nhan\s*vat|thanh\s*vien|tac\s*gia|nguoi\s*tham\s*gia|nhan\s*su|doi\s*ngu|member|author|participant|character|person|people)\b",
    r"\b(team\s*lead|teamlead|leader|project\s*manager|dev\s*manager|devmanager|hr\s*manager|hrmanager|ky\s*su|engineer|developer)\b",
    r"\b(la\s+ai|who\s+is|who'?s|nguoi\s+nao|ai\s+phu\s+trach|ai\s+quan\s+ly)\b",
)
ROLE_QUERY_PATTERN = r"\b(vai\s*tro|role)\b"
PROJECT_CHARACTER_NAMES = (
    "john",
    "carpenter",
    "chris",
    "hoang",
    "nancy",
    "snow",
    "lavrov",
)

def estimate_tokens(text: str, chars_per_token: float = 2.2) -> int:
    return int(len(text) / chars_per_token)

def trim_history(messages: list[dict[str, Any]], max_tokens: int = 2000, chars_per_token: float = 2.2) -> list[dict[str, Any]]:
    messages = list(messages)
    total = sum(estimate_tokens(m.get("content", ""), chars_per_token) for m in messages if isinstance(m.get("content"), str))
    while total > max_tokens and len(messages) > 1:
        removed = messages.pop(0)
        content = removed.get("content", "")
        if isinstance(content, str):
            total -= estimate_tokens(content, chars_per_token)
    return messages

def _normalize_for_intent(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


class DefaultPromptBuilder:
    def __init__(self, chars_per_token: float = 2.2):
        self.chars_per_token = chars_per_token

    def is_character_query(self, query: str) -> bool:
        normalized = _normalize_for_intent(query)
        if not normalized:
            return False

        if any(re.search(pattern, normalized) for pattern in CHARACTER_QUERY_PATTERNS):
            return True

        has_project_cue = bool(re.search(r"\b(du\s*an|project|e2|e3|ewoosoft|story)\b", normalized))
        has_known_character = any(re.search(rf"\b{re.escape(name)}\b", normalized) for name in PROJECT_CHARACTER_NAMES)
        if has_project_cue and has_known_character:
            return True

        return bool((has_project_cue or has_known_character) and re.search(ROLE_QUERY_PATTERN, normalized))

    def strip_citations(self, answer: str) -> str:
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

    def _build_system_prompt(self, doc_ctx: str, code_ctx: str, suppress_citations: bool, language: str) -> str:
        prompt = RAG_SYSTEM_PROMPT
        response_language = "Vietnamese" if language == "vi" else "English"
        prompt += (
            f"\n13. Respond exclusively in {response_language}, matching the current application language. "
            "Do not choose the response language from the user's message or source documents.\n"
            "14. The RAG context is private evidence. Use it to answer accurately, but do not expose "
            "its headings, filenames, numbered labels, or citations unless the user explicitly asks for a source.\n"
        )
        if suppress_citations:
            prompt += (
                "\nCHẾ ĐỘ CÂU HỎI NHÂN VẬT/VAI TRÒ ĐANG BẬT:\n"
                "- Câu hỏi hiện tại liên quan đến nhân vật, thành viên hoặc vai trò trong dự án.\n"
                "- Trả lời trực tiếp, tự nhiên; TUYỆT ĐỐI KHÔNG dùng ký hiệu nguồn như [1], [2].\n"
                "- KHÔNG viết các cụm mở đầu như 'Theo tài liệu', 'Theo nguồn', 'Tài liệu tham khảo'.\n"
                "- Vẫn dùng thông tin trong ngữ cảnh, nhưng không để lộ citation hoặc tên file nguồn trong câu trả lời.\n"
            )
        if doc_ctx:
            prompt += f"\n\n{doc_ctx}"
        if code_ctx:
            prompt += f"\n\n{code_ctx}"
        if not doc_ctx and not code_ctx:
            prompt += "\n\n[Không tìm thấy ngữ cảnh liên quan. Từ chối theo nguyên tắc số 2.]"
        return prompt

    def build_messages(self, messages: list[dict[str, Any]], doc_ctx: str, code_ctx: str, language: str = "vi", suppress_citations: bool = False) -> list[dict[str, Any]]:
        system_prompt = self._build_system_prompt(doc_ctx, code_ctx, suppress_citations, language)
        result: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        
        history = trim_history(messages[:-1], chars_per_token=self.chars_per_token)
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

    def build_vision_messages(self, messages: list[dict[str, Any]], doc_ctx: str, code_ctx: str, image_chunks: list[str] | None = None, language: str = "vi", suppress_citations: bool = False) -> list[dict[str, Any]]:
        # This requires image URI encoding which relies on RAG module. 
        # For phase 1, we will handle attachment processing inside the caller (agent) 
        # or expose a generic method here if attachments are pre-encoded.
        # Since we are extracting exactly what llm_module did, we'll keep the logic here 
        # but expect attachments to be resolved in the caller, or we import the URI helper.
        
        system_prompt = self._build_system_prompt(doc_ctx, code_ctx, suppress_citations, language)
        result: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        
        history = trim_history(messages[:-1], chars_per_token=self.chars_per_token)
        for msg in history:
            role = "user" if msg.get("role") == "user" else "assistant"
            content = msg.get("content", "")
            if isinstance(content, list):
                # Unpack if history somehow has dict parts
                content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
            result.append({"role": role, "content": content})

        last = messages[-1]
        content_parts = []
        
        text_content = last.get("content", "")
        if text_content:
            content_parts.append({"type": "text", "text": text_content})
            
        has_images = False
        
        # Attachments are expected to be pre-encoded dicts {"path": str, "b64": str} 
        # or handled gracefully.
        # To match legacy exactly, we process raw file paths if we have a helper.
        from ai_assistant.rag.image_utils import image_to_data_uri, is_image_file
        
        attachments = last.get("attachments", [])
        for att in attachments:
            if os.path.isfile(att) and is_image_file(att):
                try:
                    data_uri = image_to_data_uri(att)
                    content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                    has_images = True
                except Exception as e:
                    logger.error("Failed to encode image %s: %s", att, e)
                    content_parts.append({"type": "text", "text": f"[Lỗi đọc ảnh: {os.path.basename(att)}]"})
            else:
                content_parts.append({"type": "text", "text": f"[File đính kèm: {os.path.basename(att)}]"})
                
        if image_chunks:
            for b64 in image_chunks:
                content_parts.append({"type": "image_url", "image_url": {"url": b64}})
                has_images = True
                
        if attachments and not has_images:
            content_parts.append({"type": "text", "text": "[Không có ảnh hợp lệ trong file đính kèm.]"})
            
        if not content_parts:
            content_parts.append({"type": "text", "text": "(trống)"})
            
        result.append({"role": "user", "content": content_parts})
        return result
