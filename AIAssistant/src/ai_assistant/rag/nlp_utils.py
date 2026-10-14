"""Natural language processing utilities for RAG."""
from __future__ import annotations

import re
import unicodedata

# Vietnamese multi-word terms that must survive tokenisation as one term, so the
# BM25 index and the query share the same vocabulary.
_COMPOUNDS = {
    "tái tạo": "tái_tạo", "hình ảnh": "hình_ảnh", "mô hình": "mô_hình",
    "dữ liệu": "dữ_liệu", "hệ thống": "hệ_thống", "đầu vào": "đầu_vào",
    "đầu ra": "đầu_ra", "cấu hình": "cấu_hình", "giao diện": "giao_diện",
}
# Basic Vietnamese + English function words.
_STOPWORDS = frozenset({"là", "của", "và", "các", "trong", "được", "có", "cho", "với", "để", "những"})
# ``\w+`` with the ``re.UNICODE`` flag: one call covers Latin, Vietnamese
# diacritics, CJK and code identifiers alike (``model_idx``, ``point3d``).
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize_vn(text: str) -> list[str]:
    """Tokenizer BM25 hỗ trợ Việt + English + code.

    - NFC-normalise first: a decomposed ``ế`` (``e`` + circumflex + acute) must
      not defeat the compound rules or the stop-word list, which is what made
      two spellings of the same word index as different terms.
    - ``\\w+`` keeps ``hình_ảnh``/``model_idx`` in one token and drops
      punctuation; the previous ASCII/non-ASCII split produced junk tokens out
      of em dashes and left underscores out of code identifiers.
    """
    normalised = unicodedata.normalize("NFC", text.lower())
    for source, target in _COMPOUNDS.items():
        normalised = normalised.replace(source, target)
    return [token for token in _WORD_RE.findall(normalised) if token not in _STOPWORDS]


_PROJECT_ROLE_QUERY_ALIASES = (
    (r"\bteam\s*lead\b|\bteamlead\b|\btruong\s*nhom\b|\btrưởng\s*nhóm\b",
     "team lead trưởng nhóm leader"),
    (r"\bdev\s*manager\b|\bdevmanager\b|\bdevelopment\s*manager\b|\bquản\s*lý\s*phát\s*triển\b",
     "dev manager development manager quản lý phát triển"),
    (r"\bproject\s*manager\b|\bquan\s*ly\s*du\s*an\b|\bquản\s*lý\s*dự\s*án\b|\bpm\b",
     "project manager quản lý dự án pm"),
    (r"\bhr\s*manager\b|\bhrmanager\b|\bhuman\s*resources\s*manager\b|\bquản\s*lý\s*nhân\s*sự\b",
     "hr manager human resources manager quản lý nhân sự"),
    (r"\bky\s*su\b|\bkỹ\s*sư\b|\bengineer\b|\bdeveloper\b",
     "kỹ sư engineer developer"),
)


def expand_project_role_query(query: str) -> str:
    """Make role-title searches robust to English/Vietnamese title variants."""
    normalized = " ".join(tokenize_vn(query))
    aliases = [aliases for pattern, aliases in _PROJECT_ROLE_QUERY_ALIASES
               if re.search(pattern, normalized)]
    if not aliases:
        return query
    return f"{query} {' '.join(aliases)} thông tin vai trò nhiệm vụ trách nhiệm"
