"""Vietnamese text normalization utilities for transcript search."""

import re
from typing import Iterable

# ── Diacritic removal ────────────────────────────────────────────────────────

# Mapping: accented Vietnamese → ASCII base. Covers all tone marks and
# base-vowel variants used in modern Vietnamese orthography.
_VI_CHAR_MAP: dict[int, int | None] = {
    # a / à / á / ả / ã / ạ
    0x00E0: ord("a"), 0x00E1: ord("a"), 0x1EA3: ord("a"),
    0x00E3: ord("a"), 0x1EA1: ord("a"),
    0x00C0: ord("a"), 0x00C1: ord("a"), 0x1EA2: ord("a"),
    0x00C3: ord("a"), 0x1EA0: ord("a"),
    # ă / ằ / ắ / ẳ / ẵ / ặ
    0x0103: ord("a"), 0x1EB1: ord("a"), 0x1EAF: ord("a"),
    0x1EB3: ord("a"), 0x1EB5: ord("a"), 0x1EB7: ord("a"),
    0x0102: ord("a"), 0x1EB0: ord("a"), 0x1EAE: ord("a"),
    0x1EB2: ord("a"), 0x1EB4: ord("a"), 0x1EB6: ord("a"),
    # â / ầ / ấ / ẩ / ẫ / ậ
    0x00E2: ord("a"), 0x1EA7: ord("a"), 0x1EA5: ord("a"),
    0x1EA9: ord("a"), 0x1EAB: ord("a"), 0x1EAD: ord("a"),
    0x00C2: ord("a"), 0x1EA6: ord("a"), 0x1EA4: ord("a"),
    0x1EA8: ord("a"), 0x1EAA: ord("a"), 0x1EAC: ord("a"),
    # e / è / é / ẻ / ẽ / ẹ
    0x00E8: ord("e"), 0x00E9: ord("e"), 0x1EBB: ord("e"),
    0x1EBD: ord("e"), 0x1EB9: ord("e"),
    0x00C8: ord("e"), 0x00C9: ord("e"), 0x1EBA: ord("e"),
    0x1EBC: ord("e"), 0x1EB8: ord("e"),
    # ê / ề / ế / ể / ễ / ệ
    0x00EA: ord("e"), 0x1EC1: ord("e"), 0x1EBF: ord("e"),
    0x1EC3: ord("e"), 0x1EC5: ord("e"), 0x1EC7: ord("e"),
    0x00CA: ord("e"), 0x1EC0: ord("e"), 0x1EBE: ord("e"),
    0x1EC2: ord("e"), 0x1EC4: ord("e"), 0x1EC6: ord("e"),
    # i / ì / í / ỉ / ĩ / ị
    0x00EC: ord("i"), 0x00ED: ord("i"), 0x1EC9: ord("i"),
    0x0129: ord("i"), 0x1ECB: ord("i"),
    0x00CC: ord("i"), 0x00CD: ord("i"), 0x1EC8: ord("i"),
    0x0128: ord("i"), 0x1ECA: ord("i"),
    # o / ò / ó / ỏ / õ / ọ
    0x00F2: ord("o"), 0x00F3: ord("o"), 0x1ECF: ord("o"),
    0x00F5: ord("o"), 0x1ECD: ord("o"),
    0x00D2: ord("o"), 0x00D3: ord("o"), 0x1ECE: ord("o"),
    0x00D5: ord("o"), 0x1ECC: ord("o"),
    # ô / ồ / ố / ổ / ỗ / ộ
    0x00F4: ord("o"), 0x1ED3: ord("o"), 0x1ED1: ord("o"),
    0x1ED5: ord("o"), 0x1ED7: ord("o"), 0x1ED9: ord("o"),
    0x00D4: ord("o"), 0x1ED2: ord("o"), 0x1ED0: ord("o"),
    0x1ED4: ord("o"), 0x1ED6: ord("o"), 0x1ED8: ord("o"),
    # ơ / ờ / ớ / ở / ỡ / ợ
    0x01A1: ord("o"), 0x1EDD: ord("o"), 0x1EDB: ord("o"),
    0x1EDF: ord("o"), 0x1EE1: ord("o"), 0x1EE3: ord("o"),
    0x01A0: ord("o"), 0x1EDC: ord("o"), 0x1EDA: ord("o"),
    0x1EDE: ord("o"), 0x1EE0: ord("o"), 0x1EE2: ord("o"),
    # u / ù / ú / ủ / ũ / ụ
    0x00F9: ord("u"), 0x00FA: ord("u"), 0x1EE7: ord("u"),
    0x0169: ord("u"), 0x1EE5: ord("u"),
    0x00D9: ord("u"), 0x00DA: ord("u"), 0x1EE6: ord("u"),
    0x0168: ord("u"), 0x1EE4: ord("u"),
    # ư / ừ / ứ / ử / ữ / ự
    0x01B0: ord("u"), 0x1EEB: ord("u"), 0x1EE9: ord("u"),
    0x1EED: ord("u"), 0x1EEF: ord("u"), 0x1EF1: ord("u"),
    0x01AF: ord("u"), 0x1EEA: ord("u"), 0x1EE8: ord("u"),
    0x1EEC: ord("u"), 0x1EEE: ord("u"), 0x1EF0: ord("u"),
    # y / ỳ / ý / ỷ / ỹ / ỵ
    0x1EF3: ord("y"), 0x00FD: ord("y"), 0x1EF7: ord("y"),
    0x1EF9: ord("y"), 0x1EF5: ord("y"),
    0x1EF2: ord("y"), 0x00DD: ord("y"), 0x1EF6: ord("y"),
    0x1EF8: ord("y"), 0x1EF4: ord("y"),
    # đ
    0x0111: ord("d"),
    0x0110: ord("d"),
}

# Regex: strip punctuation and symbols, keep letters, digits, and spaces
_RE_CLEAN = re.compile(r"[^\w\s]", re.UNICODE)
_RE_WHITESPACE = re.compile(r"\s+")


def normalize_vi(text: str) -> str:
    """Lowercase, remove Vietnamese diacritics, đ→d, normalize punctuation, collapse whitespace.

    >>> normalize_vi("Thành phố Hồ Chí Minh")
    'thanh pho ho chi minh'
    >>> normalize_vi("Cầu Phước Khánh")
    'cau phuoc khanh'
    >>> normalize_vi("TP.HCM")
    'tp hcm'
    >>> normalize_vi("COVID-19")
    'covid 19'
    """
    if not text:
        return ""

    # Lowercase and translate diacritics
    result = text.lower().translate(_VI_CHAR_MAP)

    # Replace punctuation/hyphens with space, keep alphanumeric + spaces
    result = _RE_CLEAN.sub(" ", result)
    result = _RE_WHITESPACE.sub(" ", result).strip()

    return result


# ── Alias map ────────────────────────────────────────────────────────────────

_ALIAS_NORMALIZED = {
    "tp hcm": "thanh pho ho chi minh",
    "tphcm": "thanh pho ho chi minh",
    "sai gon": "thanh pho ho chi minh",
    "sg": "thanh pho ho chi minh",
    "hcm": "ho chi minh",
    "hcmc": "ho chi minh",
    "hn": "ha noi",
}


# Sort multi-word aliases by key length descending so longer matches win
_MULTI_WORD_ALIASES = sorted(
    [(k, v) for k, v in _ALIAS_NORMALIZED.items() if " " in k],
    key=lambda x: -len(x[0]),
)
_SINGLE_TOKEN_ALIASES = {k: v for k, v in _ALIAS_NORMALIZED.items() if " " not in k}


def expand_query_aliases(normalized_text: str) -> str:
    """Expand known aliases in a normalized Vietnamese query.

    Multi-word aliases are matched first (longest wins), then single tokens.
      tp hcm / tphcm / sai gon / sg → thanh pho ho chi minh
      hcm / hcmc → ho chi minh
      hn → ha noi
    """
    if not normalized_text:
        return ""
    result = normalized_text
    # Multi-word phrase aliases first
    for key, value in _MULTI_WORD_ALIASES:
        result = result.replace(key, value)
    # Single-token aliases
    tokens = result.split()
    expanded = [_SINGLE_TOKEN_ALIASES.get(t, t) for t in tokens]
    return " ".join(expanded)


# ── Stopword removal ─────────────────────────────────────────────────────────

_VI_STOPWORDS: set[str] = {
    "doan", "nao", "noi", "ve", "nhac", "toi", "co", "la", "mot",
    "cac", "nhung", "trong", "o", "tai", "ban", "tin", "video",
    "giay", "sang", "thu", "cua", "ngay", "chuong", "trinh",
    "quy", "vi", "duoc", "cho", "de", "nhu", "qua",
    "den",
    "khi", "hon", "rat", "se", "da", "dang",
}


def extract_query_terms(normalized_query: str) -> list[str]:
    """Remove stopwords from a normalized query, returning meaningful terms.

    >>> extract_query_terms("doan nao nhac den benh vien")
    ['benh', 'vien']
    >>> extract_query_terms("ban tin noi ve cau phuoc khanh")
    ['cau', 'phuoc', 'khanh']
    """
    if not normalized_query:
        return []
    return [t for t in normalized_query.split() if t not in _VI_STOPWORDS]


def extract_keywords(query: str) -> str:
    """Full pipeline: normalize → expand aliases → strip stopwords → rejoin.

    This is the main entry point called before scoring.
    """
    norm = normalize_vi(query)
    expanded = expand_query_aliases(norm)
    terms = extract_query_terms(expanded)
    return " ".join(terms) if terms else norm
