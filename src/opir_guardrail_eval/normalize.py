"""Auditable, deterministic input normalization. Original text is never modified in place.

Rule order: zero_width -> nfkc -> confusables -> base64 -> spaced.
``normalize`` returns a ``NormResult`` listing which rules fired and any failures/ambiguities.
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field, asdict

ZERO_WIDTH_CHARS = {
    "​", "‌", "‍", "⁠", "﻿", "­", "᠎", "‎", "‏",
}
# Limited Cyrillic/Greek -> Latin lookalike map (deliberately small, documented).
CONFUSABLES = {
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x",
    "у": "y", "і": "i", "ѕ": "s", "ј": "j", "в": "v", "ц": "u",
    "к": "k",
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y",
    "ο": "o", "ρ": "p", "χ": "x", "ι": "i", "ν": "v", "υ": "u",
    "κ": "k",
    "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Κ": "K", "Μ": "M",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y",
}
B64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
B64_MIN_LEN, B64_MAX_IN, B64_MAX_OUT = 16, 8192, 6144
SPACED_RUN_MIN = 3
ALL_RULES = ("zero_width", "nfkc", "confusables", "base64", "spaced")


@dataclass
class NormResult:
    original: str
    normalized: str
    fired: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # failures / ambiguous cases, never silent

    @property
    def changed(self) -> bool:
        return self.original != self.normalized

    def to_dict(self) -> dict:
        d = asdict(self)
        d["changed"] = self.changed
        return d


def _is_latin(ch: str) -> bool:
    return ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN")


def rule_zero_width(t: str) -> str:
    return "".join(c for c in t if c not in ZERO_WIDTH_CHARS)


def rule_nfkc(t: str) -> str:
    return unicodedata.normalize("NFKC", t)


def rule_confusables(t: str) -> str:
    """Map lookalikes only inside mixed-script words, plus isolated single confusables
    when the text is majority Latin. Wholly Cyrillic/Greek words are left alone."""
    letters = [c for c in t if c.isalpha()]
    latin_share = sum(_is_latin(c) for c in letters) / len(letters) if letters else 0.0
    out = []
    for tok in re.split(r"(\W+)", t):
        if tok and any(c in CONFUSABLES for c in tok):
            has_latin = any(_is_latin(c) for c in tok)
            single = len(tok) == 1
            if has_latin or (single and latin_share >= 0.5):
                tok = "".join(CONFUSABLES.get(c, c) for c in tok)
        out.append(tok)
    return "".join(out)


def rule_base64(t: str, notes: list[str]) -> str:
    s = t.strip()
    if not B64_RE.match(s) or len(s) < B64_MIN_LEN:
        return t
    if len(s) > B64_MAX_IN:
        notes.append("base64: candidate exceeds input bound; not decoded")
        return t
    if len(s) % 4 != 0:
        notes.append("base64: candidate length not multiple of 4; not decoded")
        return t
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        notes.append("base64: strict decode failed")
        return t
    if len(raw) > B64_MAX_OUT:
        notes.append("base64: decoded output exceeds bound; not decoded")
        return t
    try:
        dec = raw.decode("utf-8")
    except UnicodeDecodeError:
        notes.append("base64: decoded bytes are not valid UTF-8; not decoded")
        return t
    printable = sum(c.isprintable() or c in "\n\t" for c in dec) / max(len(dec), 1)
    texty = sum(c.isalpha() or c.isspace() or c in ".,;:!?'\"-()" for c in dec) / max(len(dec), 1)
    if printable < 0.95 or texty < 0.8 or not any(c.isspace() for c in dec):
        notes.append("base64: decoded text failed sensible-text check; not decoded")
        return t
    return dec  # exactly one decoding layer


def rule_spaced(t: str, notes: list[str]) -> str:
    """Join words written as 'w o r d' (words separated by 2+ spaces). Applied only when the text is
    predominantly in that form: >=60% of segments look spaced (or are single characters) and at least two
    segments spell >=3 letters. Otherwise single-letter runs are left alone and logged."""
    segs = re.split(r" {2,}|\n", t.strip())
    spaced_re = re.compile(r"^\S( \S)+$")
    long_sp = [bool(spaced_re.match(s)) and len(s) >= 2 * SPACED_RUN_MIN - 1 for s in segs]
    compat = [bool(spaced_re.match(s)) or len(s) == 1 for s in segs]
    if sum(long_sp) >= 2 and sum(compat) / len(segs) >= 0.6:
        return " ".join(s.replace(" ", "") if spaced_re.match(s) else s for s in segs)
    if re.search(r"(?<!\S)(?:\w ){%d,}\w(?!\S)" % (SPACED_RUN_MIN - 1), t):
        notes.append("spaced: single-letter run present but text not predominantly spaced; left unchanged")
    return t


def normalize(text: str, rules: tuple[str, ...] = ALL_RULES) -> NormResult:
    res = NormResult(original=text, normalized=text)
    cur = text
    for rule in ALL_RULES:
        if rule not in rules:
            continue
        before = cur
        if rule == "zero_width":
            cur = rule_zero_width(cur)
        elif rule == "nfkc":
            cur = rule_nfkc(cur)
        elif rule == "confusables":
            cur = rule_confusables(cur)
        elif rule == "base64":
            cur = rule_base64(cur, res.notes)
        elif rule == "spaced":
            cur = rule_spaced(cur, res.notes)
        if cur != before:
            res.fired.append(rule)
    res.normalized = cur
    return res
