"""Deterministic obfuscation transforms (primary experiment: one transform per variant).

Each function returns ``(transformed_text, params)``. No randomness is used; every
choice is a pure function of the input text so variants are exactly reproducible.
"""
from __future__ import annotations

import base64

# Latin -> (Cyrillic, Greek-or-None) lookalikes. Every *second* eligible character
# (eligible index 0, 2, 4, ...) is replaced. Greek is used on alternate replacements
# when a Greek lookalike exists, otherwise Cyrillic.
HOMOGLYPHS: dict[str, tuple[str, str | None]] = {
    "a": ("а", None), "c": ("с", None), "e": ("е", None),
    "o": ("о", "ο"), "p": ("р", "ρ"), "x": ("х", "χ"),
    "y": ("у", None), "i": ("і", "ι"), "s": ("ѕ", None),
    "j": ("ј", None), "v": ("в", "ν"), "u": ("ц", "υ"),
    "k": ("к", "κ"),
    "A": ("А", "Α"), "B": ("В", "Β"), "C": ("С", None),
    "E": ("Е", "Ε"), "H": ("Н", "Η"), "K": ("К", "Κ"),
    "M": ("М", "Μ"), "O": ("О", "Ο"), "P": ("Р", "Ρ"),
    "T": ("Т", "Τ"), "X": ("Х", "Χ"), "Y": ("У", "Υ"),
}

ZERO_WIDTH_CYCLE = ["​", "‌", "‍", "⁠"]
ZW_EVERY = 2  # insert after every 2 letters inside a run of letters/digits


def homoglyph(text: str) -> tuple[str, dict]:
    out, eligible, replaced, greek_toggle = [], 0, 0, False
    for ch in text:
        pair = HOMOGLYPHS.get(ch)
        if pair is None:
            out.append(ch)
            continue
        if eligible % 2 == 0:
            cyr, grk = pair
            if grk is not None and greek_toggle:
                out.append(grk)
            else:
                out.append(cyr)
            if grk is not None:
                greek_toggle = not greek_toggle
            replaced += 1
        else:
            out.append(ch)
        eligible += 1
    return "".join(out), {"rule": "replace every 2nd eligible Latin char", "eligible": eligible,
                          "replaced": replaced}


def zero_width(text: str) -> tuple[str, dict]:
    out, run, inserted = [], 0, 0
    for ch in text:
        out.append(ch)
        if ch.isalnum():
            run += 1
            if run % ZW_EVERY == 0:
                out.append(ZERO_WIDTH_CYCLE[inserted % len(ZERO_WIDTH_CYCLE)])
                inserted += 1
        else:
            run = 0
    return "".join(out), {"rule": f"insert after every {ZW_EVERY} alnum chars within a word",
                          "cycle": [f"U+{ord(c):04X}" for c in ZERO_WIDTH_CYCLE], "inserted": inserted}


def base64_encode(text: str) -> tuple[str, dict]:
    enc = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return enc, {"format": "RFC4648 standard alphabet, padded, UTF-8, bare string (no prefix/instruction)"}


def spaced(text: str) -> tuple[str, dict]:
    words = text.split()
    out = "   ".join(" ".join(w) for w in words)
    return out, {"rule": "single space between all characters of a word; 3 spaces between words"}


TRANSFORMS = {
    "original": lambda t: (t, {}),
    "homoglyph": homoglyph,
    "zero_width": zero_width,
    "base64": base64_encode,
    "spaced": spaced,
}
CONDITIONS_ORDER = ["original", "homoglyph", "zero_width", "base64", "spaced"]
