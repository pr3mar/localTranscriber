import unicodedata

_COMMON = {
    "a": "а", "b": "б", "c": "ц", "č": "ч", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "ј", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о",
    "p": "п", "r": "р", "s": "с", "š": "ш", "t": "т", "u": "у", "v": "в", "z": "з",
    "ž": "ж",
}
_SERBIAN = {**_COMMON, "ć": "ћ", "đ": "ђ", "dž": "џ", "lj": "љ", "nj": "њ"}
_MACEDONIAN = {
    **_COMMON,
    "ǵ": "ѓ", "ḱ": "ќ", "dzh": "џ", "dž": "џ", "dz": "ѕ", "gj": "ѓ", "kj": "ќ",
    "lj": "љ", "nj": "њ", "ch": "ч", "sh": "ш", "zh": "ж",
}
_TABLES = {"sr": _SERBIAN, "mk": _MACEDONIAN}

CYRILLIC_LANGS = frozenset(_TABLES)


def to_cyrillic(text: str, lang: str) -> str:
    """Transliterate Latin letters to Cyrillic; longest match first, case preserved."""
    table = _TABLES[lang]
    longest = max(len(key) for key in table)
    text = unicodedata.normalize("NFC", text)
    out = []
    i = 0
    while i < len(text):
        for size in range(longest, 0, -1):
            chunk = text[i : i + size]
            mapped = table.get(chunk.lower()) if len(chunk) == size else None
            if mapped is not None:
                out.append(mapped.upper() if chunk[0].isupper() else mapped)
                i += size
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)
