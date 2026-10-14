import json
import logging
from collections.abc import Callable

from localtranscribe.cues import _CLAUSE_END, _SENTENCE_END, split_to_fit
from localtranscribe.models import Cue

logger = logging.getLogger(__name__)

Complete = Callable[[str, str, dict], str]

BATCH_SIZE = 15
CONTEXT_SIZE = 3
ATTEMPTS = 2
# A sentence unit (cues translated together as one line) ends at sentence punctuation, but never grows
# beyond MAX_UNIT_CUES cues or MAX_UNIT_CHARS characters, so unpunctuated ASR output stays in small units.
MAX_UNIT_CUES = 5
MAX_UNIT_CHARS = 250
_CLOSERS = "\"'“”‘’«»)]"  # closing quotes/brackets that may follow sentence punctuation

LANGUAGE_NAMES = {"mk": "Macedonian", "sr": "Serbian", "sl": "Slovenian", "es": "Spanish", "en": "English"}

SYSTEM_PROMPT = (
    "You translate subtitles into natural, concise English. "
    "The user message is a JSON object with 'source_language', optional 'context' and 'lines'. "
    "Translate each value of 'lines' into English and reply with a JSON object with exactly the same keys, "
    "one translation per key. Never merge or split lines. Lines already in English are returned unchanged. "
    "The source is mostly the given language but may contain other languages. "
    "'context' is earlier subtitle text for understanding only: do not translate it."
)


def build_user_prompt(lines: list[str], context: list[str], source_lang: str) -> str:
    payload: dict = {"source_language": LANGUAGE_NAMES[source_lang]}
    if context:
        payload["context"] = context
    payload["lines"] = {str(i): line for i, line in enumerate(lines, 1)}
    return json.dumps(payload, ensure_ascii=False)


def build_schema(keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {k: {"type": "string"} for k in keys},
        "required": keys,
        "additionalProperties": False,
    }


def parse_reply(reply: str, expected: int) -> list[str] | None:
    try:
        data = json.loads(reply)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    out = []
    for i in range(1, expected + 1):
        value = data.get(str(i))
        if not isinstance(value, str) or not value.strip():
            return None
        out.append(value.strip())
    return out


def _translate_lines(lines: list[str], context: list[str], source_lang: str, complete: Complete) -> list[str]:
    user = build_user_prompt(lines, context, source_lang)
    schema = build_schema([str(i) for i in range(1, len(lines) + 1)])
    for _ in range(ATTEMPTS):
        parsed = parse_reply(complete(SYSTEM_PROMPT, user, schema), len(lines))
        if parsed is not None:
            return parsed
    if len(lines) == 1:
        logger.warning("could not translate line, keeping source text: %r", lines[0])
        return lines
    mid = len(lines) // 2
    left = _translate_lines(lines[:mid], context, source_lang, complete)
    right_context = (context + lines[:mid])[-CONTEXT_SIZE:]
    return left + _translate_lines(lines[mid:], right_context, source_lang, complete)


def _ends(text: str, ends: tuple[str, ...]) -> bool:
    return text.rstrip(_CLOSERS).endswith(ends)


def _units(texts: list[str]) -> list[list[int]]:
    """Group consecutive cue indices into sentence units (see MAX_UNIT_CUES / MAX_UNIT_CHARS)."""
    units: list[list[int]] = []
    current: list[int] = []
    for i, text in enumerate(texts):
        if current and (
            len(current) >= MAX_UNIT_CUES or len(" ".join(texts[j] for j in current + [i])) > MAX_UNIT_CHARS
        ):
            units.append(current)
            current = []
        current.append(i)
        if _ends(text, _SENTENCE_END):
            units.append(current)
            current = []
    if current:
        units.append(current)
    return units


def _break_near(text: str, spaces: list[int], target: float) -> int:
    """Index into spaces: sentence break, else clause break, near the target; else the nearest space."""
    window = len(text) / 8
    for ends in (_SENTENCE_END, _CLAUSE_END):
        options = [k for k, i in enumerate(spaces) if _ends(text[:i], ends) and abs(i - target) <= window]
        if options:
            return min(options, key=lambda k: abs(spaces[k] - target))
    return min(range(len(spaces)), key=lambda k: abs(spaces[k] - target))


def _distribute(english: str, sources: list[str]) -> list[str]:
    """Split english into one piece per source text, at word boundaries, proportional to source length."""
    words = english.split()
    if len(words) < len(sources):
        return words
    text = " ".join(words)
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    total = max(1, sum(len(s) for s in sources))
    pieces: list[str] = []
    begin, first, done = 0, 0, 0
    for n, source in enumerate(sources[:-1]):
        done += len(source)
        remaining = len(sources) - 2 - n
        allowed = spaces[first : len(spaces) - remaining]
        k = first + _break_near(text, allowed, len(text) * done / total)
        pieces.append(text[begin : spaces[k]])
        begin, first = spaces[k] + 1, k + 1
    pieces.append(text[begin:])
    return pieces


def translate_cues(cues: list[Cue], source_lang: str, complete: Complete) -> list[Cue]:
    """Translate whole sentence units, then distribute each unit's English back over its cues' time spans.

    Over-long pieces are split within their piece's span; if a unit's English has fewer words than
    the unit has cues, the trailing cues' spans are merged into the last piece's cue.
    """
    texts = [" ".join(c.text.split()) for c in cues]
    units = _units(texts)
    unit_texts = [" ".join(texts[i] for i in unit) for unit in units]
    translated: list[str] = []
    for start in range(0, len(unit_texts), BATCH_SIZE):
        batch = unit_texts[start : start + BATCH_SIZE]
        context = unit_texts[max(0, start - CONTEXT_SIZE) : start]
        translated.extend(_translate_lines(batch, context, source_lang, complete))
    out: list[Cue] = []
    for unit, source, english in zip(units, unit_texts, translated, strict=True):
        sources = [texts[i] for i in unit]
        pieces = sources if english == source else _distribute(english, sources)
        for n, piece in enumerate(pieces):
            last = unit[-1] if n == len(pieces) - 1 else unit[n]
            out.extend(split_to_fit(Cue(cues[unit[n]].start, cues[last].end, piece)))
    return out
