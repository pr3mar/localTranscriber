import logging
import re
from collections.abc import Callable

from localtranscribe.cues import wrap_text
from localtranscribe.models import Cue

logger = logging.getLogger(__name__)

Complete = Callable[[str, str], str]

BATCH_SIZE = 15
CONTEXT_SIZE = 3
ATTEMPTS = 2

LANGUAGE_NAMES = {"mk": "Macedonian", "sr": "Serbian", "sl": "Slovenian", "es": "Spanish", "en": "English"}

SYSTEM_PROMPT = (
    "You translate subtitles into natural, concise English. "
    "You receive numbered subtitle lines, optionally preceded by earlier lines for context. "
    "Translate ONLY the numbered lines. Reply with exactly one line per numbered input line, "
    "in the form '[N] translation', in the same order, and nothing else. "
    "Lines already in English are returned unchanged. Never merge or split lines."
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUMBERED = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[.):](?=\s|$))\s*(.*?)\s*$")


def build_user_prompt(lines: list[str], context: list[str], source_lang: str) -> str:
    parts = [f"Source language: mostly {LANGUAGE_NAMES[source_lang]}; some lines may be in other languages."]
    if context:
        parts.append("Context (do not translate):")
        parts.extend(context)
    parts.append("Translate:")
    parts.extend(f"[{i}] {line}" for i, line in enumerate(lines, 1))
    return "\n".join(parts)


def parse_numbered(reply: str, expected: int) -> list[str] | None:
    found: dict[int, str] = {}
    for line in _THINK.sub("", reply).splitlines():
        match = _NUMBERED.match(line)
        if match and match.group(3):
            n = int(match.group(1) or match.group(2))
            if 1 <= n <= expected and n not in found:
                found[n] = match.group(3)
    if len(found) != expected:
        return None
    return [found[i] for i in range(1, expected + 1)]


def _parse_single(reply: str) -> str | None:
    for line in _THINK.sub("", reply).splitlines():
        match = _NUMBERED.match(line)
        text = match.group(3) if match else line.strip()
        if text:
            return text
    return None


def _translate_lines(lines: list[str], context: list[str], source_lang: str, complete: Complete) -> list[str]:
    user = build_user_prompt(lines, context, source_lang)
    for _ in range(ATTEMPTS):
        reply = complete(SYSTEM_PROMPT, user)
        parsed = parse_numbered(reply, len(lines))
        if parsed is None and len(lines) == 1:
            single = _parse_single(reply)
            parsed = [single] if single else None
        if parsed is not None:
            return parsed
    if len(lines) == 1:
        logger.warning("could not translate line, keeping source text: %r", lines[0])
        return lines
    mid = len(lines) // 2
    left = _translate_lines(lines[:mid], context, source_lang, complete)
    right_context = (context + lines[:mid])[-CONTEXT_SIZE:]
    return left + _translate_lines(lines[mid:], right_context, source_lang, complete)


def translate_cues(cues: list[Cue], source_lang: str, complete: Complete) -> list[Cue]:
    """Translate cue texts to English; cue count and timings are always preserved."""
    texts = [" ".join(c.text.split()) for c in cues]
    translated: list[str] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        context = texts[max(0, start - CONTEXT_SIZE) : start]
        translated.extend(_translate_lines(batch, context, source_lang, complete))
    return [Cue(c.start, c.end, wrap_text(t) or t) for c, t in zip(cues, translated, strict=True)]
