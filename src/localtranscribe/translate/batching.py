import json
import logging
from collections.abc import Callable

from localtranscribe.cues import split_to_fit
from localtranscribe.models import Cue

logger = logging.getLogger(__name__)

Complete = Callable[[str, str, dict], str]

BATCH_SIZE = 15
CONTEXT_SIZE = 3
ATTEMPTS = 2

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


def translate_cues(cues: list[Cue], source_lang: str, complete: Complete) -> list[Cue]:
    """Translate cue texts to English; over-long translations are split within their source cue's span."""
    texts = [" ".join(c.text.split()) for c in cues]
    translated: list[str] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        context = texts[max(0, start - CONTEXT_SIZE) : start]
        translated.extend(_translate_lines(batch, context, source_lang, complete))
    out: list[Cue] = []
    for cue, text in zip(cues, translated, strict=True):
        out.extend(split_to_fit(Cue(cue.start, cue.end, text)))
    return out
