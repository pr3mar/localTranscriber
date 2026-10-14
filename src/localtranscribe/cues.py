from collections.abc import Iterator

from localtranscribe.models import Cue, Segment, Word

MAX_CHARS = 42
MAX_DURATION = 7.0
MIN_DURATION = 1.0
PAUSE_SPLIT = 0.6

_SENTENCE_END = (".", "?", "!", "…")
_CLAUSE_END = (",", ";", ":")


def wrap_text(text: str) -> str | None:
    """One line if it fits, else the most balanced two lines; None if it cannot fit."""
    words = text.split()
    single = " ".join(words)
    if len(single) <= MAX_CHARS:
        return single
    best: tuple[int, str] | None = None
    for i in range(1, len(words)):
        first, second = " ".join(words[:i]), " ".join(words[i:])
        if len(first) <= MAX_CHARS and len(second) <= MAX_CHARS:
            score = max(len(first), len(second))
            if best is None or score < best[0]:
                best = (score, f"{first}\n{second}")
    return best[1] if best else None


def _join(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


def _interpolate(segment: Segment) -> Iterator[Word]:
    tokens = segment.text.split()
    total = sum(len(t) for t in tokens)
    duration = segment.end - segment.start
    t = segment.start
    for token in tokens:
        end = t + duration * len(token) / total
        yield Word(t, end, token)
        t = end


def _words_of(segments: list[Segment]) -> Iterator[Word]:
    for segment in segments:
        if segment.words:
            yield from (w for w in segment.words if w.text.strip())
        elif segment.text.strip():
            yield from _interpolate(segment)


def _must_break(current: list[Word], word: Word) -> bool:
    return word.start - current[-1].end >= PAUSE_SPLIT or word.end - current[0].start > MAX_DURATION


def _clause_cut(words: list[Word]) -> int:
    """Index just after the last clause-ending word in the second half, else len(words)."""
    for i in range(len(words) - 1, len(words) // 2 - 1, -1):
        if words[i].text.endswith(_CLAUSE_END):
            return i + 1
    return len(words)


def build_cues(segments: list[Segment]) -> list[Cue]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in _words_of(segments):
        if current and _must_break(current, word):
            groups.append(current)
            current = []
        elif current and wrap_text(_join(current + [word])) is None:
            cut = _clause_cut(current)
            groups.append(current[:cut])
            current = current[cut:]
            if current and wrap_text(_join(current + [word])) is None:
                groups.append(current)
                current = []
        current.append(word)
        if word.text.endswith(_SENTENCE_END):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return _finalize(groups)


def _finalize(groups: list[list[Word]]) -> list[Cue]:
    raw = []
    for group in groups:
        text = _join(group)
        raw.append(Cue(group[0].start, group[-1].end, wrap_text(text) or text))
    cues: list[Cue] = []
    i = 0
    while i < len(raw):
        cue = raw[i]
        i += 1
        while i < len(raw) and _clamped_end(cue, raw[i]) - cue.start < MIN_DURATION:
            merged = _merge(cue, raw[i], MAX_DURATION)
            if merged is None:
                break
            cue = merged
            i += 1
        end = _clamped_end(cue, raw[i] if i < len(raw) else None)
        if end <= cue.start and cues:
            merged = _merge(cues[-1], cue, None)
            if merged is not None:
                cues[-1] = merged
                continue
        cues.append(Cue(cue.start, max(end, cue.start), cue.text))
    return cues


def _clamped_end(cue: Cue, following: Cue | None) -> float:
    end = max(cue.end, cue.start + MIN_DURATION)
    return min(end, following.start) if following else end


def _merge(first: Cue, second: Cue, max_span: float | None) -> Cue | None:
    if max_span is not None and second.end - first.start > max_span:
        return None
    text = wrap_text(f"{first.text} {second.text}")
    if text is None:
        return None
    return Cue(first.start, max(first.end, second.end), text)
