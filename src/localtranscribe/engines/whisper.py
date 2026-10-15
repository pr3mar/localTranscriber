import gc
import re
from collections.abc import Iterable
from pathlib import Path

from localtranscribe.models import Segment, Word

# Whisper (trained on subtitled video) invents these over music and silence; a segment that is only
# such a phrase is dropped. VAD stays off (it cut real speech), so this list is the guard.
_HALLUCINATION = re.compile(
    r"^\W*(хвала (вам )?што (пратите|гледате)( (канал|видео))?"
    r"|благодарам што (го )?(следите|гледате)( (каналот|видеото))?"
    r"|(продавање )?субтитр\w*|титлов\w*|аплодисмент\w*|звућање)\W*$",
    re.IGNORECASE,
)
# Skip silent stretches longer than this (seconds) when a segment looks hallucinated.
HALLUCINATION_SILENCE_THRESHOLD = 2.0


def segments_from_whisper(raw_segments: Iterable) -> list[Segment]:
    segments = []
    for s in raw_segments:
        text = s.text.strip()
        if _HALLUCINATION.match(text):
            continue
        words = tuple(
            Word(float(w.start), float(w.end), w.word.strip()) for w in (s.words or []) if w.word.strip()
        )
        segments.append(Segment(float(s.start), float(s.end), text, words))
    return segments


class WhisperEngine:
    def __init__(self, model_name: str = "large-v3", model=None):
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_name, device="cuda", compute_type="float16")
        self._model = model

    def transcribe(self, wav: Path, lang: str) -> list[Segment]:
        segments, _info = self._model.transcribe(
            str(wav),
            language=lang,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
            hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
            beam_size=5,
        )
        return segments_from_whisper(segments)

    def close(self) -> None:
        self._model = None
        gc.collect()
