import gc
from collections.abc import Iterable
from pathlib import Path

from localtranscribe.models import Segment, Word


def segments_from_whisper(raw_segments: Iterable) -> list[Segment]:
    segments = []
    for s in raw_segments:
        words = tuple(
            Word(float(w.start), float(w.end), w.word.strip()) for w in (s.words or []) if w.word.strip()
        )
        segments.append(Segment(float(s.start), float(s.end), s.text.strip(), words))
    return segments


class WhisperEngine:
    def __init__(self, model_name: str = "large-v3", model=None):
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_name, device="cuda", compute_type="float16")
        self._model = model

    def transcribe(self, wav: Path, lang: str) -> list[Segment]:
        segments, _info = self._model.transcribe(
            str(wav), language=lang, word_timestamps=True, vad_filter=True, beam_size=5
        )
        return segments_from_whisper(segments)

    def close(self) -> None:
        self._model = None
        gc.collect()
