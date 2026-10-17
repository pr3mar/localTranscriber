import gc
from pathlib import Path

from localtranscriber.models import Segment, Word

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def segments_from_parakeet(hypothesis) -> list[Segment]:
    stamps = hypothesis.timestamp
    words = [
        Word(float(w["start"]), float(w["end"]), str(w["word"]).strip())
        for w in stamps.get("word", [])
        if str(w["word"]).strip()
    ]
    raw_segments = stamps.get("segment", [])
    if not raw_segments:
        if not words:
            return []
        return [Segment(words[0].start, words[-1].end, " ".join(w.text for w in words), tuple(words))]
    segments = []
    i = 0
    for raw in raw_segments:
        start, end = float(raw["start"]), float(raw["end"])
        seg_words = []
        while i < len(words) and words[i].start < end:
            seg_words.append(words[i])
            i += 1
        segments.append(Segment(start, end, str(raw["segment"]).strip(), tuple(seg_words)))
    if i < len(words):
        last = segments[-1]
        segments[-1] = Segment(last.start, max(last.end, words[-1].end), last.text, last.words + tuple(words[i:]))
    return segments


class ParakeetEngine:
    def __init__(self, model_name: str = MODEL_NAME, model=None):
        if model is None:
            import nemo.collections.asr as nemo_asr

            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        # Local attention handles hours of audio in one pass; full attention tops out around 24 minutes.
        model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        self._model = model

    def transcribe(self, wav: Path, lang: str) -> list[Segment]:
        # Parakeet v3 detects the spoken language itself; `lang` only selected this engine.
        hypotheses = self._model.transcribe([str(wav)], timestamps=True)
        return segments_from_parakeet(hypotheses[0])

    def close(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
