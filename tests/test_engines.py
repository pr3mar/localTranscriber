from pathlib import Path
from types import SimpleNamespace as NS

from localtranscriber.engines.parakeet import ParakeetEngine, segments_from_parakeet
from localtranscriber.engines.whisper import WhisperEngine, segments_from_whisper
from localtranscriber.models import Segment, Word

PARAKEET_STAMPS = {
    "word": [
        {"start": 0.0, "end": 0.4, "word": "Hola,"},
        {"start": 0.5, "end": 0.9, "word": "amigo."},
        {"start": 1.5, "end": 2.0, "word": "Adiós."},
    ],
    "segment": [
        {"start": 0.0, "end": 0.9, "segment": "Hola, amigo."},
        {"start": 1.5, "end": 2.0, "segment": "Adiós."},
    ],
}


def test_segments_from_whisper_strips_word_spaces():
    raw = [NS(start=0.0, end=1.0, text=" Здраво свет.",
              words=[NS(start=0.0, end=0.4, word=" Здраво"), NS(start=0.5, end=1.0, word=" свет.")])]
    assert segments_from_whisper(raw) == [
        Segment(0.0, 1.0, "Здраво свет.", (Word(0.0, 0.4, "Здраво"), Word(0.5, 1.0, "свет.")))
    ]


def test_segments_from_whisper_without_words():
    raw = [NS(start=0.0, end=1.0, text=" Здраво", words=None)]
    assert segments_from_whisper(raw) == [Segment(0.0, 1.0, "Здраво", ())]


def test_segments_from_parakeet_assigns_words_to_segments():
    segments = segments_from_parakeet(NS(timestamp=PARAKEET_STAMPS))
    assert [s.text for s in segments] == ["Hola, amigo.", "Adiós."]
    assert [[w.text for w in s.words] for s in segments] == [["Hola,", "amigo."], ["Adiós."]]


def test_segments_from_parakeet_words_without_segments():
    stamps = {"word": PARAKEET_STAMPS["word"], "segment": []}
    [segment] = segments_from_parakeet(NS(timestamp=stamps))
    assert (segment.start, segment.end, segment.text) == (0.0, 2.0, "Hola, amigo. Adiós.")
    assert len(segment.words) == 3


def test_segments_from_parakeet_empty():
    assert segments_from_parakeet(NS(timestamp={"word": [], "segment": []})) == []


class FakeNemoModel:
    def __init__(self):
        self.attention = None
        self.calls = []

    def change_attention_model(self, self_attention_model, att_context_size):
        self.attention = (self_attention_model, att_context_size)

    def transcribe(self, paths, timestamps):
        self.calls.append((paths, timestamps))
        return [NS(timestamp=PARAKEET_STAMPS)]


def test_parakeet_engine_always_uses_local_attention_for_long_audio():
    model = FakeNemoModel()
    engine = ParakeetEngine(model=model)
    assert model.attention == ("rel_pos_local_attn", [256, 256])
    segments = engine.transcribe(Path("a.wav"), "sl")
    assert model.calls == [(["a.wav"], True)]
    assert len(segments) == 2
    engine.close()


class FakeWhisperModel:
    def transcribe(self, path, **kwargs):
        self.path, self.kwargs = path, kwargs
        return iter([NS(start=0.0, end=1.0, text=" Здраво", words=None)]), NS(language="mk")


def test_whisper_engine_forces_language_and_word_timestamps():
    model = FakeWhisperModel()
    engine = WhisperEngine(model=model)
    assert engine.transcribe(Path("a.wav"), "mk") == [Segment(0.0, 1.0, "Здраво", ())]
    assert model.path == "a.wav"
    assert model.kwargs["language"] == "mk"
    assert model.kwargs["word_timestamps"] is True
    assert model.kwargs["vad_filter"] is False
    assert model.kwargs["condition_on_previous_text"] is False
    assert model.kwargs["hallucination_silence_threshold"] > 0
    engine.close()


def test_segments_from_whisper_drops_known_hallucinations():
    raw = [
        NS(start=0.0, end=1.0, text=" Хвала што пратите канал.", words=None),
        NS(start=1.0, end=2.0, text=" Субтитрујуће", words=None),
        NS(start=2.0, end=3.0, text=" Хвала што си дошао.", words=None),
    ]
    assert segments_from_whisper(raw) == [Segment(2.0, 3.0, "Хвала што си дошао.", ())]
