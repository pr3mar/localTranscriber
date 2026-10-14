import re
import shutil
from pathlib import Path

import pytest

from localtranscribe import cli
from localtranscribe.audio import extract_audio
from localtranscribe.cues import build_cues
from localtranscribe.cyrillic import CYRILLIC_LANGS, to_cyrillic
from localtranscribe.router import load_engine
from localtranscribe.srt import read_srt

pytestmark = pytest.mark.gpu

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
LANGS = ["en", "es", "sl", "mk", "sr"]


def fixture_clip(lang):
    clips = [p for p in FIXTURES.glob(f"{lang}.*") if p.suffix != ".txt"]
    assert len(clips) == 1, f"expected exactly one clip for {lang} in {FIXTURES}"
    return clips[0], (FIXTURES / f"{lang}.txt").read_text(encoding="utf-8")


def normalise(text, lang):
    if lang in CYRILLIC_LANGS:
        text = to_cyrillic(text, lang)
    return set(re.findall(r"\w+", text.lower()))


@pytest.mark.parametrize("lang", LANGS)
def test_transcription_matches_reference(lang, tmp_path):
    clip, reference = fixture_clip(lang)
    wav = tmp_path / "audio.wav"
    extract_audio(clip, wav)
    engine = load_engine(lang)
    try:
        cues = build_cues(engine.transcribe(wav, lang))
    finally:
        engine.close()
    assert cues
    for a, b in zip(cues, cues[1:]):
        assert a.start <= a.end <= b.start
    expected, got = normalise(reference, lang), normalise(" ".join(c.text for c in cues), lang)
    assert len(expected & got) / len(expected) >= 0.6, (sorted(expected), sorted(got))


def test_cli_end_to_end_with_translation(tmp_path):
    clip, _ = fixture_clip("mk")
    media = tmp_path / f"Епизода 1{clip.suffix}"
    shutil.copy(clip, media)
    assert cli.main([str(media), "--lang", "mk", "--translate"]) == 0
    source = read_srt(tmp_path / "Епизода 1.mk.srt")
    target = read_srt(tmp_path / "Епизода 1.en.srt")
    assert source
    assert not re.search(r"[A-Za-z]", " ".join(c.text for c in source))
    assert [(c.start, c.end) for c in source] == [(c.start, c.end) for c in target]
    assert all(c.text.strip() for c in target)
