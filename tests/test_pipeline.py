import json
from pathlib import Path

import pytest

from localtranscribe import pipeline
from localtranscribe.audio import AudioError
from localtranscribe.models import Cue, Segment, Word
from localtranscribe.pipeline import discover_media, run, source_srt_path, target_srt_path
from localtranscribe.srt import read_srt, write_srt

SEGMENTS = [Segment(0.0, 1.2, "Zdravo svete.", (Word(0.0, 0.5, "Zdravo"), Word(0.6, 1.2, "svete.")))]


class FakeEngine:
    def __init__(self, segments=SEGMENTS):
        self.segments = segments
        self.transcribed = []
        self.closed = False

    def transcribe(self, wav, lang):
        self.transcribed.append(lang)
        return self.segments

    def close(self):
        self.closed = True


def echo_complete(system, user, schema=None):
    lines = json.loads(user)["lines"]
    return json.dumps({k: f"EN {v}" for k, v in lines.items()}, ensure_ascii=False)


def never(*args):
    raise AssertionError("must not be called")


@pytest.fixture(autouse=True)
def fake_extract(monkeypatch):
    failing = set()

    def extract(src, dst):
        if src.name in failing:
            raise AudioError(f"{src.name} has no audio stream")
        dst.write_bytes(b"")

    monkeypatch.setattr(pipeline, "extract_audio", extract)
    return failing


def make_media(tmp_path, name="ep1.mkv"):
    path = tmp_path / name
    path.write_bytes(b"x")
    return path


def go(files, lang="mk", engine=None, **overrides):
    kwargs = dict(
        translate=False,
        overwrite=False,
        from_srt=False,
        engine_factory=lambda _lang: engine or FakeEngine(),
        complete_factory=lambda: echo_complete,
    )
    kwargs.update(overrides)
    return run(files, lang, **kwargs)


def test_discover_media(tmp_path):
    for name in ["ep2.mkv", "ep1.MP4", "notes.txt", "ep1.mk.srt"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "ep3.mkv").write_bytes(b"x")
    assert [p.name for p in discover_media(tmp_path)] == ["ep1.MP4", "ep2.mkv"]
    assert discover_media(tmp_path / "ep2.mkv") == [tmp_path / "ep2.mkv"]
    with pytest.raises(FileNotFoundError):
        discover_media(tmp_path / "missing")


def test_output_paths_keep_dotted_stems():
    media = Path("show.s01e01.mkv")
    assert source_srt_path(media, "mk").name == "show.s01e01.mk.srt"
    assert target_srt_path(media).name == "show.s01e01.en.srt"


def test_transcribes_macedonian_to_cyrillic_and_closes_engine(tmp_path):
    media, engine = make_media(tmp_path), FakeEngine()
    [result] = go([media], engine=engine)
    assert result.ok
    assert read_srt(source_srt_path(media, "mk")) == [Cue(0.0, 1.2, "Здраво свете.")]
    assert engine.closed


def test_translation_keeps_timings(tmp_path):
    media = make_media(tmp_path)
    [result] = go([media], translate=True)
    assert result.ok
    assert read_srt(target_srt_path(media)) == [Cue(0.0, 1.2, "EN Здраво свете.")]


def test_existing_source_is_skipped_without_loading_engine(tmp_path):
    media = make_media(tmp_path)
    write_srt(source_srt_path(media, "mk"), [Cue(0.0, 1.0, "старо")])
    [result] = go([media], engine_factory=never)
    assert result.ok
    assert read_srt(source_srt_path(media, "mk")) == [Cue(0.0, 1.0, "старо")]


def test_overwrite_retranscribes(tmp_path):
    media = make_media(tmp_path)
    write_srt(source_srt_path(media, "mk"), [Cue(0.0, 1.0, "старо")])
    go([media], overwrite=True)
    assert read_srt(source_srt_path(media, "mk")) == [Cue(0.0, 1.2, "Здраво свете.")]


def test_rerun_after_failed_translation_only_translates(tmp_path):
    media = make_media(tmp_path)
    write_srt(source_srt_path(media, "mk"), [Cue(0.0, 1.0, "Здраво")])
    [result] = go([media], translate=True, engine_factory=never)
    assert result.ok
    assert read_srt(target_srt_path(media)) == [Cue(0.0, 1.0, "EN Здраво")]


def test_one_failing_file_does_not_stop_the_batch(tmp_path, fake_extract):
    bad, good = make_media(tmp_path, "bad.mkv"), make_media(tmp_path, "good.mkv")
    fake_extract.add("bad.mkv")
    results = go([bad, good], translate=True)
    assert "no audio stream" in results[0].error
    assert results[1].ok
    assert not target_srt_path(bad).exists()
    assert target_srt_path(good).exists()


def test_engine_load_failure_fails_all_pending_files(tmp_path):
    def broken(lang):
        raise RuntimeError("CUDA out of memory")

    files = [make_media(tmp_path, "a.mkv"), make_media(tmp_path, "b.mkv")]
    results = go(files, engine_factory=broken)
    assert all("CUDA out of memory" in r.error for r in results)
    assert not any(source_srt_path(f, "mk").exists() for f in files)


def test_translator_unavailable_keeps_source_and_reports(tmp_path):
    def unavailable():
        raise RuntimeError("Ollama not reachable")

    media = make_media(tmp_path)
    [result] = go([media], translate=True, complete_factory=unavailable)
    assert result.error == "translation unavailable: Ollama not reachable"
    assert source_srt_path(media, "mk").exists()


def test_from_srt(tmp_path):
    missing, present = make_media(tmp_path, "a.mkv"), make_media(tmp_path, "b.mkv")
    write_srt(source_srt_path(present, "mk"), [Cue(0.0, 1.0, "Здраво")])
    results = go([missing, present], translate=True, from_srt=True, engine_factory=never)
    assert "a.mk.srt not found" in results[0].error
    assert results[1].ok
    assert read_srt(target_srt_path(present)) == [Cue(0.0, 1.0, "EN Здраво")]


def test_english_source_is_not_translated(tmp_path):
    media = make_media(tmp_path)
    [result] = go([media], lang="en", translate=True, complete_factory=never)
    assert result.ok
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ep1.en.srt", "ep1.mkv"]


def test_unicode_and_spaces_in_filename(tmp_path):
    media = make_media(tmp_path, "Епизода 1.mkv")
    [result] = go([media])
    assert result.ok
    assert (tmp_path / "Епизода 1.mk.srt").exists()


def test_no_speech_writes_empty_subtitles(tmp_path):
    media = make_media(tmp_path)
    [result] = go([media], engine=FakeEngine(segments=[]), translate=True)
    assert result.ok
    assert read_srt(source_srt_path(media, "mk")) == []
    assert read_srt(target_srt_path(media)) == []


def test_same_stem_collision_fails_loudly(tmp_path):
    mkv, mp4 = make_media(tmp_path, "ep1.mkv"), make_media(tmp_path, "ep1.mp4")
    engine = FakeEngine()
    results = go([mkv, mp4], engine=engine)
    assert results[0].ok
    assert "ep1.mk.srt" in results[1].error and "ep1.mkv" in results[1].error
    assert engine.transcribed == ["mk"]
