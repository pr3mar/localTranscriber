import json
from pathlib import Path

import pytest

from localtranscriber import pipeline
from localtranscriber.audio import AudioError
from localtranscriber.models import Cue, Segment, Word
from localtranscriber.pipeline import discover_media, run, source_srt_path, target_srt_path
from localtranscriber.srt import read_srt, write_srt

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
        translate=[],
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
    assert target_srt_path(media, "en").name == "show.s01e01.en.srt"


def test_transcribes_macedonian_to_cyrillic_and_closes_engine(tmp_path):
    media, engine = make_media(tmp_path), FakeEngine()
    [result] = go([media], engine=engine)
    assert result.ok
    assert read_srt(source_srt_path(media, "mk")) == [Cue(0.0, 1.2, "Здраво свете.")]
    assert engine.closed


def test_translation_keeps_timings(tmp_path):
    media = make_media(tmp_path)
    [result] = go([media], translate=["en"])
    assert result.ok
    assert read_srt(target_srt_path(media, "en")) == [Cue(0.0, 1.2, "EN Здраво свете.")]


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
    [result] = go([media], translate=["en"], engine_factory=never)
    assert result.ok
    assert read_srt(target_srt_path(media, "en")) == [Cue(0.0, 1.0, "EN Здраво")]


def test_one_failing_file_does_not_stop_the_batch(tmp_path, fake_extract):
    bad, good = make_media(tmp_path, "bad.mkv"), make_media(tmp_path, "good.mkv")
    fake_extract.add("bad.mkv")
    results = go([bad, good], translate=["en"])
    assert "no audio stream" in results[0].error
    assert results[1].ok
    assert not target_srt_path(bad, "en").exists()
    assert target_srt_path(good, "en").exists()


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
    [result] = go([media], translate=["en"], complete_factory=unavailable)
    assert result.error == "translation unavailable: Ollama not reachable"
    assert source_srt_path(media, "mk").exists()


def test_from_srt(tmp_path):
    missing, present = make_media(tmp_path, "a.mkv"), make_media(tmp_path, "b.mkv")
    write_srt(source_srt_path(present, "mk"), [Cue(0.0, 1.0, "Здраво")])
    results = go([missing, present], translate=["en"], from_srt=True, engine_factory=never)
    assert "a.mk.srt not found" in results[0].error
    assert results[1].ok
    assert read_srt(target_srt_path(present, "en")) == [Cue(0.0, 1.0, "EN Здраво")]


def test_english_source_is_not_translated(tmp_path):
    media = make_media(tmp_path)
    [result] = go([media], lang="en", translate=["en"], complete_factory=never)
    assert result.ok
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ep1.en.srt", "ep1.mkv"]


def test_unicode_and_spaces_in_filename(tmp_path):
    media = make_media(tmp_path, "Епизода 1.mkv")
    [result] = go([media])
    assert result.ok
    assert (tmp_path / "Епизода 1.mk.srt").exists()


def test_no_speech_writes_empty_subtitles(tmp_path):
    media = make_media(tmp_path)
    [result] = go([media], engine=FakeEngine(segments=[]), translate=["en"])
    assert result.ok
    assert read_srt(source_srt_path(media, "mk")) == []
    assert read_srt(target_srt_path(media, "en")) == []


def test_same_stem_collision_fails_loudly(tmp_path):
    mkv, mp4 = make_media(tmp_path, "ep1.mkv"), make_media(tmp_path, "ep1.mp4")
    engine = FakeEngine()
    results = go([mkv, mp4], engine=engine)
    assert results[0].ok
    assert "ep1.mk.srt" in results[1].error and "ep1.mkv" in results[1].error
    assert engine.transcribed == ["mk"]


def targeted_complete(system, user, schema=None):
    tag = system.split("into natural, concise ")[1].split(".")[0]
    lines = json.loads(user)["lines"]
    return json.dumps({k: f"{tag} {v}" for k, v in lines.items()}, ensure_ascii=False)


def test_several_targets_write_all_files(tmp_path):
    media = make_media(tmp_path)
    [result] = go([media], translate=["en", "es"], complete_factory=lambda: targeted_complete)
    assert result.ok
    assert read_srt(target_srt_path(media, "en")) == [Cue(0.0, 1.2, "English Здраво свете.")]
    assert read_srt(target_srt_path(media, "es")) == [Cue(0.0, 1.2, "Spanish Здраво свете.")]


def test_existing_target_skipped_unless_overwrite(tmp_path):
    media = make_media(tmp_path)
    write_srt(source_srt_path(media, "mk"), [Cue(0.0, 1.0, "Здраво")])
    write_srt(target_srt_path(media, "en"), [Cue(0.0, 1.0, "old")])
    go([media], translate=["en", "es"], from_srt=True, complete_factory=lambda: targeted_complete)
    assert read_srt(target_srt_path(media, "en")) == [Cue(0.0, 1.0, "old")]
    assert target_srt_path(media, "es").exists()
    go([media], translate=["en"], from_srt=True, overwrite=True, complete_factory=lambda: targeted_complete)
    assert read_srt(target_srt_path(media, "en")) == [Cue(0.0, 1.0, "English Здраво")]


def test_failing_target_does_not_stop_other_targets_or_files(tmp_path):
    files = [make_media(tmp_path, "a.mkv"), make_media(tmp_path, "b.mkv")]
    write_srt(source_srt_path(files[0], "mk"), [Cue(0.0, 1.0, "Лош")])
    write_srt(source_srt_path(files[1], "mk"), [Cue(0.0, 1.0, "Добар")])

    def complete(system, user, schema=None):
        if "Spanish" in system and "Лош" in user:
            raise RuntimeError("boom")
        return targeted_complete(system, user, schema)

    results = go(files, translate=["es", "en"], from_srt=True, complete_factory=lambda: complete)
    assert "boom" in results[0].error and "es" in results[0].error
    assert results[1].ok
    assert target_srt_path(files[0], "en").exists()
    assert not target_srt_path(files[0], "es").exists()
    assert target_srt_path(files[1], "es").exists() and target_srt_path(files[1], "en").exists()


def test_complete_factory_called_once_for_several_targets(tmp_path):
    media = make_media(tmp_path)
    count = []

    def factory():
        count.append(1)
        return targeted_complete

    go([media], translate=["en", "es", "sl"], complete_factory=factory)
    assert len(count) == 1


def test_source_language_target_is_never_translated(tmp_path):
    media = make_media(tmp_path)
    [result] = go([media], translate=["mk"], complete_factory=never)
    assert result.ok
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ep1.mk.srt", "ep1.mkv"]
