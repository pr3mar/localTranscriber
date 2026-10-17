import shutil
import subprocess
import wave

import pytest

from localtranscriber import audio
from localtranscriber.audio import AudioError, extract_audio


class FakeRun:
    def __init__(self, probe_stdout="0\n", probe_rc=0, ffmpeg_rc=0):
        self.calls = []
        self.probe_stdout, self.probe_rc, self.ffmpeg_rc = probe_stdout, probe_rc, ffmpeg_rc

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, self.probe_rc, self.probe_stdout, "probe broke")
        return subprocess.CompletedProcess(cmd, self.ffmpeg_rc, "", "ffmpeg exploded")


def test_extract_builds_ffmpeg_command_with_path_as_single_argument(monkeypatch, tmp_path):
    fake = FakeRun()
    monkeypatch.setattr(audio.subprocess, "run", fake)
    src, dst = tmp_path / "Епизода 1 (2024).s01e01.mkv", tmp_path / "out.wav"
    extract_audio(src, dst)
    probe, ffmpeg = fake.calls
    assert probe[-1] == str(src)
    assert ffmpeg[0] == "ffmpeg"
    assert ffmpeg[ffmpeg.index("-i") + 1] == str(src)
    assert ffmpeg[ffmpeg.index("-ac") + 1] == "1"
    assert ffmpeg[ffmpeg.index("-ar") + 1] == "16000"
    assert ffmpeg[-1] == str(dst)


def test_no_audio_stream(monkeypatch, tmp_path):
    fake = FakeRun(probe_stdout="")
    monkeypatch.setattr(audio.subprocess, "run", fake)
    with pytest.raises(AudioError, match="no audio stream"):
        extract_audio(tmp_path / "a.mkv", tmp_path / "out.wav")
    assert len(fake.calls) == 1


def test_ffprobe_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(audio.subprocess, "run", FakeRun(probe_rc=1))
    with pytest.raises(AudioError, match="ffprobe failed.*probe broke"):
        extract_audio(tmp_path / "a.mkv", tmp_path / "out.wav")


def test_ffmpeg_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(audio.subprocess, "run", FakeRun(ffmpeg_rc=1))
    with pytest.raises(AudioError, match="ffmpeg exploded"):
        extract_audio(tmp_path / "a.mkv", tmp_path / "out.wav")


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed; runs in the container")


@needs_ffmpeg
def test_real_extract(tmp_path):
    src = tmp_path / "tone.flac"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-ac", "2", "-ar", "44100", str(src)],
        check=True,
    )
    dst = tmp_path / "out.wav"
    extract_audio(src, dst)
    with wave.open(str(dst)) as w:
        assert (w.getnchannels(), w.getframerate()) == (1, 16000)


@needs_ffmpeg
def test_real_video_without_audio(tmp_path):
    src = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
         "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    with pytest.raises(AudioError, match="no audio stream"):
        extract_audio(src, tmp_path / "out.wav")
