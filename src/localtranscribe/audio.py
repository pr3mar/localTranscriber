import subprocess
from pathlib import Path


class AudioError(Exception):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def has_audio_stream(src: Path) -> bool:
    proc = _run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                 "-of", "csv=p=0", str(src)])
    if proc.returncode != 0:
        raise AudioError(f"ffprobe failed for {src.name}: {proc.stderr.strip()[-500:]}")
    return proc.stdout.strip() != ""


def extract_audio(src: Path, dst: Path) -> None:
    """Decode any ffmpeg-readable media into 16 kHz mono 16-bit PCM WAV."""
    if not has_audio_stream(src):
        raise AudioError(f"{src.name} has no audio stream")
    proc = _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                 "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)])
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed for {src.name}: {proc.stderr.strip()[-500:]}")
