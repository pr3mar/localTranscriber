import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from localtranscriber.audio import extract_audio
from localtranscriber.cues import build_cues
from localtranscriber.cyrillic import CYRILLIC_LANGS, to_cyrillic
from localtranscriber.engines.base import Engine
from localtranscriber.models import Cue
from localtranscriber.srt import read_srt, write_srt
from localtranscriber.translate.batching import Complete, translate_cues

logger = logging.getLogger(__name__)

MEDIA_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac", ".ogg"})


@dataclass
class FileResult:
    media: Path
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def discover_media(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)
    raise FileNotFoundError(path)


def source_srt_path(media: Path, lang: str) -> Path:
    return media.with_name(f"{media.stem}.{lang}.srt")


def target_srt_path(media: Path, target_lang: str) -> Path:
    return source_srt_path(media, target_lang)


def run(
    media_files: list[Path],
    lang: str,
    *,
    translate: list[str],
    overwrite: bool,
    from_srt: bool,
    engine_factory: Callable[[str], Engine],
    complete_factory: Callable[[], Complete],
) -> list[FileResult]:
    results = [FileResult(m) for m in media_files]
    _reject_shared_outputs(results, lang)
    if from_srt:
        for r in results:
            source = source_srt_path(r.media, lang)
            if r.ok and not source.exists():
                r.error = f"--from-srt: {source.name} not found"
    else:
        _transcribe_stage(results, lang, overwrite, engine_factory)
    targets = [t for t in dict.fromkeys(translate) if t != lang]
    if targets:
        _translate_stage(results, lang, targets, overwrite, complete_factory)
    return results


def _reject_shared_outputs(results: list[FileResult], lang: str) -> None:
    owners: dict[Path, Path] = {}
    for r in results:
        out = source_srt_path(r.media, lang)
        if out in owners:
            r.error = f"{out.name} would also be written for {owners[out].name}; rename one of them"
        else:
            owners[out] = r.media


def _transcribe_stage(results, lang, overwrite, engine_factory) -> None:
    pending = []
    for r in results:
        if not r.ok:
            continue
        if source_srt_path(r.media, lang).exists() and not overwrite:
            logger.info("skip transcription, exists: %s", source_srt_path(r.media, lang).name)
        else:
            pending.append(r)
    if not pending:
        return
    try:
        engine = engine_factory(lang)
    except Exception as e:  # one broken model load fails the stage, not the whole run
        for r in pending:
            r.error = f"could not load speech model: {e}"
        return
    try:
        for r in pending:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    wav = Path(tmp) / "audio.wav"
                    extract_audio(r.media, wav)
                    cues = build_cues(engine.transcribe(wav, lang))
                if lang in CYRILLIC_LANGS:
                    cues = [Cue(c.start, c.end, to_cyrillic(c.text, lang)) for c in cues]
                write_srt(source_srt_path(r.media, lang), cues)
                logger.info("transcribed %s (%d cues)", r.media.name, len(cues))
            except Exception as e:  # per-file isolation: log, record, continue with the batch
                logger.exception("transcription failed: %s", r.media.name)
                r.error = f"transcription failed: {e}"
    finally:
        engine.close()


def _translate_stage(results, lang, targets, overwrite, complete_factory) -> None:
    pending = []  # (result, targets still to write)
    for r in results:
        if not r.ok:
            continue
        todo = []
        for t in targets:
            if target_srt_path(r.media, t).exists() and not overwrite:
                logger.info("skip translation, exists: %s", target_srt_path(r.media, t).name)
            else:
                todo.append(t)
        if todo:
            pending.append((r, todo))
    if not pending:
        return
    try:
        complete = complete_factory()
    except Exception as e:
        for r, _ in pending:
            r.error = f"translation unavailable: {e}"
        return
    for r, todo in pending:
        errors = []
        try:
            cues = read_srt(source_srt_path(r.media, lang))
        except Exception as e:  # per-file isolation
            logger.exception("translation failed: %s", r.media.name)
            r.error = f"translation failed: {e}"
            continue
        for t in todo:
            try:
                write_srt(target_srt_path(r.media, t), translate_cues(cues, lang, t, complete))
                logger.info("translated %s -> %s", r.media.name, t)
            except Exception as e:  # per-target isolation
                logger.exception("translation to %s failed: %s", t, r.media.name)
                errors.append(f"translation to {t} failed: {e}")
        if errors:
            r.error = "; ".join(errors)
