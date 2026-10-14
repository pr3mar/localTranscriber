# localTranscribe

Local, Dockerised CLI that turns audio/video files (or folders of them) into `.srt` subtitles,
with optional translation to English. Supported source languages: Macedonian (mk), Serbian (sr),
Slovenian (sl), Spanish (es), English (en).

Full design: `docs/superpowers/specs/2026-10-14-localtranscribe-design.md` — read it before
changing behaviour or architecture.

## Hard constraints

- **Local models only.** No paid APIs, subscriptions or cloud inference.
- **Everything runs in Docker.** Host needs only the NVIDIA driver + Docker with GPU support.
- **Targets: x86-64 + NVIDIA GPU only.** No CPU fallback, no ARM builds.

## Pipeline

media → ffmpeg (16 kHz mono WAV) → ASR → subtitle cues → `<name>.<lang>.srt`
→ (optional `--translate`) LLM translation → `<name>.en.srt` (same cue count and timings).

Folders run stage by stage (transcribe all files, unload ASR, then translate all) so the ASR model
and the LLM never compete for VRAM.

## Tech stack

| Area | Choice |
|---|---|
| Language / tooling | Python ≥3.12, `uv`, `pytest` |
| ASR, sl/es/en | NVIDIA Parakeet TDT 0.6B v3 via NeMo (it does not support mk/sr) |
| ASR, mk/sr | Whisper large-v3 via `faster-whisper` |
| Audio | ffmpeg (inside the image) |
| Translation | Local LLM served by Ollama (`ollama/ollama` container), behind a translator interface |
| Runtime | Docker image on a CUDA 12.8+ base + `docker-compose.yml` (`transcribe` + `ollama` services) |
| Models | Downloaded on first run into Docker volumes, not baked into the image |
| Distribution | Image built and pushed manually; compose reads `LOCALTRANSCRIBE_IMAGE` |

## Behaviour rules worth remembering

- `--lang` is always given by the user; whole file is treated as that language (no per-segment
  language detection yet).
- Serbian output is always Cyrillic (deterministic Latin→Cyrillic transliteration).
- English translation must keep exactly the source cue count and timings; on LLM line-count
  mismatch: retry, then split the batch, down to single cues.
- Stages skip when their output exists (resumable) unless `--overwrite`; `.srt` writes are atomic.
- One failing file never stops a batch; summary at the end, non-zero exit if any failed.

## Testing

- Unit tests run on the host without GPU/Docker: `uv run pytest`.
- Integration tests run inside the container on GPU with short CC0 clips (Mozilla Common Voice).

## Status

Design approved; implementation not started. Commands, module layout and Docker files described
in the spec do not exist yet — update this file as they land.
