# localTranscribe

Local, Dockerised CLI that turns audio/video files (or folders of them) into `.srt` subtitles,
with optional translation to any of en/es/mk/sl/sr (English by default). Supported source languages: Macedonian (mk), Serbian (sr),
Slovenian (sl), Spanish (es), English (en).

Full design: `docs/superpowers/specs/2026-10-14-localtranscribe-design.md` — read it before
changing behaviour or architecture.

## Hard constraints

- **Local models only.** No paid APIs, subscriptions or cloud inference.
- **Everything runs in Docker.** Host needs only the NVIDIA driver + Docker with GPU support.
- **Targets: x86-64 + NVIDIA GPU only.** No CPU fallback, no ARM builds.

## Pipeline

media → ffmpeg (16 kHz mono WAV) → ASR → subtitle cues → `<name>.<lang>.srt`
→ (optional `--translate`) LLM translation → `<name>.<target>.srt` per target (`--translate en,mk,es`; bare = `en`; translated cues start and end inside source cue spans, usually one source cue's span, and can cover several cues of the same sentence unit; long translations are split).

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
- Macedonian and Serbian output is always Cyrillic (deterministic per-language Latin→Cyrillic
  transliteration).
- Translation uses JSON keyed by sentence unit with Ollama structured output; on invalid reply (incl. one still in Serbian): retry, then split
  the batch, down to single units. Translated cues start and end inside source cue spans, usually one source cue's span (long ones are split).
- Whisper runs without VAD (it cut real speech) but with `hallucination_silence_threshold`, and drops segments that are only
  known hallucinated phrases ("Хвала што пратите канал", "Субтитр…").
- Stages skip when their output exists (resumable) unless `--overwrite`; `.srt` writes are atomic.
- One failing file never stops a batch; summary at the end, non-zero exit if any failed.

## Testing

- Unit tests run on the host without GPU/Docker: `uv run pytest`.
- Integration tests run inside the container on GPU with short CC0 clips (Mozilla Common Voice).

## Commands

```bash
uv sync                                   # host dev environment (no GPU libs)
uv run pytest                             # unit tests on the host
docker compose build transcribe           # build the GPU image
docker compose run --rm transcribe /media/<path> --lang mk --translate
docker compose run --rm -e LOCALTRANSCRIBE_GPU_TESTS=1 --entrypoint python transcribe -m pytest -v
docker compose run --rm --entrypoint python transcribe scripts/spike_gpu.py /media/<clip>   # GPU smoke test
```

When running these docker commands from Git Bash on Windows, prefix them with `MSYS_NO_PATHCONV=1` — otherwise Git Bash rewrites `/media/...` paths.

Local runs use `docker-compose.override.yml`, which mounts `./src` and `./tests` into the container — code changes need no rebuild; rebuild only when dependencies change.

Regenerate `docker/requirements.lock` from Git Bash, not PowerShell (`>` there writes UTF-16):
`{ echo "--extra-index-url https://download.pytorch.org/whl/cu128"; docker compose run --rm -T --entrypoint pip transcribe freeze; } > docker/requirements.lock`

## Layout

- `src/localtranscribe/` — `audio` (ffmpeg), `engines/` (Parakeet, Whisper), `router`, `cues`,
  `cyrillic`, `srt`, `translate/` (batching + Ollama client), `pipeline` (staged run), `cli`.
- Heavy GPU libraries are imported only inside engine constructors so unit tests run on the host.
- `tests/integration/` runs only in the container with `LOCALTRANSCRIBE_GPU_TESTS=1`.
- Default LLM lives in two places that must match: `cli.DEFAULT_LLM` and `docker-compose.yml`.
