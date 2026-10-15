# localTranscribe — Design

Date: 2026-10-14
Status: Approved

## 1. Goal

Turn any audio/video file (or a folder of them) into subtitle files, fully locally:

- Transcribe **Macedonian (mk), Serbian (sr), Slovenian (sl), Spanish (es), English (en)**.
- Optionally translate the transcript **into one or more of the same five languages** (English by default).
- Output: `.srt` files.

### Hard constraints

- **Local models only.** No paid APIs, no subscriptions, no cloud inference.
- **Everything runs in Docker.** Pull the image + compose file onto any supported machine and run; nothing else installed on the host besides the NVIDIA driver and Docker with GPU support.
- **Target machines: x86-64 with an NVIDIA GPU only.** No CPU fallback, no ARM/Apple Silicon builds.

### Content profile

Mixed: TV shows, movies, talks, interviews. General-purpose defaults; no speaker labels.

### Out of scope (for now)

- Per-segment detection of foreign-language passages (a show is treated entirely as `--lang`).
- Speaker diarization.
- Re-encoding media or burning subtitles into video.
- Web UI.
- CPU-only and ARM targets.
- CI image builds / automated registry publishing.

## 2. User interface

CLI, single file or folder:

```
docker compose run --rm transcribe /media/<file-or-folder> --lang mk [--translate] [--overwrite] [--from-srt]
```

| Flag | Meaning |
|---|---|
| `--lang {mk,sr,sl,es,en}` | **Required.** Primary language of the media. |
| `--translate [LANGS]` | Also translate into a comma-separated list of targets from `mk,sr,sl,es,en` (bare = `en`). A target equal to `--lang` is skipped. |
| `--overwrite` | Redo stages even if their output already exists. |
| `--from-srt` | Skip transcription; translate an existing `<name>.<lang>.srt`. |

Folder input: process all media files in the folder (non-recursive), by extension, using ffmpeg-readable formats (e.g. mp4, mkv, avi, mov, webm, mp3, wav, m4a, flac, ogg).

### Outputs

Written next to the input file:

- `<name>.<lang>.srt` — source-language subtitles.
- `<name>.<target>.srt` — one file per translation target (with `--translate`, e.g. `<name>.en.srt`); `mk`/`sr` targets are Cyrillic. Every translated cue starts and ends inside source cue spans, usually a single source cue's span (it can cover several cues of the same sentence unit). Over-long translations are split into several cues, so the translated file may have more cues than the source.

Macedonian and Serbian output is always **Cyrillic**.

## 3. Pipeline

```
input file/folder
  ├─ 1. extract audio      ffmpeg → 16 kHz mono WAV (temp file)
  ├─ 2. transcribe         router → engine → timed segments with word timestamps
  │                          sl / es / en → Parakeet TDT 0.6B v3 (NVIDIA NeMo)
  │                          mk / sr      → Whisper large-v3 (faster-whisper)
  ├─ 3. build cues         segments → readable subtitle cues
  ├─ 4. post-process       mk / sr: Latin → Cyrillic transliteration
  ├─ 5. write source srt   <name>.<lang>.srt
  ├─ 6. translate          (--translate) batched cues + context → Ollama LLM → English
  └─ 7. write target srt   <name>.en.srt
```

### Why two ASR engines

Parakeet TDT 0.6B v3 is fast and accurate with word timestamps (CC-BY-4.0), but covers only 25 European languages and **does not include Macedonian or Serbian**. Whisper large-v3 covers all five. Parakeet is used where it applies; Whisper elsewhere.

### Batch ordering

For a folder, run stages across all files rather than file by file:

1. Transcribe every file (ASR model loaded once), writing source `.srt` files.
2. Unload the ASR model and free GPU memory.
3. Translate every file (LLM loaded once in Ollama).

This prevents the ASR model and the LLM from competing for VRAM and avoids reloading models per file.

### Cue building rules

- Max 42 characters per line, max 2 lines per cue.
- Cue duration roughly 1–7 s; never overlapping; monotonically increasing.
- Break preferentially at sentence punctuation, then at pauses between words (word timestamps), then at clause punctuation.

### Translation

- Translator is an interface (`translate(cues, source_lang) -> english_texts`), with one implementation: **Ollama** over HTTP.
- Consecutive cues are first grouped into **sentence units** (a unit ends at a cue ending in `.?!…`, capped at 5 cues / 250 characters), because the cue builder splits long sentences across cues and translating fragments separately lets the LLM merge them and shift later lines. Units are sent in batches of ~10–20 with a few preceding units as read-only context, as a JSON object keyed by 1-based unit id; the response is constrained with Ollama structured output (a JSON schema requiring exactly those keys), so each unit gets exactly one translation. Each unit's English is distributed back over its cues at word boundaries (preferring sentence/clause punctuation) proportionally to each cue's source length; if it has fewer words than the unit has cues, the trailing cues' spans are merged into the last piece.
- Prompt notes that the text is mostly `<lang>` and may contain other languages; lines already in the target language pass through unchanged.
- **Alignment guarantee:** on invalid JSON, missing keys, empty values or a value that still reads as Serbian (ћ/ђ or Serbian-only words in Macedonian; Cyrillic in a Latin-script target; Serbian function words in English/Spanish) → retry the batch once, then split it in half recursively, down to single units (a unit that still fails keeps its cues' source text, with a warning; one whose only fault is reading as Serbian keeps that reply, with a warning). Translations that do not fit two lines of 42 characters are split at the sentence boundary closest to the middle (else clause boundary, else space), with the source cue's time divided proportionally to text length, so English cues always start and end inside source cue spans, usually one source cue's span (a cue can cover several cues of the same sentence unit).
- The LLM model name is configuration (env var). The default model is chosen by comparison (see §7), not fixed in this spec; the current default is `gemma3:27b`.
- Before the translate stage starts, the CLI checks that Ollama is reachable and the configured model is present, pulling it via the Ollama API if missing.

### Cyrillic output (mk, sr)

Whisper may emit Macedonian or Serbian in Latin or Cyrillic. Output for both is normalised to Cyrillic with a deterministic, per-language Latin→Cyrillic transliteration; text already in Cyrillic is left untouched.

- Serbian: standard Gaj's Latin mapping, including digraphs lj→љ, nj→њ, dž→џ and đ→ђ, ć→ћ.
- Macedonian: standard romanisation, including lj→љ, nj→њ, dž/dzh→џ, dz→ѕ, gj/ǵ→ѓ, kj/ḱ→ќ, ch/č→ч, sh/š→ш, zh/ž→ж.

Loanwords the ASR writes in Latin script (e.g. "custom shop") are transliterated letter by letter ("цустом схоп"); this is an accepted limitation.

Known limitation: rare words where a digraph is really two letters (e.g. Serbian "injekcija", Macedonian "podzemen") will be transliterated incorrectly.

## 4. Components

Python package `localtranscribe` (rename `pyproject.toml` project name from `localtranscoder`). Each module has one job and plain-data interfaces:

| Module | Responsibility |
|---|---|
| `audio` | ffmpeg wrapper: media → 16 kHz mono WAV; detect missing audio stream. |
| `engines/base` | ASR interface: `transcribe(wav_path, lang) -> list[Segment]`; `Segment` has start, end, text, words (with timings). |
| `engines/parakeet` | Parakeet TDT via NeMo. |
| `engines/whisper` | Whisper large-v3 via faster-whisper. |
| `router` | Maps language → engine. |
| `cues` | Segments → subtitle cues per the rules above. |
| `cyrillic` | Per-language (mk, sr) Latin → Cyrillic transliteration. |
| `srt` | Format/write SRT, atomic write (temp file + rename). |
| `translate/base` | Translator interface. |
| `translate/ollama` | Ollama implementation: batching, prompt, parsing, alignment fallback. |
| `cli` | Argument parsing, file discovery, stage orchestration, skip/overwrite logic, summary. |

Engines and translator are loaded lazily and only for the languages/stages actually needed.

## 5. Docker

`docker-compose.yml` with two services:

| Service | Image | Notes |
|---|---|---|
| `transcribe` | `${LOCALTRANSCRIBE_IMAGE:-localtranscribe:latest}`, built from our `Dockerfile` | CUDA 12.8+ base (needed for current-generation NVIDIA GPUs), Python, NeMo, faster-whisper, ffmpeg. GPU reserved. |
| `ollama` | `ollama/ollama` (pinned tag) | GPU reserved. Reached from `transcribe` at `http://ollama:11434`. |

Volumes:

- `models` → Hugging Face / NeMo cache for ASR weights (downloaded on first run).
- `ollama` → Ollama model store (LLM pulled on first use).
- Bind mount: host media directory → `/media` (configurable via env var).

Distribution: build and push the image manually (`docker compose build`, `docker push`) to a registry of the user's choice; other machines set `LOCALTRANSCRIBE_IMAGE` and `docker compose pull`. No CI.

### Known risk — dependency conflict

NeMo and faster-whisper (CTranslate2) each have their own PyTorch / CUDA / cuDNN requirements. **First implementation task is a spike** proving both run on GPU in a single image. Fallback if they conflict: split Whisper into its own container/service with the same engine interface.

## 6. Error handling

- **Per-file isolation:** a failure on one file is logged and the batch continues. End-of-run summary lists successes and failures with reasons; exit code non-zero if any file failed.
- **Resumable:** a stage is skipped when its output exists (unless `--overwrite`). A re-run after a translation failure only translates.
- **Atomic writes:** no partial `.srt` files are ever left behind.
- **Fail fast:** invalid `--lang`, unreadable input path, missing audio stream → clear error for that file; Ollama unreachable/model unavailable → detected before the translate stage starts.
- **Translation alignment:** see §3 — never produces a misaligned English file.

## 7. Testing

- **Unit tests** (host, `uv run pytest`, no GPU/Docker): cue building, SRT formatting, router, Macedonian and Serbian transliteration, translation batching/parsing/alignment fallback with a fake LLM, CLI skip/overwrite logic.
- **Integration tests** (in the container, GPU): one short CC0 clip per language (Mozilla Common Voice), asserting the `.srt` exists, timings are monotonic and non-overlapping, and expected words appear (fuzzy match).
- **LLM selection script** (manual, not pass/fail): `scripts/eval_translation.py` translates hand-corrected source subtitles with candidate models and scores them against the hand-corrected translations (chrF, plus the share of lines still in the source language); the winner becomes the default model setting.
