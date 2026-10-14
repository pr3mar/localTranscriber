# localTranscribe

Turn audio/video into `.srt` subtitles locally — Macedonian, Serbian, Slovenian, Spanish, English —
with optional translation into any of those languages (English by default). Runs entirely in Docker on an NVIDIA GPU; no paid APIs.

## Requirements

- x86-64 machine with an NVIDIA GPU and a current driver
- Docker with GPU support (Docker Desktop + WSL2 on Windows, or NVIDIA Container Toolkit on Linux)

## Run on a new machine

1. Copy `docker-compose.yml` to the machine.
2. Set where your media lives and which image to use:
   ```bash
   export MEDIA_DIR=/path/to/videos
   export LOCALTRANSCRIBE_IMAGE=<registry>/<user>/localtranscribe:<tag>
   docker compose pull
   ```
3. Transcribe (and optionally translate):
   ```bash
   docker compose run --rm transcribe /media/episode.mkv --lang mk --translate
   docker compose run --rm transcribe /media/episode.mkv --lang sr --translate en,mk,es
   docker compose run --rm transcribe /media/season1 --lang sr
   ```

The first run downloads the speech and translation models (several GB) into Docker volumes.
`docker compose down` stops the Ollama service afterwards.

## Options

| Flag | Meaning |
|---|---|
| `--lang {mk,sr,sl,es,en}` | Primary spoken language (required) |
| `--translate [LANGS]` | Also translate into a comma-separated list of languages from `mk,sr,sl,es,en` (bare `--translate` means `en`), writing `<name>.<lang>.srt` for each; a target equal to `--lang` is skipped. Each translated cue starts and ends within its source cues' time span, usually one cue; long translations are split. Macedonian and Serbian targets are Cyrillic |
| `--overwrite` | Redo outputs that already exist |
| `--from-srt` | Translate an existing `<name>.<lang>.srt` without transcribing |

Outputs are written next to each input. Re-running skips finished work, so an interrupted batch
resumes where it stopped. Macedonian and Serbian subtitles are always Cyrillic.

## Build and publish the image

```bash
docker compose build transcribe
docker tag localtranscribe:latest <registry>/<user>/localtranscribe:<tag>
docker push <registry>/<user>/localtranscribe:<tag>
```
