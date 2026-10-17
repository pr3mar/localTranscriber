# localTranscriber

Turn audio/video into `.srt` subtitles locally — Macedonian, Serbian, Slovenian, Spanish, English —
with optional translation into any of those languages (English by default). Runs entirely in Docker on an NVIDIA GPU; no paid APIs.

## Requirements

- x86-64 machine with an NVIDIA GPU and a current driver
- Docker with GPU support (Docker Desktop + WSL2 on Windows, or NVIDIA Container Toolkit on Linux)

## Run on a new machine

1. Copy `docker-compose.yml` to the machine.
2. Set where your media lives and pull the image (`pr3mar/local-transcriber:latest` by default;
   set `LOCALTRANSCRIBER_IMAGE` to use another tag, e.g. `pr3mar/local-transcriber:0.1.1`):
   ```bash
   export MEDIA_DIR=/path/to/videos
   docker compose pull
   ```
3. Transcribe (and optionally translate):
   ```bash
   docker compose run --rm transcribe /media/episode.mkv --lang mk --translate
   docker compose run --rm transcribe /media/episode.mkv --lang sr --translate en,mk,es
   docker compose run --rm transcribe /media/season1 --lang sr
   ```

The first run downloads the speech and translation models (several GB) into Docker volumes.
Translation uses `gemma3:27b` through Ollama by default; set `LOCALTRANSCRIBER_LLM` to use another Ollama model.
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
docker compose build transcribe            # tags pr3mar/local-transcriber:latest
docker tag pr3mar/local-transcriber:latest pr3mar/local-transcriber:<version>
docker push pr3mar/local-transcriber:<version>
docker push pr3mar/local-transcriber:latest
```

## Licence

MIT, see `LICENSE`. The Docker image also bundles [FFmpeg](https://ffmpeg.org/), licensed under
the LGPL/GPL; its source is available from <https://ffmpeg.org/download.html>.
