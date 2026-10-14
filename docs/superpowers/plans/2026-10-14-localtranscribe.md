# localTranscribe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Dockerised CLI that turns audio/video files (or folders) into `.srt` subtitles in mk/sr/sl/es/en, with optional local-LLM translation to English.

**Architecture:** A staged pipeline (ffmpeg → ASR engine chosen by language → subtitle cues → `.srt` → optional translation → `.en.srt`) built from small pure-Python modules with plain-data interfaces. Heavy GPU libraries (NeMo, faster-whisper) are imported lazily so all logic is unit-testable on the host without a GPU. Runtime is a Docker Compose stack: our `transcribe` image plus the official `ollama` image.

**Tech Stack:** Python ≥3.12, uv, pytest, httpx, NVIDIA NeMo (Parakeet TDT 0.6B v3), faster-whisper (Whisper large-v3), ffmpeg, Ollama, Docker Compose on a CUDA 12.8+ base image.

**Spec:** `docs/superpowers/specs/2026-10-14-localtranscribe-design.md`

## Global Constraints

- Local models only — no paid APIs, subscriptions or cloud inference.
- Everything runs in Docker; host needs only the NVIDIA driver + Docker with GPU support.
- Targets: x86-64 + NVIDIA GPU only. No CPU fallback, no ARM builds.
- Base image: CUDA 12.8+ (`nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`).
- Python ≥3.12; package name `localtranscribe`; `src/` layout; console script `localtranscribe`.
- Supported `--lang` values exactly: `mk`, `sr`, `sl`, `es`, `en`. Translation target: English only.
- Engines: `sl`/`es`/`en` → Parakeet TDT 0.6B v3; `mk`/`sr` → Whisper large-v3.
- Outputs next to the input: `<stem>.<lang>.srt`, and with `--translate` `<stem>.en.srt` whose English cues start and end inside source cue spans (not necessarily one-to-one with the source cues).
- Macedonian and Serbian output is always Cyrillic.
- Cue rules: ≤42 characters per line, ≤2 lines, ~1–7 s, never overlapping.
- Unit tests run on the host with `uv run pytest` — no GPU, Docker, ffmpeg, NeMo or faster-whisper needed. Heavy imports stay inside functions/constructors.
- Model weights download on first run into Docker volumes (`HF_HOME=/models/hf`); never baked into the image.
- Never write secrets, hardware specs or machine details into repo content.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **Filenames with spaces, Cyrillic or several dots** (`Епизода 1.mkv`, `show.s01e01.mkv`) → outputs `Епизода 1.mk.srt`, `show.s01e01.mk.srt`; ffmpeg gets the path as one argv element. Pinned in Task 5 and Task 9.
2. **Two media files sharing a stem** (`ep1.mkv` + `ep1.mp4`) → both would write `ep1.mk.srt`; the second must fail loudly, not be silently "skipped as existing". Pinned in Task 9.
3. **Media with no speech or no audio track** → no speech gives a valid empty `.srt` (and empty `.en.srt`); no audio stream is a clear per-file error. Pinned in Task 5 and Task 9.
4. **Feature-length media (>24 min) on Parakeet** → full attention would run out of memory; local attention must always be enabled. Pinned in Task 6.
5. **`--from-srt` with SRT files made by other tools** (UTF-8 BOM, CRLF, `.` millisecond separator, position tags) → parsed correctly. Pinned in Task 2.

---

## File Structure

```
.gitignore
pyproject.toml                         # package metadata, deps, pytest config
docker-compose.yml                     # transcribe + ollama services
docker/Dockerfile                      # our CUDA image
docker/requirements.lock               # pinned deps produced by the spike
scripts/spike_gpu.py                   # Task 1 feasibility probe (kept as a GPU smoke test)
scripts/compare_llms.py                # Task 12 translation model comparison
src/localtranscribe/__init__.py
src/localtranscribe/models.py          # Word, Segment, Cue dataclasses
src/localtranscribe/srt.py             # SRT format/parse/atomic write
src/localtranscribe/cyrillic.py        # mk/sr Latin → Cyrillic
src/localtranscribe/cues.py            # segments → subtitle cues, wrap_text
src/localtranscribe/audio.py           # ffprobe/ffmpeg wrapper
src/localtranscribe/router.py          # language → engine
src/localtranscribe/engines/__init__.py
src/localtranscribe/engines/base.py    # Engine protocol
src/localtranscribe/engines/whisper.py # faster-whisper engine + adapter
src/localtranscribe/engines/parakeet.py# NeMo Parakeet engine + adapter
src/localtranscribe/translate/__init__.py
src/localtranscribe/translate/batching.py # prompt, parsing, alignment-safe translate_cues
src/localtranscribe/translate/ollama.py   # Ollama HTTP client
src/localtranscribe/pipeline.py        # discovery, output paths, staged run
src/localtranscribe/cli.py             # argparse entry point, summary, exit codes
tests/conftest.py                      # gpu-marker gating
tests/test_*.py                        # host unit tests
tests/integration/test_gpu_pipeline.py # in-container GPU tests
tests/fixtures/audio/                  # CC0 clips <lang>.<ext> + <lang>.txt + README.md
README.md                              # usage on a new machine
CLAUDE.md                              # updated with real commands
```

---

### Task 1: GPU feasibility spike: NeMo and faster-whisper in one image

This task answers the spec's main risk. It produces the Docker scaffolding the rest of the plan builds on.

**Files:**
- Create: `.gitignore`, `docker/Dockerfile`, `docker-compose.yml`, `scripts/spike_gpu.py`, `docker/requirements.lock`

**Interfaces:**
- Consumes: nothing.
- Produces: a buildable `transcribe` service with `torch`, `nemo_toolkit[asr]`, `faster-whisper`, `httpx`, `pytest` pinned in `docker/requirements.lock`; `ollama` service; volumes `models` (mounted `/models`, `HF_HOME=/models/hf`) and `ollama`; media bind mount `${MEDIA_DIR:-./media}` → `/media`.

- [ ] **Step 1: Verify Docker can see the GPU**

Run: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`
Expected: the nvidia-smi table listing the GPU. If it fails, STOP and report the error to the user (Docker Desktop GPU support / WSL2 / driver issue). Do not continue.

- [ ] **Step 2: Create `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
media/
*.srt.tmp
```

- [ ] **Step 3: Create the spike `docker/Dockerfile` (unpinned)**

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir "nemo_toolkit[asr]" faster-whisper httpx pytest

WORKDIR /app
COPY scripts/spike_gpu.py scripts/spike_gpu.py
```

- [ ] **Step 4: Create `docker-compose.yml`**

Look up the latest stable release tag at https://github.com/ollama/ollama/releases and replace `OLLAMA_TAG` below with that exact tag (for example `0.12.6`). Never use `latest`.

```yaml
services:
  transcribe:
    image: ${LOCALTRANSCRIBE_IMAGE:-localtranscribe:latest}
    build:
      context: .
      dockerfile: docker/Dockerfile
    volumes:
      - models:/models
      - ${MEDIA_DIR:-./media}:/media
    environment:
      OLLAMA_URL: http://ollama:11434
      LOCALTRANSCRIBE_LLM: ${LOCALTRANSCRIBE_LLM:-gemma3:12b}
    depends_on:
      - ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  ollama:
    image: ollama/ollama:OLLAMA_TAG
    volumes:
      - ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  models:
  ollama:
```

`gemma3:12b` is a provisional default. Task 12 replaces it with a model chosen by comparison.

- [ ] **Step 5: Create `scripts/spike_gpu.py`**

```python
"""GPU smoke test: both ASR engines load and transcribe on CUDA in this image.

Usage (inside the container): python scripts/spike_gpu.py /media/<clip>
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main(clip: str) -> None:
    import torch

    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    print("device", torch.cuda.get_device_name(0))

    import ctranslate2

    print("ctranslate2", ctranslate2.__version__, "cuda devices", ctranslate2.get_cuda_device_count())

    with tempfile.TemporaryDirectory() as tmp:
        wav = str(Path(tmp) / "clip.wav")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", clip, "-vn", "-ac", "1", "-ar", "16000", wav],
            check=True,
        )

        from faster_whisper import WhisperModel

        started = time.time()
        whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
        segments, info = whisper.transcribe(wav, word_timestamps=True, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments)
        print(f"whisper [{info.language}] {time.time() - started:.1f}s: {text[:300]}")
        del whisper

        import nemo.collections.asr as nemo_asr

        started = time.time()
        parakeet = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        parakeet.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        output = parakeet.transcribe([wav], timestamps=True)
        words = output[0].timestamp["word"]
        print(f"parakeet {time.time() - started:.1f}s, {len(words)} words: {output[0].text[:300]}")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 6: Get a sample clip from the user**

Ask the user to put a short (under 1 minute) English, Spanish or Slovenian speech clip at `media/spike.<ext>` in the repo (`media/` is git-ignored). Wait until it exists.

- [ ] **Step 7: Build and run the spike**

Run:
```bash
docker compose build transcribe
docker compose run --rm --entrypoint python transcribe scripts/spike_gpu.py /media/spike.<ext>
```
Expected:
- `available True`.
- `ctranslate2 ... cuda devices 1` or more.
- A `whisper [...]` line and a `parakeet ... words:` line, both with sensible text.

The first run downloads several GB of models into the `models` volume.

- [ ] **Step 8: If it fails, make at most two targeted fix attempts, then escalate**

Allowed fixes:
- Pin `ctranslate2`/`faster-whisper` to versions matching the installed cuDNN.
- Add the pip-installed cuDNN/cuBLAS directories (`python -c "import nvidia.cudnn.lib, nvidia.cublas.lib, os; print(os.path.dirname(nvidia.cudnn.lib.__file__), os.path.dirname(nvidia.cublas.lib.__file__))"`) to `LD_LIBRARY_PATH` via `ENV`.
- Reinstall the cu128 `torch` after NeMo if NeMo replaced it.

If it still fails after two attempts, STOP and report the exact errors to the user. The spec's fallback (Whisper in its own container) changes the design and needs their approval.

- [ ] **Step 9: Pin dependencies**

Run in Git Bash. PowerShell's `>` writes UTF-16, which pip can't read.
```bash
{ echo "--extra-index-url https://download.pytorch.org/whl/cu128"; docker compose run --rm -T --entrypoint pip transcribe freeze; } > docker/requirements.lock
```
Expected: `docker/requirements.lock` has `torch==...+cu128`, `nemo_toolkit==...`, `faster-whisper==...`, `httpx==...`, `pytest==...`.

- [ ] **Step 10: Switch the Dockerfile to the lock file and re-verify**

Replace the two `RUN pip install ...` lines in `docker/Dockerfile` with:

```dockerfile
COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
```

Keep any `ENV LD_LIBRARY_PATH=...` added in Step 8 after this line. Then run:
```bash
docker compose build transcribe
docker compose run --rm --entrypoint python transcribe scripts/spike_gpu.py /media/spike.<ext>
```
Expected: same output as Step 7.

- [ ] **Step 11: Commit**

```bash
git add .gitignore docker/Dockerfile docker/requirements.lock docker-compose.yml scripts/spike_gpu.py
git commit -m "build: GPU image with NeMo and faster-whisper, compose stack with Ollama" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Report the spike output (versions, timings, transcripts) to the user before continuing.

---

### Task 2: Package scaffold, data models, SRT read/write

**Files:**
- Modify: `pyproject.toml` (replace entirely)
- Create: `src/localtranscribe/__init__.py`, `src/localtranscribe/models.py`, `src/localtranscribe/srt.py`, `tests/conftest.py`
- Test: `tests/test_srt.py`

**Interfaces:**
- Produces:
  - `Word(start: float, end: float, text: str)`, `Segment(start: float, end: float, text: str, words: tuple[Word, ...] = ())`, `Cue(start: float, end: float, text: str)`. All frozen dataclasses in `localtranscribe.models`. `Cue.text` may contain `"\n"` between lines.
  - `localtranscribe.srt`: `format_timestamp(seconds: float) -> str`, `format_srt(cues: list[Cue]) -> str`, `parse_srt(text: str) -> list[Cue]`, `write_srt(path: Path, cues: list[Cue]) -> None` (atomic), `read_srt(path: Path) -> list[Cue]`.

- [ ] **Step 1: Replace `pyproject.toml`**

```toml
[project]
name = "localtranscribe"
version = "0.1.0"
description = "Local, Dockerised audio/video to .srt subtitles with optional English translation"
requires-python = ">=3.12"
dependencies = ["httpx>=0.27"]

[project.optional-dependencies]
asr = ["faster-whisper", "nemo_toolkit[asr]"]

[project.scripts]
localtranscribe = "localtranscribe.cli:main"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/localtranscribe"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["gpu: needs an NVIDIA GPU and model downloads; runs inside the container"]
```

- [ ] **Step 2: Create the package, models and test gating**

`src/localtranscribe/__init__.py`:
```python
"""Local audio/video to subtitles with optional English translation."""
```

`src/localtranscribe/models.py`:
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str  # lines separated by "\n"
```

`tests/conftest.py`:
```python
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("LOCALTRANSCRIBE_GPU_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set LOCALTRANSCRIBE_GPU_TESTS=1 inside the container to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
```

Run: `uv sync`
Expected: installs `httpx` and `pytest` into `.venv` and `localtranscribe` in editable mode.

- [ ] **Step 3: Write the failing SRT tests**

`tests/test_srt.py`:
```python
from localtranscribe.models import Cue
from localtranscribe.srt import format_srt, format_timestamp, parse_srt, read_srt, write_srt


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(59.9996) == "00:01:00,000"
    assert format_timestamp(-0.2) == "00:00:00,000"


def test_format_srt():
    cues = [Cue(0.0, 1.5, "Здраво"), Cue(2.0, 3.25, "first line\nsecond line")]
    assert format_srt(cues) == (
        "1\n00:00:00,000 --> 00:00:01,500\nЗдраво\n\n"
        "2\n00:00:02,000 --> 00:00:03,250\nfirst line\nsecond line\n"
    )


def test_format_srt_empty():
    assert format_srt([]) == ""
    assert parse_srt("") == []


def test_round_trip():
    cues = [Cue(0.0, 1.5, "Здраво"), Cue(2.0, 3.25, "first line\nsecond line")]
    assert parse_srt(format_srt(cues)) == cues


def test_parse_external_srt_with_bom_crlf_and_position_tags():
    raw = (
        "﻿1\r\n00:00:01,000 --> 00:00:02,500 X1:10 X2:20\r\nHello\r\n\r\n"
        "2\r\n00:00:03.000 --> 00:00:04,000\r\nTwo\r\nlines\r\n"
    )
    assert parse_srt(raw) == [Cue(1.0, 2.5, "Hello"), Cue(3.0, 4.0, "Two\nlines")]


def test_write_srt_is_atomic_and_overwrites(tmp_path):
    path = tmp_path / "a.mk.srt"
    path.write_text("old", encoding="utf-8")
    write_srt(path, [Cue(0.0, 1.0, "ново")])
    assert read_srt(path) == [Cue(0.0, 1.0, "ново")]
    assert [p.name for p in tmp_path.iterdir()] == ["a.mk.srt"]
```

- [ ] **Step 4: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_srt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'localtranscribe.srt'`.

- [ ] **Step 5: Implement `src/localtranscribe/srt.py`**

```python
import os
import re
from pathlib import Path

from localtranscribe.models import Cue

_TIMESTAMP = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})")


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{i}\n{format_timestamp(c.start)} --> {format_timestamp(c.end)}\n{c.text}\n"
        for i, c in enumerate(cues, 1)
    ]
    return "\n".join(blocks)


def _parse_timestamp(value: str) -> float:
    match = _TIMESTAMP.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"bad SRT timestamp: {value!r}")
    hours, minutes, secs, ms = (int(g) for g in match.groups())
    return hours * 3600 + minutes * 60 + secs + ms / 1000


def parse_srt(text: str) -> list[Cue]:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n[ \t]*\n", text.strip()):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            if "-->" in line:
                start, end = line.split("-->", 1)
                body = "\n".join(lines[i + 1 :]).strip()
                cues.append(Cue(_parse_timestamp(start), _parse_timestamp(end.split()[0]), body))
                break
    return cues


def write_srt(path: Path, cues: list[Cue]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(format_srt(cues), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_srt(path: Path) -> list[Cue]:
    return parse_srt(path.read_text(encoding="utf-8-sig"))
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_srt.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/localtranscribe/__init__.py src/localtranscribe/models.py src/localtranscribe/srt.py tests/conftest.py tests/test_srt.py
git commit -m "feat: package scaffold, data models and SRT read/write" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Macedonian and Serbian Latin → Cyrillic

**Files:**
- Create: `src/localtranscribe/cyrillic.py`
- Test: `tests/test_cyrillic.py`

**Interfaces:**
- Produces: `CYRILLIC_LANGS: frozenset[str]` (`{"mk", "sr"}`), `to_cyrillic(text: str, lang: str) -> str`. Raises `KeyError` for other languages. Cyrillic, digits and punctuation pass through unchanged.

- [ ] **Step 1: Write the failing tests**

`tests/test_cyrillic.py`:
```python
import pytest

from localtranscribe.cyrillic import CYRILLIC_LANGS, to_cyrillic


def test_cyrillic_languages():
    assert CYRILLIC_LANGS == {"mk", "sr"}


@pytest.mark.parametrize(
    ("latin", "expected"),
    [
        ("Ljubav i njegova džamija", "Љубав и његова џамија"),
        ("Ćao, đače!", "Ћао, ђаче!"),
        ("LJUBAV", "ЉУБАВ"),
        ("Zdravo, svete 123", "Здраво, свете 123"),
        ("Већ ћирилица", "Већ ћирилица"),
    ],
)
def test_serbian(latin, expected):
    assert to_cyrillic(latin, "sr") == expected


@pytest.mark.parametrize(
    ("latin", "expected"),
    [
        ("shto zhivot", "што живот"),
        ("gjavol kjerka", "ѓавол ќерка"),
        ("ǵavol ḱerka", "ѓавол ќерка"),
        ("ǵavol", "ѓавол"),
        ("dzvezda dzhamija", "ѕвезда џамија"),
        ("Čovek, ljubov, njiva", "Човек, љубов, њива"),
        ("Веќе е кирилица", "Веќе е кирилица"),
    ],
)
def test_macedonian(latin, expected):
    assert to_cyrillic(latin, "mk") == expected


def test_other_language_rejected():
    with pytest.raises(KeyError):
        to_cyrillic("hola", "es")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_cyrillic.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/localtranscribe/cyrillic.py`**

```python
import unicodedata

_COMMON = {
    "a": "а", "b": "б", "c": "ц", "č": "ч", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "ј", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о",
    "p": "п", "r": "р", "s": "с", "š": "ш", "t": "т", "u": "у", "v": "в", "z": "з",
    "ž": "ж",
}
_SERBIAN = {**_COMMON, "ć": "ћ", "đ": "ђ", "dž": "џ", "lj": "љ", "nj": "њ"}
_MACEDONIAN = {
    **_COMMON,
    "ǵ": "ѓ", "ḱ": "ќ", "dzh": "џ", "dž": "џ", "dz": "ѕ", "gj": "ѓ", "kj": "ќ",
    "lj": "љ", "nj": "њ", "ch": "ч", "sh": "ш", "zh": "ж",
}
_TABLES = {"sr": _SERBIAN, "mk": _MACEDONIAN}

CYRILLIC_LANGS = frozenset(_TABLES)


def to_cyrillic(text: str, lang: str) -> str:
    """Transliterate Latin letters to Cyrillic; longest match first, case preserved."""
    table = _TABLES[lang]
    longest = max(len(key) for key in table)
    text = unicodedata.normalize("NFC", text)
    out = []
    i = 0
    while i < len(text):
        for size in range(longest, 0, -1):
            chunk = text[i : i + size]
            mapped = table.get(chunk.lower()) if len(chunk) == size else None
            if mapped is not None:
                out.append(mapped.upper() if chunk[0].isupper() else mapped)
                i += size
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_cyrillic.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add src/localtranscribe/cyrillic.py tests/test_cyrillic.py
git commit -m "feat: Macedonian and Serbian Latin to Cyrillic transliteration" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Subtitle cue builder

**Files:**
- Create: `src/localtranscribe/cues.py`
- Test: `tests/test_cues.py`

**Interfaces:**
- Consumes: `Word`, `Segment`, `Cue` (Task 2).
- Produces:
  - Constants `MAX_CHARS = 42`, `MAX_DURATION = 7.0`, `MIN_DURATION = 1.0`, `PAUSE_SPLIT = 0.6`.
  - `wrap_text(text: str) -> str | None`: one line if it fits; otherwise the most balanced two lines, each ≤`MAX_CHARS`; `None` if two lines can't hold it. Collapses whitespace; `""` for blank input.
  - `build_cues(segments: list[Segment]) -> list[Cue]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cues.py`:
```python
import pytest

from localtranscribe.cues import MAX_CHARS, MAX_DURATION, build_cues, wrap_text
from localtranscribe.models import Cue, Segment, Word


def seg(*words):
    ws = tuple(Word(s, e, t) for s, e, t in words)
    return Segment(ws[0].start, ws[-1].end, " ".join(w.text for w in ws), ws)


def evenly(texts, start=0.0, dur=0.3, gap=0.05):
    out, t = [], start
    for text in texts:
        out.append((t, t + dur, text))
        t += dur + gap
    return out


def flat(cues):
    return " ".join(c.text.replace("\n", " ") for c in cues)


def test_wrap_text():
    assert wrap_text("Hello   world") == "Hello world"
    line = " ".join(["aaaa"] * 6)
    assert wrap_text(" ".join(["aaaa"] * 12)) == f"{line}\n{line}"
    assert wrap_text("x" * 50) is None
    assert wrap_text(" ".join(["abcdefghij"] * 10)) is None
    assert wrap_text("  ") == ""


def test_single_short_sentence_is_one_cue_extended_to_min_duration():
    cues = build_cues([seg((0.0, 0.4, "Hello"), (0.45, 0.9, "world."))])
    assert cues == [Cue(0.0, 1.0, "Hello world.")]


def test_sentence_end_starts_new_cue():
    cues = build_cues([seg(*evenly(["Hi.", "There", "you", "go."]))])
    assert [c.text for c in cues] == ["Hi.", "There you go."]


def test_pause_starts_new_cue():
    cues = build_cues([seg((0.0, 0.4, "one"), (0.45, 0.9, "two"), (2.0, 2.4, "three"), (2.45, 2.9, "four"))])
    assert [c.text for c in cues] == ["one two", "three four"]


def test_max_duration_is_respected():
    texts = ["ab"] * 30
    cues = build_cues([seg(*evenly(texts, dur=0.4, gap=0.1))])
    assert len(cues) >= 2
    assert all(c.end - c.start <= MAX_DURATION for c in cues)
    assert flat(cues) == " ".join(texts)


def test_long_text_wraps_to_two_lines_of_max_chars():
    texts = [f"w{i:03d}" for i in range(20)]
    cues = build_cues([seg(*evenly(texts, dur=0.2, gap=0.05))])
    for cue in cues:
        lines = cue.text.split("\n")
        assert len(lines) <= 2
        assert all(len(line) <= MAX_CHARS for line in lines)
    assert flat(cues) == " ".join(texts)


def test_overflow_prefers_clause_punctuation():
    names = [f"word{i:02d}" for i in range(1, 15)]
    names[8] = "word09,"
    cues = build_cues([seg(*evenly(names, dur=0.25, gap=0.05))])
    assert cues[0].text.replace("\n", " ") == " ".join(names[:9])
    assert cues[1].text.replace("\n", " ").startswith("word10")


def test_min_duration_never_overlaps_next_cue():
    cues = build_cues([seg((0.0, 0.3, "Yes.")), seg((0.8, 1.5, "No."))])
    assert cues == [Cue(0.0, 0.8, "Yes."), Cue(0.8, 1.8, "No.")]


def test_overlapping_asr_timings_are_clamped():
    cues = build_cues([seg((0.0, 1.5, "A.")), seg((1.2, 2.5, "B."))])
    assert cues[0].end == 1.2


def test_segment_without_word_timings_is_interpolated():
    text = "one two three four five six seven eight nine ten eleven twelve"
    cues = build_cues([Segment(0.0, 6.0, text, ())])
    assert flat(cues) == text
    assert cues[0].start == 0.0
    assert cues[-1].end == pytest.approx(6.0)
    assert all(len(line) <= MAX_CHARS for c in cues for line in c.text.split("\n"))


def test_no_speech_gives_no_cues():
    assert build_cues([]) == []
    assert build_cues([Segment(0.0, 1.0, "  ", ())]) == []
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_cues.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/localtranscribe/cues.py`**

```python
from collections.abc import Iterator

from localtranscribe.models import Cue, Segment, Word

MAX_CHARS = 42
MAX_DURATION = 7.0
MIN_DURATION = 1.0
PAUSE_SPLIT = 0.6

_SENTENCE_END = (".", "?", "!", "…")
_CLAUSE_END = (",", ";", ":")


def wrap_text(text: str) -> str | None:
    """One line if it fits, else the most balanced two lines; None if it cannot fit."""
    words = text.split()
    single = " ".join(words)
    if len(single) <= MAX_CHARS:
        return single
    best: tuple[int, str] | None = None
    for i in range(1, len(words)):
        first, second = " ".join(words[:i]), " ".join(words[i:])
        if len(first) <= MAX_CHARS and len(second) <= MAX_CHARS:
            score = max(len(first), len(second))
            if best is None or score < best[0]:
                best = (score, f"{first}\n{second}")
    return best[1] if best else None


def _join(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


def _interpolate(segment: Segment) -> Iterator[Word]:
    tokens = segment.text.split()
    total = sum(len(t) for t in tokens)
    duration = segment.end - segment.start
    t = segment.start
    for token in tokens:
        end = t + duration * len(token) / total
        yield Word(t, end, token)
        t = end


def _words_of(segments: list[Segment]) -> Iterator[Word]:
    for segment in segments:
        if segment.words:
            yield from (w for w in segment.words if w.text.strip())
        elif segment.text.strip():
            yield from _interpolate(segment)


def _must_break(current: list[Word], word: Word) -> bool:
    return word.start - current[-1].end >= PAUSE_SPLIT or word.end - current[0].start > MAX_DURATION


def _clause_cut(words: list[Word]) -> int:
    """Index just after the last clause-ending word in the second half, else len(words)."""
    for i in range(len(words) - 1, len(words) // 2 - 1, -1):
        if words[i].text.endswith(_CLAUSE_END):
            return i + 1
    return len(words)


def build_cues(segments: list[Segment]) -> list[Cue]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in _words_of(segments):
        if current and _must_break(current, word):
            groups.append(current)
            current = []
        elif current and wrap_text(_join(current + [word])) is None:
            cut = _clause_cut(current)
            groups.append(current[:cut])
            current = current[cut:]
            if current and wrap_text(_join(current + [word])) is None:
                groups.append(current)
                current = []
        current.append(word)
        if word.text.endswith(_SENTENCE_END):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return _finalize(groups)


def _finalize(groups: list[list[Word]]) -> list[Cue]:
    raw = []
    for group in groups:
        text = _join(group)
        raw.append(Cue(group[0].start, group[-1].end, wrap_text(text) or text))
    cues = []
    for i, cue in enumerate(raw):
        end = max(cue.end, cue.start + MIN_DURATION)
        if i + 1 < len(raw):
            end = min(end, raw[i + 1].start)
        cues.append(Cue(cue.start, max(end, cue.start), cue.text))
    return cues
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_cues.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/localtranscribe/cues.py tests/test_cues.py
git commit -m "feat: build readable subtitle cues from timed segments" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Audio extraction (ffmpeg)

**Files:**
- Create: `src/localtranscribe/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Produces: `AudioError(Exception)`; `has_audio_stream(src: Path) -> bool`; `extract_audio(src: Path, dst: Path) -> None` (writes 16 kHz mono 16-bit PCM WAV; raises `AudioError` with a readable reason).

- [ ] **Step 1: Write the failing tests**

`tests/test_audio.py`:
```python
import shutil
import subprocess
import wave

import pytest

from localtranscribe import audio
from localtranscribe.audio import AudioError, extract_audio


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
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError`. The two `needs_ffmpeg` tests are skipped on the host.

- [ ] **Step 3: Implement `src/localtranscribe/audio.py`**

```python
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_audio.py -v`
Expected: 4 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/localtranscribe/audio.py tests/test_audio.py
git commit -m "feat: extract 16 kHz mono audio with ffmpeg" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: ASR engines and language router

**Files:**
- Create: `src/localtranscribe/engines/__init__.py` (empty), `src/localtranscribe/engines/base.py`, `src/localtranscribe/engines/whisper.py`, `src/localtranscribe/engines/parakeet.py`, `src/localtranscribe/router.py`
- Test: `tests/test_engines.py`, `tests/test_router.py`

**Interfaces:**
- Consumes: `Word`, `Segment` (Task 2).
- Produces:
  - `Engine` protocol: `transcribe(wav: Path, lang: str) -> list[Segment]`, `close() -> None`.
  - `segments_from_whisper(raw_segments) -> list[Segment]`; `WhisperEngine(model_name: str = "large-v3", model=None)`.
  - `segments_from_parakeet(hypothesis) -> list[Segment]`; `ParakeetEngine(model_name: str = "nvidia/parakeet-tdt-0.6b-v3", model=None)`.
  - `router.SUPPORTED_LANGS: tuple[str, ...]` (sorted: `("en", "es", "mk", "sl", "sr")`), `engine_name(lang: str) -> str` (raises `ValueError`), `load_engine(lang: str) -> Engine`.

- [ ] **Step 1: Write the failing tests**

`tests/test_engines.py`:
```python
from pathlib import Path
from types import SimpleNamespace as NS

from localtranscribe.engines.parakeet import ParakeetEngine, segments_from_parakeet
from localtranscribe.engines.whisper import WhisperEngine, segments_from_whisper
from localtranscribe.models import Segment, Word

PARAKEET_STAMPS = {
    "word": [
        {"start": 0.0, "end": 0.4, "word": "Hola,"},
        {"start": 0.5, "end": 0.9, "word": "amigo."},
        {"start": 1.5, "end": 2.0, "word": "Adiós."},
    ],
    "segment": [
        {"start": 0.0, "end": 0.9, "segment": "Hola, amigo."},
        {"start": 1.5, "end": 2.0, "segment": "Adiós."},
    ],
}


def test_segments_from_whisper_strips_word_spaces():
    raw = [NS(start=0.0, end=1.0, text=" Здраво свет.",
              words=[NS(start=0.0, end=0.4, word=" Здраво"), NS(start=0.5, end=1.0, word=" свет.")])]
    assert segments_from_whisper(raw) == [
        Segment(0.0, 1.0, "Здраво свет.", (Word(0.0, 0.4, "Здраво"), Word(0.5, 1.0, "свет.")))
    ]


def test_segments_from_whisper_without_words():
    raw = [NS(start=0.0, end=1.0, text=" Здраво", words=None)]
    assert segments_from_whisper(raw) == [Segment(0.0, 1.0, "Здраво", ())]


def test_segments_from_parakeet_assigns_words_to_segments():
    segments = segments_from_parakeet(NS(timestamp=PARAKEET_STAMPS))
    assert [s.text for s in segments] == ["Hola, amigo.", "Adiós."]
    assert [[w.text for w in s.words] for s in segments] == [["Hola,", "amigo."], ["Adiós."]]


def test_segments_from_parakeet_words_without_segments():
    stamps = {"word": PARAKEET_STAMPS["word"], "segment": []}
    [segment] = segments_from_parakeet(NS(timestamp=stamps))
    assert (segment.start, segment.end, segment.text) == (0.0, 2.0, "Hola, amigo. Adiós.")
    assert len(segment.words) == 3


def test_segments_from_parakeet_empty():
    assert segments_from_parakeet(NS(timestamp={"word": [], "segment": []})) == []


class FakeNemoModel:
    def __init__(self):
        self.attention = None
        self.calls = []

    def change_attention_model(self, self_attention_model, att_context_size):
        self.attention = (self_attention_model, att_context_size)

    def transcribe(self, paths, timestamps):
        self.calls.append((paths, timestamps))
        return [NS(timestamp=PARAKEET_STAMPS)]


def test_parakeet_engine_always_uses_local_attention_for_long_audio():
    model = FakeNemoModel()
    engine = ParakeetEngine(model=model)
    assert model.attention == ("rel_pos_local_attn", [256, 256])
    segments = engine.transcribe(Path("a.wav"), "sl")
    assert model.calls == [(["a.wav"], True)]
    assert len(segments) == 2
    engine.close()


class FakeWhisperModel:
    def transcribe(self, path, **kwargs):
        self.path, self.kwargs = path, kwargs
        return iter([NS(start=0.0, end=1.0, text=" Здраво", words=None)]), NS(language="mk")


def test_whisper_engine_forces_language_and_word_timestamps():
    model = FakeWhisperModel()
    engine = WhisperEngine(model=model)
    assert engine.transcribe(Path("a.wav"), "mk") == [Segment(0.0, 1.0, "Здраво", ())]
    assert model.path == "a.wav"
    assert model.kwargs["language"] == "mk"
    assert model.kwargs["word_timestamps"] is True
    assert model.kwargs["vad_filter"] is True
    engine.close()
```

`tests/test_router.py`:
```python
import pytest

from localtranscribe.router import SUPPORTED_LANGS, engine_name


def test_engine_routing():
    assert {lang: engine_name(lang) for lang in SUPPORTED_LANGS} == {
        "en": "parakeet", "es": "parakeet", "sl": "parakeet", "mk": "whisper", "sr": "whisper",
    }


def test_unsupported_language():
    with pytest.raises(ValueError, match="unsupported language 'bg'"):
        engine_name("bg")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_engines.py tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the engines and router**

`src/localtranscribe/engines/__init__.py`: empty file.

`src/localtranscribe/engines/base.py`:
```python
from pathlib import Path
from typing import Protocol

from localtranscribe.models import Segment


class Engine(Protocol):
    def transcribe(self, wav: Path, lang: str) -> list[Segment]: ...

    def close(self) -> None: ...
```

`src/localtranscribe/engines/whisper.py`:
```python
import gc
from collections.abc import Iterable
from pathlib import Path

from localtranscribe.models import Segment, Word


def segments_from_whisper(raw_segments: Iterable) -> list[Segment]:
    segments = []
    for s in raw_segments:
        words = tuple(
            Word(float(w.start), float(w.end), w.word.strip()) for w in (s.words or []) if w.word.strip()
        )
        segments.append(Segment(float(s.start), float(s.end), s.text.strip(), words))
    return segments


class WhisperEngine:
    def __init__(self, model_name: str = "large-v3", model=None):
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_name, device="cuda", compute_type="float16")
        self._model = model

    def transcribe(self, wav: Path, lang: str) -> list[Segment]:
        segments, _info = self._model.transcribe(
            str(wav), language=lang, word_timestamps=True, vad_filter=True, beam_size=5
        )
        return segments_from_whisper(segments)

    def close(self) -> None:
        self._model = None
        gc.collect()
```

`src/localtranscribe/engines/parakeet.py`:
```python
import gc
from pathlib import Path

from localtranscribe.models import Segment, Word

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def segments_from_parakeet(hypothesis) -> list[Segment]:
    stamps = hypothesis.timestamp
    words = [
        Word(float(w["start"]), float(w["end"]), str(w["word"]).strip())
        for w in stamps.get("word", [])
        if str(w["word"]).strip()
    ]
    raw_segments = stamps.get("segment", [])
    if not raw_segments:
        if not words:
            return []
        return [Segment(words[0].start, words[-1].end, " ".join(w.text for w in words), tuple(words))]
    segments = []
    i = 0
    for raw in raw_segments:
        start, end = float(raw["start"]), float(raw["end"])
        seg_words = []
        while i < len(words) and words[i].start < end:
            seg_words.append(words[i])
            i += 1
        segments.append(Segment(start, end, str(raw["segment"]).strip(), tuple(seg_words)))
    if i < len(words):
        last = segments[-1]
        segments[-1] = Segment(last.start, max(last.end, words[-1].end), last.text, last.words + tuple(words[i:]))
    return segments


class ParakeetEngine:
    def __init__(self, model_name: str = MODEL_NAME, model=None):
        if model is None:
            import nemo.collections.asr as nemo_asr

            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        # Local attention handles hours of audio in one pass; full attention tops out around 24 minutes.
        model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        self._model = model

    def transcribe(self, wav: Path, lang: str) -> list[Segment]:
        # Parakeet v3 detects the spoken language itself; `lang` only selected this engine.
        hypotheses = self._model.transcribe([str(wav)], timestamps=True)
        return segments_from_parakeet(hypotheses[0])

    def close(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

`src/localtranscribe/router.py`:
```python
from localtranscribe.engines.base import Engine

ENGINE_FOR_LANG = {"en": "parakeet", "es": "parakeet", "sl": "parakeet", "mk": "whisper", "sr": "whisper"}
SUPPORTED_LANGS = tuple(sorted(ENGINE_FOR_LANG))


def engine_name(lang: str) -> str:
    try:
        return ENGINE_FOR_LANG[lang]
    except KeyError:
        raise ValueError(f"unsupported language {lang!r}; choose one of {', '.join(SUPPORTED_LANGS)}") from None


def load_engine(lang: str) -> Engine:
    if engine_name(lang) == "parakeet":
        from localtranscribe.engines.parakeet import ParakeetEngine

        return ParakeetEngine()
    from localtranscribe.engines.whisper import WhisperEngine

    return WhisperEngine()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_engines.py tests/test_router.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/localtranscribe/engines src/localtranscribe/router.py tests/test_engines.py tests/test_router.py
git commit -m "feat: Parakeet and Whisper engines with language router" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Alignment-safe batched translation

**Files:**
- Create: `src/localtranscribe/translate/__init__.py` (empty), `src/localtranscribe/translate/batching.py`
- Test: `tests/test_batching.py`

**Interfaces:**
- Consumes: `Cue` (Task 2), `wrap_text` (Task 4).
- Produces:
  - `Complete = Callable[[str, str], str]`: `(system_prompt, user_prompt) -> reply text`.
  - Constants `BATCH_SIZE = 15`, `CONTEXT_SIZE = 3`, `ATTEMPTS = 2`, `SYSTEM_PROMPT`, `LANGUAGE_NAMES`.
  - `build_user_prompt(lines: list[str], context: list[str], source_lang: str) -> str`.
  - `parse_numbered(reply: str, expected: int) -> list[str] | None`.
  - `translate_cues(cues: list[Cue], source_lang: str, complete: Complete) -> list[Cue]`: same length, same timings, English text re-wrapped.

- [ ] **Step 1: Write the failing tests**

`tests/test_batching.py`:
```python
import logging

from localtranscribe.cues import MAX_CHARS
from localtranscribe.models import Cue
from localtranscribe.translate.batching import (
    BATCH_SIZE,
    CONTEXT_SIZE,
    build_user_prompt,
    parse_numbered,
    translate_cues,
)


def numbered_lines(user):
    body = user.split("Translate:\n", 1)[1]
    return [line.split("] ", 1)[1] for line in body.splitlines()]


def echo(system, user):
    return "\n".join(f"[{i}] EN {text}" for i, text in enumerate(numbered_lines(user), 1))


def test_build_user_prompt():
    assert build_user_prompt(["Здраво"], ["Пред"], "mk") == (
        "Source language: mostly Macedonian; some lines may be in other languages.\n"
        "Context (do not translate):\nПред\nTranslate:\n[1] Здраво"
    )
    assert "Context" not in build_user_prompt(["Hola"], [], "es")


def test_parse_numbered_variants():
    assert parse_numbered("Here you go:\n1. one\n2) two\n[3] three", 3) == ["one", "two", "three"]
    assert parse_numbered("[1] a\n[3] c", 3) is None
    assert parse_numbered("<think>[1] wrong</think>\n[1] right", 1) == ["right"]
    assert parse_numbered("[1] a\n[1] b", 1) == ["a"]


def test_translation_keeps_timings_and_count():
    cues = [Cue(0.0, 1.0, "Здраво"), Cue(1.5, 2.5, "Две\nлинии")]
    assert translate_cues(cues, "mk", echo) == [Cue(0.0, 1.0, "EN Здраво"), Cue(1.5, 2.5, "EN Две линии")]


def test_batches_with_context():
    calls = []

    def complete(system, user):
        calls.append(user)
        return echo(system, user)

    cues = [Cue(float(i), i + 0.5, f"line{i}") for i in range(BATCH_SIZE * 2 + 1)]
    out = translate_cues(cues, "sr", complete)
    assert len(calls) == 3
    assert [c.text for c in out] == [f"EN line{i}" for i in range(len(cues))]
    context = "\n".join(f"line{i}" for i in range(BATCH_SIZE - CONTEXT_SIZE, BATCH_SIZE))
    assert f"Context (do not translate):\n{context}\nTranslate:" in calls[1]


def test_retries_on_line_count_mismatch():
    replies = iter(["[1] only one", None])

    def complete(system, user):
        reply = next(replies)
        return reply if reply is not None else echo(system, user)

    out = translate_cues([Cue(0.0, 1.0, "a"), Cue(1.0, 2.0, "b")], "es", complete)
    assert [c.text for c in out] == ["EN a", "EN b"]


def test_splits_batch_until_counts_match():
    def complete(system, user):
        if len(numbered_lines(user)) > 2:
            return "[1] garbage"
        return echo(system, user)

    cues = [Cue(float(i), i + 0.5, f"l{i}") for i in range(5)]
    assert [c.text for c in translate_cues(cues, "sl", complete)] == [f"EN l{i}" for i in range(5)]


def test_single_line_accepts_unnumbered_reply():
    out = translate_cues([Cue(0.0, 1.0, "Hola")], "es", lambda s, u: "Hello there")
    assert out == [Cue(0.0, 1.0, "Hello there")]


def test_unrecoverable_line_keeps_source_text_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        out = translate_cues([Cue(0.0, 1.0, "Здраво")], "mk", lambda s, u: "")
    assert out == [Cue(0.0, 1.0, "Здраво")]
    assert "keeping source text" in caplog.text


def test_long_translation_is_rewrapped():
    long_text = " ".join(["word"] * 15)
    [cue] = translate_cues([Cue(0.0, 3.0, "x")], "es", lambda s, u: f"[1] {long_text}")
    assert all(len(line) <= MAX_CHARS for line in cue.text.split("\n"))
    assert cue.text.replace("\n", " ") == long_text
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_batching.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/localtranscribe/translate/batching.py`**

`src/localtranscribe/translate/__init__.py`: empty file.

`src/localtranscribe/translate/batching.py`:
```python
import logging
import re
from collections.abc import Callable

from localtranscribe.cues import wrap_text
from localtranscribe.models import Cue

logger = logging.getLogger(__name__)

Complete = Callable[[str, str], str]

BATCH_SIZE = 15
CONTEXT_SIZE = 3
ATTEMPTS = 2

LANGUAGE_NAMES = {"mk": "Macedonian", "sr": "Serbian", "sl": "Slovenian", "es": "Spanish", "en": "English"}

SYSTEM_PROMPT = (
    "You translate subtitles into natural, concise English. "
    "You receive numbered subtitle lines, optionally preceded by earlier lines for context. "
    "Translate ONLY the numbered lines. Reply with exactly one line per numbered input line, "
    "in the form '[N] translation', in the same order, and nothing else. "
    "Lines already in English are returned unchanged. Never merge or split lines."
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUMBERED = re.compile(r"^\s*\[?(\d+)[\].):]?\s*(.*?)\s*$")


def build_user_prompt(lines: list[str], context: list[str], source_lang: str) -> str:
    parts = [f"Source language: mostly {LANGUAGE_NAMES[source_lang]}; some lines may be in other languages."]
    if context:
        parts.append("Context (do not translate):")
        parts.extend(context)
    parts.append("Translate:")
    parts.extend(f"[{i}] {line}" for i, line in enumerate(lines, 1))
    return "\n".join(parts)


def parse_numbered(reply: str, expected: int) -> list[str] | None:
    found: dict[int, str] = {}
    for line in _THINK.sub("", reply).splitlines():
        match = _NUMBERED.match(line)
        if match and match.group(2):
            n = int(match.group(1))
            if 1 <= n <= expected and n not in found:
                found[n] = match.group(2)
    if len(found) != expected:
        return None
    return [found[i] for i in range(1, expected + 1)]


def _parse_single(reply: str) -> str | None:
    for line in _THINK.sub("", reply).splitlines():
        match = _NUMBERED.match(line)
        text = match.group(2) if match else line.strip()
        if text:
            return text
    return None


def _translate_lines(lines: list[str], context: list[str], source_lang: str, complete: Complete) -> list[str]:
    user = build_user_prompt(lines, context, source_lang)
    for _ in range(ATTEMPTS):
        reply = complete(SYSTEM_PROMPT, user)
        parsed = parse_numbered(reply, len(lines))
        if parsed is None and len(lines) == 1:
            single = _parse_single(reply)
            parsed = [single] if single else None
        if parsed is not None:
            return parsed
    if len(lines) == 1:
        logger.warning("could not translate line, keeping source text: %r", lines[0])
        return lines
    mid = len(lines) // 2
    left = _translate_lines(lines[:mid], context, source_lang, complete)
    right_context = (context + lines[:mid])[-CONTEXT_SIZE:]
    return left + _translate_lines(lines[mid:], right_context, source_lang, complete)


def translate_cues(cues: list[Cue], source_lang: str, complete: Complete) -> list[Cue]:
    """Translate cue texts to English; cue count and timings are always preserved."""
    texts = [" ".join(c.text.split()) for c in cues]
    translated: list[str] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        context = texts[max(0, start - CONTEXT_SIZE) : start]
        translated.extend(_translate_lines(batch, context, source_lang, complete))
    return [Cue(c.start, c.end, wrap_text(t) or t) for c, t in zip(cues, translated, strict=True)]
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_batching.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/localtranscribe/translate tests/test_batching.py
git commit -m "feat: batched, alignment-safe subtitle translation" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Ollama client

**Files:**
- Create: `src/localtranscribe/translate/ollama.py`
- Test: `tests/test_ollama.py`

**Interfaces:**
- Consumes: httpx.
- Produces: `OllamaError(Exception)`; `OllamaClient(base_url: str, model: str, *, client: httpx.Client | None = None, timeout: float = 600.0)` with `ensure_model() -> None` (pulls the model if missing) and `complete(system: str, user: str) -> str` (matches `Complete` from Task 7).

- [ ] **Step 1: Write the failing tests**

`tests/test_ollama.py`:
```python
import json

import httpx
import pytest

from localtranscribe.translate.ollama import OllamaClient, OllamaError

BASE = "http://ollama:11434"


def make_client(handler, model="gemma3:12b"):
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE)
    return OllamaClient(BASE, model, client=http)


def test_ensure_model_present_does_not_pull():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"models": [{"name": "gemma3:12b"}]})

    make_client(handler).ensure_model()
    assert seen == ["/api/tags"]


def test_ensure_model_matches_latest_tag():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "mymodel:latest"}]})

    make_client(handler, model="mymodel").ensure_model()


def test_ensure_model_pulls_when_missing():
    pulled = []

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        pulled.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "success"})

    make_client(handler).ensure_model()
    assert pulled == [{"model": "gemma3:12b", "stream": False}]


def test_failed_pull_raises():
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"error": "pull model manifest: file does not exist"})

    with pytest.raises(OllamaError, match="could not pull model 'gemma3:12b'"):
        make_client(handler).ensure_model()


def test_unreachable_server_raises():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(OllamaError, match="not reachable"):
        make_client(handler).ensure_model()


def test_complete_sends_chat_request():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "[1] Hello"}})

    assert make_client(handler).complete("sys", "usr") == "[1] Hello"
    assert sent[0]["model"] == "gemma3:12b"
    assert sent[0]["stream"] is False
    assert sent[0]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_complete_http_error_raises():
    def handler(request):
        return httpx.Response(500, json={"error": "out of memory"})

    with pytest.raises(OllamaError, match="translation request failed"):
        make_client(handler).complete("sys", "usr")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_ollama.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/localtranscribe/translate/ollama.py`**

```python
import httpx


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, *, client: httpx.Client | None = None, timeout: float = 600.0):
        self.model = model
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def ensure_model(self) -> None:
        """Fail fast if Ollama is down; pull the model if it is not present yet."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama not reachable at {self._client.base_url}: {e}") from e
        names = {m["name"] for m in response.json().get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return
        try:
            response = self._client.post("/api/pull", json={"model": self.model, "stream": False}, timeout=None)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"could not pull model {self.model!r}: {e}") from e
        status = response.json()
        if status.get("status") != "success":
            raise OllamaError(f"could not pull model {self.model!r}: {status.get('error', status)}")

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"translation request failed: {e}") from e
        return response.json()["message"]["content"]
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_ollama.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/localtranscribe/translate/ollama.py tests/test_ollama.py
git commit -m "feat: Ollama client with model auto-pull" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Staged pipeline

**Files:**
- Create: `src/localtranscribe/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `extract_audio`, `AudioError` (Task 5); `build_cues` (Task 4); `CYRILLIC_LANGS`, `to_cyrillic` (Task 3); `read_srt`, `write_srt` (Task 2); `Engine` (Task 6); `Complete`, `translate_cues` (Task 7).
- Produces:
  - `MEDIA_EXTENSIONS: frozenset[str]`.
  - `FileResult(media: Path, error: str | None = None)` with property `ok: bool`.
  - `discover_media(path: Path) -> list[Path]` (raises `FileNotFoundError`).
  - `source_srt_path(media: Path, lang: str) -> Path`; `target_srt_path(media: Path) -> Path`.
  - `run(media_files: list[Path], lang: str, *, translate: bool, overwrite: bool, from_srt: bool, engine_factory: Callable[[str], Engine], complete_factory: Callable[[], Complete]) -> list[FileResult]`. `complete_factory` must check that the backend is ready before returning (for example, pull the model).

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:
```python
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


def echo_complete(system, user):
    body = user.split("Translate:\n", 1)[1]
    return "\n".join(f"[{i}] EN {line.split('] ', 1)[1]}" for i, line in enumerate(body.splitlines(), 1))


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
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/localtranscribe/pipeline.py`**

```python
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from localtranscribe.audio import extract_audio
from localtranscribe.cues import build_cues
from localtranscribe.cyrillic import CYRILLIC_LANGS, to_cyrillic
from localtranscribe.engines.base import Engine
from localtranscribe.models import Cue
from localtranscribe.srt import read_srt, write_srt
from localtranscribe.translate.batching import Complete, translate_cues

logger = logging.getLogger(__name__)

MEDIA_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac", ".ogg"})
TARGET_LANG = "en"


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


def target_srt_path(media: Path) -> Path:
    return source_srt_path(media, TARGET_LANG)


def run(
    media_files: list[Path],
    lang: str,
    *,
    translate: bool,
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
    if translate and lang == TARGET_LANG:
        logger.info("source is already English; nothing to translate")
    elif translate:
        _translate_stage(results, lang, overwrite, complete_factory)
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


def _translate_stage(results, lang, overwrite, complete_factory) -> None:
    pending = []
    for r in results:
        if not r.ok:
            continue
        if target_srt_path(r.media).exists() and not overwrite:
            logger.info("skip translation, exists: %s", target_srt_path(r.media).name)
        else:
            pending.append(r)
    if not pending:
        return
    try:
        complete = complete_factory()
    except Exception as e:
        for r in pending:
            r.error = f"translation unavailable: {e}"
        return
    for r in pending:
        try:
            cues = read_srt(source_srt_path(r.media, lang))
            write_srt(target_srt_path(r.media), translate_cues(cues, lang, complete))
            logger.info("translated %s", r.media.name)
        except Exception as e:  # per-file isolation
            logger.exception("translation failed: %s", r.media.name)
            r.error = f"translation failed: {e}"
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 15 passed.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -v`
Expected: all pass; only the `needs_ffmpeg` tests are skipped.

- [ ] **Step 6: Commit**

```bash
git add src/localtranscribe/pipeline.py tests/test_pipeline.py
git commit -m "feat: staged, resumable transcribe-then-translate pipeline" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: CLI

**Files:**
- Create: `src/localtranscribe/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `discover_media`, `run`, `FileResult` (Task 9); `SUPPORTED_LANGS`, `load_engine` (Task 6); `OllamaClient` (Task 8).
- Produces: `main(argv: list[str] | None = None) -> int` (0 = all ok, 1 = any file failed, 2 = usage error); `DEFAULT_LLM`, `DEFAULT_OLLAMA_URL`; env vars `OLLAMA_URL`, `LOCALTRANSCRIBE_LLM`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
import pytest

from localtranscribe import cli
from localtranscribe.pipeline import FileResult


def test_rejects_unknown_language(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main([str(tmp_path), "--lang", "bg"])
    assert exc.value.code == 2


def test_missing_path(tmp_path, capsys):
    assert cli.main([str(tmp_path / "nope"), "--lang", "mk"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_folder_without_media(tmp_path, capsys):
    (tmp_path / "notes.txt").write_text("x")
    assert cli.main([str(tmp_path), "--lang", "mk"]) == 2
    assert "no media files" in capsys.readouterr().err


def test_passes_flags_and_reports_success(monkeypatch, tmp_path, capsys):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"x")
    seen = {}

    def fake_run(files, lang, **kwargs):
        seen.update(files=files, lang=lang, **kwargs)
        return [FileResult(files[0])]

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli.main([str(media), "--lang", "sr", "--translate", "--overwrite"]) == 0
    assert seen["files"] == [media]
    assert seen["lang"] == "sr"
    assert seen["translate"] and seen["overwrite"] and not seen["from_srt"]
    assert "1 succeeded, 0 failed" in capsys.readouterr().out


def test_failure_sets_exit_code(monkeypatch, tmp_path, capsys):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"x")
    monkeypatch.setattr(cli, "run", lambda files, lang, **kw: [FileResult(files[0], error="boom")])
    assert cli.main([str(media), "--lang", "mk"]) == 1
    out = capsys.readouterr().out
    assert "FAIL a.mkv: boom" in out
    assert "0 succeeded, 1 failed" in out


def test_complete_factory_reads_env_and_ensures_model(monkeypatch):
    created = {}

    class FakeClient:
        def __init__(self, url, model):
            created.update(url=url, model=model)

        def ensure_model(self):
            created["ensured"] = True

        def complete(self, system, user):
            return ""

    monkeypatch.setattr(cli, "OllamaClient", FakeClient)
    monkeypatch.setenv("OLLAMA_URL", "http://x:1")
    monkeypatch.setenv("LOCALTRANSCRIBE_LLM", "m:1")
    cli._complete_factory()
    assert created == {"url": "http://x:1", "model": "m:1", "ensured": True}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/localtranscribe/cli.py`**

```python
import argparse
import logging
import os
import sys
from pathlib import Path

from localtranscribe.pipeline import FileResult, discover_media, run
from localtranscribe.router import SUPPORTED_LANGS, load_engine
from localtranscribe.translate.batching import Complete
from localtranscribe.translate.ollama import OllamaClient

DEFAULT_OLLAMA_URL = "http://ollama:11434"
DEFAULT_LLM = "gemma3:12b"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localtranscribe", description="Media to .srt subtitles, locally.")
    parser.add_argument("path", type=Path, help="media file or folder (non-recursive)")
    parser.add_argument("--lang", required=True, choices=SUPPORTED_LANGS, help="primary spoken language")
    parser.add_argument("--translate", action="store_true", help="also write an English .en.srt")
    parser.add_argument("--overwrite", action="store_true", help="redo stages whose output already exists")
    parser.add_argument("--from-srt", action="store_true", help="translate an existing <name>.<lang>.srt")
    return parser


def _complete_factory() -> Complete:
    client = OllamaClient(
        os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        os.environ.get("LOCALTRANSCRIBE_LLM", DEFAULT_LLM),
    )
    client.ensure_model()
    return client.complete


def _print_summary(results: list[FileResult]) -> None:
    for r in results:
        print(f"OK   {r.media.name}" if r.ok else f"FAIL {r.media.name}: {r.error}")
    failed = sum(not r.ok for r in results)
    print(f"{len(results) - failed} succeeded, {failed} failed")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        media = discover_media(args.path)
    except FileNotFoundError:
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2
    if not media:
        print(f"error: no media files found in {args.path}", file=sys.stderr)
        return 2
    results = run(
        media,
        args.lang,
        translate=args.translate,
        overwrite=args.overwrite,
        from_srt=args.from_srt,
        engine_factory=load_engine,
        complete_factory=_complete_factory,
    )
    _print_summary(results)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -v`
Expected: all pass (6 new CLI tests); only the `needs_ffmpeg` tests are skipped.

- [ ] **Step 5: Commit**

```bash
git add src/localtranscribe/cli.py tests/test_cli.py
git commit -m "feat: localtranscribe CLI with run summary and exit codes" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: App image and in-container GPU integration tests

**Files:**
- Modify: `docker/Dockerfile` (append app install and entrypoint)
- Create: `tests/integration/test_gpu_pipeline.py`, `tests/fixtures/audio/README.md`, `tests/fixtures/audio/<lang>.<ext>` + `<lang>.txt` for each of en, es, sl, mk, sr

**Interfaces:**
- Consumes: everything above.
- Produces: an image whose entrypoint is `localtranscribe`. `docker compose run --rm transcribe /media/<path> --lang <l> [--translate]` works end to end.

- [ ] **Step 1: Get CC0 fixture clips (needs the user)**

Ask the user to download one short clip per language (en, es, sl, mk, sr) from Mozilla Common Voice (CC0) along with each clip's sentence. Save them as `tests/fixtures/audio/<lang>.<ext>` (e.g. `mk.mp3`) and the sentence as `tests/fixtures/audio/<lang>.txt` (UTF-8). Wait until all ten files exist.

Create `tests/fixtures/audio/README.md`:
```markdown
# Audio fixtures

One short clip per language with its reference sentence (`<lang>.<ext>` + `<lang>.txt`),
taken from Mozilla Common Voice, released under CC0 1.0. Used only by the in-container
GPU integration tests (`tests/integration/`).
```

- [ ] **Step 2: Write the integration tests**

`tests/integration/test_gpu_pipeline.py`:
```python
import re
import shutil
from pathlib import Path

import pytest

from localtranscribe import cli
from localtranscribe.audio import extract_audio
from localtranscribe.cues import build_cues
from localtranscribe.cyrillic import CYRILLIC_LANGS, to_cyrillic
from localtranscribe.router import load_engine
from localtranscribe.srt import read_srt

pytestmark = pytest.mark.gpu

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
LANGS = ["en", "es", "sl", "mk", "sr"]


def fixture_clip(lang):
    clips = [p for p in FIXTURES.glob(f"{lang}.*") if p.suffix != ".txt"]
    assert len(clips) == 1, f"expected exactly one clip for {lang} in {FIXTURES}"
    return clips[0], (FIXTURES / f"{lang}.txt").read_text(encoding="utf-8")


def normalise(text, lang):
    if lang in CYRILLIC_LANGS:
        text = to_cyrillic(text, lang)
    return set(re.findall(r"\w+", text.lower()))


@pytest.mark.parametrize("lang", LANGS)
def test_transcription_matches_reference(lang, tmp_path):
    clip, reference = fixture_clip(lang)
    wav = tmp_path / "audio.wav"
    extract_audio(clip, wav)
    engine = load_engine(lang)
    try:
        cues = build_cues(engine.transcribe(wav, lang))
    finally:
        engine.close()
    assert cues
    for a, b in zip(cues, cues[1:]):
        assert a.start <= a.end <= b.start
    expected, got = normalise(reference, lang), normalise(" ".join(c.text for c in cues), lang)
    assert len(expected & got) / len(expected) >= 0.6, (sorted(expected), sorted(got))


def test_cli_end_to_end_with_translation(tmp_path):
    clip, _ = fixture_clip("mk")
    media = tmp_path / f"Епизода 1{clip.suffix}"
    shutil.copy(clip, media)
    assert cli.main([str(media), "--lang", "mk", "--translate"]) == 0
    source = read_srt(tmp_path / "Епизода 1.mk.srt")
    target = read_srt(tmp_path / "Епизода 1.en.srt")
    assert source
    assert not re.search(r"[A-Za-z]", " ".join(c.text for c in source))
    assert [(c.start, c.end) for c in source] == [(c.start, c.end) for c in target]
    assert all(c.text.strip() for c in target)
```

- [ ] **Step 3: Append the app install to `docker/Dockerfile`**

Append after the existing `RUN pip install --no-cache-dir -r /tmp/requirements.lock` line and any `ENV LD_LIBRARY_PATH` line from Task 1:

```dockerfile
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY tests ./tests
COPY scripts ./scripts

ENTRYPOINT ["localtranscribe"]
```

Remove the now-redundant `COPY scripts/spike_gpu.py scripts/spike_gpu.py` line.

- [ ] **Step 4: Build and run the full suite in the container**

Run:
```bash
docker compose build transcribe
docker compose run --rm -e LOCALTRANSCRIBE_GPU_TESTS=1 --entrypoint python transcribe -m pytest -v
```
Expected: all unit tests pass (including the two real-ffmpeg tests, which aren't skipped in the container) and 6 `gpu` tests pass. The end-to-end test pulls the default LLM through Ollama on first run.

If a language falls below the 0.6 overlap threshold, show the user the printed expected/got word sets before changing anything. A threshold change is their decision.

- [ ] **Step 5: Real-use smoke check**

Run with a real episode the user chooses (`MEDIA_DIR` pointing at its folder):
```bash
docker compose run --rm transcribe /media/<file> --lang <lang> --translate
```
Expected: `<file>.<lang>.srt` and `<file>.en.srt` next to the file, and `1 succeeded, 0 failed`. Ask the user to open both in a player and judge timing and readability.

- [ ] **Step 6: Commit**

```bash
git add docker/Dockerfile tests/integration tests/fixtures
git commit -m "build: app image entrypoint and in-container GPU integration tests" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Choose the default translation model

**Files:**
- Create: `scripts/compare_llms.py`
- Modify: `src/localtranscribe/cli.py` (`DEFAULT_LLM`), `docker-compose.yml` (`LOCALTRANSCRIBE_LLM` default)

**Interfaces:**
- Consumes: `OllamaClient` (Task 8), `translate_cues` (Task 7), `read_srt` (Task 2), fixture texts (Task 11).
- Produces: the default model name, used in both `cli.DEFAULT_LLM` and the compose default.

- [ ] **Step 1: Create `scripts/compare_llms.py`**

```python
"""Translate sample subtitles with several Ollama models and print them side by side.

Usage (inside the container):
  python scripts/compare_llms.py gemma3:12b qwen3:14b --srt mk=/media/ep1.mk.srt
"""

import argparse
import os
from pathlib import Path

from localtranscribe.models import Cue
from localtranscribe.srt import read_srt
from localtranscribe.translate.batching import translate_cues
from localtranscribe.translate.ollama import OllamaClient

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "audio"
MAX_CUES = 30


def samples(srt_args: list[str]) -> list[tuple[str, list[Cue]]]:
    out = []
    for ref in sorted(FIXTURES.glob("*.txt")):
        if ref.stem != "en":
            out.append((ref.stem, [Cue(0.0, 1.0, ref.read_text(encoding="utf-8").strip())]))
    for arg in srt_args:
        lang, path = arg.split("=", 1)
        out.append((lang, read_srt(Path(path))[:MAX_CUES]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--srt", action="append", default=[], help="lang=path to a real source .srt")
    args = parser.parse_args()
    url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    clients = []
    for model in args.models:
        client = OllamaClient(url, model)
        client.ensure_model()
        clients.append(client)
    for lang, cues in samples(args.srt):
        results = {c.model: translate_cues(cues, lang, c.complete) for c in clients}
        for i, cue in enumerate(cues):
            print(f"\n[{lang} #{i + 1}] {cue.text.replace(chr(10), ' ')}")
            for model, translated in results.items():
                print(f"  {model:<24} {translated[i].text.replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Pick candidates and run the comparison**

Check https://ollama.com/library for current multilingual instruction models that fit in GPU memory alongside nothing else (the ASR model is unloaded before translation). Include `gemma3:12b` as the baseline plus 2–3 current candidates. Ask the user for one or two real `.mk.srt`/`.sr.srt` files produced in Task 11. Rebuild so the script is in the image, then run:
```bash
docker compose build transcribe
docker compose run --rm --entrypoint python transcribe scripts/compare_llms.py <model-1> <model-2> <model-3> --srt mk=/media/<file>.mk.srt
```

- [ ] **Step 3: User chooses**

Show the user the side-by-side output and the time each model took. The choice is theirs. Do not pick silently.

- [ ] **Step 4: Set the chosen default in both places**

In `src/localtranscribe/cli.py`, set `DEFAULT_LLM = "<chosen-model>"`. In `docker-compose.yml`, set `LOCALTRANSCRIBE_LLM: ${LOCALTRANSCRIBE_LLM:-<chosen-model>}`. The two values must be identical.

Run: `uv run pytest -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_llms.py src/localtranscribe/cli.py docker-compose.yml
git commit -m "feat: choose default translation model by side-by-side comparison" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Documentation

**Files:**
- Create: `README.md`
- Modify: `CLAUDE.md` (replace the "Status" section; add commands and layout)

- [ ] **Step 1: Write `README.md`**

````markdown
# localTranscribe

Turn audio/video into `.srt` subtitles locally — Macedonian, Serbian, Slovenian, Spanish, English —
with optional translation to English. Runs entirely in Docker on an NVIDIA GPU; no paid APIs.

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
   docker compose run --rm transcribe /media/season1 --lang sr
   ```

The first run downloads the speech and translation models (several GB) into Docker volumes.
`docker compose down` stops the Ollama service afterwards.

## Options

| Flag | Meaning |
|---|---|
| `--lang {mk,sr,sl,es,en}` | Primary spoken language (required) |
| `--translate` | Also write `<name>.en.srt` with English cues inside the source cue spans |
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
````

- [ ] **Step 2: Update `CLAUDE.md`**

Replace the `## Status` section with:

````markdown
## Commands

```bash
uv sync                                   # host dev environment (no GPU libs)
uv run pytest                             # unit tests on the host
docker compose build transcribe           # build the GPU image
docker compose run --rm transcribe /media/<path> --lang mk --translate
docker compose run --rm -e LOCALTRANSCRIBE_GPU_TESTS=1 --entrypoint python transcribe -m pytest -v
docker compose run --rm --entrypoint python transcribe scripts/spike_gpu.py /media/<clip>   # GPU smoke test
```

Regenerate `docker/requirements.lock` from Git Bash, not PowerShell (`>` there writes UTF-16):
`{ echo "--extra-index-url https://download.pytorch.org/whl/cu128"; docker compose run --rm -T --entrypoint pip transcribe freeze; } > docker/requirements.lock`

## Layout

- `src/localtranscribe/` — `audio` (ffmpeg), `engines/` (Parakeet, Whisper), `router`, `cues`,
  `cyrillic`, `srt`, `translate/` (batching + Ollama client), `pipeline` (staged run), `cli`.
- Heavy GPU libraries are imported only inside engine constructors so unit tests run on the host.
- `tests/integration/` runs only in the container with `LOCALTRANSCRIBE_GPU_TESTS=1`.
- Default LLM lives in two places that must match: `cli.DEFAULT_LLM` and `docker-compose.yml`.
````

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: README usage and CLAUDE.md commands" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
