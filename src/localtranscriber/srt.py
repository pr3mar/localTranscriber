import os
import re
from pathlib import Path

from localtranscriber.models import Cue

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
