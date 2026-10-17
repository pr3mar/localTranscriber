from pathlib import Path
from typing import Protocol

from localtranscriber.models import Segment


class Engine(Protocol):
    def transcribe(self, wav: Path, lang: str) -> list[Segment]: ...

    def close(self) -> None: ...
