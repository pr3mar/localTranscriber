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
