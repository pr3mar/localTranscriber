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


def _translate_targets(value: str) -> list[str]:
    targets = [t.strip() for t in value.split(",") if t.strip()]
    bad = [t for t in targets if t not in SUPPORTED_LANGS]
    if bad or not targets:
        raise argparse.ArgumentTypeError(
            f"invalid translation target(s) {','.join(bad) or value!r}; choose from {', '.join(SUPPORTED_LANGS)}"
        )
    return list(dict.fromkeys(targets))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localtranscribe", description="Media to .srt subtitles, locally.")
    parser.add_argument("path", type=Path, help="media file or folder (non-recursive)")
    parser.add_argument("--lang", required=True, choices=SUPPORTED_LANGS, help="primary spoken language")
    parser.add_argument(
        "--translate",
        nargs="?",
        const=["en"],
        default=[],
        type=_translate_targets,
        metavar="LANGS",
        help="also translate into these comma-separated languages (default en), writing <name>.<lang>.srt",
    )
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
    translate = [t for t in args.translate if t != args.lang]
    if len(translate) != len(args.translate):
        logging.info("--translate %s skipped: the source file already is in that language", args.lang)
    results = run(
        media,
        args.lang,
        translate=translate,
        overwrite=args.overwrite,
        from_srt=args.from_srt,
        engine_factory=load_engine,
        complete_factory=_complete_factory,
    )
    _print_summary(results)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
