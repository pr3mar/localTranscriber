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
