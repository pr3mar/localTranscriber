import json
import logging

from localtranscribe.cues import MAX_CHARS
from localtranscribe.models import Cue
from localtranscribe.translate.batching import (
    BATCH_SIZE,
    CONTEXT_SIZE,
    build_schema,
    build_user_prompt,
    parse_reply,
    translate_cues,
)


def request_lines(user):
    return json.loads(user)["lines"]


def echo(system, user, schema=None):
    return json.dumps({k: f"EN {v}" for k, v in request_lines(user).items()}, ensure_ascii=False)


def test_build_user_prompt():
    prompt = json.loads(build_user_prompt(["Здраво", "Свето"], ["Пред"], "mk"))
    assert prompt == {"source_language": "Macedonian", "context": ["Пред"], "lines": {"1": "Здраво", "2": "Свето"}}
    assert "Здраво" in build_user_prompt(["Здраво"], [], "mk")


def test_build_schema():
    assert build_schema(["1", "2"]) == {
        "type": "object",
        "properties": {"1": {"type": "string"}, "2": {"type": "string"}},
        "required": ["1", "2"],
        "additionalProperties": False,
    }


def test_parse_reply():
    assert parse_reply('{"1": " a ", "2": "b"}', 2) == ["a", "b"]
    assert parse_reply('{"1": "a"}', 2) is None
    assert parse_reply('{"1": "a", "2": "  "}', 2) is None
    assert parse_reply('{"1": "a", "2": 5}', 2) is None
    assert parse_reply("not json", 1) is None
    assert parse_reply('["a"]', 1) is None


def test_translation_keeps_timings_for_short_text():
    cues = [Cue(0.0, 1.0, "Здраво"), Cue(1.5, 2.5, "Две\nлинии")]
    assert translate_cues(cues, "mk", echo) == [Cue(0.0, 1.0, "EN Здраво"), Cue(1.5, 2.5, "EN Две линии")]


def test_request_carries_schema_for_all_keys():
    seen = []

    def complete(system, user, schema):
        seen.append(schema)
        return echo(system, user)

    translate_cues([Cue(0.0, 1.0, "a"), Cue(1.0, 2.0, "b")], "es", complete)
    assert seen == [build_schema(["1", "2"])]


def test_batches_with_context():
    calls = []

    def complete(system, user, schema):
        calls.append(user)
        return echo(system, user)

    cues = [Cue(float(i), i + 0.5, f"line{i}") for i in range(BATCH_SIZE * 2 + 1)]
    out = translate_cues(cues, "sr", complete)
    assert len(calls) == 3
    assert [c.text for c in out] == [f"EN line{i}" for i in range(len(cues))]
    assert json.loads(calls[1])["context"] == [f"line{i}" for i in range(BATCH_SIZE - CONTEXT_SIZE, BATCH_SIZE)]


def test_index_echo_in_text_is_not_stripped_but_keys_decide_alignment():
    out = translate_cues([Cue(0.0, 1.0, "a")], "es", lambda s, u, sc: '{"1": "5 people came"}')
    assert out == [Cue(0.0, 1.0, "5 people came")]


def test_retries_on_invalid_reply():
    replies = iter(['{"1": "only one"}', None])

    def complete(system, user, schema):
        reply = next(replies)
        return reply if reply is not None else echo(system, user)

    out = translate_cues([Cue(0.0, 1.0, "a"), Cue(1.0, 2.0, "b")], "es", complete)
    assert [c.text for c in out] == ["EN a", "EN b"]


def test_splits_batch_until_replies_are_valid():
    def complete(system, user, schema):
        if len(request_lines(user)) > 2:
            return "garbage"
        return echo(system, user)

    cues = [Cue(float(i), i + 0.5, f"l{i}") for i in range(5)]
    assert [c.text for c in translate_cues(cues, "sl", complete)] == [f"EN l{i}" for i in range(5)]


def test_unrecoverable_line_keeps_source_text_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        out = translate_cues([Cue(0.0, 1.0, "Здраво")], "mk", lambda s, u, sc: "")
    assert out == [Cue(0.0, 1.0, "Здраво")]
    assert "keeping source text" in caplog.text


def test_long_translation_is_split_within_source_span():
    long_text = "First sentence goes here for a while. " + " ".join(["word"] * 30) + "."
    out = translate_cues([Cue(10.0, 16.0, "x")], "es", lambda s, u, sc: json.dumps({"1": long_text}))
    assert len(out) >= 2
    assert out[0].start == 10.0 and out[-1].end == 16.0
    assert all(a.end <= b.start for a, b in zip(out, out[1:]))
    assert all(c.start < c.end for c in out)
    assert all(len(line) <= MAX_CHARS for c in out for line in c.text.split("\n"))
    assert " ".join(c.text.replace("\n", " ") for c in out) == long_text
