import json
import logging

from localtranscribe.cues import MAX_CHARS
from localtranscribe.models import Cue
from localtranscribe.translate.batching import (
    BATCH_SIZE,
    CONTEXT_SIZE,
    MAX_UNIT_CHARS,
    MAX_UNIT_CUES,
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
    cues = [Cue(0.0, 1.0, "Здраво."), Cue(1.5, 2.5, "Две\nлинии.")]
    assert translate_cues(cues, "mk", "en", echo) == [Cue(0.0, 1.0, "EN Здраво."), Cue(1.5, 2.5, "EN Две линии.")]


def test_request_carries_schema_for_all_keys():
    seen = []

    def complete(system, user, schema):
        seen.append(schema)
        return echo(system, user)

    translate_cues([Cue(0.0, 1.0, "a."), Cue(1.0, 2.0, "b.")], "es", "en", complete)
    assert seen == [build_schema(["1", "2"])]


def test_batches_with_context():
    calls = []

    def complete(system, user, schema):
        calls.append(user)
        return echo(system, user)

    cues = [Cue(float(i), i + 0.5, f"line{i}.") for i in range(BATCH_SIZE * 2 + 1)]
    out = translate_cues(cues, "sr", "en", complete)
    assert len(calls) == 3
    assert [c.text for c in out] == [f"EN line{i}." for i in range(len(cues))]
    assert json.loads(calls[1])["context"] == [f"line{i}." for i in range(BATCH_SIZE - CONTEXT_SIZE, BATCH_SIZE)]


def test_index_echo_in_text_is_not_stripped_but_keys_decide_alignment():
    out = translate_cues([Cue(0.0, 1.0, "a")], "es", "en", lambda s, u, sc: '{"1": "5 people came"}')
    assert out == [Cue(0.0, 1.0, "5 people came")]


def test_retries_on_invalid_reply():
    replies = iter(['{"1": "only one"}', None])

    def complete(system, user, schema):
        reply = next(replies)
        return reply if reply is not None else echo(system, user)

    out = translate_cues([Cue(0.0, 1.0, "a."), Cue(1.0, 2.0, "b.")], "es", "en", complete)
    assert [c.text for c in out] == ["EN a.", "EN b."]


def test_splits_batch_until_replies_are_valid():
    def complete(system, user, schema):
        if len(request_lines(user)) > 2:
            return "garbage"
        return echo(system, user)

    cues = [Cue(float(i), i + 0.5, f"l{i}.") for i in range(5)]
    assert [c.text for c in translate_cues(cues, "sl", "en", complete)] == [f"EN l{i}." for i in range(5)]


def test_unrecoverable_line_keeps_source_text_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        out = translate_cues([Cue(0.0, 1.0, "Здраво")], "mk", "en", lambda s, u, sc: "")
    assert out == [Cue(0.0, 1.0, "Здраво")]
    assert "keeping source text" in caplog.text


def test_long_translation_is_split_within_source_span():
    long_text = "First sentence goes here for a while. " + " ".join(["word"] * 30) + "."
    out = translate_cues([Cue(10.0, 16.0, "x")], "es", "en", lambda s, u, sc: json.dumps({"1": long_text}))
    assert len(out) >= 2
    assert out[0].start == 10.0 and out[-1].end == 16.0
    assert all(a.end <= b.start for a, b in zip(out, out[1:]))
    assert all(c.start < c.end for c in out)
    assert all(len(line) <= MAX_CHARS for c in out for line in c.text.split("\n"))
    assert " ".join(c.text.replace("\n", " ") for c in out) == long_text


def flat(text):
    return " ".join(text.split())


def check_invariants(source, out):
    assert all(c.start < c.end for c in out)
    assert all(a.end <= b.start for a, b in zip(out, out[1:]))
    assert all(c.text.strip() for c in out)
    assert all(any(s.start <= c.start < s.end for s in source) for c in out)
    assert all(any(s.start < c.end <= s.end for s in source) for c in out)


def fake_translator(table, seen=None):
    def complete(system, user, schema):
        lines = request_lines(user)
        if seen is not None:
            seen.append(list(lines.values()))
        return json.dumps({k: table[v] for k, v in lines.items()}, ensure_ascii=False)

    return complete


SLATKAR = [
    Cue(0.0, 4.18, "Овозможено од Оилтек, лиценцирана\nкомпанија за рециклирање на отпадно"),
    Cue(4.18, 5.18, "моторно масло."),
    Cue(5.18, 7.84, "Како е Австралија која се\nнастапува пред македонци?"),
    Cue(8.58, 9.94, "Какво искусство е тоа?"),
]


def test_sentence_split_across_cues_is_translated_as_one_unit_without_shift():
    table = {
        "Овозможено од Оилтек, лиценцирана компанија за рециклирање на отпадно моторно масло.": (
            "Enabled by Oiltek, a licensed company for recycling waste motor oil."
        ),
        "Како е Австралија која се настапува пред македонци?": "How is Australia, performing before Macedonians?",
        "Какво искусство е тоа?": "What kind of experience is that?",
    }
    seen = []
    out = translate_cues(SLATKAR, "mk", "en", fake_translator(table, seen))
    assert seen == [list(table)]
    check_invariants(SLATKAR, out)

    def text_in(cue):
        return " ".join(flat(c.text) for c in out if cue.start <= c.start and c.end <= cue.end)

    assert text_in(SLATKAR[0]) and text_in(SLATKAR[1])
    assert text_in(SLATKAR[0]) + " " + text_in(SLATKAR[1]) == table[list(table)[0]]
    assert text_in(SLATKAR[2]) == "How is Australia, performing before Macedonians?"
    assert text_in(SLATKAR[3]) == "What kind of experience is that?"


def test_unit_translation_is_distributed_proportionally_at_a_clause_boundary():
    source = [
        Cue(10.0, 16.0, "Мислам, ние имаме не знам колку илјадници километри дистанци"),
        Cue(16.0, 18.0, "пак сме ист народ."),
    ]
    english = "I mean, we have I don't know how many thousands of kilometers, but we are one people."
    out = translate_cues(source, "mk", "en", fake_translator({flat(" ".join(c.text for c in source)): english}))
    check_invariants(source, out)
    first = [c for c in out if c.end <= 16.0]
    second = [c for c in out if c.start >= 16.0]
    assert len(first) + len(second) == len(out)
    assert flat(" ".join(c.text for c in first)) == "I mean, we have I don't know how many thousands of kilometers,"
    assert flat(" ".join(c.text for c in second)) == "but we are one people."
    assert all(len(line) <= MAX_CHARS for c in out for line in c.text.split("\n"))


def test_unit_split_without_punctuation_falls_on_a_word_boundary():
    source = [Cue(0.0, 3.0, "еден два три четири пет шест"), Cue(3.0, 4.0, "седум.")]
    english = "one two three four five six seven"
    out = translate_cues(source, "mk", "en", fake_translator({"еден два три четири пет шест седум.": english}))
    check_invariants(source, out)
    assert [flat(c.text) for c in out] == ["one two three four five six", "seven"]


def test_fewer_english_words_than_cues_merges_trailing_spans():
    source = [Cue(0.0, 1.0, "Ајде"), Cue(1.2, 2.0, "да одиме"), Cue(2.5, 3.0, "дома.")]
    out = translate_cues(source, "mk", "en", fake_translator({"Ајде да одиме дома.": "Home!"}))
    assert out == [Cue(0.0, 3.0, "Home!")]
    out = translate_cues(source, "mk", "en", fake_translator({"Ајде да одиме дома.": "Go home!"}))
    assert out == [Cue(0.0, 1.0, "Go"), Cue(1.2, 3.0, "home!")]


def test_unpunctuated_input_respects_unit_cap():
    seen = []
    short = [Cue(float(i), i + 0.9, f"збор{i} збор") for i in range(12)]
    translate_cues(short, "mk", "en", lambda s, u, sc: (seen.extend(request_lines(u).values()), echo(s, u))[1])
    assert " ".join(seen) == " ".join(flat(c.text) for c in short)
    assert max(len(u.split()) for u in seen) == 2 * MAX_UNIT_CUES

    seen = []
    long_cues = [Cue(float(i), i + 0.9, ("долг " * 16).strip()) for i in range(8)]
    translate_cues(long_cues, "mk", "en", lambda s, u, sc: (seen.extend(request_lines(u).values()), echo(s, u))[1])
    assert len(seen) > 1
    assert all(len(u) <= MAX_UNIT_CHARS for u in seen)


def test_closing_quote_after_sentence_end_closes_unit():
    seen = []
    source = [Cue(0.0, 1.0, "Рече „Добро.“"), Cue(1.0, 2.0, "Ајде.")]
    out = translate_cues(source, "mk", "en", fake_translator({"Рече „Добро.“": "He said 'Fine.'", "Ајде.": "Go."}, seen))
    assert seen == [["Рече „Добро.“", "Ајде."]]
    assert out == [Cue(0.0, 1.0, "He said 'Fine.'"), Cue(1.0, 2.0, "Go.")]


def test_untranslatable_unit_keeps_each_cue_source_text(caplog):
    source = [Cue(0.0, 1.0, "Ајде\nна"), Cue(1.0, 2.0, "море.")]
    with caplog.at_level(logging.WARNING):
        out = translate_cues(source, "mk", "en", lambda s, u, sc: "")
    assert out == [Cue(0.0, 1.0, "Ајде на"), Cue(1.0, 2.0, "море.")]
    assert "keeping source text" in caplog.text


def test_prompt_names_target_language():
    seen = []

    def complete(system, user, schema=None):
        seen.append(system)
        return json.dumps({"1": "Hola."})

    out = translate_cues([Cue(0.0, 1.0, "Здраво.")], "mk", "es", complete)
    assert out == [Cue(0.0, 1.0, "Hola.")]
    assert "Spanish" in seen[0] and "English" not in seen[0]


def test_macedonian_target_is_transliterated_to_cyrillic():
    out = translate_cues([Cue(0.0, 2.0, "Hello there.")], "en", "mk", lambda s, u, sc: json.dumps({"1": "Zdravo sveti."}))
    assert out == [Cue(0.0, 2.0, "Здраво свети.")]


def test_serbian_target_is_transliterated_to_cyrillic():
    out = translate_cues([Cue(0.0, 2.0, "Hello.")], "en", "sr", lambda s, u, sc: json.dumps({"1": "Dobar dan, đače."}))
    assert out == [Cue(0.0, 2.0, "Добар дан, ђаче.")]
