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
    cues = build_cues([seg((0.0, 1.2, "Hi."), *evenly(["There", "you", "go."], start=1.25))])
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
    cues = build_cues([seg((0.0, 0.3, "Yes.")), seg((0.8, 7.9, "No."))])
    assert cues == [Cue(0.0, 0.8, "Yes."), Cue(0.8, 7.9, "No.")]


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


def test_zero_length_word_does_not_produce_zero_duration_cue():
    cues = build_cues([seg((71.27, 72.67, "Уф!")), seg((72.67, 72.67, "Уф!")), seg((72.67, 73.0, "Уф!"))])
    assert all(c.end > c.start for c in cues)
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))
    assert flat(cues).split() == ["Уф!"] * 3


def test_short_cues_close_together_are_merged():
    cues = build_cues([seg((74.75, 75.2, "Трни се!")), seg((75.71, 76.4, "Чегове, трни се!"))])
    assert len(cues) == 1
    assert cues[0].start == 74.75
    assert cues[0].text == "Трни се! Чегове, трни се!"


def test_short_cue_is_not_merged_when_text_would_not_fit():
    first = "a" * 30 + "."
    long = [(0.5, 1.5, "b" * 20), (1.5, 2.5, "c" * 20), (2.5, 3.0, "d" * 20 + ".")]
    cues = build_cues([seg((0.0, 0.3, first)), seg(*long)])
    assert [(c.start, c.end) for c in cues] == [(0.0, 0.5), (0.5, 3.0)]
