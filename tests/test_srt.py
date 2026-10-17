from localtranscriber.models import Cue
from localtranscriber.srt import format_srt, format_timestamp, parse_srt, read_srt, write_srt


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
