import pytest

from localtranscriber.router import SUPPORTED_LANGS, engine_name


def test_engine_routing():
    assert {lang: engine_name(lang) for lang in SUPPORTED_LANGS} == {
        "en": "parakeet", "es": "parakeet", "sl": "parakeet", "mk": "whisper", "sr": "whisper",
    }


def test_unsupported_language():
    with pytest.raises(ValueError, match="unsupported language 'bg'"):
        engine_name("bg")
