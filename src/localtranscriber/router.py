from localtranscriber.engines.base import Engine

ENGINE_FOR_LANG = {"en": "parakeet", "es": "parakeet", "sl": "parakeet", "mk": "whisper", "sr": "whisper"}
SUPPORTED_LANGS = tuple(sorted(ENGINE_FOR_LANG))


def engine_name(lang: str) -> str:
    try:
        return ENGINE_FOR_LANG[lang]
    except KeyError:
        raise ValueError(f"unsupported language {lang!r}; choose one of {', '.join(SUPPORTED_LANGS)}") from None


def load_engine(lang: str) -> Engine:
    if engine_name(lang) == "parakeet":
        from localtranscriber.engines.parakeet import ParakeetEngine

        return ParakeetEngine()
    from localtranscriber.engines.whisper import WhisperEngine

    return WhisperEngine()
