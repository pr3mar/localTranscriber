import pytest

from localtranscriber import cli
from localtranscriber.pipeline import FileResult


def test_rejects_unknown_language(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main([str(tmp_path), "--lang", "bg"])
    assert exc.value.code == 2


def test_missing_path(tmp_path, capsys):
    assert cli.main([str(tmp_path / "nope"), "--lang", "mk"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_folder_without_media(tmp_path, capsys):
    (tmp_path / "notes.txt").write_text("x")
    assert cli.main([str(tmp_path), "--lang", "mk"]) == 2
    assert "no media files" in capsys.readouterr().err


def test_passes_flags_and_reports_success(monkeypatch, tmp_path, capsys):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"x")
    seen = {}

    def fake_run(files, lang, **kwargs):
        seen.update(files=files, lang=lang, **kwargs)
        return [FileResult(files[0])]

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli.main([str(media), "--lang", "sr", "--translate", "--overwrite"]) == 0
    assert seen["files"] == [media]
    assert seen["lang"] == "sr"
    assert seen["translate"] == ["en"] and seen["overwrite"] and not seen["from_srt"]
    assert "1 succeeded, 0 failed" in capsys.readouterr().out


def test_failure_sets_exit_code(monkeypatch, tmp_path, capsys):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"x")
    monkeypatch.setattr(cli, "run", lambda files, lang, **kw: [FileResult(files[0], error="boom")])
    assert cli.main([str(media), "--lang", "mk"]) == 1
    out = capsys.readouterr().out
    assert "FAIL a.mkv: boom" in out
    assert "0 succeeded, 1 failed" in out


def test_complete_factory_reads_env_and_ensures_model(monkeypatch):
    created = {}

    class FakeClient:
        def __init__(self, url, model):
            created.update(url=url, model=model)

        def ensure_model(self):
            created["ensured"] = True

        def complete(self, system, user, schema):
            return ""

    monkeypatch.setattr(cli, "OllamaClient", FakeClient)
    monkeypatch.setenv("OLLAMA_URL", "http://x:1")
    monkeypatch.setenv("LOCALTRANSCRIBER_LLM", "m:1")
    cli._complete_factory()
    assert created == {"url": "http://x:1", "model": "m:1", "ensured": True}


def _translate_arg(monkeypatch, tmp_path, *flags):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"x")
    seen = {}

    def fake_run(files, lang, **kwargs):
        seen.update(kwargs)
        return [FileResult(files[0])]

    monkeypatch.setattr(cli, "run", fake_run)
    code = cli.main([str(media), "--lang", "sr", *flags])
    return code, seen["translate"]


def test_translate_targets(monkeypatch, tmp_path):
    assert _translate_arg(monkeypatch, tmp_path, "--translate") == (0, ["en"])
    assert _translate_arg(monkeypatch, tmp_path, "--translate", "en,mk,es") == (0, ["en", "mk", "es"])
    assert _translate_arg(monkeypatch, tmp_path, "--translate", "mk,en,mk") == (0, ["mk", "en"])
    assert _translate_arg(monkeypatch, tmp_path) == (0, [])


def test_translate_skips_source_language(monkeypatch, tmp_path, caplog):
    with caplog.at_level("INFO"):
        assert _translate_arg(monkeypatch, tmp_path, "--translate", "sr,en") == (0, ["en"])
    assert "sr" in caplog.text


def test_translate_rejects_unknown_target(tmp_path):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(media), "--lang", "sr", "--translate", "xx"])
    assert exc.value.code == 2
