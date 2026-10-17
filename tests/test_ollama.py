import json

import httpx
import pytest

from localtranscriber.translate.ollama import OllamaClient, OllamaError

BASE = "http://ollama:11434"


def make_client(handler, model="gemma3:12b"):
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE)
    return OllamaClient(BASE, model, client=http)


def test_ensure_model_present_does_not_pull():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"models": [{"name": "gemma3:12b"}]})

    make_client(handler).ensure_model()
    assert seen == ["/api/tags"]


def test_ensure_model_matches_latest_tag():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "mymodel:latest"}]})

    make_client(handler, model="mymodel").ensure_model()


def test_ensure_model_pulls_when_missing():
    pulled = []

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        pulled.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "success"})

    make_client(handler).ensure_model()
    assert pulled == [{"model": "gemma3:12b", "stream": False}]


def test_failed_pull_raises():
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"error": "pull model manifest: file does not exist"})

    with pytest.raises(OllamaError, match="could not pull model 'gemma3:12b'"):
        make_client(handler).ensure_model()


def test_unreachable_server_raises():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(OllamaError, match="not reachable"):
        make_client(handler).ensure_model()


def test_complete_sends_chat_request():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "{}"}})

    assert make_client(handler).complete("sys", "usr", {"type": "object"}) == "{}"
    assert sent[0]["model"] == "gemma3:12b"
    assert sent[0]["stream"] is False
    assert sent[0]["format"] == {"type": "object"}
    assert sent[0]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_complete_http_error_raises():
    def handler(request):
        return httpx.Response(500, json={"error": "out of memory"})

    with pytest.raises(OllamaError, match="translation request failed"):
        make_client(handler).complete("sys", "usr", {"type": "object"})


def test_complete_aborted_prediction_returns_empty_reply():
    # A degenerate source line (e.g. a 200-letter "Шаваааа...") can make the model loop; Ollama then
    # aborts with a 500. That is a bad reply, not a broken server: let batching retry/split/keep source.
    def handler(request):
        return httpx.Response(500, json={"error": "prediction aborted, token repeat limit reached"})

    assert make_client(handler).complete("sys", "usr", {"type": "object"}) == ""
