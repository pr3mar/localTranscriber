import httpx


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, *, client: httpx.Client | None = None, timeout: float = 600.0):
        self.model = model
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def ensure_model(self) -> None:
        """Fail fast if Ollama is down; pull the model if it is not present yet."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama not reachable at {self._client.base_url}: {e}") from e
        names = {m["name"] for m in response.json().get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return
        try:
            response = self._client.post("/api/pull", json={"model": self.model, "stream": False}, timeout=None)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"could not pull model {self.model!r}: {e}") from e
        status = response.json()
        if status.get("status") != "success":
            raise OllamaError(f"could not pull model {self.model!r}: {status.get('error', status)}")

    def complete(self, system: str, user: str, schema: dict) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"translation request failed: {e}") from e
        return response.json()["message"]["content"]
