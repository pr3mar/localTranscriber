"""Compare Ollama models on subtitle translation against hand-corrected reference subtitles.

For each episode, the first N cues of the corrected source .srt are translated with the pipeline's own
translate_cues; the output is scored against the reference cues in the same time span with chrF
(character 6-gram F-score, beta 2), and the share of lines still reading as the source language is counted.

Run in the container (models are pulled on first use):
  MSYS_NO_PATHCONV=1 docker compose run --rm --entrypoint python transcribe scripts/eval_translation.py \
      "/media/vraticeserode/Vratice se rode S01E0[1-9]" --src sr --langs mk,es,en --cues 120 \
      --models gemma3:12b,qwen3:14b,gemma3:27b,mistral-small3.2:24b
"""
import argparse
import glob
import os
import time
from collections import Counter
from pathlib import Path

from localtranscriber.srt import read_srt
from localtranscriber.translate.batching import translate_cues, untranslated
from localtranscriber.translate.ollama import OllamaClient


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    hyp, ref = "".join(hyp.split()), "".join(ref.split())
    precision = recall = 0.0
    for k in range(1, n + 1):
        h = Counter(hyp[i : i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i : i + k] for i in range(len(ref) - k + 1))
        overlap = sum((h & r).values())
        precision += overlap / max(sum(h.values()), 1)
        recall += overlap / max(sum(r.values()), 1)
    precision, recall = precision / n, recall / n
    if not precision or not recall:
        return 0.0
    return 100 * (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


class NoThinkClient(OllamaClient):
    """Qwen3 and similar models reason before answering unless told not to; translation does not need it."""

    def complete(self, system: str, user: str, schema: dict) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prefix_glob", help="episode path prefix glob, e.g. '/media/show/Name S01E0[1-9]'")
    p.add_argument("--src", default="sr")
    p.add_argument("--langs", default="mk,es,en")
    p.add_argument("--models", required=True)
    p.add_argument("--cues", type=int, default=120, help="source cues per episode")
    args = p.parse_args()
    prefixes = sorted(f[: -len(f".{args.src}.srt")] for f in glob.glob(f"{args.prefix_glob}.{args.src}.srt"))
    url = os.environ["OLLAMA_URL"]
    print(f"{len(prefixes)} episodes, {args.cues} cues each", flush=True)
    for model in args.models.split(","):
        client = (NoThinkClient if model.startswith("qwen3") else OllamaClient)(url, model)
        client.ensure_model()
        for lang in args.langs.split(","):
            hyp_all, ref_all, lines, bad, t0 = [], [], 0, 0, time.time()
            for prefix in prefixes:
                src = read_srt(Path(f"{prefix}.{args.src}.srt"))[: args.cues]
                end = src[-1].end
                ref = [c.text for c in read_srt(Path(f"{prefix}.{lang}.srt")) if (c.start + c.end) / 2 <= end]
                out = translate_cues(src, args.src, lang, client.complete)
                hyp_all += [c.text for c in out]
                ref_all += ref
                lines += len(out)
                bad += sum(untranslated(c.text, lang) for c in out)
            score = chrf(" ".join(hyp_all), " ".join(ref_all))
            print(f"{model:24} {lang}  chrF {score:5.1f}  still-source {bad / max(lines, 1):6.1%}  "
                  f"{time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    main()
