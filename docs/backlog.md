# Backlog

## Evaluate stronger translation models
- Compare candidates with `scripts/eval_translation.py`. It translates corrected source subtitles and scores the result against hand-corrected references with chrF, plus the share of lines that still read as the source language.
- Candidates: `qwen3:14b`, `gemma3:27b`, `mistral-small3.2:24b`, against the earlier default `gemma3:12b`. Also worth trying: `translategemma` (check that the tag exists) and `qwen3:30b-a3b`.
- Caveat: the references were made by correcting gemma3:12b output, so chrF is biased toward it. The still-source rate is not biased.
- The default model lives in two places that must match: `cli.DEFAULT_LLM` and `docker-compose.yml`.
- First run (sr source, first 120 cues of 9 hand-corrected episodes; chrF / still-source share):

  | Model | mk | es | en |
  |---|---|---|---|
  | gemma3:12b | 79.6 / 1.7% | 78.7 / 0% | 81.9 / 0% |
  | gemma3:27b | 80.5 / 0.1% | 77.4 / 0% | 81.2 / 0% |
  | mistral-small3.2:24b | 76.4 / 1.0% | 75.4 / 0.1% | 78.8 / 0% |
  | qwen3:14b (thinking off) | 66.6 / 12.9% | 73.3 / 0% | 77.7 / 0% |

  gemma3:27b became the default: it is best on Macedonian and about even on es/en, given the bias above.
  Still to try: `translategemma`, `qwen3:30b-a3b`.

## Evaluate alternative ASR for Serbian/Macedonian (currently Whisper large-v3)
- Serbian Whisper fine-tunes:
  - [Sagicc/whisper-large-v3-sr-cmb](https://huggingface.co/Sagicc/whisper-large-v3-sr-cmb): Common Voice + FLEURS + JuzneVesti. Most promising.
  - [Sagicc/whisper-large-v3-sr-combined](https://huggingface.co/Sagicc/whisper-large-v3-sr-combined)
  - [Sagicc/whisper-large-v3-turbo-sr-v2](https://huggingface.co/Sagicc/whisper-large-v3-turbo-sr-v2)
  - ~~BuzzASR/serbian~~: its own card shows no gain over zero-shot.
- Macedonian Whisper fine-tunes:
  - [BuzzASR/macedonian](https://huggingface.co/BuzzASR/macedonian): Transformers format, strongest self-reported numbers.
  - [Macedonian-ASR/whisper-large-v3-macedonian-asr](https://huggingface.co/Macedonian-ASR/whisper-large-v3-macedonian-asr) and
    [Macedonian-ASR/buki-whisper-2.0](https://huggingface.co/Macedonian-ASR/buki-whisper-2.0): SpeechBrain, so they need a separate engine.
- Transformers-format models need a one-time conversion (`ct2-transformers-converter`) before faster-whisper can load them.
- Alternative for Serbian: Parakeet TDT v3 or Canary-1B v2 run as Croatian (`hr`), then Latin→Cyrillic. The catch is Croatian
  lexical forms (tko/kruh/mlijeko).
- Later: Meta Omnilingual ASR (covers mk/sr; no built-in word timestamps).
- Evaluate WER/CER against hand-corrected Serbian subtitles. Skip lines marked unclear. The references came from Whisper output, so the scores are biased toward it.

## Translation fixes found in real use
- `untranslated()` flags Serbian names spelled with ђ/ћ/ć in Macedonian output (e.g. "Синђелић"), and Cyrillic names
  in Spanish/English output, which causes needless retries. Ignore those letters inside capitalised words.
- The model sometimes keeps a Serbian name in Cyrillic inside a Spanish/English line ("Кристина. Encontramos…").
  Transliterate leftover Cyrillic words to Latin for Latin-script targets.
- Names in Spanish/English sometimes keep Serbian case endings ("Štefa", "Vražalića"). A `--glossary` (below) would help.
- Try parallel requests (`OLLAMA_NUM_PARALLEL` plus concurrent batches). Measure throughput and GPU load first.

## Ideas
- Optional `--glossary` file (names, terms), used both as the Whisper `initial_prompt` and in the translation prompt.
- Investigate translations shifted by one cue, seen in manual review.
