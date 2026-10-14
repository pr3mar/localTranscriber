"""GPU smoke test: both ASR engines load and transcribe on CUDA in this image.

Usage (inside the container): python scripts/spike_gpu.py /media/<clip>
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main(clip: str) -> None:
    import torch

    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    print("device", torch.cuda.get_device_name(0))

    import ctranslate2

    print("ctranslate2", ctranslate2.__version__, "cuda devices", ctranslate2.get_cuda_device_count())

    with tempfile.TemporaryDirectory() as tmp:
        wav = str(Path(tmp) / "clip.wav")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", clip, "-vn", "-ac", "1", "-ar", "16000", wav],
            check=True,
        )

        from faster_whisper import WhisperModel

        started = time.time()
        whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
        segments, info = whisper.transcribe(wav, word_timestamps=True, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments)
        print(f"whisper [{info.language}] {time.time() - started:.1f}s: {text[:300]}")
        del whisper

        import nemo.collections.asr as nemo_asr

        started = time.time()
        parakeet = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        parakeet.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        output = parakeet.transcribe([wav], timestamps=True)
        words = output[0].timestamp["word"]
        print(f"parakeet {time.time() - started:.1f}s, {len(words)} words: {output[0].text[:300]}")


if __name__ == "__main__":
    main(sys.argv[1])
