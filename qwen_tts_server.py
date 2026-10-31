"""Qwen3-TTS behind the same Fish-like Gradio API as simple_tts_server.

Clones the character voice from `reference_audio` + `reference_text` sent by
tts/fish/fish_gr.py (refs from resources/Characters/<Name>/Voice). Runs natively
on Apple Silicon via MLX, so no Docker here. Separate env, see docs/wisdom-man/progress.md:

    QWEN_TTS_PORT=22231 python qwen_tts_server.py
"""

import os
import tempfile
import time
import traceback

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

from simple_tts_server import build_and_launch

MODEL_NAME = os.environ.get("QWEN_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit")
LANG = os.environ.get("QWEN_TTS_LANG", "russian")

print(f"[STARTUP] Loading {MODEL_NAME}...")
_t = time.time()
_model = load_model(MODEL_NAME)
print(f"[STARTUP] Model loaded in {time.time() - _t:.1f}s")


def predict(
    text: str,
    normalize: bool,
    reference_id: str,
    reference_audio: str | None,
    reference_text: str,
    max_new_tokens: int,
    chunk_length: int,
    top_p: float,
    repetition_penalty: float,
    temperature: float,
    seed: int,
    use_memory_cache: str,
) -> tuple[str | None, str]:
    """Synthesize `text` in the reference voice, return (wav_path, error_html).

    Gradio names API params after this signature, so all Fish inputs are listed;
    only text and the reference are used."""
    text = (text or "").strip()
    if not text:
        return None, "<div style='color:red'>Empty text</div>"
    if reference_audio and os.path.isdir(reference_audio):
        reference_audio = None
    ref_kwargs = {}
    if reference_audio and reference_text:
        ref_kwargs = {"ref_audio": reference_audio, "ref_text": reference_text}

    try:
        start = time.time()
        results = list(_model.generate(text=text, lang_code=LANG, **ref_kwargs))
        audio = np.concatenate([np.array(r.audio) for r in results])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, _model.sample_rate, subtype="PCM_16")
        took = time.time() - start
        print(f"[GENERATE] {took:.2f}s for {len(audio) / _model.sample_rate:.1f}s audio: '{text[:50]}'")
        return tmp.name, ""
    except Exception as e:
        traceback.print_exc()
        return None, f"<div style='color:red'>Synthesis failed: {e}</div>"


if __name__ == "__main__":
    build_and_launch(server_port=int(os.environ.get("QWEN_TTS_PORT", "22235")), predict_fn=predict)
