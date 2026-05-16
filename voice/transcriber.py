import os
import tempfile

import whisper

_model = None


def _get_model() -> whisper.Whisper:
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model


def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        result = _get_model().transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result["text"].strip()
