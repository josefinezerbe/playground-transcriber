from typing import Optional

class TranscribeError(Exception):
    pass

def _auto_device(user_device: Optional[str]) -> str:
        # Prefer user choice; otherwise try CUDA if torch is available
        if user_device in {"cpu", "cuda"}:
            return user_device
        try:
            import torch  # local import to avoid heavy import at package import time
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

def transcribe(
    input_path: str,
    model: str = "small",
    language: Optional[str] = None,
    device: Optional[str] = None,
    backend: str = "whisper",
) -> str:
    """
    Return plain transcript text for a media file.
    backend: 'whisper' (openai-whisper) or 'faster' (faster-whisper).
    """
    if backend == "faster":
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise TranscribeError("faster-whisper not installed. Try: pip install '.[faster]'") from e

        dev = _auto_device(device)
        compute_type = "float16" if dev == "cuda" else "int8"
        try:
            model_obj = WhisperModel(model, device=dev, compute_type=compute_type)
            segments, _ = model_obj.transcribe(
                input_path,
                language=language,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            return "".join(s.text for s in segments).strip()
        except Exception as e:
            raise TranscribeError(f"Transcription failed (faster-whisper): {e}") from e

    # openai-whisper as default
    try:
        import whisper
        dev = _auto_device(device)
        model_obj = whisper.load_model(model, device=dev)
        result = model_obj.transcribe(
            input_path,
            language=language,
            verbose=False,
            condition_on_previous_text=False,
        )
        return (result.get("text") or "").strip()
    except ImportError as e:
        raise TranscribeError("openai-whisper not installed. Try: pip install openai-whisper") from e
    except Exception as e:
        raise TranscribeError(f"Transcription failed (whisper): {e}") from e