# CLI wrapper (arg parsing, I/O, messages)
import argparse
import os
import subprocess
import sys
from .core import transcribe, TranscribeError

def _check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        sys.exit("ffmpeg not found. Install it and ensure it's on PATH.")

def main():
    p = argparse.ArgumentParser(description="Transcribe video/audio to plain text.")
    p.add_argument("input", help="Path to input video/audio file")
    p.add_argument("-o", "--output", help="Path to output .txt (default: <input_basename>.txt)")
    p.add_argument("-m", "--model", default="small", help="Model name (tiny, base, small, medium, large, etc.)")
    p.add_argument("-l", "--language", default=None, help="Language code (e.g., en). If omitted, auto-detect.")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Force device (default: auto-detect)")
    p.add_argument("--backend", choices=["whisper", "faster"], default="whisper", help="Backend engine")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Input not found: {args.input}")

    _check_ffmpeg()

    try:
        text = transcribe(
            args.input,
            model=args.model,
            language=args.language,
            device=args.device,
            backend=args.backend,
        )
    except TranscribeError as e:
        sys.exit(str(e))
    except Exception as e:
        sys.exit(f"Unexpected error: {e}")

    out_path = args.output or os.path.splitext(args.input)[0] + ".txt"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + ("\n" if text and not text.endswith("\n") else ""))
    print(f"Done. Transcript written to: {out_path}")

if __name__ == "__main__":
    main()