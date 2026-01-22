"""
CLI entry point for the transcriber package.

Responsibilities:
- Parse command-line arguments (what the user typed).
- Validate inputs and environment (file exists, ffmpeg available).
- Call the core library function (transcribe) to do the actual work.
- Handle user-friendly errors and write the output file.
- Exit with an appropriate message and status code.

Note: focused on I/O and UX; business logic lives in core.py.
"""

import argparse
import os
import subprocess  # lightweight check for ffmpeg (python based)
import sys
from typing import Union # python <3.10 does not work with or "|" so this solves it hopefully
from .core import transcribe, TranscribeError


def _check_ffmpeg():
    """
    Verify that 'ffmpeg' is installed and on PATH by invoking 'ffmpeg -version'.
    If it's missing or fails, exit with a clear message.

    Why this check:
    - Whisper/faster-whisper rely on ffmpeg to decode media files.
    - Failing early with a clear message improves UX.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],          # give input as list of strings for safety
            stdout=subprocess.DEVNULL,       # normal output is trashed in the DEVNULL black hole
            stderr=subprocess.DEVNULL,       # standard error as well (stop cluttering)
            check=True,                      # fail fast on detected error
        )
    except Exception:
        # FileNotFoundError (ffmpeg not installed) or CalledProcessError (unexpected failure)
        # sys.exit with a string prints the message and exits with code 1.
        sys.exit("ffmpeg not found. Install it and ensure it's on $PATH.")

def _resolve_input_path(user_input: str, indir: str) -> str:
    """
    If user_input exists as given, use it.
    If user_input has a directory component, treat it as a path (even if missing).
    Otherwise, treat it as a bare filename and look in indir/<filename>.
    """
    if os.path.isfile(user_input):
        return user_input
    if os.path.dirname(user_input):  # user provided a subpath like sub/f.mp4
        return user_input  # let the existence check fail later with a clear message
    return os.path.join(indir, user_input)

def _resolve_output_path(input_path: str, output_arg: Union[str, None], outdir: Union[str, None]) -> str: #str | None, outdir: str) -> str:
    """
    If -o/--output is a path with directories, use it as-is.
    If -o is just a filename, write it into outdir.
    If -o is not provided, write <stem>.txt into outdir.
    """
    if output_arg:
        if os.path.dirname(output_arg):
            os.makedirs(os.path.dirname(output_arg), exist_ok=True)
            return output_arg
        os.makedirs(outdir, exist_ok=True)
        return os.path.join(outdir, output_arg)

    name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, f"{name}.txt")

def main():
    """
    Main entry point for the CLI. Defined as a function so it can be reused by
    the console script entry point (pyproject.toml) and Python -m execution.
    """
    # Build the argument parser with helpful descriptions for -h/--help.
    p = argparse.ArgumentParser(description="Transcribe video/audio to plain text.")

    # Positional argument: required path to the input media file.
    p.add_argument("input",
                   help="Video/audio filename or complete path. If just filename, it is expected to be in --indir (default: media)")

    # Optional output file. If omitted, <input_basename>.txt is derived in outputs folder
    p.add_argument("-o", "--output",
                   help="Output .txt filename or path. If only filename, it is written to --outdir (./outputs/<input_basename.txt>).")

    # Optional directory to look for input filename if not specified as path in --output
    p.add_argument("--indir", default="media",
                   help="Directory to look for inputs when given a bare filename (default: media)")

    # Optional directory to write the output file to if --output is only specified as filename
    p.add_argument("--outdir", default="outputs",
                   help="Directory to write outputs by default (default: outputs)")

    # Whisper model name; influences speed/accuracy trade-off.
    p.add_argument("-m", "--model", default="small",
                   help="Model name (tiny, base, small, medium, large, etc.)")

    # Language code. If omitted, the backend will try to auto-detect.
    p.add_argument("-l", "--language", default=None,
                   help="Language code (e.g., en). If omitted, auto-detect.")

    # Device selection. If omitted, core will auto-detect (CUDA if available, else CPU).
    p.add_argument("--device", choices=["cpu", "cuda"], default=None,
                   help="Force device (default: auto-detect)")

    # Backend engine: 'whisper' (openai-whisper) or 'faster' (faster-whisper).
    p.add_argument("--backend", choices=["whisper", "faster"], default="whisper",
                   help="Backend engine")
    
    # transcriber/cli.py (changes)
    p.add_argument("--diarize", action="store_true", help="Enable speaker diarization (WhisperX + pyannote)")
    p.add_argument("--hf-token", default=None, help="Hugging Face token for pyannote diarization")

    # Parse CLI args from sys.argv.
    args = p.parse_args()

    # Ensure ffmpeg is available before running transcription.
    _check_ffmpeg()

    input_path = _resolve_input_path(args.input, args.indir)

    # check input path early on and exit with clear error message if missing
    if not os.path.isfile(input_path):
        # Helpful error showing where we looked
        tried = [args.input]
        if args.input == os.path.basename(args.input):  # bare filename
            tried.append(os.path.join(args.indir, args.input))
        sys.exit("Input not found. Tried:\n- " + "\n- ".join(tried))

    # Call the core transcription logic. We separate user-facing errors (TranscribeError)
    # from unexpected ones to provide clearer messages.
    # ...
    try:
        if args.diarize:
            from .core import transcribe_diarized
            text = transcribe_diarized(
                input_path,
                model=args.model,
                language=args.language,
                device=args.device,
                hf_token=args.hf_token
            )
        else:
            text = transcribe(
                input_path,
                model=args.model,
                language=args.language,
                device=args.device,
                backend=args.backend
            )
    except TranscribeError as e:
        # Known, user-fixable error (missing dependency, bad params, etc.)
        sys.exit(str(e))
    except Exception as e:
        # Unexpected error; include the message for debugging.
        sys.exit(f"Unexpected error: {e}")

    # Decide where to write the transcript:
    # - If user provided -o/--output, use that.
    # - Otherwise, use <input_basename>.txt and put into /outputs.
    out_path = _resolve_output_path(input_path, args.output, args.outdir)

    # Ensure the output directory exists. If no directory is present (just a filename),
    # os.path.dirname(...) returns "", so we fallback to "." (current directory).
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Write the transcript as UTF-8 text. Append a newline if the text doesn’t already end with one,
    # which makes CLI pipelines and editors a bit happier.
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + ("\n" if text and not text.endswith("\n") else ""))

    # Success message with the output path.
    print(f"Done. Transcript written to: {out_path}")


# This guard ensures main() only runs when:
# - The module is executed directly (python -m transcriber.cli), or
# - Invoked via the console script generated by pyproject.toml.
# Importing this module from elsewhere won't trigger the CLI.
if __name__ == "__main__":
    main()