# transcriber

Simple CLI to transcribe video/audio to plain text using local Whisper.

## Install
- Prereqs: ffmpeg
  - macOS: brew install ffmpeg
  - Ubuntu/Debian: sudo apt-get install -y ffmpeg
  - Windows: choco install ffmpeg or download from ffmpeg.org and add to PATH
- Create venv and install:
  - python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
  - pip install -e .   # or: pip install -r requirements.txt

Optional: Faster backend
- pip install faster-whisper
- Use: --backend faster

## Usage
- vid2txt "input.mp4"
- vid2txt input.mov -o transcript.txt
- vid2txt input.mp4 -m medium -l en --device cuda