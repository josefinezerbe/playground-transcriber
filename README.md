# playground-transcriber

Simple CLI to transcribe video/audio to plain text using local Whisper from OpenAI.

## Install
- Prerequisites: ffmpeg
  - macOS: brew install ffmpeg
  - Ubuntu/Debian: sudo apt-get install -y ffmpeg
- Create virtual environment and install (Linux/OS):
  - python -m transcriber-env .transcriber-env
  - source .transcriber-env/bin/activate
  - pip install -e .   #pip install -r requirements.txt

Optional: Faster backend
- pip install faster-whisper
- Use: --backend faster

Optional: Diarization of speakers
- pip install '.[diarize]'
- Get a Hugging Face token:
  - Create a HF account; on the model pages (e.g., pyannote/segmentation and/or pyannote/speaker-diarization), click “Access repository” to accept terms.
  - Copy your token from hf.co/settings/tokens.
- Use: --diarize --hf-token YOUR_HF_TOKEN

## Usage
Call from repository main and have your input in a folder called "media" for easiest usage. You can also input the full path if you want to keep the video somewhere else.
- vid2txt input.mp4
- vid2txt /path/to/input.mp4

Give the output a specific name or also save it at another place.
- vid2txt input.mov -o output.txt
- vid2txt input.mov -o /path/to/output.txt

Find some more specifications with: `vid2txt --help`. For example:
- vid2txt input.mp4 -m medium -l en --device cuda

Optional:
- vid2txt input.mp4 --backend faster
- vid2txt input.mp4 --diarize --hf-token YOUR_HF_TOKEN -o output.txt
