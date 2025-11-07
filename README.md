# EchoLabel - Voice-to-Text Auto‑Tagger for Events & Media

EchoLabel performs voice to text (speech to text) on the first seconds of a video or audio clip using Whisper, then names the file with date, participant name, and a tag. Use it for sports sessions, classes, rehearsals, talks, meetings, or any event where a quick spoken label helps organize footage.

## Quickstart
1) Create venv and install deps (Python 3.10+):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   pip install --upgrade pip
   pip install openai-whisper rapidfuzz pyyaml python-slugify
   ```
2) Put a test video in `./input/clip001.mp4`.
3) Dry run to preview what will happen:
   ```bash
   python echolabel.py ./input/clip001.mp4 --seconds 6 --use-silence-gate --print-only
   ```
4) Process all files in the input folder:
   ```bash
   python echolabel.py
   ```
   Notes:
   - First run auto-installs the required Python packages.
   - If FFmpeg is not found, place it under `./ffmpeg/bin/ffmpeg(.exe)` or on Windows run: `winget install Gyan.FFmpeg`.

## Usage
- Voice to text (single video):
  ```bash
  python echolabel.py ./input/clip001.mp4 --use-silence-gate --seconds 6 --print-only
  ```
- Voice to text (batch newest):
  ```bash
  python echolabel.py --input-dir ./input --newest 5 --use-silence-gate
  ```
- Single file:
  ```bash
  python echolabel.py ./input/clip001.mp4 --use-silence-gate
  ```
- Batch latest N clips:
  ```bash
  python echolabel.py --input-dir ./input --newest 5 --use-silence-gate
  ```
- Watch a folder for new files:
  ```bash
  python echolabel.py --input-dir ./input --watch --use-silence-gate --model tiny
  ```

## Configuration
Edit `data/config.yaml` to set defaults. Key fields:
- `input_dir`: folder for incoming clips (e.g., `./input`).
- `out_root`: destination root for organized clips (e.g., `./output`).
- `seconds`: duration sampled for the voice note.
- `pattern`: filename pattern (e.g., `{date}_{name}_{skill}_run{run}.{ext}`).
- `model`: Whisper model (`tiny`, `base`, `small`, ...).
- `use_silence_gate`: seek to first speech before sampling.

Defaults example:
```yaml
defaults:
  input_dir: ./input
  seconds: 6
  out_root: ./output
  pattern: "{date}_{name}_{skill}_run{run}.{ext}"
  model: base
  use_silence_gate: true
```

### Names and Tags
- `data/roster.yaml` - list names (Title Case)
  ```yaml
  students:
    - Alice
    - Jack
    - Coach Kim
  ```
- `data/skills.yaml` - list tags/categories (lowercase)
  ```yaml
  skills:
    - warmup
    - keynote
    - high-jump
  ```
- Auto-add: during real runs (not `--print-only`), unknown names/tags are appended. Fuzzy checks help avoid adding close mishears.

## Behavior
- Silence gate finds first speech before transcription.
- Filenames include timestamp `YYYYMMDD_HHMM` from metadata (or file time).
- Output path: `./output/<Name>/<date_name_skill_runn>.mp4` (treat `skill` as your tag/category).
- After successful copy, originals move to `./input/Processed/`.

## Requirements
- Python 3.10+
- FFmpeg: auto-detected from `./ffmpeg/bin/ffmpeg(.exe)` or `./Tools/ffmpeg/bin/ffmpeg(.exe)`. On Windows you can `winget install Gyan.FFmpeg`.
- Python packages: `openai-whisper`, `rapidfuzz`, `pyyaml`, `python-slugify` (installed automatically on first run)


## Topics
Suggested GitHub topics for discovery: voice-to-text, speech-to-text, transcription, whisper, ffmpeg, audio-processing, video-processing, auto-tagging.
