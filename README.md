# EchoLabel - Voice Autotag for Ski Clips

## Quickstart
1) Python 3.10+:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install openai-whisper rapidfuzz pyyaml python-slugify
   ```
2) Put a test video in `./input/clip001.mp4`.
3) Run (single file):
   ```bash
   python echolabel.py ./input/clip001.mp4 --seconds 6 --use-silence-gate --print-only
   ```
4) If the plan looks right, run 
    python echolabel.py

## Simple Steps
1) Download or clone this package to a local folder.
2) Place a sample video file into the `input` folder.
3) Open PowerShell in the project folder and run:
   `python echolabel.py`
4) The app processes the clip(s) and saves a new, labeled file under `output/<PlayerName>/`.
5) After a successful copy, the original file is moved from `input` to `input/Processed/`.

## Setup Files
- `data/config.yaml`
  - Set global defaults used by the app.
  - Key fields you may want to set:
    - `input_dir`: path to the folder of incoming clips (e.g., `./input`)
    - `out_root`: destination root for organized clips (e.g., `./output`)
    - `seconds`: how many seconds from the head to sample for voice note
    - `pattern`: filename pattern (e.g., `{date}_{name}_{skill}_run{run}.{ext}`)
    - `model`: whisper model to use (e.g., `tiny`, `base`, `small`)
    - `use_silence_gate`: true/false to detect first speech before sampling

- `data/roster.yaml` (sometimes called "roaster.yaml")
  - List of student names stored in Title Case.
  - Example:
    ```yaml
    students:
      - Emily
      - Jack
      - Liam
    ```

- `data/skills.yaml`
  - List of ski skills stored in lowercase for robust matching.
  - Example:
    ```yaml
    skills:
      - parallel-turn
      - carving
      - slalom
    ```

### Auto-adding new values
- During real runs (not `--print-only`), if the script detects a name or skill that is not in the respective list:
  - Name is appended to `data/roster.yaml` in Title Case.
  - Skill is appended to `data/skills.yaml` in lowercase.
  - For skills, a fuzzy check prevents adding close misheard variants (e.g., "dawn" won't be added if "turn" exists and is similar).
- You can edit these YAML files later to fix/rename items or remove any entries.

## Batch & Watch
- Batch newest N clips in a folder (no dummy video needed):
  `python echolabel.py --input-dir ./input --newest 5 --use-silence-gate`
- Watch a folder for new files:
  `python echolabel.py --input-dir ./input --watch --use-silence-gate --model tiny`

## Behavior
- Silence gate: detects first speech and seeks before extracting audio for Whisper.
- Timestamps: filenames include date+time from video metadata (or file time) as `YYYYMMDD_HHMM`.
- Output structure: files land under `./output/<Name>/<date_name_skill_runn>.mp4`.
- Input housekeeping: after a successful copy, originals move to `./input/Processed/`.

## Config
- Defaults: `data/config.yaml`
  ```yaml
  defaults:
    input_dir: ./input
    seconds: 6
    out_root: ./output
    pattern: "{date}_{name}_{skill}_run{run}.{ext}"
    model: base
    use_silence_gate: true
  ```

## Tools Used
- FFmpeg (auto-detected locally from `./Tools/ffmpeg/bin/ffmpeg(.exe)` or `./ffmpeg/bin/ffmpeg(.exe)`; no PATH changes required)
- Python 3.10+
- OpenAI Whisper (openai-whisper)
- RapidFuzz, PyYAML, python-slugify
