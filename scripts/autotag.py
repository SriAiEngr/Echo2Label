import argparse, os, sys, subprocess, tempfile, shutil, datetime, re, json, logging, pathlib
import math, time
from collections import defaultdict
from slugify import slugify
from rapidfuzz import process, fuzz
import yaml

# --- logging ---
LOG_DIR = pathlib.Path("./logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "autotag.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def find_ffmpeg() -> str:
    """Return a path to an ffmpeg executable.
    Looks in PATH first, then common local folders inside the repo.
    Raises RuntimeError if not found.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # local candidates (cross‑platform)
    cwd = os.getcwd()
    candidates = [
        # ./ffmpeg
        os.path.join(cwd, "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(cwd, "ffmpeg", "bin", "ffmpeg"),
        os.path.join(cwd, "ffmpeg", "ffmpeg.exe"),
        os.path.join(cwd, "ffmpeg", "ffmpeg"),
        os.path.join(cwd, "ffmpeg.exe"),
        os.path.join(cwd, "ffmpeg"),
        # ./Tools/ffmpeg
        os.path.join(cwd, "Tools", "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(cwd, "Tools", "ffmpeg", "bin", "ffmpeg"),
        os.path.join(cwd, "Tools", "ffmpeg", "ffmpeg.exe"),
        os.path.join(cwd, "Tools", "ffmpeg", "ffmpeg"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError("ffmpeg not found. Place ffmpeg in ./ffmpeg/bin or add to PATH.")


def find_ffprobe() -> str:
    """Return a path to an ffprobe executable, checking PATH and local folders."""
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, "ffmpeg", "bin", "ffprobe.exe"),
        os.path.join(cwd, "ffmpeg", "bin", "ffprobe"),
        os.path.join(cwd, "ffmpeg", "ffprobe.exe"),
        os.path.join(cwd, "ffmpeg", "ffprobe"),
        os.path.join(cwd, "Tools", "ffmpeg", "bin", "ffprobe.exe"),
        os.path.join(cwd, "Tools", "ffmpeg", "bin", "ffprobe"),
        os.path.join(cwd, "Tools", "ffmpeg", "ffprobe.exe"),
        os.path.join(cwd, "Tools", "ffmpeg", "ffprobe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError("ffprobe not found. Place ffmpeg/ffprobe in ./ffmpeg/bin or ./Tools/ffmpeg/bin, or add to PATH.")


def ffmpeg_detect_speech_start(video_path, probe_seconds=12, threshold="-35dB", min_silence=0.25):
    """
    Returns float seconds where speech likely starts, or 0.0 if unknown.
    We scan first `probe_seconds` seconds for end of initial silence.
    """
    # Run a short audio probe with silencedetect
    # Note: ffmpeg logs to stderr; parse lines like: "silence_end: 1.28"
    cmd = [
        find_ffmpeg(), "-hide_banner", "-nostats",
        "-i", video_path, "-t", str(probe_seconds),
        "-af", f"silencedetect=noise={threshold}:d={min_silence}",
        "-f", "null", "-"
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start = 0.0
    for line in p.stderr.splitlines():
        if "silence_end:" in line:
            try:
                # pick first silence_end as start of sound
                val = float(line.split("silence_end:")[1].split("|")[0].strip())
                start = max(0.0, val)
                break
            except Exception:
                pass
    return start


def ffmpeg_extract_head(video_path, out_wav, seconds=6, use_silence_gate=True):
    start = 0.0
    if use_silence_gate:
        try:
            start = ffmpeg_detect_speech_start(video_path)
        except Exception:
            start = 0.0
    # mono 16k
    cmd = [
        find_ffmpeg(), "-y",
        "-ss", f"{start:.2f}",
        "-i", video_path,
        "-t", str(seconds),
        "-vn", "-ac", "1", "-ar", "16000",
        out_wav
    ]
    run(cmd)


def load_list(file_path, key=None):
    if not os.path.exists(file_path):
        return []
    if file_path.lower().endswith((".yaml", ".yml")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if key and key in data:
            return [str(x).lower() for x in data[key]]
        if isinstance(data, list):
            return [str(x).lower() for x in data]
        return []
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return [ln.strip().lower() for ln in f if ln.strip()]


def load_yaml(p):
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(p, data):
    ensure_dir(os.path.dirname(p) or ".")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=True)


def _ensure_value_in_yaml_list(file_path: str, key: str, value: str, mode: str = "lower") -> bool:
    """Ensure `value` is present under list `key` in YAML file_path.
    - mode: 'lower' writes lowercase, 'title' writes Title Case.
    Returns True if file updated, False otherwise.
    """
    if not value:
        return False
    data = load_yaml(file_path) or {}
    seq = data.get(key)
    if not isinstance(seq, list):
        seq = []
    # Check membership case-insensitively
    existing_norm = {str(x).strip().lower() for x in seq}
    v_norm = value.strip().lower()
    if v_norm in existing_norm:
        return False
    # Transform for storage
    if mode == "title":
        stored = str(value).strip().title()
    elif mode == "lower":
        stored = str(value).strip().lower()
    else:
        stored = str(value).strip()
    seq.append(stored)
    data[key] = seq
    save_yaml(file_path, data)
    return True


def load_config(path: str):
    defaults = {
        "seconds": 6,
        "out_root": "./output",
        "pattern": "{date}_{name}_{skill}_run{run}.{ext}",
        "model": "base",
        "use_silence_gate": True,
    }
    cfg = {"defaults": defaults, "counters": {}}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        # merge shallowly
        cfg["defaults"].update(file_cfg.get("defaults", {}) or {})
        cfg["counters"].update(file_cfg.get("counters", {}) or {})
    return cfg


def save_config(path: str, cfg: dict):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def best_match(token, candidates, cutoff=80):
    if not candidates:
        return None
    res = process.extractOne(token, candidates, scorer=fuzz.token_set_ratio)
    if res and res[1] >= cutoff:
        return res[0]
    return None


_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50
}

_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
}

# Additional public mappings and helper from spec (kept alongside existing)
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
}
WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50
}


def words_to_int(s):
    s = s.lower().strip().replace("-", " ")
    parts = s.split()
    total = 0
    for w in parts:
        if w in ORDINALS:
            total = ORDINALS[w]
            return total
    for w in parts:
        if w in WORDS:
            total += WORDS[w]
    return total if total > 0 else None


def _parse_run_value(val: str):
    val = val.strip().lower()
    if val.isdigit():
        return int(val)
    # broader words/ordinal handling
    maybe = words_to_int(val)
    if maybe is not None:
        return maybe
    if val in _WORD_NUMS:
        return _WORD_NUMS[val]
    if val in _ORDINALS:
        return _ORDINALS[val]
    # simple hyphenated like twenty-one
    if "-" in val:
        parts = val.split("-")
        if len(parts) == 2 and parts[0] in _WORD_NUMS and parts[1] in _WORD_NUMS:
            return _WORD_NUMS[parts[0]] + _WORD_NUMS[parts[1]]
    return None


def parse_transcript(t, students, skills):
    low = t.lower()

    # run number: accept digits or word/ordinal after 'run'
    run_num = None
    # allow punctuation between 'run' and the value (e.g., 'run, three')
    m = re.search(r"\brun(?:\s+number)?[\s,:-]+([a-z\-]+|[0-9]+)\b", low)
    if m:
        run_num = _parse_run_value(m.group(1))
    if run_num is None:
        # try broader word sequences e.g., "run three", "run, twenty one"
        mw = re.search(r"\brun[\s,:-]+([a-z\- ]+)\b", low)
        if mw:
            run_num = words_to_int(mw.group(1))

    # skill: best fuzzy match over phrases found in transcript n-grams
    found_skill = None
    for s in skills:
        if s in low:
            found_skill = s
            break
    if not found_skill:
        words = re.findall(r"[a-z]+", low)
        grams = set(words)
        grams.update([" ".join(words[i:i+2]) for i in range(len(words)-1)])
        grams.update([" ".join(words[i:i+3]) for i in range(len(words)-2)])
        candidate = None; score = -1
        for g in grams:
            res = process.extractOne(g, skills, scorer=fuzz.token_set_ratio)
            if res and res[1] > score:
                candidate, score = res[0], res[1]
        # accept a reasonably close match so common mis-hearings map correctly
        if score >= 70:
            found_skill = candidate

    # name: fuzzy over tokens
    name = None
    token_list = re.findall(r"[a-z]+", low)
    for tok in set(token_list):
        mname = best_match(tok, students, cutoff=90)
        if mname:
            name = mname
            break

    # Fallbacks: use actual spoken values if not in lists
    # Try capitalized token from original transcript for name
    if not name:
        caps = re.findall(r"\b([A-Z][a-z]+)\b", t)
        stop_caps = {"Hey", "Hi", "Hello", "Run"}
        for c in caps:
            if c not in stop_caps:
                name = c.lower()
                break
    # If still no name, take first non-filler token before 'run'
    if not name and token_list:
        fillers = {"hey", "hi", "hello", "run"}
        for w in token_list:
            if w not in fillers:
                name = w
                break

    # If skill not found from list, extract words immediately before 'run'
    if not found_skill:
        words = token_list
        skill_phrase = None
        try:
            run_idx = words.index("run")
        except ValueError:
            run_idx = -1
        if run_idx > 0:
            # prefer words between name appearance and 'run'
            start_idx = 0
            if name and name in words:
                try:
                    start_idx = words.index(name) + 1
                except ValueError:
                    start_idx = max(0, run_idx - 3)
            window = words[start_idx:run_idx]
            # remove common fillers
            fillers2 = {"hey", "hi", "hello", "and", "then", "the", "a"}
            window = [w for w in window if w not in fillers2]
            if window:
                # take last up to 3 words as the spoken skill phrase
                skill_phrase = " ".join(window[-3:])
        # If no 'run' keyword, take up to first 3 non-filler words
        if not skill_phrase and token_list:
            keep = [w for w in token_list if w not in {"hey", "hi", "hello", "run"}]
            if keep:
                skill_phrase = " ".join(keep[:3])
        if skill_phrase:
            found_skill = skill_phrase

    # Final skill correction: try fuzzy-correcting to known skills; otherwise keep original
    if found_skill and (found_skill not in skills):
        try:
            res = process.extractOne(found_skill, skills, scorer=fuzz.token_set_ratio)
            if res and res[1] >= 70:
                found_skill = res[0]
        except Exception:
            pass

    return {
        "name": name,
        "skill": found_skill,
        "run": run_num
    }


def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def human_date_from_video(path):
    """Return date label as YYYYMMDD_HHMM.
    Prefer container metadata (creation_time via ffprobe); fallback to file mtime.
    """
    dt = None
    # Try ffprobe creation_time
    try:
        probe = subprocess.run([
            find_ffprobe(), "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if probe.returncode == 0 and probe.stdout:
            info = json.loads(probe.stdout)
            # format.tags.creation_time
            ct = None
            try:
                ct = (info.get("format", {}).get("tags", {}) or {}).get("creation_time")
            except Exception:
                ct = None
            if not ct:
                # search streams
                for st in (info.get("streams") or []):
                    tags = st.get("tags") or {}
                    if tags.get("creation_time"):
                        ct = tags.get("creation_time"); break
            if ct:
                s = str(ct).strip().replace("Z", "+00:00")
                try:
                    dt = datetime.datetime.fromisoformat(s)
                except Exception:
                    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?", s)
                    if m:
                        y, mo, d, h, mi, se = m.groups()
                        dt = datetime.datetime(int(y), int(mo), int(d), int(h), int(mi), int(se or 0))
                # Normalize to local time if aware
                if dt is not None and dt.tzinfo is not None:
                    dt = dt.astimezone().replace(tzinfo=None)
    except Exception:
        dt = None

    if dt is None:
        # fallback to file mtime (local time)
        dt = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    return dt.strftime("%Y%m%d_%H%M")


def safe_filename(pattern, meta, ext):
    # fill defaults
    name_val = meta.get("name") or "unknown"
    skill_val = meta.get("skill") or "skill"
    run = meta.get("run") or 1
    date = meta.get("date")
    # Title-case name and skill, preserve casing via slugify(lowercase=False)
    name_slug = slugify(str(name_val).title(), separator="_", lowercase=False)
    skill_slug = slugify(str(skill_val).title(), separator="_", lowercase=False)
    base = pattern.format(
        date=date,
        name=name_slug,
        skill=skill_slug,
        run=run,
        ext=ext
    )
    # Ensure each underscore-separated token starts with a capital letter; keep extension casing
    if "." in base:
        root, dot, extpart = base.rpartition(".")
        tokens = [t[:1].upper() + t[1:] if t else t for t in root.split("_")]
        root_tc = "_".join(tokens)
        return root_tc + dot + extpart
    else:
        tokens = [t[:1].upper() + t[1:] if t else t for t in base.split("_")]
        return "_".join(tokens)


# --- Batch/Watch helpers ---
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def list_videos(folder):
    files = []
    for p in sorted(pathlib.Path(folder).glob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(str(p))
    return files


def _speak_back(meta):
    text = f"{(meta.get('name') or 'unknown')}, {meta.get('skill') or 'skill'}, run {meta.get('run') or 1}"
    try:
        if sys.platform == "darwin":
            subprocess.run(["say", text], check=False)
        elif sys.platform == "win32":
            ps_text = text.replace("'", "''")
            cmd = [
                "powershell", "-NoProfile", "-Command",
                f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{ps_text}')"
            ]
            subprocess.run(cmd, check=False)
        else:
            subprocess.run(["espeak", text], check=False)
    except Exception:
        pass


def process_one(video_path, **kwargs):
    """Process a single video and return destination path.
    kwargs: seconds, roster, skills, out_root, pattern, delete_original, print_only, use_silence_gate, chosen_model, say, config
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")

    seconds = kwargs.get("seconds", 6)
    roster = kwargs.get("roster", "./data/roster.yaml")
    skills_file = kwargs.get("skills", "./data/skills.yaml")
    out_root = kwargs.get("out_root", "./output")
    pattern = kwargs.get("pattern", "{date}_{name}_{skill}_run{run}.{ext}")
    delete_original = bool(kwargs.get("delete_original", False))
    print_only = bool(kwargs.get("print_only", False))
    use_silence_gate = bool(kwargs.get("use_silence_gate", True))
    chosen_model = kwargs.get("chosen_model", "base")
    do_say = bool(kwargs.get("say", False))
    config_path = kwargs.get("config", "./data/config.yaml")

    # 1) extract head audio
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "head.wav")
        ffmpeg_extract_head(video_path, wav, seconds=seconds, use_silence_gate=use_silence_gate)

        # 2) transcribe
        try:
            import whisper
        except ImportError:
            raise RuntimeError("openai-whisper not installed. pip install openai-whisper")
        model = whisper.load_model(chosen_model)
        # Avoid FP16 warning on CPU by forcing fp16=False
        result = model.transcribe(wav, fp16=False)
        transcript = (result.get("text") or "").strip()
        if not transcript:
            raise RuntimeError("empty transcript")

    # 3) parse
    students = load_list(roster, "students")
    skills = load_list(skills_file, "skills")
    meta = parse_transcript(transcript, students, skills)
    meta["date"] = human_date_from_video(video_path)
    logging.info(json.dumps({"video": video_path, "transcript": transcript, "meta": meta}, ensure_ascii=False))

    # No counters: if run missing, default to 1 (handled in safe_filename)
    name_key = (meta.get("name") or "unknown").lower()

    # 4) destination
    ext = os.path.splitext(video_path)[1].lstrip(".") or "mp4"
    fname = safe_filename(pattern, meta, ext)
    name_dir = (meta.get("name") or "unknown").title()
    # Store directly under the name folder (no Run <n> subfolder)
    dest_dir = os.path.join(out_root, name_dir)
    ensure_dir(dest_dir)
    dest_path = os.path.join(dest_dir, fname)

    # 5) report
    print(f"TRANSCRIPT: {transcript}")
    print(f"PARSED: name={meta.get('name')} skill={meta.get('skill')} run={meta.get('run')}")
    print(f"OUTPUT: {dest_path}")

    # Optionally append newly detected name/skill to roster/skills files (skip during print-only)
    if not print_only:
        try:
            detected_name = meta.get("name")
            if detected_name and detected_name.strip().lower() not in (students or []):
                _ensure_value_in_yaml_list(roster, "students", detected_name, mode="title")
        except Exception as e:
            print(f"WARN: could not update roster file: {e}", file=sys.stderr)
        try:
            detected_skill = meta.get("skill")
            if detected_skill:
                det_low = detected_skill.strip().lower()
                if det_low not in (skills or []):
                    # Only append if it doesn't look similar to any existing skill (avoid adding mishears like 'dawn')
                    try:
                        sim = process.extractOne(detected_skill, skills or [], scorer=fuzz.token_set_ratio)
                        if not sim or sim[1] < 60:
                            _ensure_value_in_yaml_list(skills_file, "skills", detected_skill, mode="lower")
                    except Exception:
                        pass
        except Exception as e:
            print(f"WARN: could not update skills file: {e}", file=sys.stderr)

    if print_only:
        return dest_path

    # optional confidence read-back
    if kwargs.get("say"):
        phrase = f"{(meta.get('name') or 'unknown').title()} run {meta.get('run') or 1}, {meta.get('skill') or 'skill'}"
        try:
            if sys.platform == "darwin":
                subprocess.run(["say", phrase])
            elif os.name == "nt":
                # Powershell TTS
                ps = f'Add-Type -AssemblyName System.Speech;$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;$speak.Speak("{phrase}");'
                subprocess.run(["powershell", "-Command", ps])
        except Exception:
            pass

    # copy to destination
    shutil.copy2(video_path, dest_path)
    if delete_original:
        os.remove(video_path)
    else:
        # Move original into a sibling 'Processed' folder under its source directory
        src_dir = os.path.dirname(video_path)
        processed_dir = os.path.join(src_dir, "Processed")
        ensure_dir(processed_dir)
        base_name = os.path.basename(video_path)
        target_path = os.path.join(processed_dir, base_name)
        if os.path.exists(target_path):
            stem, ext = os.path.splitext(base_name)
            i = 1
            while True:
                candidate = os.path.join(processed_dir, f"{stem}_processed{i}{ext}")
                if not os.path.exists(candidate):
                    target_path = candidate
                    break
                i += 1
        try:
            shutil.move(video_path, target_path)
        except Exception as e:
            print(f"WARN: could not move source to Processed: {e}", file=sys.stderr)

    print("DONE")
    return dest_path


def watch_folder(folder, poll=2.0, seen=None, **kwargs):
    seen = seen or set()
    print(f"Watching {folder} ... Ctrl+C to stop.")
    while True:
        try:
            for f in list_videos(folder):
                if f not in seen:
                    try:
                        dp = process_one(f, **kwargs)
                        print(f"Processed: {dp}")
                    except Exception as e:
                        print(f"ERROR processing {f}: {e}", file=sys.stderr)
                    seen.add(f)
            time.sleep(poll)
        except KeyboardInterrupt:
            print("Stopping watcher.")
            break


def main():
    parser = argparse.ArgumentParser(description="Auto-tag ski videos from voice note at head of clip.")
    parser.add_argument("video", nargs="?", help="Path to video")
    parser.add_argument("--seconds", type=int, default=None, help="Seconds to sample from head")
    parser.add_argument("--roster", default="./data/roster.yaml", help="student roster file")
    parser.add_argument("--skills", default="./data/skills.yaml", help="skills file")
    parser.add_argument("--out-root", default=None, help="destination root folder")
    parser.add_argument("--pattern", default=None, help="filename pattern")
    parser.add_argument("--input-dir", help="Batch process a folder of videos instead of a single file")
    parser.add_argument("--newest", type=int, default=0, help="Process only newest N files in --input-dir (0 = all)")
    parser.add_argument("--watch", action="store_true", help="Watch --input-dir for new files")
    parser.add_argument("--model", default=None, help="Whisper model: tiny/base/small/medium (overrides config)")
    parser.add_argument("--use-silence-gate", action="store_true", help="Enable silence gate for first-speech")
    parser.add_argument("--no-silence-gate", action="store_true", help="Disable silence gate")
    parser.add_argument("--config", default="./data/config.yaml", help="Global config (defaults)")
    parser.add_argument("--say", action="store_true", help="Speak back parsed fields before copy (macOS 'say' / PowerShell TTS)")
    parser.add_argument("--delete-original", action="store_true", help="delete source file after copy")
    parser.add_argument("--print-only", action="store_true", help="do not copy/move; just print plan")
    args = parser.parse_args()

    video_path = args.video
    # Load config and apply defaults
    cfg = load_yaml(args.config)
    defaults = cfg.get("defaults", {})
    seconds = args.seconds or defaults.get("seconds", 6)
    out_root = args.out_root or defaults.get("out_root", "./output")
    pattern = args.pattern or defaults.get("pattern", "{date}_{name}_{skill}_run{run}.{ext}")
    use_silence_gate = defaults.get("use_silence_gate", True)
    if args.use_silence_gate:
        use_silence_gate = True
    if args.no_silence_gate:
        use_silence_gate = False
    chosen_model = (args.model or defaults.get("model", "base")).lower()

    # Batch or single-file branch
    if args.input_dir:
        folder = args.input_dir
        if not os.path.isdir(folder):
            print(f"ERROR: input dir not found: {folder}", file=sys.stderr); sys.exit(2)
        vids = list_videos(folder)
        if args.newest > 0:
            vids = sorted(vids, key=lambda p: os.path.getmtime(p), reverse=True)[:args.newest]
        for v in vids:
            try:
                process_one(v,
                    seconds=seconds, roster=args.roster, skills=args.skills,
                    out_root=out_root, pattern=pattern,
                    delete_original=args.delete_original, print_only=args.print_only,
                    use_silence_gate=use_silence_gate, chosen_model=chosen_model, say=args.say,
                    config=args.config)
            except Exception as e:
                print(f"ERROR processing {v}: {e}", file=sys.stderr)
        if args.watch:
            watch_folder(folder,
                seconds=seconds, roster=args.roster, skills=args.skills,
                out_root=out_root, pattern=pattern,
                delete_original=args.delete_original, print_only=False,
                use_silence_gate=use_silence_gate, chosen_model=chosen_model, say=args.say,
                config=args.config)
    else:
        if not video_path or not os.path.exists(video_path):
            print(f"ERROR: video not found: {video_path}", file=sys.stderr); sys.exit(2)
        process_one(video_path,
            seconds=seconds, roster=args.roster, skills=args.skills,
            out_root=out_root, pattern=pattern,
            delete_original=args.delete_original, print_only=args.print_only,
            use_silence_gate=use_silence_gate, chosen_model=chosen_model, say=args.say,
            config=args.config)


if __name__ == "__main__":
    main()
