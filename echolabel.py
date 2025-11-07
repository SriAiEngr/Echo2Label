import os
import sys
import shutil
import subprocess
from pathlib import Path

# Dependencies to bootstrap on first run
DEPS = [
    "openai-whisper==20231117",
    "rapidfuzz==3.9.6",
    "pyyaml==6.0.2",
    "python-slugify==8.0.4",
]


def ensure_dependencies_once():
    data_dir = Path("data"); data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / ".deps_installed"
    try:
        import whisper  # noqa: F401
        import rapidfuzz  # noqa: F401
        import yaml  # noqa: F401
        import slugify  # noqa: F401
        marker.write_text("ok", encoding="utf-8")
        return
    except Exception:
        pass
    print("Installing required Python packages (first run)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", *DEPS])
        marker.write_text("ok", encoding="utf-8")
    except Exception as e:
        print(f"WARNING: Failed to auto-install dependencies: {e}")
        # Continue; scripts/autotag.py will still raise a clear ImportError if missing


def ensure_data_initialized_once():
    """On first run, ensure roster and skills files exist and are non-empty.
    If missing/empty, create sample content and prompt the user.
    """
    import yaml  # type: ignore

    data_dir = Path("data"); data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / ".data_initialized"
    if marker.exists():
        return

    roster_path = data_dir / "roster.yaml"
    skills_path = data_dir / "skills.yaml"

    def read_yaml_safe(p: Path):
        if not p.exists() or p.stat().st_size == 0:
            return None
        try:
            return yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    roster = read_yaml_safe(roster_path)
    skills = read_yaml_safe(skills_path)

    need_roster = not isinstance(roster, dict) or not roster.get("students")
    need_skills = not isinstance(skills, dict) or not skills.get("skills")

    sample_roster = """
students:
  - Emily
  - Jack
  - Liam
  - Sophia
  - Ava
  - Noah
""".strip()

    sample_skills = """
skills:
  - warmup
  - rehearsal
  - demo
  - keynote
  - q&a
  - meeting
""".strip()

    changed = False
    if need_roster:
        roster_path.write_text(sample_roster + "\n", encoding="utf-8")
        changed = True
    if need_skills:
        skills_path.write_text(sample_skills + "\n", encoding="utf-8")
        changed = True

    if changed:
        print("It looks like this is your first run and your data files were empty or missing.")
        print(f"A sample roster was written to: {roster_path}")
        print(f"A sample skills list was written to: {skills_path}")
        print("Edit these files to match your names and tags.")
        try:
            resp = input("Press Enter to continue with the samples, or type 'n' to abort and edit now: ").strip().lower()
        except KeyboardInterrupt:
            print("\nCanceled."); sys.exit(1)
        if resp == 'n':
            print("Please edit the YAML files and run again.")
            sys.exit(1)

    try:
        marker.write_text("ok", encoding="utf-8")
    except Exception:
        pass


def ensure_ffmpeg_on_path():
    """Ensure ffmpeg is invokable in this process PATH on Windows/macOS/Linux."""
    # Also accept a local copy inside ./ffmpeg or repo root
    cwd = os.getcwd()
    local_candidates = [
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
    for path in local_candidates:
        if os.path.isfile(path):
            # prepend its directory to PATH for this process
            os.environ["PATH"] = os.path.dirname(path) + os.pathsep + os.environ.get("PATH", "")
            return
    if shutil.which("ffmpeg"):
        return
    # Try common Windows locations (winget links / WindowsApps / C:\\Tools\\ffmpeg)
    candidates = []
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        candidates.extend([
            os.path.join(local, "Microsoft", "WinGet", "Links"),
            os.path.join(local, "Microsoft", "WindowsApps"),
            r"C:\\Tools\\ffmpeg\\bin",
        ])
    for c in candidates:
        if c and os.path.isdir(c) and c not in os.environ.get("PATH", ""):
            os.environ["PATH"] = c + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("ffmpeg"):
                return
    # Final check; if still missing, raise a helpful error
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found. Place ffmpeg in ./ffmpeg (ffmpeg/bin) or install via winget (winget install Gyan.FFmpeg).")
        sys.exit(2)


def _load_yaml_default_config():
    cfg_path = Path("data/config.yaml")
    if not cfg_path.exists():
        return {"defaults": {}}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {"defaults": {}}
    except Exception:
        return {"defaults": {}}


def _prompt_path(prompt: str, default: str) -> str:
    try:
        val = input(f"{prompt} [{default}]: ").strip()
    except KeyboardInterrupt:
        print("\nCanceled.")
        sys.exit(1)
    return val or default


def cli_wrapper():
    # If user provided CLI args, pass-through to underlying main
    if len(sys.argv) > 1:
        ensure_dependencies_once()
        ensure_data_initialized_once()
        ensure_ffmpeg_on_path()
        from scripts.autotag import main
        main()
        return

    # Interactive mode: use config defaults if present, otherwise prompt
    ensure_dependencies_once()
    ensure_data_initialized_once()
    cfg = _load_yaml_default_config()
    dfl = cfg.get("defaults", {})
    in_dir_default = dfl.get("input_dir") or "./input"
    out_dir_default = dfl.get("out_root") or "./output"

    if dfl.get("input_dir") and dfl.get("out_root"):
        # Use config-specified paths without prompting
        in_dir = in_dir_default
        out_dir = out_dir_default
        print(f"Using config paths: input_dir={in_dir} out_root={out_dir}")
    else:
        in_dir = _prompt_path("Enter input folder path", in_dir_default)
        out_dir = _prompt_path("Enter output folder path", out_dir_default)
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    ensure_ffmpeg_on_path()

    sys.argv = [
        sys.argv[0],
        "--input-dir", in_dir,
        "--out-root", out_dir,
    ]
    from scripts.autotag import main
    main()


if __name__ == "__main__":
    cli_wrapper()
