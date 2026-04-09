from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_NAME = "YouTubeIntelDesktop"
SPEC_FILE = ROOT / f"{APP_NAME}.spec"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def _make_release_paths() -> tuple[str, str]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dist_path = DIST / "releases" / stamp
    work_path = BUILD / f"pyinstaller-{stamp}"
    dist_path.mkdir(parents=True, exist_ok=False)
    work_path.mkdir(parents=True, exist_ok=True)
    return str(dist_path), str(work_path)


def main() -> None:
    if shutil.which("pyinstaller") is None:
        raise SystemExit("PyInstaller is not installed. Run: pip install -r requirements-build.txt")
    if not SPEC_FILE.exists():
        raise SystemExit(f"PyInstaller spec is missing: {SPEC_FILE}")

    frontend_dist = ROOT / "frontend" / "dist"
    if not frontend_dist.exists():
        if shutil.which("npm") is None:
            raise SystemExit("npm is required to build the frontend before packaging.")
        run(["npm", "install"], cwd=ROOT / "frontend")
        run(["npm", "run", "build"], cwd=ROOT / "frontend")

    dist_path, work_path = _make_release_paths()

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        dist_path,
        "--workpath",
        work_path,
        str(SPEC_FILE),
    ]

    run(cmd)
    print(f"Done. EXE folder: {Path(dist_path) / APP_NAME}")


if __name__ == "__main__":
    main()
