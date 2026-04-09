from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_NAME = "YouTubeIntelDesktop"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def main() -> None:
    if shutil.which("pyinstaller") is None:
        raise SystemExit("PyInstaller is not installed. Run: pip install -r requirements-build.txt")

    frontend_dist = ROOT / "frontend" / "dist"
    if not frontend_dist.exists():
        if shutil.which("npm") is None:
            raise SystemExit("npm is required to build the frontend before packaging.")
        run(["npm", "install"], cwd=ROOT / "frontend")
        run(["npm", "run", "build"], cwd=ROOT / "frontend")

    data_sep = ";" if os.name == "nt" else ":"
    add_data = [
        f"frontend/dist{data_sep}frontend/dist",
        f"desktop/defaults.env{data_sep}desktop",
    ]

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        APP_NAME,
        "--onedir",
        "--collect-submodules",
        "app",
        "--collect-submodules",
        "sentence_transformers",
        "--collect-submodules",
        "transformers",
        "--collect-submodules",
        "sklearn",
        "--collect-submodules",
        "scipy",
        "--collect-submodules",
        "hdbscan",
        "--collect-data",
        "sentence_transformers",
        "--collect-data",
        "transformers",
        "--collect-data",
        "torch",
    ]
    for item in add_data:
        cmd.extend(["--add-data", item])
    cmd.append("desktop_main.py")

    run(cmd)
    print(f"Done. EXE folder: {DIST / APP_NAME}")


if __name__ == "__main__":
    main()
