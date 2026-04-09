# YouTube Intel Desktop

Desktop packaging of the main **YouTube Intel** codebase.

This folder is now kept intentionally close to the main application logic, while preserving the desktop-only runtime layer required for local Windows delivery.

## What is shared with the main project
- `app/` mirrors the main backend services, schemas, and analytics pipelines.
- `frontend/` mirrors the main React UI, including toxic review and moderation flows.
- `tests/` reuses the main regression coverage where it makes sense for desktop.

## What stays desktop-specific
- `desktop/` — launcher, runtime paths, first-run setup, local secret storage, local queue.
- `desktop_main.py` — desktop entrypoint.
- `scripts/build_windows_exe.py` — PyInstaller packaging helper.
- SQLite/local runtime wiring in a few adapter files under `app/`.

## Current desktop architecture
- **Backend:** FastAPI
- **Database:** SQLite in the local app-data directory
- **Queue:** single-process sequential queue (no Redis/Celery required)
- **Frontend:** React + Vite, served by the local FastAPI process
- **Secrets:** stored locally via `desktop/bootstrap.py` + `desktop/security.py`

## Local app data on Windows
The packaged app writes to:

`%LOCALAPPDATA%/YouTubeIntelDesktop/`

Main contents:
- `config/desktop.env` — editable non-secret runtime settings
- `config/secrets.json` — locally stored secrets payload
- `data/youtube_intel.db` — SQLite database
- `data/reports/` — generated reports
- `data/cache/` — embeddings/cache artifacts
- `data/raw/` — raw fetched payloads

## Development run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
npm --prefix desktop/frontend install
npm --prefix desktop/frontend run build
python desktop_main.py
```

Then open the local URL printed by the launcher.

## First-run setup
On first launch the desktop UI asks for:
- OpenAI API key
- YouTube API key
- optional YouTube Playlist ID
- optional `YOUTUBE_OAUTH_CLIENT_ID`
- optional `YOUTUBE_OAUTH_CLIENT_SECRET`
- optional `YOUTUBE_OAUTH_REFRESH_TOKEN`

OAuth values are **not required** for the app to run, but they unlock channel-level
ban / best-effort unban actions through YouTube moderation API.

After setup you can manage the same values later from the **Dashboard** page.

## Windows build
Run this on a Windows machine:

```bash
pip install -r requirements.txt -r requirements-build.txt
npm --prefix frontend install
npm --prefix frontend run build
python scripts/build_windows_exe.py
```

Output:
- `dist/releases/<timestamp>/YouTubeIntelDesktop/`

Deliver the whole folder, not only a single executable file.

## What should be committed to GitHub
Commit:
- source code
- docs
- tests
- build scripts
- `YouTubeIntelDesktop.spec`

Do **not** commit:
- `dist/`
- `build/`
- local DB files
- secrets/env files
- caches and reports

## Repository naming note
The folder is now named `desktop/`, which is much clearer than the old `DesktopVersion/`.
If you later migrate to a larger monorepo, a natural next step would be `apps/desktop/`.
