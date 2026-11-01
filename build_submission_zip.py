#!/usr/bin/env python3
"""One-click script to package your clean submission ZIP from VS Code.

Excludes .env, .venv, .git, __pycache__, and dataset copies as required by Page 14 of the Capstone Brief.
"""
import zipfile
from pathlib import Path

EXCLUDED_DIRS = {".venv", "venv", "env", ".git", "__pycache__", ".pytest_cache", "Capstone_Project_Dataset_FTTT_2026"}
EXCLUDED_FILES = {".env", ".env.local", ".DS_Store", "Thumbs.db"}


def main():
    root = Path(__file__).resolve().parent
    team_folder_name = root.name
    out_zip = root.parent / f"{team_folder_name}.zip"

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if any(part in EXCLUDED_DIRS for part in p.parts):
                continue
            if p.name in EXCLUDED_FILES or p.suffix in (".pyc", ".zip"):
                continue
            if p.is_file():
                arcname = f"{team_folder_name}/{p.relative_to(root).as_posix()}"
                zf.write(p, arcname)

    print(f"[SUCCESS] Created clean submission ZIP: {out_zip}")
    print("Included files:")
    with zipfile.ZipFile(out_zip, "r") as zf:
        for info in zf.infolist():
            print(f"  - {info.filename} ({info.file_size} bytes)")


if __name__ == "__main__":
    main()
