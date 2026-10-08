"""Listening page: one self-contained HTML file to play a mix and its tracks in sync (manual testing).

A song folder holds ``mix.<ext>`` and tracks, either next to it or in subfolders (``pistes/``,
``sources/``, a separation output...). Each subfolder becomes a group of rows, so ground truth and
separated tracks can sit on the same page. Audio is embedded (base64) so the page works from file://;
lossless files are re-encoded to MP3 with ffmpeg to keep the page small.
"""

from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
from pathlib import Path

AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aif", ".aiff"}
COMPRESSED = {".mp3", ".ogg", ".opus", ".m4a", ".aac"}
MIME = {".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".opus": "audio/ogg", ".m4a": "audio/mp4", ".aac": "audio/aac",
        ".wav": "audio/wav", ".flac": "audio/flac", ".aif": "audio/aiff", ".aiff": "audio/aiff"}
PAGE = "ecoute.html"


def _audio_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO)


def is_song(folder: Path) -> bool:
    return any(p.stem.lower() == "mix" for p in _audio_files(folder))


def scan_song(folder: str | Path) -> dict:
    """Mix file and track groups of a song folder: {"name", "mix", "groups": [(name, [files])]}."""
    folder = Path(folder)
    top = _audio_files(folder)
    mix = next((p for p in top if p.stem.lower() == "mix"), None)
    groups = []
    loose = [p for p in top if p != mix]
    if loose:
        groups.append(("", loose))
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO)
        if files:
            groups.append((sub.name, files))
    if mix is None and not groups:
        raise ValueError(f"no audio in {folder}")
    return {"name": folder.name, "mix": mix, "groups": groups}


def _encode(path: Path, bitrate: str | None) -> tuple[str, str]:
    """(mime, base64) of a file, re-encoded to MP3 when lossless (or always, when a bitrate is forced)."""
    ext = path.suffix.lower()
    if (bitrate or ext not in COMPRESSED) and shutil.which("ffmpeg"):
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c:a", "libmp3lame",
                              "-b:a", bitrate or "192k", "-f", "mp3", "-"], capture_output=True, check=True).stdout
        return "audio/mpeg", base64.b64encode(out).decode()
    return MIME.get(ext, "application/octet-stream"), base64.b64encode(path.read_bytes()).decode()


def render(song: dict, bitrate: str | None = None, back_link: str | None = None) -> str:
    def entry(p: Path) -> dict:
        mime, data = _encode(p, bitrate)
        return {"name": p.stem, "mime": mime, "data": data}

    data = {"name": song["name"], "back": back_link,
            "mix": entry(song["mix"]) if song["mix"] else None,
            "groups": [{"name": g, "tracks": [entry(p) for p in files]} for g, files in song["groups"]]}
    template = (Path(__file__).parent / "listen.html").read_text()
    payload = json.dumps(data).replace("</", "<\\/")
    return template.replace("__TITLE__", html.escape(song["name"])).replace("__DATA__", payload)


def build(folders: list[str | Path], out: str | Path | None = None, bitrate: str | None = None) -> list[Path]:
    """Write a page per song. A folder without a mix is a collection: one page per song inside plus an index."""
    written = []
    for folder in map(Path, folders):
        if is_song(folder) or not any(is_song(p) for p in folder.iterdir() if p.is_dir()):
            dest = Path(out) if out and len(folders) == 1 else folder / PAGE
            dest.write_text(render(scan_song(folder), bitrate))
            written.append(dest)
            continue
        songs = sorted(p for p in folder.iterdir() if p.is_dir() and is_song(p))
        for s in songs:
            dest = s / PAGE
            dest.write_text(render(scan_song(s), bitrate, back_link="../index.html"))
            written.append(dest)
        links = "\n".join(f'<li><a href="{html.escape(s.name)}/{PAGE}">{html.escape(s.name)}</a></li>' for s in songs)
        index = folder / "index.html"
        index.write_text(INDEX.replace("__TITLE__", html.escape(folder.name)).replace("__LINKS__", links))
        written.append(index)
    return written


INDEX = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Écoute · __TITLE__</title>
<style>
:root{--bg:#f7f7f5;--fg:#1c1c1a;--accent:#2f6fde}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--accent:#7aa7ff}}
body{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:640px;margin:auto}a{color:var(--accent)}
</style></head>
<body><main><h1>__TITLE__</h1><ul>
__LINKS__
</ul></main></body></html>
"""
