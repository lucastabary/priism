import base64
import json
import re

import numpy as np
import soundfile as sf

from priism.listen import build, scan_song


def _song(folder, tracks=("00_kick", "01_bass")):
    (folder / "pistes").mkdir(parents=True)
    x = np.zeros((2205, 2), dtype=np.float32)
    sf.write(folder / "mix.wav", x, 22050)
    for t in tracks:
        sf.write(folder / "pistes" / f"{t}.wav", x, 22050)


def _data(page):
    raw = re.search(r'<script type="application/json" id="data">(.*?)</script>', page.read_text(), re.S).group(1)
    return json.loads(raw)


def test_scan_groups_subfolders(tmp_path):
    _song(tmp_path / "s")
    sf.write(tmp_path / "s" / "extra.wav", np.zeros(100, dtype=np.float32), 22050)
    song = scan_song(tmp_path / "s")
    assert song["mix"].name == "mix.wav"
    assert [(g, [p.stem for p in f]) for g, f in song["groups"]] == [("", ["extra"]), ("pistes", ["00_kick", "01_bass"])]


def test_single_song_page_embeds_audio(tmp_path):
    _song(tmp_path / "s")
    [page] = build([tmp_path / "s"])
    assert page == tmp_path / "s" / "ecoute.html"
    d = _data(page)
    assert d["mix"]["name"] == "mix" and [t["name"] for t in d["groups"][0]["tracks"]] == ["00_kick", "01_bass"]
    assert len(base64.b64decode(d["mix"]["data"])) > 0


def test_collection_writes_index(tmp_path):
    _song(tmp_path / "c" / "a")
    _song(tmp_path / "c" / "b", tracks=("00_lead",))
    pages = build([tmp_path / "c"])
    assert tmp_path / "c" / "index.html" in pages
    assert 'href="a/ecoute.html"' in (tmp_path / "c" / "index.html").read_text()
    assert _data(tmp_path / "c" / "b" / "ecoute.html")["back"] == "../index.html"
