import io
import zipfile

from priism.data.fma import RangeFile, member_name, select_tracks


def test_range_file_reads_a_zip_lazily():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("fma_full/000/000002.mp3", b"a" * 5000)
        z.writestr("fma_full/000/000005.mp3", bytes(range(256)) * 40)
    data = buf.getvalue()
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return data[start:end + 1]

    with zipfile.ZipFile(RangeFile(fetch, len(data), block=256)) as z:
        assert z.read(member_name(5, "full")) == bytes(range(256)) * 40
        assert z.read(member_name(2, "full")) == b"a" * 5000
    assert calls and all(e - s < len(data) for s, e in calls)


def test_select_tracks_balances_genres():
    tracks = [{"id": i, "genres_all": [1] if i % 2 else [2], "duration": 200} for i in range(20)]
    tracks.append({"id": 99, "genres_all": [1, 2], "duration": 30})  # too short
    picked = select_tracks(tracks, {"A": 1, "B": 2}, ["A", "B"], limit_per_genre=3)
    assert len(picked) == 6
    assert {t["picked_for"] for t in picked} == {"A", "B"}
    assert all(t["id"] != 99 for t in picked)
