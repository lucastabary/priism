import numpy as np
import pytest

from priism.data.tempo_key import estimate_tempo, find_pairs, pairing

SR = 22050


def clicks(bpm: float, seconds: float = 30.0) -> np.ndarray:
    """A kick-like thump on every beat and a hat on every off-beat."""
    x = np.zeros(int(seconds * SR))
    beat = 60 / bpm
    t = np.arange(int(0.08 * SR)) / SR
    kick = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 40)
    hat = np.random.default_rng(0).normal(size=len(t)) * np.exp(-t * 200) * 0.3
    for k in range(int(seconds / beat)):
        i = int(k * beat * SR)
        x[i:i + len(t)] += kick[: len(x) - i]
        j = int((k + 0.5) * beat * SR)
        if j < len(x):
            x[j:j + len(t)] += hat[: len(x) - j]
    return x


@pytest.mark.parametrize("bpm", [124.0, 138.0, 174.0])
def test_tempo_of_a_beat(bpm):
    est = estimate_tempo(clicks(bpm), SR)
    assert min(abs(est / (bpm * m) - 1) for m in (0.5, 1, 2)) < 0.02


def test_pairing_allows_small_shifts_and_half_tempo():
    a = {"bpm": 128.0, "tonic": 9, "mode": "minor"}
    assert pairing(a, {"bpm": 126.0, "tonic": 9, "mode": "minor"}) == {"semitones": 0, "stretch": pytest.approx(128 / 126)}
    # C major is the relative major of A minor: same notes, no shift needed.
    assert pairing(a, {"bpm": 128.0, "tonic": 0, "mode": "major"})["semitones"] == 0
    assert pairing(a, {"bpm": 64.5, "tonic": 7, "mode": "minor"})["semitones"] == 2
    assert pairing(a, {"bpm": 128.0, "tonic": 3, "mode": "minor"}) is None  # a tritone away
    assert pairing(a, {"bpm": 140.0, "tonic": 9, "mode": "minor"}) is None  # 9 % faster
    tags = {"x": a, "y": {"bpm": 127.0, "tonic": 10, "mode": "minor"}, "z": {"bpm": 90.0, "tonic": 9, "mode": "minor"}}
    assert [(p[0], p[1]) for p in find_pairs(tags)] == [("x", "y")]


def test_key_of_a_minor_chord_loop(tmp_path):
    pytest.importorskip("essentia")
    sf = pytest.importorskip("soundfile")
    from priism.data.tempo_key import estimate_key

    t = np.arange(int(2 * SR)) / SR
    chords = [[57, 60, 64], [53, 57, 60], [55, 59, 62], [52, 56, 59]]  # Am F G E
    y = np.concatenate([sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * t) for m in c) for c in chords] * 4)
    sf.write(tmp_path / "am.wav", 0.2 * y, SR)
    tonic, mode, _ = estimate_key(tmp_path / "am.wav", SR)
    assert (tonic, mode) == (9, "minor")
