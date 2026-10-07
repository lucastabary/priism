import json

import numpy as np

from priism.skank.generate import render, sample_params
from priism.skank.params import STYLES, SkankParams
from priism.sources import SOURCES, generate, render_params


def test_render_is_deterministic_and_well_formed():
    p = sample_params(5, duration_s=2.0)
    a, b = render(p), render(sample_params(5, duration_s=2.0))
    assert a.shape == (88200, 2) and a.dtype == np.float32
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    assert np.isclose(np.abs(a).max(), 10 ** (p.peak_dbfs / 20), rtol=1e-4)


def test_every_instrument_renders_and_has_no_bass():
    seen = set()
    for seed in range(40):
        p = sample_params(seed, duration_s=3.0)
        if p.tone.instrument in seen:
            continue
        seen.add(p.tone.instrument)
        a = render(p).mean(axis=1)
        spec = np.abs(np.fft.rfft(a))
        freqs = np.fft.rfftfreq(len(a), 1 / p.sample_rate)
        assert spec[freqs < 80].sum() < 0.05 * spec.sum(), p.tone.instrument
    assert seen == {"guitar", "organ", "piano", "stab"}


def test_hits_fall_on_the_style_grid():
    p = sample_params(11, duration_s=8.0)
    step = 60.0 / p.bpm / 4.0
    allowed = set(STYLES[p.style])
    for h in p.hits:
        pos = round((h.time_s % (16 * step)) / step) % 16
        assert pos in allowed or (pos - 1) % 16 in allowed  # swing and timing jitter stay within a step


def test_params_roundtrip_through_json():
    p = sample_params(3, duration_s=1.0)
    q = SkankParams.from_dict(json.loads(json.dumps(p.to_dict())))
    np.testing.assert_array_equal(render(p), render(q))


def test_generic_generate_writes_every_source(tmp_path):
    for name in SOURCES:
        files = generate(name, tmp_path / name, count=2, start_seed=10, duration_s=1.0)
        assert [f.split("/")[-1] for f in files] == [f"{name}_0000010.flac", f"{name}_0000011.flac"]
        params = json.loads((tmp_path / name / f"{name}_0000010.json").read_text())
        assert render_params(name, params).shape == (44100, 2)
