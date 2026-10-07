import json

import numpy as np
import soundfile as sf

from priism.acid.generate import generate, render
from priism.acid.params import AcidParams, Step, sample_params
from priism.acid.synth import control_signals, render_dry


def test_render_is_deterministic_and_well_formed():
    p = sample_params(7, duration_s=2.0)
    a, b = render(p), render(sample_params(7, duration_s=2.0))
    assert a.shape == (88200, 2) and a.dtype == np.float32
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    assert np.isclose(np.abs(a).max(), 10 ** (p.peak_dbfs / 20), rtol=1e-4)


def test_seeds_give_different_lines():
    assert not np.allclose(render(sample_params(1, 1.0)), render(sample_params(2, 1.0)))


def test_params_roundtrip_through_json():
    p = sample_params(3, duration_s=1.0)
    q = AcidParams.from_dict(json.loads(json.dumps(p.to_dict())))
    np.testing.assert_array_equal(render(p), render(q))


def _two_step(slide: bool) -> AcidParams:
    p = sample_params(0, duration_s=1.0)
    p.bpm = 120.0
    p.steps = [Step(note=36, gate=True, accent=False, slide=slide), Step(note=48, gate=True, accent=False, slide=False)]
    return p


def test_slide_glides_pitch_and_skips_retrigger():
    n = 44100
    step = int(60 / 120 / 4 * 44100)
    slid = control_signals(_two_step(True), n)
    plain = control_signals(_two_step(False), n)
    # With a slide, pitch is between the two notes just after the step boundary.
    assert 36 < slid["note"][step + 50] < 48
    assert plain["note"][step + 50] == 48
    # The gate holds through the slide but closes between plain notes.
    assert slid["gate"][step - 1] == 1.0 and plain["gate"][step - 1] == 0.0
    # The filter envelope is retriggered on the second note only without slide.
    assert plain["filt_env"][step] > slid["filt_env"][step]


def test_accent_opens_filter_and_raises_level():
    p = _two_step(False)
    p.steps = [Step(note=36, gate=True, accent=a, slide=False) for a in (True, False)]
    c = control_signals(p, 44100)
    step = int(60 / 120 / 4 * 44100)
    assert c["acc_env"][10] > 0 and c["acc_env"][step + 10] == 0
    assert c["amp"][step - 200] > c["amp"][2 * step - 200]


def test_resonance_boosts_cutoff_region():
    p = sample_params(5, duration_s=1.0)
    p.steps = [Step(note=36, gate=True, accent=False, slide=True)]  # one held note
    p.synth.env_mod_oct = 0.0
    p.synth.cutoff_hz = 800.0
    spectra = []
    for res in (0.0, 0.9):
        p.synth.resonance = res
        y = render_dry(p)[4410:]
        spectra.append(np.abs(np.fft.rfft(y * np.hanning(len(y)))))
    freqs = np.fft.rfftfreq(len(y), 1 / 44100)
    band = (freqs > 600) & (freqs < 1000)
    low = (freqs > 50) & (freqs < 200)
    ratio = [s[band].sum() / s[low].sum() for s in spectra]
    assert ratio[1] > 2 * ratio[0]


def test_generate_writes_audio_and_metadata(tmp_path):
    files = generate(tmp_path, count=2, start_seed=10, duration_s=0.5)
    assert len(files) == 2
    audio, sr = sf.read(files[0])
    assert sr == 44100 and audio.shape == (22050, 2)
    meta = json.loads((tmp_path / "acid_0000010.json").read_text())
    assert meta["seed"] == 10
