import pytest
import yaml

from priism.train.init_model import _Loader, make_config, remap_heads

torch = pytest.importorskip("torch")


def _sd(n_heads):
    sd = {"layers.0.w": torch.zeros(2)}
    for i in range(n_heads):
        sd[f"mask_estimators.{i}.to_freqs.0.weight"] = torch.full((2,), float(i))
    return sd


def test_remap_heads_copies_the_chosen_heads():
    src = ["bass", "drums", "other", "vocals"]
    out = remap_heads(_sd(4), src, {"drums": "drums", "bass": "bass", "acid": "other", "rest": "other"})
    vals = [out[f"mask_estimators.{i}.to_freqs.0.weight"][0].item() for i in range(4)]
    assert vals == [1.0, 0.0, 2.0, 2.0]
    assert "mask_estimators.4.to_freqs.0.weight" not in out
    assert "layers.0.w" in out
    # Heads initialised from the same source must not share memory.
    out["mask_estimators.2.to_freqs.0.weight"][0] = 9
    assert out["mask_estimators.3.to_freqs.0.weight"][0] == 2


def test_remap_heads_rejects_unknown_source():
    with pytest.raises(KeyError):
        remap_heads(_sd(2), ["bass", "drums"], {"acid": "synth"})


def test_make_config_sets_stems_and_keeps_tuples():
    base = {
        "model": {"num_stems": 6, "freqs_per_bands": (2, 2, 4)},
        "training": {"instruments": ["bass", "drums"], "lr": 1e-4, "batch_size": 2},
        "augmentations": {"all": {"channel_shuffle": 0.5}, "vocals": {"pitch_shift": 0.1}, "bass": {"pitch_shift": 0.1}},
    }
    cfg = make_config(base, ["drums", "bass", "acid", "rest"], {"training": {"lr": 1e-5}})
    assert cfg["model"]["num_stems"] == 4
    assert cfg["training"]["batch_size"] == 2 and cfg["training"]["lr"] == 1e-5
    assert "vocals" not in cfg["augmentations"] and "bass" in cfg["augmentations"]
    dumped = yaml.dump(cfg, Dumper=yaml.Dumper)
    assert "python/tuple" in dumped
    assert yaml.load(dumped, Loader=_Loader)["model"]["freqs_per_bands"] == (2, 2, 4)
    assert base["model"]["num_stems"] == 6  # base untouched
