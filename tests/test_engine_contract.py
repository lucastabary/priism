import copy
import json

import jsonschema
import pytest

from engine_helpers import validate
from priism.engine.manifest import ContractError, Manifest, Project, write_json_atomic
from priism.engine.profile import ProfileError, bundled_profiles, load_profile, parse_profile

MANIFEST = {
    "schema_version": "1.1",
    "priism_version": "0.2.0",
    "profile": {"name": "fake", "version": "1"},
    "source": {"path": "x.wav", "sha256": "0" * 64, "sample_rate": 44100, "channels": 2, "frames": 100,
               "decoder": "mixxx:SoundSourceMp3"},
    "models": {"fake:bands": {"backend": "fake", "name": "bands", "version": "1"}},
    "stems": [
        {"id": "drums", "label": "Drums", "kind": "drums", "file": "stems/drums.flac", "models": ["fake:bands"],
         "parent": "mix"},
        {"id": "bass", "label": "Bass", "kind": "bass", "file": "stems/bass.flac", "models": ["fake:bands"]},
    ],
    "residual": {"file": "stems/residual.flac"},
    "suggested_slots": [["drums"], ["bass"], ["residual"]],
}

PROJECT = {
    "schema_version": "1.1",
    "manifest": "manifest.json",
    "slots": [
        {"name": "Drums", "color": "#FF5A3C", "stems": ["drums"], "gain": [1.0]},
        {"name": "Bass", "color": "#3C8CFF", "stems": ["bass"]},
        {"name": "Music", "color": "#FFD23C", "stems": ["residual"]},
        {"name": "Vocals", "color": "#3CFF8C", "stems": []},
    ],
    "discarded": [],
}


def test_valid_manifest_round_trips_and_matches_schema():
    m = Manifest.from_dict(MANIFEST)
    assert m.to_dict() == MANIFEST
    validate(MANIFEST, "manifest.schema.json")


def test_newer_minor_and_unknown_fields_are_accepted():
    d = copy.deepcopy(MANIFEST)
    d["schema_version"] = "1.9"
    d["future"] = 1
    d["stems"][0]["future"] = "x"
    assert Manifest.from_dict(d).stems[0].id == "drums"


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(schema_version="2.0"), "not supported"),
    (lambda d: d["stems"][1].update(id="drums"), "duplicate"),
    (lambda d: d["stems"][1].update(id="residual"), "reserved"),
    (lambda d: d["stems"][1].update(id="Bass"), "does not match"),
    (lambda d: d["stems"][1].update(file="../bass.flac"), "relative"),
    (lambda d: d["stems"][1].update(file="C:/bass.flac"), "relative"),
    (lambda d: d["stems"][1].update(models=["nope"]), "models table"),
    (lambda d: d.update(suggested_slots=[["vocals"]]), "unknown stems"),
])
def test_manifest_rules(change, message):
    d = copy.deepcopy(MANIFEST)
    change(d)
    with pytest.raises(ContractError, match=message):
        Manifest.from_dict(d)


def test_schema_rejects_what_the_rules_reject():
    for change in (lambda d: d["stems"][1].update(file="../x.flac"), lambda d: d["stems"][1].update(id="mix"),
                   lambda d: d.update(schema_version="2.0")):
        d = copy.deepcopy(MANIFEST)
        change(d)
        with pytest.raises(jsonschema.ValidationError):
            validate(d, "manifest.schema.json")


def test_project_rules():
    manifest = Manifest.from_dict(MANIFEST)
    validate(PROJECT, "project.schema.json")
    Project.from_dict(PROJECT).validate(manifest)
    bad = [
        (lambda d: d["slots"].pop(), "exactly 4"),
        (lambda d: d["slots"][2]["stems"].append("drums"), "more than one slot"),
        (lambda d: d["slots"][2]["stems"].append("vocals"), "unknown stems"),
        (lambda d: d["slots"][0].update(gain=[1.0, 0.5]), "gains"),
        (lambda d: d["slots"][0].update(color="red"), "color"),
        (lambda d: d["slots"][1].update(name="Drums"), "unique"),
        (lambda d: d.update(discarded=["bass"]), "both used and discarded"),
    ]
    for change, message in bad:
        d = copy.deepcopy(PROJECT)
        change(d)
        with pytest.raises(ContractError, match=message):
            Project.from_dict(d).validate(manifest)


def test_atomic_write_leaves_only_the_file(tmp_path):
    write_json_atomic(tmp_path / "manifest.json", {"a": 1})
    write_json_atomic(tmp_path / "manifest.json", {"a": 2})
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"a": 2}


def test_bundled_profiles_load():
    assert {"fake", "fast", "standard"} <= set(bundled_profiles())
    for name in bundled_profiles():
        assert load_profile(name).steps


def _step(input="mix", outputs=None, rest=None):
    s = {"input": input, "model": "fake:bands",
         "outputs": outputs or [{"id": "low", "kind": "bass", "label": "Low", "from": ["low"]}]}
    if rest:
        s["rest"] = rest
    return s


@pytest.mark.parametrize("steps, message", [
    ([], "no steps"),
    ([_step(input="upper")], "not 'mix' or an earlier output"),
    ([_step(rest={"id": "upper"}), _step(input="upper", outputs=[{"id": "a", "kind": "x", "label": "A", "from": ["mid"]}]),
      _step(input="upper", outputs=[{"id": "b", "kind": "x", "label": "B", "from": ["mid"]}])], "already split"),
    ([_step(rest={"id": "upper"})], "needs a kind and a label"),
    ([_step(rest={"id": "r", "from": ["mid"]})], "has no 'from'"),
    ([_step(outputs=[{"id": "residual", "kind": "x", "label": "X", "from": ["low"]}])], "reserved"),
    ([_step(outputs=[{"id": "a", "kind": "x", "label": "A", "from": ["low"]},
                     {"id": "b", "kind": "x", "label": "B", "from": ["low"]}])], "two outputs"),
])
def test_invalid_profiles(steps, message):
    with pytest.raises(ProfileError, match=message):
        parse_profile({"name": "t", "steps": steps}, "t")


def test_intermediate_nodes_are_not_stems():
    p = parse_profile({"name": "t", "steps": [
        _step(rest={"id": "upper"}),
        _step(input="upper", outputs=[{"id": "mid", "kind": "other", "label": "Mid", "from": ["mid"]}]),
    ]}, "t")
    assert [o.id for _, o in p.final_outputs()] == ["low", "mid"]


def test_shared_fixture_matches_the_contract():
    """fixtures/contract is what MyMixxx tests against; regenerate it with scripts/make_fixture.py."""
    from pathlib import Path

    import numpy as np
    import soundfile as sf

    root = Path(__file__).resolve().parents[1] / "fixtures" / "contract"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    project = json.loads((root / "project.json").read_text(encoding="utf-8"))
    validate(manifest, "manifest.schema.json")
    validate(project, "project.schema.json")
    m = Manifest.from_dict(manifest)
    Project.from_dict(project).validate(m)
    src, _ = sf.read(root / "source.wav", dtype="float64")
    total = sum(sf.read(root / f, dtype="float64")[0] for f in [s.file for s in m.stems] + [m.residual_file])
    assert np.array_equal(total, src)
