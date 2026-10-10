"""Profiles: a separation as a tree of subtractive steps.

Each step takes a signal (the source, ``mix``, or a node an earlier step produced), runs one
model on it and names some sums of the model's stems. What the named outputs leave of the
step's input is its *rest*: either a named node (a stem, or a signal a later step splits
further) or, when unnamed, part of the final residual. Since every split is "input = outputs
+ rest", the stems plus the residual sum back to the source by construction, whatever models
are combined; no stem is counted twice.

    [[steps]]                      # vocals first, with the model best at vocals...
    input = "mix"
    model = "demucs:htdemucs_ft"
    outputs = [{ id = "vocals", kind = "vocals", label = "Vocals", from = ["vocals"] }]
    rest = { id = "instrumental" } # ...then a later step splits what is left

An output or rest that no later step consumes is a final stem and needs ``kind`` and
``label``; one that a later step consumes is an intermediate node and is not written.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import ID_RE, RESIDUAL, ROOT

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


class ProfileError(ValueError):
    pass


@dataclass
class Output:
    id: str
    sources: list[str] = field(default_factory=list)  # model stems summed into it; empty for a rest
    kind: str | None = None
    label: str | None = None


@dataclass
class Step:
    input: str
    model: str
    outputs: list[Output]
    rest: Output | None = None


@dataclass
class Profile:
    name: str
    version: str
    description: str
    steps: list[Step]
    suggested_slots: list[list[str]] | None = None

    @property
    def models(self) -> list[str]:
        return list(dict.fromkeys(s.model for s in self.steps))

    def consumed(self) -> set[str]:
        return {s.input for s in self.steps}

    def final_outputs(self) -> list[tuple[Step, Output]]:
        """Stems the profile writes, in step order."""
        used = self.consumed()
        out = []
        for step in self.steps:
            for o in step.outputs + ([step.rest] if step.rest else []):
                if o.id not in used:
                    out.append((step, o))
        return out


def _output(d: dict, where: str, is_rest: bool) -> Output:
    if not isinstance(d, dict) or "id" not in d:
        raise ProfileError(f"{where}: needs an id")
    unknown = set(d) - {"id", "from", "kind", "label"}
    if unknown:
        raise ProfileError(f"{where}: unknown keys {sorted(unknown)}")
    sources = d.get("from", [])
    if is_rest and sources:
        raise ProfileError(f"{where}: a rest takes what is left, it has no 'from'")
    if not is_rest and not sources:
        raise ProfileError(f"{where}: 'from' must list the model stems summed into it")
    return Output(id=d["id"], sources=list(sources), kind=d.get("kind"), label=d.get("label"))


def parse_profile(data: dict, name: str) -> Profile:
    steps = []
    for i, s in enumerate(data.get("steps", [])):
        where = f"profile {name}, step {i + 1}"
        try:
            outputs = [_output(o, f"{where}, output {j + 1}", False) for j, o in enumerate(s.get("outputs", []))]
            rest = _output(s["rest"], f"{where}, rest", True) if "rest" in s else None
            steps.append(Step(input=s["input"], model=s["model"], outputs=outputs, rest=rest))
        except KeyError as e:
            raise ProfileError(f"{where}: missing {e}") from e
    p = Profile(name=data.get("name", name), version=str(data.get("version", "1")),
                description=data.get("description", ""), steps=steps, suggested_slots=data.get("suggested_slots"))
    validate_profile(p)
    return p


def validate_profile(p: Profile) -> None:
    if not p.steps:
        raise ProfileError(f"profile {p.name}: no steps")
    seen = {ROOT}
    consumed: set[str] = set()
    for i, step in enumerate(p.steps):
        where = f"profile {p.name}, step {i + 1}"
        if step.input not in seen:
            raise ProfileError(f"{where}: input {step.input!r} is not 'mix' or an earlier output")
        if step.input in consumed:
            raise ProfileError(f"{where}: {step.input!r} is already split by an earlier step")
        consumed.add(step.input)
        model_stems = [s for o in step.outputs for s in o.sources]
        if len(model_stems) != len(set(model_stems)):
            raise ProfileError(f"{where}: a model stem goes to two outputs")
        for o in step.outputs + ([step.rest] if step.rest else []):
            if not ID_RE.match(o.id) or o.id in (ROOT, RESIDUAL):
                raise ProfileError(f"{where}: invalid or reserved id {o.id!r}")
            if o.id in seen:
                raise ProfileError(f"{where}: id {o.id!r} used twice")
            seen.add(o.id)
    for step, o in p.final_outputs():
        if not o.kind or not o.label:
            raise ProfileError(f"profile {p.name}: stem {o.id!r} needs a kind and a label")
    finals = {o.id for _, o in p.final_outputs()} | {RESIDUAL}
    for slot in p.suggested_slots or []:
        missing = [i for i in slot if i not in finals]
        if missing:
            raise ProfileError(f"profile {p.name}: suggested_slots refers to {missing}, not stems of the profile")


def load_profile(name_or_path: str | Path) -> Profile:
    path = Path(name_or_path)
    if path.suffix != ".toml" or not path.exists():
        path = PROFILES_DIR / f"{name_or_path}.toml"
    if not path.exists():
        raise ProfileError(f"unknown profile {name_or_path!r}; bundled: {', '.join(bundled_profiles())}")
    return parse_profile(tomllib.loads(path.read_text(encoding="utf-8")), path.stem)


def bundled_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.toml"))
