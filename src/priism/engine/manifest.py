"""``manifest.json`` (output of ``separate``) and ``project.json`` (input of ``export-ni``).

The JSON Schemas in ``schema/`` describe the shape; this module adds the rules a schema
cannot express (unique ids, references between fields, safe relative paths) and the atomic
write. ``docs/contract.md`` is the human-readable specification.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from . import SCHEMA_VERSION, SUPPORTED_SCHEMA_MAJORS

ID_RE = re.compile(r"^[a-z0-9_]+$")
COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
RESIDUAL = "residual"  # reserved id: the residual is referenced by it but is not in ``stems``
ROOT = "mix"  # parent of the stems split straight from the source


class ContractError(ValueError):
    """A manifest or project that breaks the contract."""


@dataclass
class Source:
    path: str
    sha256: str
    sample_rate: int
    channels: int
    frames: int
    decoder: str | None = None


@dataclass
class ModelInfo:
    backend: str
    name: str
    version: str
    sha256: str | None = None


@dataclass
class Stem:
    id: str
    label: str
    kind: str
    file: str
    models: list[str] = field(default_factory=list)
    parent: str | None = None
    confidence: float | None = None


@dataclass
class Profile:
    name: str
    version: str


@dataclass
class Manifest:
    source: Source
    stems: list[Stem]
    residual_file: str
    profile: Profile
    models: dict[str, ModelInfo]
    suggested_slots: list[list[str]]
    priism_version: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        def clean(d: dict) -> dict:
            return {k: v for k, v in d.items() if v is not None}

        return {
            "schema_version": self.schema_version,
            "priism_version": self.priism_version,
            "profile": asdict(self.profile),
            "source": clean(asdict(self.source)),
            "models": {k: clean(asdict(m)) for k, m in self.models.items()},
            "stems": [clean(asdict(s)) for s in self.stems],
            "residual": {"file": self.residual_file},
            "suggested_slots": self.suggested_slots,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        check_schema_version(d.get("schema_version"))
        try:
            m = cls(
                schema_version=d["schema_version"],
                priism_version=d["priism_version"],
                profile=Profile(**d["profile"]) if "profile" in d else Profile("unknown", "0"),
                source=Source(**_known(Source, d["source"])),
                models={k: ModelInfo(**_known(ModelInfo, v)) for k, v in d.get("models", {}).items()},
                stems=[Stem(**_known(Stem, s)) for s in d["stems"]],
                residual_file=d["residual"]["file"],
                suggested_slots=d.get("suggested_slots", []),
            )
        except (KeyError, TypeError) as e:
            raise ContractError(f"malformed manifest: {e!r}") from e
        m.validate()
        return m

    def validate(self) -> None:
        ids = [s.id for s in self.stems]
        for i in ids:
            if not ID_RE.match(i):
                raise ContractError(f"stem id {i!r} does not match [a-z0-9_]+")
            if i in (RESIDUAL, ROOT):
                raise ContractError(f"stem id {i!r} is reserved")
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ContractError(f"duplicate stem ids: {sorted(dup)}")
        for s in self.stems:
            check_relative(s.file)
            unknown = [m for m in s.models if m not in self.models]
            if unknown:
                raise ContractError(f"stem {s.id!r} refers to models missing from the models table: {unknown}")
        check_relative(self.residual_file)
        known = set(ids) | {RESIDUAL}
        for slot in self.suggested_slots:
            missing = [i for i in slot if i not in known]
            if missing:
                raise ContractError(f"suggested_slots refers to unknown stems {missing}")


@dataclass
class Slot:
    name: str
    color: str
    stems: list[str]
    gain: list[float] | None = None

    def gains(self) -> list[float]:
        return list(self.gain) if self.gain is not None else [1.0] * len(self.stems)


@dataclass
class Project:
    manifest: str
    slots: list[Slot]
    discarded: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Project:
        check_schema_version(d.get("schema_version"))
        try:
            p = cls(schema_version=d["schema_version"], manifest=d["manifest"],
                    slots=[Slot(**_known(Slot, s)) for s in d["slots"]], discarded=list(d.get("discarded", [])))
        except (KeyError, TypeError) as e:
            raise ContractError(f"malformed project: {e!r}") from e
        return p

    def validate(self, manifest: Manifest) -> None:
        """Rules that need the manifest: 4 slots, known stems, each stem used at most once."""
        check_relative(self.manifest)
        if len(self.slots) != 4:
            raise ContractError(f"an NI stem file has exactly 4 slots, got {len(self.slots)}")
        names = [s.name for s in self.slots]
        if len(set(names)) != len(names):
            raise ContractError(f"slot names must be unique: {names}")
        known = {s.id for s in manifest.stems} | {RESIDUAL}
        used: list[str] = []
        for slot in self.slots:
            if not slot.name:
                raise ContractError("slot names cannot be empty")
            if not COLOR_RE.match(slot.color):
                raise ContractError(f"slot {slot.name!r}: color {slot.color!r} is not #RRGGBB")
            if slot.gain is not None and len(slot.gain) != len(slot.stems):
                raise ContractError(f"slot {slot.name!r}: {len(slot.gain)} gains for {len(slot.stems)} stems")
            missing = [i for i in slot.stems if i not in known]
            if missing:
                raise ContractError(f"slot {slot.name!r} refers to unknown stems {missing}")
            used += slot.stems
        twice = {i for i in used if used.count(i) > 1}
        if twice:
            raise ContractError(f"stems used in more than one slot: {sorted(twice)}")
        both = set(used) & set(self.discarded)
        if both:
            raise ContractError(f"stems both used and discarded: {sorted(both)}")


def _known(cls, d: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys: a newer minor version may add fields this priism ignores."""
    names = cls.__dataclass_fields__.keys()
    return {k: v for k, v in d.items() if k in names}


def check_schema_version(v: Any) -> None:
    if not isinstance(v, str) or not re.match(r"^\d+\.\d+$", v):
        raise ContractError(f"missing or malformed schema_version: {v!r}")
    if int(v.split(".")[0]) not in SUPPORTED_SCHEMA_MAJORS:
        raise ContractError(f"schema_version {v} is not supported (majors: {SUPPORTED_SCHEMA_MAJORS})")


def check_relative(p: str) -> None:
    """Paths in the contract are POSIX, relative to the file holding them, and stay below it."""
    pp = PurePosixPath(p)
    if not p or "\\" in p or pp.is_absolute() or ".." in pp.parts or re.match(r"^[A-Za-z]:", p):
        raise ContractError(f"path {p!r} must be relative, with '/' separators and no '..'")


def write_json_atomic(path: str | Path, data: dict[str, Any]) -> None:
    """Write next to ``path`` then rename over it, so a reader never sees a partial file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ContractError(f"{path}: not valid JSON ({e})") from e
