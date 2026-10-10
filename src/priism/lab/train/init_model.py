"""Start a fine-tune from a pretrained RoFormer with a new set of stems.

The body of the network is kept as is. Each new stem gets the output head
(mask estimator) of a pretrained stem rather than random weights, e.g. the
new "acid" head starts as a copy of "other": the model already sends most
303 lines there, so training only has to teach it what to take away.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))

_HEAD = re.compile(r"^(mask_estimators)\.(\d+)\.(.*)$")


def load_config(path: str | Path) -> dict:
    return yaml.load(Path(path).read_text(), Loader=_Loader)


def remap_heads(state_dict: dict, src_instruments: list[str], mapping: dict[str, str]) -> dict:
    """New state dict whose mask estimator ``i`` is copied from the source stem ``mapping[new[i]]``."""
    for new, src in mapping.items():
        if src not in src_instruments:
            raise KeyError(f"{new!r} maps to {src!r}, which the pretrained model does not have ({src_instruments})")
    out = {}
    heads: dict[int, dict[str, object]] = {}
    for key, value in state_dict.items():
        m = _HEAD.match(key)
        if m:
            heads.setdefault(int(m.group(2)), {})[m.group(3)] = value
        else:
            out[key] = value
    if sorted(heads) != list(range(len(src_instruments))):
        raise ValueError(f"checkpoint has {len(heads)} heads but the config lists {len(src_instruments)} instruments")
    for i, new in enumerate(mapping):
        src = src_instruments.index(mapping[new])
        for rest, value in heads[src].items():
            out[f"mask_estimators.{i}.{rest}"] = value.clone() if hasattr(value, "clone") else value
    return out


def make_config(base: dict, instruments: list[str], overrides: dict | None = None) -> dict:
    """Copy of ``base`` set up for ``instruments``, with nested ``overrides`` applied."""
    cfg = yaml.load(yaml.dump(base, Dumper=yaml.Dumper), Loader=_Loader)
    cfg["model"]["num_stems"] = len(instruments)
    cfg["training"]["instruments"] = list(instruments)
    cfg["training"]["target_instrument"] = None
    # Per-stem augmentations of the base config may name stems that no longer exist.
    if "augmentations" in cfg:
        for name in list(cfg["augmentations"]):
            if isinstance(cfg["augmentations"][name], dict) and name != "all" and name not in instruments:
                del cfg["augmentations"][name]

    def merge(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    merge(cfg, overrides or {})
    return cfg


def prepare(base_config: str | Path, base_ckpt: str | Path, mapping: dict[str, str], out_dir: str | Path,
            overrides: dict | None = None) -> tuple[Path, Path]:
    """Write ``config.yaml`` and ``init.ckpt`` for MSST ``train.py`` into ``out_dir``."""
    import torch

    base = load_config(base_config)
    sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = remap_heads(sd, list(base["training"]["instruments"]), mapping)
    cfg = make_config(base, list(mapping), overrides)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path, ckpt_path = out_dir / "config.yaml", out_dir / "init.ckpt"
    cfg_path.write_text(yaml.dump(cfg, Dumper=yaml.Dumper, sort_keys=False))
    torch.save(new_sd, ckpt_path)
    return cfg_path, ckpt_path
