"""Stem separation engine: what MyMixxx drives through the ``priism`` command line.

The contract with MyMixxx (commands, ``manifest.json``, ``project.json``, progress events)
is specified in ``docs/contract.md`` and checked by the JSON Schemas in ``schema/``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

SCHEMA_VERSION = "1.3"
# Major versions this priism reads and writes; any minor of a supported major is accepted.
SUPPORTED_SCHEMA_MAJORS = (1,)

try:
    __version__ = version("priism")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"
