"""Synthetic songs for identity-based separation (generator v2).

A song is a short arranged piece (techno, dub, acid, dnb, ...) where every
source (each drum element, each synth part) is rendered on its own track.
Tracks are not named classes for the separator: they are just distinct
sources, and the mix is exactly their sum. Classes and roles are kept in the
metadata for statistics and for an optional track classifier.

See docs/spec-generateur-v2.md in the project folder for the design.
"""

from .song import render_song, write_song  # noqa: F401
