"""Class-agnostic separator with attractors (see docs/architecture-identite.md in the project folder).

mix -> STFT -> band-split RoFormer-style core -> attractor decoder (up to K
identity vectors, each with an existence probability) -> one shared head
conditioned on each vector by FiLM -> one complex mask per source.

Trained with permutation-invariant matching (Hungarian), so outputs carry no
class: the model only has to find each source once. ``rest = mix - sum`` keeps
the reconstruction exact at inference.

This is the CPU-sized skeleton: the core is our own compact implementation;
loading pretrained BS-RoFormer weights into it is a later step.
"""

from .separator import AttractorSeparator, SeparatorConfig  # noqa: F401
