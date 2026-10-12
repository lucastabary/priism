"""The heavy, data-parallel parts of the clusterings in torch, for a GPU: the Lloyd iterations of the micro
k-means and the final token -> centroid assignment of ``whole_song``. Drop-ins for the numpy code in
``track`` (``kmeans=``, ``assign=`` of the sweeps), same algorithm step by step.

What stays on the CPU, and why:

- the k-means++ seeding (``track.kmeanspp``): it draws from the numpy generator, so the GPU k-means starts
  from exactly the centroids of the CPU one, and the generator is left in the same state for what follows;
- the average linkage on the (at most 256) micro centroids and its cuts: a few milliseconds in scipy;
- the online registries: a chunk-by-chunk loop of small operations, launch-bound on a GPU.

Equivalence: float32 like the CPU path, TF32 off, but another library's summation order (and atomic,
unordered sums in ``index_add_``), so dot products differ from numpy's in the last bits (~1e-7). An argmax
flips only where two centroids are tied to that precision; a flip during the k-means can move it to a
nearby local optimum, so labels are not guaranteed bit-identical: the benchmark
(``scripts/bench_registry_eval.py``) reports how often songs differ, and the metrics' differences.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from .track import kmeanspp


def torch_device(device: str) -> torch.device:
    """"cuda", "cuda:1"..., or "torch:cpu" (the torch code on the CPU, to test it without a GPU)."""
    return torch.device(device.removeprefix("torch:"))


def _norm(c: torch.Tensor) -> torch.Tensor:
    return c / torch.clamp(torch.linalg.vector_norm(c, dim=-1, keepdim=True), min=1e-8)


def spherical_kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 15,
                     device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """``track.spherical_kmeans`` with its iterations on ``device``."""
    c0 = kmeanspp(x, k, rng)
    dev = torch_device(device)
    xt = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev)
    c = torch.from_numpy(c0).to(dev)
    for _ in range(iters):
        a = (xt @ c.T).argmax(1)  # first maximum on ties, as numpy
        new = torch.zeros_like(c).index_add_(0, a, xt)
        keep = torch.bincount(a, minlength=len(c)) > 0
        c = _norm(new[keep])
    a = (xt @ c.T).argmax(1)
    return c.cpu().numpy(), a.cpu().numpy()


def assign(xa: np.ndarray, cents: list[np.ndarray], device: str = "cuda") -> list[np.ndarray]:
    """[``(xa @ cent.T).argmax(1)`` for each cent] on ``device``: a sweep assigns the same tokens at every
    threshold, so they are copied to the device once per song."""
    dev = torch_device(device)
    xt = torch.from_numpy(np.ascontiguousarray(xa, np.float32)).to(dev)
    return [(xt @ torch.from_numpy(np.ascontiguousarray(c, np.float32)).to(dev).T).argmax(1).cpu().numpy()
            for c in cents]


@lru_cache(maxsize=None)
def gpu_ops(device: str) -> dict:
    """{"kmeans": ..., "assign": ...} for ``track.*_sweep`` on ``device`` (one set per process)."""
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 would round the dot products to ~1e-3
    torch.backends.cudnn.allow_tf32 = False

    def kmeans(x, k, rng, iters: int = 15):
        return spherical_kmeans(x, k, rng, iters, device)

    return {"kmeans": kmeans, "assign": lambda xa, cents: assign(xa, cents, device)}
