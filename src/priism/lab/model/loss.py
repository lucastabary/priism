"""Permutation-invariant training loss for the attractor separator.

For each example, the K outputs are matched to the N present sources with the
Hungarian algorithm on a negative-SNR cost. Matched outputs learn their source
and "exists"; unmatched outputs learn silence and "absent". A reconstruction
term asks the sum of all outputs to give back the mix.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


def neg_snr(est: torch.Tensor, ref: torch.Tensor, soft_db: float = 30.0, eps: float = 1e-8) -> torch.Tensor:
    """-SNR in dB over the last two dims, with a soft cap (``soft_db``) so easy sources stop dominating.

    est (..., C, S), ref (..., C, S) broadcastable.
    """
    tau = 10 ** (-soft_db / 10)
    num = ref.pow(2).sum((-2, -1))
    den = (ref - est).pow(2).sum((-2, -1)) + tau * num
    return -10 * torch.log10((num + eps) / (den + eps))


def pit_loss(sources: torch.Tensor, exist_logits: torch.Tensor, targets: torch.Tensor, n_targets: torch.Tensor,
             mix: torch.Tensor, silence_weight: float = 0.1, exist_weight: float = 1.0, recon_weight: float = 0.5,
             match: list | None = None) -> tuple[torch.Tensor, dict]:
    """sources (B, K, C, S), exist_logits (B, K), targets (B, Nmax, C, S) zero-padded, n_targets (B,), mix (B, C, S).

    ``match``, a list, receives per example the (output rows, target cols) of the assignment.
    """
    B, K = exist_logits.shape
    total_sep, total_sil, exist_target = 0.0, 0.0, torch.zeros_like(exist_logits)
    matched = 0
    for b in range(B):
        n = int(n_targets[b])
        if n == 0:
            if match is not None:
                match.append(((), ()))
            total_sil = total_sil + sources[b].abs().mean()
            continue
        tgt = targets[b, :n]
        cost = neg_snr(sources[b][:, None], tgt[None])  # (K, n)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        if match is not None:
            match.append((rows, cols))
        rows_t = torch.as_tensor(rows, device=sources.device)
        total_sep = total_sep + cost[rows, cols].sum()
        matched += n
        exist_target[b, rows_t] = 1.0
        free = torch.ones(K, dtype=torch.bool, device=sources.device)
        free[rows_t] = False
        if free.any():
            # Unmatched outputs should be silent: penalise their energy relative to the mix.
            e = sources[b][free].pow(2).mean((-2, -1))
            total_sil = total_sil + (e / (mix[b].pow(2).mean() + 1e-8)).mean()
    sep = total_sep / max(matched, 1)
    sil = total_sil / B
    ex = F.binary_cross_entropy_with_logits(exist_logits, exist_target)
    recon = neg_snr(sources.sum(1), mix).mean()
    loss = sep + silence_weight * sil + exist_weight * ex + recon_weight * recon
    stats = {"loss": _f(loss), "sep_snr": -_f(sep), "silence": _f(sil), "exist_bce": _f(ex), "recon_snr": -_f(recon)}
    return loss, stats


def _f(x) -> float:
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def count_accuracy(exist_logits: torch.Tensor, n_targets: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(exist_logits) > threshold).sum(-1)
    return float((pred == n_targets.to(pred.device)).float().mean())
