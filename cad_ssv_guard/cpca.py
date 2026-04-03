"""Contrastive PCA (cPCA) for refining concept directions.

The raw refusal/steering vector computed from safe/unsafe activation
differences may be entangled with unrelated semantics (e.g., the
"nudity" direction may partially overlap with "woman" or "skin").

Contrastive PCA (Abid et al., 2018) addresses this by:
  1. Maximizing variance specific to the target concept (unsafe vs. safe).
  2. Minimizing variance associated with a neutral set (unrelated prompts).

This yields a low-rank subspace that better isolates the unwanted concept
without interfering with unrelated generation capabilities.

Reference:
  - LRR-V (ICLR'26): applies cPCA to refine refusal vectors in video models.
  - Abid et al., 2018: "Exploring Patterns Enriched in a Dataset with
    Contrastive PCA", Nature Communications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class CPCAResult:
    """Result of contrastive PCA refinement.

    Attributes
    ----------
    refined_direction : torch.Tensor
        Refined concept direction in channel space, shape (k,).
    subspace_basis : torch.Tensor
        Low-rank subspace basis U_rho, shape (k, rho).
    projection_matrix : torch.Tensor
        Projection matrix P_rho = U_rho @ U_rho^T, shape (k, k).
    singular_values : torch.Tensor
        Singular values of the contrastive covariance, shape (rho,).
    raw_direction : torch.Tensor
        Original (unrefined) concept direction, shape (k,).
    """
    refined_direction: torch.Tensor
    subspace_basis: torch.Tensor
    projection_matrix: torch.Tensor
    singular_values: torch.Tensor
    raw_direction: torch.Tensor


def compute_cpca(
    concept_diffs: torch.Tensor,
    neutral_activations: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    rank: int = 5,
    min_explained_variance: float = 0.0,
) -> CPCAResult:
    """Compute contrastive PCA to refine the concept direction.

    Parameters
    ----------
    concept_diffs : torch.Tensor
        Activation differences between unsafe and safe prompts for
        the selected channels, shape (N, k) where N is the number of
        prompt pairs and k is the number of selected channels.
    neutral_activations : torch.Tensor, optional
        Activations from neutral/unrelated prompts for the same
        channels, shape (M, k).  If None, standard PCA is used.
    alpha : float
        Contrastive strength: controls how much neutral variance is
        subtracted.  Higher alpha → stronger disentanglement from
        neutral semantics.  Typical range: 0.5 to 5.0.
    rank : int
        Number of top singular vectors to retain (cPCA subspace rank).
    min_explained_variance : float
        If > 0, dynamically choose rank to explain at least this
        fraction of variance.  Overrides ``rank`` when set.

    Returns
    -------
    CPCAResult with the refined direction and subspace information.
    """
    concept_diffs = concept_diffs.float()
    N, k = concept_diffs.shape

    # Center the concept differences
    mu = concept_diffs.mean(dim=0)
    centered = concept_diffs - mu

    # Concept covariance
    C_r = (centered.T @ centered) / max(N - 1, 1)

    # Contrastive covariance
    if neutral_activations is not None:
        neutral_activations = neutral_activations.float()
        M = neutral_activations.shape[0]
        nu = neutral_activations.mean(dim=0)
        neutral_centered = neutral_activations - nu
        C_e = (neutral_centered.T @ neutral_centered) / max(M - 1, 1)
        C = C_r - alpha * C_e
    else:
        C = C_r

    # Eigendecomposition (symmetric matrix → use eigh for stability)
    eigenvalues, eigenvectors = torch.linalg.eigh(C)

    # eigh returns eigenvalues in ascending order; reverse for descending
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    # Only keep positive eigenvalues (negative ones correspond to
    # directions dominated by neutral variance)
    positive_mask = eigenvalues > 0
    eigenvalues = eigenvalues[positive_mask]
    eigenvectors = eigenvectors[:, positive_mask]

    if eigenvalues.numel() == 0:
        # Fallback: if all eigenvalues are negative, use raw direction
        raw_dir = mu / (mu.norm() + 1e-8)
        return CPCAResult(
            refined_direction=raw_dir,
            subspace_basis=raw_dir.unsqueeze(1),
            projection_matrix=raw_dir.unsqueeze(1) @ raw_dir.unsqueeze(0),
            singular_values=torch.ones(1),
            raw_direction=mu,
        )

    # Select rank
    if min_explained_variance > 0:
        total_var = eigenvalues.sum()
        cumvar = eigenvalues.cumsum(0) / total_var
        rank = int((cumvar < min_explained_variance).sum().item()) + 1

    rank = min(rank, eigenvalues.numel())

    # Low-rank subspace basis
    U_rho = eigenvectors[:, :rank]  # (k, rho)
    sigma = eigenvalues[:rank].sqrt()

    # Projection matrix
    P_rho = U_rho @ U_rho.T  # (k, k)

    # Refine the concept direction by projecting onto the subspace
    projected = P_rho @ mu
    norm = projected.norm()
    refined = projected / (norm + 1e-8)

    return CPCAResult(
        refined_direction=refined,
        subspace_basis=U_rho,
        projection_matrix=P_rho,
        singular_values=sigma,
        raw_direction=mu,
    )


def collect_concept_diffs(
    positive_activations: List[torch.Tensor],
    negative_activations: List[torch.Tensor],
    channel_indices: List[int],
) -> torch.Tensor:
    """Collect per-pair activation differences for selected channels.

    Parameters
    ----------
    positive_activations : list of Tensor
        Per-prompt mean activations under unsafe conditioning, each shape (d_ff,).
    negative_activations : list of Tensor
        Per-prompt mean activations under safe conditioning, each shape (d_ff,).
    channel_indices : list of int
        Selected channel indices.

    Returns
    -------
    Tensor of shape (N, k) where N = min(len(pos), len(neg)).
    """
    N = min(len(positive_activations), len(negative_activations))
    idx = torch.tensor(channel_indices, dtype=torch.long)
    diffs = []
    for i in range(N):
        pos = positive_activations[i].float()
        neg = negative_activations[i].float()
        diff = pos[idx] - neg[idx]
        diffs.append(diff)
    return torch.stack(diffs, dim=0)


def collect_neutral_activations(
    activations: List[torch.Tensor],
    channel_indices: List[int],
) -> torch.Tensor:
    """Collect neutral prompt activations for selected channels.

    Parameters
    ----------
    activations : list of Tensor
        Per-prompt mean activations under neutral prompts, each shape (d_ff,).
    channel_indices : list of int
        Selected channel indices.

    Returns
    -------
    Tensor of shape (M, k).
    """
    idx = torch.tensor(channel_indices, dtype=torch.long)
    return torch.stack([act.float()[idx] for act in activations], dim=0)
