"""Closed-form sparse weight editing for permanent concept erasure.

This module implements the core innovation of UniErase: converting
inference-time activation steering into permanent weight modifications.

Mathematical derivation:
    FFN structure:  h(x) = Activate(x W_1 + b_1)   [post-activation, sparse]
                    y(x) = h(x) W_2 + b_2           [output]

    SSV-Guard steers activations at inference time:
        h'[C_top] = h[C_top] - alpha * gamma * v_top
        where gamma = cos(h[C_top], v_top)

    We prove this is equivalent to a left-side projection on W_2:
        W_2_new[C_top, :] = (I_k - lambda * P_rho * r_hat * r_hat^T * P_rho
                             / (r_hat^T * P_rho * r_hat)) @ W_2[C_top, :]

    For concept-aligned input:   r_hat^T @ W_2_new[C_top, :] ≈ 0  (suppressed)
    For unrelated input h ⊥ r:   h^T @ W_2_new[C_top, :] = h^T @ W_2[C_top, :]  (preserved)

    This update is:
    - Permanent: baked into weights, survives model distribution
    - Precise: only affects concept-relevant channel rows of W_2
    - Zero-overhead: no inference-time computation added
    - Non-reversible: cannot be undone by removing inference hooks

References:
    - LRR-V (ICLR'26) Eq. 8: closed-form weight update via refusal vectors
    - UCE (WACV'24): closed-form cross-attention editing
    - ROME (NeurIPS'22): rank-one model editing in LLMs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .cpca import CPCAResult


@dataclass
class WeightEditResult:
    """Result of a weight edit operation on one FFN layer.

    Attributes
    ----------
    layer_name : str
        Name of the edited FFN layer (e.g., "up_blocks.1.attentions.0.ff").
    projection_module_name : str
        Name of the W_2 projection module that was modified.
    channel_indices : List[int]
        Indices of the modified channel rows in W_2.
    concept_direction : torch.Tensor
        The refined concept direction used for the edit.
    weight_delta_norm : float
        Frobenius norm of the weight change (for monitoring).
    original_weight_norm : float
        Frobenius norm of the original W_2 sub-matrix.
    """
    layer_name: str
    projection_module_name: str
    channel_indices: List[int]
    concept_direction: torch.Tensor
    weight_delta_norm: float
    original_weight_norm: float


def compute_weight_update_matrix(
    concept_direction: torch.Tensor,
    cpca_result: Optional[CPCAResult] = None,
    suppression_strength: float = 1.0,
) -> torch.Tensor:
    """Compute the left-multiplication matrix for weight editing.

    Parameters
    ----------
    concept_direction : torch.Tensor
        Refined concept direction in channel space, shape (k,).
    cpca_result : CPCAResult, optional
        If provided, uses the cPCA projection matrix for more precise
        concept isolation.  If None, uses direct projection.
    suppression_strength : float
        Lambda parameter controlling suppression magnitude.
        1.0 = full suppression; < 1.0 = partial suppression.

    Returns
    -------
    Tensor of shape (k, k): the matrix M such that W_2_new = M @ W_2_old.
    """
    r = concept_direction.float()
    k = r.shape[0]

    if cpca_result is not None and cpca_result.projection_matrix is not None:
        P = cpca_result.projection_matrix.float()  # (k, k)
        # Projected direction
        Pr = P @ r
        denominator = r @ Pr
        if denominator.abs() < 1e-10:
            return torch.eye(k, dtype=r.dtype, device=r.device)
        # M = I - lambda * (P r r^T P) / (r^T P r)
        update = suppression_strength * torch.outer(Pr, Pr) / denominator
    else:
        # Direct projection without cPCA subspace
        norm_sq = r @ r
        if norm_sq.abs() < 1e-10:
            return torch.eye(k, dtype=r.dtype, device=r.device)
        # M = I - lambda * (r r^T) / (r^T r)
        update = suppression_strength * torch.outer(r, r) / norm_sq

    M = torch.eye(k, dtype=r.dtype, device=r.device) - update
    return M


def edit_ffn_weights(
    backbone: nn.Module,
    projection_module_name: str,
    channel_indices: List[int],
    concept_direction: torch.Tensor,
    cpca_result: Optional[CPCAResult] = None,
    suppression_strength: float = 1.0,
) -> WeightEditResult:
    """Apply closed-form weight edit to one FFN layer's W_2 projection.

    This modifies W_2[channel_indices, :] in-place, permanently removing
    the concept direction from the weight matrix's row subspace.

    Parameters
    ----------
    backbone : nn.Module
        The denoising backbone (UNet or Transformer).
    projection_module_name : str
        Full name of the W_2 linear module (e.g.,
        ``"up_blocks.1.attentions.0.ff.net.2"``).
    channel_indices : list of int
        Row indices of W_2 corresponding to concept-relevant channels.
    concept_direction : torch.Tensor
        Refined concept direction in the channel subspace, shape (k,).
    cpca_result : CPCAResult, optional
        cPCA result for subspace-constrained editing.
    suppression_strength : float
        Controls how strongly the concept is suppressed (0=none, 1=full).

    Returns
    -------
    WeightEditResult with edit metadata.
    """
    # Look up the projection module
    module_lookup = dict(backbone.named_modules())
    if projection_module_name not in module_lookup:
        raise KeyError(
            f"Module {projection_module_name!r} not found in backbone. "
            f"Available modules with 'ff': "
            f"{[n for n in module_lookup if 'ff' in n][:10]}"
        )

    proj_module = module_lookup[projection_module_name]
    if not isinstance(proj_module, nn.Linear):
        raise TypeError(
            f"Expected nn.Linear, got {type(proj_module).__name__} "
            f"for {projection_module_name}"
        )

    # W_2 shape: (d_out, d_ff) for nn.Linear — note the transposition!
    # nn.Linear stores weight as (out_features, in_features)
    # so W_2[i, :] = weight[:, i] in the stored tensor
    # We operate on the input dimension (in_features = d_ff)
    weight = proj_module.weight.data  # (d_out, d_ff)
    idx = torch.tensor(channel_indices, dtype=torch.long, device=weight.device)

    # Extract the sub-matrix: rows in channel space = columns of weight
    # W_2_sub shape: (d_out, k) — these are the columns we want to modify
    W_2_sub = weight[:, idx].clone().float()  # (d_out, k)
    original_norm = W_2_sub.norm().item()

    # Compute update matrix M: shape (k, k)
    M = compute_weight_update_matrix(
        concept_direction=concept_direction,
        cpca_result=cpca_result,
        suppression_strength=suppression_strength,
    ).to(weight.device)

    # Apply: W_2_new_sub = W_2_sub @ M^T
    # Because W_2_sub is (d_out, k) and M operates on the k-dim (input side):
    # For each output neuron j:  new_w_j = M @ old_w_j  (where w_j ∈ R^k)
    # In matrix form:  W_2_new_sub^T = M @ W_2_sub^T
    #                  W_2_new_sub = (M @ W_2_sub^T)^T = W_2_sub @ M^T
    W_2_new_sub = W_2_sub @ M.T

    # Compute delta norm
    delta_norm = (W_2_new_sub - W_2_sub).norm().item()

    # Write back in the original dtype
    weight[:, idx] = W_2_new_sub.to(weight.dtype)

    return WeightEditResult(
        layer_name=projection_module_name.rsplit(".net.", 1)[0]
        if ".net." in projection_module_name
        else projection_module_name,
        projection_module_name=projection_module_name,
        channel_indices=channel_indices,
        concept_direction=concept_direction.cpu(),
        weight_delta_norm=delta_norm,
        original_weight_norm=original_norm,
    )


def edit_multiple_layers(
    backbone: nn.Module,
    layer_edits: List[Dict],
    suppression_strength: float = 1.0,
) -> List[WeightEditResult]:
    """Apply weight edits to multiple FFN layers.

    Parameters
    ----------
    backbone : nn.Module
        The denoising backbone.
    layer_edits : list of dict
        Each dict must contain:
        - ``"projection_module_name"``: str
        - ``"channel_indices"``: list of int
        - ``"concept_direction"``: Tensor of shape (k,)
        - ``"cpca_result"``: Optional[CPCAResult]
    suppression_strength : float
        Global suppression strength applied to all layers.

    Returns
    -------
    List of WeightEditResult, one per layer.
    """
    results = []
    for edit in layer_edits:
        result = edit_ffn_weights(
            backbone=backbone,
            projection_module_name=edit["projection_module_name"],
            channel_indices=edit["channel_indices"],
            concept_direction=edit["concept_direction"],
            cpca_result=edit.get("cpca_result"),
            suppression_strength=suppression_strength,
        )
        results.append(result)
    return results


def verify_edit(
    backbone: nn.Module,
    edit_result: WeightEditResult,
) -> Dict[str, float]:
    """Verify that the weight edit successfully suppresses the concept direction.

    Checks that the concept direction's projection through the modified
    W_2 sub-matrix is near zero.

    Returns
    -------
    Dict with verification metrics.
    """
    module_lookup = dict(backbone.named_modules())
    proj_module = module_lookup[edit_result.projection_module_name]
    weight = proj_module.weight.data.float()

    idx = torch.tensor(
        edit_result.channel_indices,
        dtype=torch.long,
        device=weight.device,
    )
    W_2_sub = weight[:, idx]  # (d_out, k)

    r = edit_result.concept_direction.float().to(weight.device)

    # The concept direction's contribution through W_2
    # output_concept = W_2_sub @ r → should be near zero
    output_concept = W_2_sub @ r
    suppression_ratio = output_concept.norm().item() / (
        W_2_sub.norm().item() * r.norm().item() + 1e-10
    )

    return {
        "concept_output_norm": output_concept.norm().item(),
        "suppression_ratio": suppression_ratio,
        "weight_change_ratio": edit_result.weight_delta_norm / (
            edit_result.original_weight_norm + 1e-10
        ),
    }
