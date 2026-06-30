"""Closed-form sparse weight editing for permanent concept erasure.

Mathematical derivation
-----------------------
FFN structure:
    h(x) = Activate(x W_1 + b_1)    [post-activation hidden, shape (d_ff,)]
    y(x) = h(x) W_2 + b_2           [output projection]

**Rank-1 edit** (original SSV-Guard / UniErase):
    We want to erase one concept direction r ∈ R^k (k = |selected channels|).
    The update matrix is:
        M₁ = I - λ · (r rᵀ) / ‖r‖²
    Applied to W_2:
        W_new[:, idx] = W_old[:, idx] @ M₁
    Effect on concept-aligned activation:
        W_new @ r = (1 - λ) · W_old @ r
        λ = 1 → W_new @ r = 0   (complete nullification)
        λ = 2 → W_new @ r = -W_old @ r  (Householder reflection / inversion)

**Rank-d edit** (PCA-based, this module):
    Given d orthonormal concept directions R = [r₁, …, r_d] (rows, from SVD),
    apply successive rank-1 projections:
        M = M_d @ … @ M_1    where  M_i = I - λ · r_i rᵢᵀ / ‖r_i‖²
    Because the r_i are orthonormal, the combined effect is:
        M = I - λ · Rᵀ R    (projection onto the orthogonal complement of span{R})
    This completely removes the k-dimensional concept subspace from W_2
    when λ = 1, regardless of how many directions d ≤ k there are.

Verification metric (primary direction):
    suppression_factor = ‖W_new @ r₁‖ / ‖W_old @ r₁‖  = |1 - λ|
    direction_cosine   = cosine(W_old @ r₁, W_new @ r₁) = sign(1 - λ)

References:
    - LRR-V (ICLR'26) Eq. 8: closed-form weight update via refusal vectors
    - UCE (WACV'24): closed-form cross-attention editing
    - ROME (NeurIPS'22): rank-one model editing in LLMs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .cpca import CPCAResult


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WeightEditResult:
    """Result of a rank-d weight edit operation on one FFN layer.

    Attributes
    ----------
    layer_name : str
        Human-readable layer name.
    projection_module_name : str
        Full path of the edited nn.Linear (W_2).
    channel_indices : List[int]
        Edited column indices of W_2 (= selected hidden channels).
    concept_direction : torch.Tensor
        Primary concept direction (first PCA component), shape (k,).
        Used for verification metrics.
    num_directions : int
        Total number of PCA directions used in the rank-d edit.
    weight_delta_norm : float
        ‖W_new − W_old‖_F.
    original_weight_norm : float
        ‖W_old[:, idx]‖_F.
    original_concept_response_norm : float
        ‖W_old[:, idx] @ r₁‖  (primary direction, before edit).
    edited_concept_response_norm : float
        ‖W_new[:, idx] @ r₁‖  (primary direction, after edit).
        Theoretically |1 − λ| × original_concept_response_norm.
    direction_cosine : float
        cos(W_old @ r₁, W_new @ r₁):
          +1  same direction  (λ < 1)
           0  orthogonal      (λ = 1, fully suppressed)
          -1  inverted        (λ > 1)
    suppression_strength : float
        λ value used.
    use_aggressive_mode : bool
        Whether λ > 1 was allowed.
    """
    layer_name: str
    projection_module_name: str
    channel_indices: List[int]
    concept_direction: torch.Tensor      # primary direction (k,)
    num_directions: int
    weight_delta_norm: float
    original_weight_norm: float
    original_concept_response_norm: float
    edited_concept_response_norm: float
    direction_cosine: float
    suppression_strength: float
    use_aggressive_mode: bool


# ─────────────────────────────────────────────────────────────────────────────
# Update matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_weight_update_matrix(
    concept_directions: torch.Tensor,
    suppression_strength: float = 1.0,
    use_aggressive_mode: bool = False,
) -> torch.Tensor:
    """Compute the rank-d left-multiplication matrix for weight editing.

    Parameters
    ----------
    concept_directions : Tensor of shape (d, k) or (k,)
        PCA concept directions (rows).  For a single direction pass a 1-D
        tensor; it will be promoted to (1, k) internally.
        Rows are assumed to be unit-norm and mutually orthogonal (as produced
        by SVD / ``_pca_concept_directions`` in builder.py).
    suppression_strength : float
        λ — controls suppression magnitude per direction:
        λ = 1 → nullify (W_new @ r = 0)
        λ = 2 → Householder reflection (W_new @ r = −W_old @ r)
        λ < 1 → partial suppression
    use_aggressive_mode : bool
        When False, λ is clipped to max 1.0 (clean nullification).
        When True, λ > 1 is allowed (concept direction inverted).

    Returns
    -------
    M : Tensor (k, k) such that W_new[:, idx] = W_old[:, idx] @ M.
    """
    R = concept_directions.float()
    if R.ndim == 1:
        R = R.unsqueeze(0)              # (1, k)

    k = R.shape[1]
    lam = suppression_strength if use_aggressive_mode else min(suppression_strength, 1.0)

    M = torch.eye(k, dtype=R.dtype, device=R.device)
    for r in R:
        norm_sq = (r @ r).item()
        if norm_sq < 1e-10:
            continue
        # M_i = I - λ · r rᵀ / ‖r‖²
        M = M @ (torch.eye(k, dtype=R.dtype, device=R.device)
                 - lam * torch.outer(r, r) / norm_sq)

    return M                            # (k, k)


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer edit
# ─────────────────────────────────────────────────────────────────────────────

def edit_ffn_weights(
    backbone: nn.Module,
    projection_module_name: str,
    channel_indices: List[int],
    concept_directions: torch.Tensor,   # (d, k) or (k,)
    suppression_strength: float = 1.0,
    use_aggressive_mode: bool = False,
    # kept for API compatibility; no longer used internally
    cpca_result: Optional[CPCAResult] = None,
) -> WeightEditResult:
    """Apply rank-d closed-form weight edit to one FFN layer's W_2.

    Modifies ``W_2[:, channel_indices]`` in-place.

    Parameters
    ----------
    concept_directions : Tensor (d, k) or (k,)
        PCA concept directions for the selected channel subspace.
        Pass a 2-D tensor from builder.py's PCA output.
    """
    module_lookup = dict(backbone.named_modules())
    if projection_module_name not in module_lookup:
        raise KeyError(
            f"Module {projection_module_name!r} not found in backbone. "
            f"Modules containing 'ff': "
            f"{[n for n in module_lookup if 'ff' in n][:10]}"
        )
    proj_module = module_lookup[projection_module_name]
    if not isinstance(proj_module, nn.Linear):
        raise TypeError(
            f"Expected nn.Linear, got {type(proj_module).__name__} "
            f"for {projection_module_name}"
        )

    # nn.Linear: weight shape = (out_features, in_features)
    # "channels" are along the in_features (d_ff) dimension
    weight = proj_module.weight.data                      # (d_out, d_ff)
    idx = torch.tensor(channel_indices, dtype=torch.long, device=weight.device)

    # Normalise and promote concept_directions to (d, k)
    R = concept_directions.float().to(weight.device)
    if R.ndim == 1:
        R = R.unsqueeze(0)                                # (1, k)
    norms = R.norm(dim=1, keepdim=True).clamp_min(1e-8)
    R = R / norms                                         # unit-norm rows

    primary_r = R[0]                                      # primary direction (k,)
    num_directions = R.shape[0]

    # Extract W_2 sub-matrix: columns = selected channels
    W_2_sub = weight[:, idx].clone().float()              # (d_out, k)
    original_norm = W_2_sub.norm().item()

    # ── Measure concept response BEFORE edit (primary direction) ──────────────
    orig_response = W_2_sub @ primary_r                   # (d_out,)
    original_concept_response_norm = orig_response.norm().item()

    # ── Compute and apply rank-d update matrix ────────────────────────────────
    M = compute_weight_update_matrix(
        concept_directions=R,
        suppression_strength=suppression_strength,
        use_aggressive_mode=use_aggressive_mode,
    )                                                      # (k, k)

    W_2_new_sub = W_2_sub @ M.T                           # M is symmetric → M.T = M

    # ── Measure concept response AFTER edit (primary direction) ───────────────
    new_response = W_2_new_sub @ primary_r                # (d_out,)
    edited_concept_response_norm = new_response.norm().item()

    # Direction cosine (primary direction only)
    if original_concept_response_norm > 1e-10 and edited_concept_response_norm > 1e-10:
        direction_cosine = float(
            (orig_response @ new_response)
            / (original_concept_response_norm * edited_concept_response_norm)
        )
    else:
        direction_cosine = 0.0

    # ── Write back ────────────────────────────────────────────────────────────
    delta_norm = (W_2_new_sub - W_2_sub).norm().item()

    # Direct scatter assignment: weight[:, idx] = ... is an in-place scatter
    # operation that modifies the original tensor.
    # NOTE: weight[:, idx] with a tensor index returns a COPY (advanced indexing),
    # so weight_col_sub = weight[:, idx]; weight_col_sub.copy_(...) would NOT
    # write back to the original weight — never use that pattern here.
    weight[:, idx] = W_2_new_sub.to(weight.dtype)

    # Verify write
    verify_read = weight[:, idx].float()
    write_error = (verify_read - W_2_new_sub.float()).norm().item()
    if write_error > 1e-3:
        print(f"  [WARNING] Weight write-back error: {write_error:.6f}")

    layer_name = (
        projection_module_name.rsplit(".net.", 1)[0]
        if ".net." in projection_module_name
        else projection_module_name
    )
    return WeightEditResult(
        layer_name=layer_name,
        projection_module_name=projection_module_name,
        channel_indices=channel_indices,
        concept_direction=primary_r.cpu(),
        num_directions=num_directions,
        weight_delta_norm=delta_norm,
        original_weight_norm=original_norm,
        original_concept_response_norm=original_concept_response_norm,
        edited_concept_response_norm=edited_concept_response_norm,
        direction_cosine=direction_cosine,
        suppression_strength=suppression_strength,
        use_aggressive_mode=use_aggressive_mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def edit_multiple_layers(
    backbone: nn.Module,
    layer_edits: List[Dict],
    suppression_strength: float = 1.0,
    use_aggressive_mode: bool = False,
) -> List[WeightEditResult]:
    """Apply rank-d weight edits to multiple FFN layers.

    Each dict in ``layer_edits`` must contain:
    - ``"projection_module_name"``: str
    - ``"channel_indices"``: list of int
    - ``"concept_directions"``: Tensor (d, k) or (k,)
    - ``"cpca_result"`` (optional, ignored): kept for API compat
    """
    results = []
    for edit in layer_edits:
        result = edit_ffn_weights(
            backbone=backbone,
            projection_module_name=edit["projection_module_name"],
            channel_indices=edit["channel_indices"],
            concept_directions=edit["concept_directions"],
            suppression_strength=suppression_strength,
            use_aggressive_mode=use_aggressive_mode,
            cpca_result=edit.get("cpca_result"),
        )
        results.append(result)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_edit(
    backbone: nn.Module,
    edit_result: WeightEditResult,
) -> Dict[str, float]:
    """Compute suppression verification metrics (primary direction).

    Key metrics
    -----------
    suppression_factor : float
        ‖W_new @ r₁‖ / ‖W_old @ r₁‖ = |1 − λ|:
          λ = 1.0 → 0.00  (complete suppression)
          λ = 2.0 → 1.00  (reflection, same magnitude, direction inverted)
    direction_cosine : float
        cos(W_old @ r₁, W_new @ r₁):
          +1 same direction (λ < 1)   0 suppressed (λ = 1)   −1 inverted (λ > 1)
    weight_change_ratio : float
        ‖W_new − W_old‖_F / ‖W_old‖_F
    """
    module_lookup = dict(backbone.named_modules())
    proj_module = module_lookup[edit_result.projection_module_name]
    weight = proj_module.weight.data.float()
    idx = torch.tensor(
        edit_result.channel_indices, dtype=torch.long, device=weight.device
    )
    W_2_sub = weight[:, idx]                              # edited weight
    r = edit_result.concept_direction.float().to(weight.device)

    current_response_norm = (W_2_sub @ r).norm().item()
    original_response_norm = edit_result.original_concept_response_norm
    suppression_factor = current_response_norm / (original_response_norm + 1e-10)
    weight_change_ratio = edit_result.weight_delta_norm / (
        edit_result.original_weight_norm + 1e-10
    )
    return {
        "suppression_factor":         suppression_factor,
        "direction_cosine":           edit_result.direction_cosine,
        "concept_response_before":    original_response_norm,
        "concept_response_after":     current_response_norm,
        "weight_change_ratio":        weight_change_ratio,
        "suppression_strength":       edit_result.suppression_strength,
        "use_aggressive_mode":        float(edit_result.use_aggressive_mode),
        "num_directions":             float(edit_result.num_directions),
    }


def print_verification(
    edit_result: WeightEditResult,
    verification: Dict[str, float],
) -> None:
    """Pretty-print suppression verification for one layer."""
    sf   = verification["suppression_factor"]
    dc   = verification["direction_cosine"]
    rb   = verification["concept_response_before"]
    ra   = verification["concept_response_after"]
    wcr  = verification["weight_change_ratio"]
    lam  = verification["suppression_strength"]
    agg  = bool(verification["use_aggressive_mode"])
    nd   = int(verification["num_directions"])

    # Direction interpretation
    if ra < 1e-6:
        direction_tag = "已归零 (完全抑制)"
    elif dc > 0.5:
        direction_tag = f"同向  (部分抑制)   cos={dc:+.3f}"
    elif dc < -0.5:
        direction_tag = f"已反转 (激进模式)  cos={dc:+.3f}"
    else:
        direction_tag = f"近正交             cos={dc:+.3f}"

    pct_suppressed = max(0.0, 1.0 - sf) * 100
    bar_len = 30
    filled = int(round(pct_suppressed / 100 * bar_len))
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"\n  层: {edit_result.layer_name}")
    print(f"  ├─ 修改通道数    : {len(edit_result.channel_indices)}")
    print(f"  ├─ PCA 方向数    : {nd}  (rank-{nd} 编辑)")
    print(f"  ├─ 编辑参数      : λ={lam:.1f}  aggressive={'是' if agg else '否'}")
    print(f"  ├─ 概念响应幅度  : {rb:.4f} → {ra:.4f}  (主方向 r₁)")
    print(f"  │    抑制因子    : {sf:.4f}  (理论值 |1−λ|={abs(1-lam):.2f})")
    print(f"  │    [{bar}] {pct_suppressed:.1f}% 幅度抑制")
    print(f"  ├─ 响应方向      : {direction_tag}")
    print(f"  └─ 权重扰动比    : {wcr*100:.2f}%")
