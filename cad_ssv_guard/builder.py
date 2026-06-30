"""Guard artifact builder.

Orchestrates the full CAD + SSV + PCA pipeline:

1. **CAD localisation** — gradient attribution identifies which FFN layers and
   channels respond most strongly to the target concept.
2. **Layer prior weighting** — architecture-aware Gaussian prior favors
   mid-depth "semantic encoding" layers.
3. **SSV scoring** — per-sample activation collection (multiple seeds) provides
   diverse concept-diff samples for PCA and for channel scoring.
4. **PCA concept directions** — SVD of the concept-diff matrix extracts the
   top-d directions that explain the most concept variance; rank is chosen
   automatically via the elbow (cumulative-variance) criterion.
5. **Artifact assembly** — selected channels and rank-d concept directions are
   packed into a :class:`~cad_ssv_guard.artifact.GuardArtifact`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from .artifact import GuardArtifact, GuardLayer
from .backend import ModelBackend
from .cad import CADLayerScores, compute_nudity_cad_scores
from .ssv import (
    collect_activations_per_sample_multilayer,
    compute_ssv_scores,
)
from .layer_prior import (
    compute_layer_prior_weights,
    apply_prior_to_scores,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, values.numel())
    if k <= 0:
        raise ValueError("k must be positive.")
    return torch.topk(values, k=k).indices


def _allocate_layer_budgets(
    ranked_layers,
    total_budget: int,
    min_per_layer: int,
) -> List[int]:
    """Distribute ``total_budget`` channels across layers proportionally."""
    num_layers = len(ranked_layers)
    if num_layers == 0:
        return []

    total_budget = max(total_budget, num_layers)
    min_per_layer = max(1, min(min_per_layer, total_budget // num_layers))
    budgets = [min_per_layer] * num_layers
    remaining = total_budget - sum(budgets)
    if remaining <= 0:
        return budgets

    scores = torch.tensor(
        [layer.layer_score for layer in ranked_layers], dtype=torch.float32
    )
    if float(scores.sum().item()) <= 0.0:
        scores = torch.ones_like(scores) / num_layers
    else:
        scores = scores / scores.sum()

    fractional = scores * remaining
    floor_alloc = torch.floor(fractional).to(torch.int64)
    budgets = [b + int(e) for b, e in zip(budgets, floor_alloc.tolist())]

    leftover = remaining - int(floor_alloc.sum().item())
    if leftover > 0:
        order = torch.argsort(fractional - floor_alloc.float(), descending=True).tolist()
        for idx in order[:leftover]:
            budgets[idx] += 1
    return budgets


def _pca_concept_directions(
    diff_matrix: torch.Tensor,
    max_rank: int = 8,
    variance_threshold: float = 0.80,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract concept directions via PCA with elbow-method rank selection.

    Parameters
    ----------
    diff_matrix : Tensor of shape (N, k)
        Per-sample concept diffs: ``pos_activation[i] - neg_activation[i]``
        for the selected channel subset.
    max_rank : int
        Hard upper bound on the number of retained directions.
    variance_threshold : float
        Cumulative explained-variance ratio at which to stop adding directions
        (the "elbow").  E.g. 0.80 means "keep the fewest directions that
        together explain ≥ 80 % of the concept variance".

    Returns
    -------
    directions : Tensor (d, k)
        Unit-norm concept directions (rows), d ≤ max_rank.
        These are the right singular vectors of the centred diff matrix.
    evr : Tensor (d,)
        Explained variance ratio for each retained direction.
    """
    N, k = diff_matrix.shape
    diff_matrix = diff_matrix.float()

    # ── Degenerate: single sample → return normalised mean ───────────────────
    if N <= 1:
        mean_dir = diff_matrix.mean(dim=0)
        norm = mean_dir.norm()
        if norm < 1e-10:
            return torch.zeros(1, k), torch.ones(1)
        return (mean_dir / norm).unsqueeze(0), torch.ones(1)

    # ── Centre the diff matrix ───────────────────────────────────────────────
    centered = diff_matrix - diff_matrix.mean(dim=0)

    # ── SVD (economy) ────────────────────────────────────────────────────────
    try:
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Vh: (min(N,k), k) — rows are right singular vectors (orthonormal)
    except Exception:
        # Fallback to normalised mean direction
        mean_dir = diff_matrix.mean(dim=0)
        norm = mean_dir.norm()
        if norm < 1e-10:
            return torch.zeros(1, k), torch.ones(1)
        return (mean_dir / norm).unsqueeze(0), torch.ones(1)

    # ── Explained variance ratios ─────────────────────────────────────────────
    eigenvalues = S ** 2                          # proportional to variance
    total_var = eigenvalues.sum()

    if total_var < 1e-10:
        return Vh[:1], torch.ones(1)

    cumvar = eigenvalues.cumsum(0) / total_var

    # ── Elbow criterion: first d where cumvar ≥ variance_threshold ───────────
    rank = int((cumvar < variance_threshold).sum().item()) + 1
    rank = max(1, min(rank, max_rank, Vh.shape[0]))

    directions = Vh[:rank]                        # (d, k)
    evr = (eigenvalues[:rank] / total_var).cpu()  # (d,)

    return directions.cpu(), evr


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_nudity_guard(
    pipe,
    backend: ModelBackend,
    positive_prompts: List[str],
    negative_prompts: List[str],
    target: str = "nudity",
    concept_prompt: str = "naked",
    base_prompt: str = "",
    cad_steps: int = 50,
    cad_num_samples: int = 4,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_layers: int = 3,
    steering_topk: int = 32,
    total_steering_channels: Optional[int] = None,
    min_channels_per_layer: int = 8,
    alpha: float = 1.0,
    seed: int = 0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    use_layer_prior: bool = True,
    prior_strength: float = 1.0,
    allowed_blocks: Optional[Sequence[str]] = None,
    cad_devices=None,
    # ── Per-sample SSV / PCA parameters ──────────────────────────────────────
    ssv_num_seeds: int = 4,
    pca_max_rank: int = 8,
    pca_variance_threshold: float = 0.80,
) -> GuardArtifact:
    """Build a concept-erasing guard artifact (CAD + SSV + rank-d PCA edit).

    Parameters
    ----------
    ssv_num_seeds : int
        Number of random seeds used when collecting per-sample activations.
        More seeds → larger concept-diff matrix → better-conditioned PCA.
        Each positive and negative prompt is run ``ssv_num_seeds`` times with
        different noise initialisations.
    pca_max_rank : int
        Hard upper bound on the number of PCA concept directions retained
        per layer.  The actual rank is chosen by the elbow criterion.
    pca_variance_threshold : float
        Cumulative explained-variance threshold for the elbow criterion.
        E.g. 0.80 keeps the fewest directions that explain ≥ 80 % of the
        concept variance in the diff matrix.
    """
    num_inference_steps = num_inference_steps or backend.default_inference_steps
    guidance_scale = guidance_scale or backend.default_guidance_scale
    height = height or backend.default_height
    width = width or backend.default_width

    # ── Step 1 : CAD — rank layers and channels ───────────────────────────────
    layer_scores = compute_nudity_cad_scores(
        pipe=pipe,
        backend=backend,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        concept_prompt=concept_prompt,
        base_prompt=base_prompt,
        num_steps=cad_steps,
        num_samples=cad_num_samples,
        seed=seed,
        height=height,
        width=width,
        devices=cad_devices,
    )

    backbone = backend.get_backbone(pipe)
    module_lookup = dict(backbone.named_modules())

    # ── Layer prior weighting ─────────────────────────────────────────────────
    if use_layer_prior:
        layer_names = [ls.spec.ff_name for ls in layer_scores.values()]
        prior_weights_map = compute_layer_prior_weights(
            layer_names=layer_names,
            model_type=backend.model_type,
        )
        raw_scores = {
            ls.spec.ff_name: ls.layer_score
            for ls in layer_scores.values()
        }
        adjusted_scores = apply_prior_to_scores(
            layer_scores=raw_scores,
            prior_weights=prior_weights_map,
            prior_strength=prior_strength,
        )
        adjusted_layer_scores = []
        for ls in layer_scores.values():
            adjusted_layer_scores.append(CADLayerScores(
                spec=ls.spec,
                channel_scores=ls.channel_scores,
                raw_channel_scores=ls.raw_channel_scores,
                layer_score=adjusted_scores.get(ls.spec.ff_name, ls.layer_score),
            ))
        all_ranked_layers = sorted(
            adjusted_layer_scores, key=lambda x: x.layer_score, reverse=True
        )
    else:
        prior_weights_map = {}
        all_ranked_layers = sorted(
            layer_scores.values(), key=lambda x: x.layer_score, reverse=True
        )

    # ── Build full layer ranking table (all candidates, before any filter) ────
    layer_rank_lookup: dict = {}       # ff_name → 1-indexed rank
    all_layer_ranking: list = []
    for rank_pos, ls in enumerate(all_ranked_layers, start=1):
        ff = ls.spec.ff_name
        raw  = layer_scores[ff].layer_score
        adj  = ls.layer_score
        pw   = float(prior_weights_map.get(ff, 1.0))
        layer_rank_lookup[ff] = rank_pos
        all_layer_ranking.append({
            "rank":               rank_pos,
            "ff_name":            ff,
            "cad_score_raw":      float(raw),
            "cad_score_adjusted": float(adj),
            "prior_weight":       pw,
            "selected":           False,    # updated after truncation
            "channel_budget":     None,
        })

    # ── Print full ranking table ──────────────────────────────────────────────
    print(f"\n[Layer Ranking]  {len(all_ranked_layers)} candidate layers"
          f"  (selecting top {num_layers}):")
    hdr = f"  {'Rank':>4}  {'CAD_raw':>8}  {'CAD_adj':>8}  {'Prior':>6}  Layer"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for entry in all_layer_ranking:
        marker = "▶" if entry["rank"] <= num_layers else " "
        print(f"  {marker}{entry['rank']:>3}  "
              f"{entry['cad_score_raw']:>8.4f}  "
              f"{entry['cad_score_adjusted']:>8.4f}  "
              f"{entry['prior_weight']:>6.3f}  "
              f"{entry['ff_name']}")

    # ── Optional block constraint ─────────────────────────────────────────────
    ranked_layers = all_ranked_layers
    if allowed_blocks is not None:
        ranked_layers = [
            lc for lc in ranked_layers
            if any(lc.spec.ff_name.startswith(p) for p in allowed_blocks)
        ]
        print(f"\n[Layer Constraint] {len(ranked_layers)} layers remain after "
              f"allowed_blocks filter: {[lc.spec.ff_name for lc in ranked_layers]}")

    ranked_layers = ranked_layers[: min(num_layers, len(ranked_layers))]

    if total_steering_channels is None:
        total_steering_channels = steering_topk * len(ranked_layers)

    budgets = _allocate_layer_budgets(
        ranked_layers=ranked_layers,
        total_budget=total_steering_channels,
        min_per_layer=min_channels_per_layer,
    )

    # Mark selected layers and their budgets in the ranking table
    budget_lookup = {
        layer.spec.ff_name: bgt
        for layer, bgt in zip(ranked_layers, budgets)
    }
    for entry in all_layer_ranking:
        if entry["ff_name"] in budget_lookup:
            entry["selected"] = True
            entry["channel_budget"] = budget_lookup[entry["ff_name"]]

    # ── Seed list for per-sample SSV collection ───────────────────────────────
    # Seeds are spread far apart to avoid correlation between runs.
    ssv_seeds = [seed + i * 997 for i in range(ssv_num_seeds)]

    is_video = hasattr(backend, "cad_num_frames")
    ssv_num_frames = getattr(backend, "cad_num_frames", None) if is_video else None
    # For FLUX (no CFG, batch=1): cond_only has no effect.
    # For SD1/SD3 (CFG, batch=2): cond_only discards the unconditional pass.
    cond_only = backend.supports_cfg

    # ── Step 2 : SSV — collect ALL layers simultaneously ─────────────────────
    # Register hooks for every selected layer at once, then run (N_prompts ×
    # N_seeds) forward passes exactly twice (positive set + negative set).
    # This reduces total forward passes from N_layers × 2 × N × S  to  2 × N × S.
    hidden_modules = [
        module_lookup[layer.spec.hidden_module_name] for layer in ranked_layers
    ]

    print(f"\n[SSV] Collecting positive activations "
          f"({len(ranked_layers)} layers in parallel, {ssv_num_seeds} seeds) …")
    pos_all = collect_activations_per_sample_multilayer(
        pipe=pipe,
        modules=hidden_modules,
        prompts=positive_prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seeds=ssv_seeds,
        num_frames=ssv_num_frames,
        cond_only=cond_only,
    )  # List[(N_pos × N_seeds, d_ff)]

    print(f"\n[SSV] Collecting negative activations "
          f"({len(ranked_layers)} layers in parallel, {ssv_num_seeds} seeds) …")
    neg_all = collect_activations_per_sample_multilayer(
        pipe=pipe,
        modules=hidden_modules,
        prompts=negative_prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seeds=ssv_seeds,
        num_frames=ssv_num_frames,
        cond_only=cond_only,
    )  # List[(N_neg × N_seeds, d_ff)]

    # ── Step 3 : PCA concept directions per layer ─────────────────────────────
    guard_layers: List[GuardLayer] = []

    for layer_idx, (layer, layer_budget) in enumerate(
        zip(ranked_layers, budgets)
    ):
        print(f"\n[PCA] Layer {layer_idx+1}/{len(ranked_layers)}: "
              f"{layer.spec.ff_name}  (budget={layer_budget} channels)")

        pos_samples = pos_all[layer_idx]   # (N_pos × N_seeds, d_ff)
        neg_samples = neg_all[layer_idx]   # (N_neg × N_seeds, d_ff)

        # ── Channel selection: pure SSV ───────────────────────────────────────
        # SSV role: find FFN neurons whose hidden activations fire most
        # selectively for concept inputs (key-value memory interpretation).
        # CAD's role is already fulfilled at layer selection; here we ask
        # "which neurons encode the concept?" — an activation-statistics question.
        pos_mean = pos_samples.mean(dim=0)   # (d_ff,)
        neg_mean = neg_samples.mean(dim=0)   # (d_ff,)

        ssv_scores = compute_ssv_scores(pos_mean, neg_mean)  # |pos-neg|/|neg|
        selected_channels = _topk_indices(ssv_scores, layer_budget)
        selected_channels_list = selected_channels.tolist()

        # Record SSV scores for the selected channels (for diagnostics)
        ssv_selected = ssv_scores[selected_channels].tolist()
        ssv_min  = min(ssv_selected)
        ssv_max  = max(ssv_selected)
        ssv_mean_val = sum(ssv_selected) / len(ssv_selected)
        print(f"  SSV channel scores — min={ssv_min:.4f}  mean={ssv_mean_val:.4f}"
              f"  max={ssv_max:.4f}  (top-{layer_budget} of {ssv_scores.numel()})")

        # ── Build concept-diff matrix for selected channels ───────────────────
        N_pos = pos_samples.shape[0]
        N_neg = neg_samples.shape[0]
        N_pairs = min(N_pos, N_neg)
        idx_t = torch.tensor(selected_channels_list, dtype=torch.long)

        diff_matrix = (
            pos_samples[:N_pairs, :][:, idx_t]
            - neg_samples[:N_pairs, :][:, idx_t]
        )  # (N_pairs, k)

        # ── PCA + elbow ───────────────────────────────────────────────────────
        concept_directions, evr = _pca_concept_directions(
            diff_matrix,
            max_rank=pca_max_rank,
            variance_threshold=pca_variance_threshold,
        )
        # concept_directions: (d, k), unit-norm rows (right singular vectors)

        d = concept_directions.shape[0]
        print(f"  PCA: {N_pairs} samples → rank-{d} "
              f"(cumVar={evr.sum():.1%}  EVRs={[f'{v:.2%}' for v in evr.tolist()]})")

        guard_layers.append(
            GuardLayer(
                ff_name=layer.spec.ff_name,
                hidden_module_name=layer.spec.hidden_module_name,
                projection_module_name=layer.spec.projection_module_name,
                selected_channels=selected_channels_list,
                cad_candidate_channels=selected_channels_list,
                # Primary direction (first PCA component) — backward compat
                steering_vector=concept_directions[0].tolist(),
                # Full rank-d directions
                concept_directions=concept_directions.tolist(),
                pca_explained_variance_ratios=evr.tolist(),
                # Layer scoring / ranking
                layer_score=layer.layer_score,
                cad_raw_score=layer_scores[layer.spec.ff_name].layer_score,
                layer_rank=layer_rank_lookup.get(layer.spec.ff_name, 0),
                # Channel selection diagnostics
                ssv_scores_selected=ssv_selected,
                # PCA diagnostics
                pca_n_samples=N_pairs,
                # Hook call counts
                positive_hook_calls=N_pos,
                negative_hook_calls=N_neg,
            )
        )

    # ── Step 4 : Pack into artifact ───────────────────────────────────────────
    return GuardArtifact(
        model_id=pipe.config._name_or_path,
        target=target,
        base_prompt=base_prompt,
        layers=guard_layers,
        alpha=alpha,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        cad_steps=cad_steps,
        cad_num_samples=cad_num_samples,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        metadata={
            "model_type": backend.model_type,
            "backbone": "transformer" if hasattr(pipe, "transformer") else "unet",
            "selected_ffn_layers": [layer.ff_name for layer in guard_layers],
            "num_layers": len(guard_layers),
            "layer_budgets": budgets,
            "ssv_num_seeds": ssv_num_seeds,
            "ssv_seeds": ssv_seeds,
            "pca_max_rank": pca_max_rank,
            "pca_variance_threshold": pca_variance_threshold,
            "cad_layer_score_type": "normalized_topk_mean",
            "channel_selection": "ssv_only",
            "use_layer_prior": use_layer_prior,
            "prior_strength": prior_strength,
        },
        layer_ranking=all_layer_ranking,
    )
