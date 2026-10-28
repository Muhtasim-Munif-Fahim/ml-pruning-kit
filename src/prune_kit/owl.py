"""OWL: outlier-weighed layer-wise sparsity for Wanda (Yin et al., ICML 2024).

"Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for
Pruning LLMs to High Sparsity". Wanda (:mod:`prune_kit.wanda`) prunes every
layer to the *same* sparsity. OWL observes that layers differ in how many
*outlier* connections they carry -- Wanda scores ``|W| * ||X||`` far above
the layer mean -- and that those layers are the ones that break first at
high sparsity. It therefore gives outlier-heavy layers more density:

1. For each layer ``l`` compute the **layerwise outlier distribution**
   ``D_l = mean(A_l > M * mean(A_l))`` over its Wanda scores ``A_l``
   (``M`` is the outlier threshold, 5 in the paper; 3-10 is typical).
2. Map ``D`` linearly onto ``[0, 2 * lam]`` (``lam`` bounds how far any
   layer's density may drift from the target, 0.08 in the paper), then
   shift so the layers' mean density equals the target ``density``.
   With ``weighted=True`` the shift instead matches the *parameter-weighted*
   mean, so the overall kept fraction is the target even when layer sizes
   differ (the paper's LLM blocks are equal-sized, where the two coincide).
3. Prune each layer with Wanda at its own density
   (``wanda_prune_model(..., per_layer=densities)``).

Densities are clipped to ``(0, 1]``; when clipping binds, the realised
overall density is reported by :func:`owl_prune_summary`. If every layer has
the same outlier ratio, OWL reduces to uniform Wanda.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .layers import LayerSpec, layer_weight_count
from .wanda import _layer_scores, _resolve_saliency, _validate_density, wanda_prune_summary

__all__ = [
    "layer_outlier_ratio",
    "owl_outlier_ratios",
    "owl_layer_densities",
    "owl_prune_model",
    "owl_prune_summary",
]


_MIN_DENSITY = 1e-12


def _check_threshold(m: float) -> float:
    m = float(m)
    if not m > 0.0:
        raise ValueError("outlier threshold m must be positive")
    return m


def _check_lam(lam: float) -> float:
    lam = float(lam)
    if not 0.0 <= lam <= 0.5:
        raise ValueError("lam must be in [0, 0.5]")
    return lam


def layer_outlier_ratio(scores: Sequence[float], m: float = 5.0) -> float:
    """Fraction of ``scores`` strictly above ``m`` times their mean (0 when all are 0)."""
    m = _check_threshold(m)
    if not scores:
        raise ValueError("scores must not be empty")
    values = [abs(float(v)) for v in scores]
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    cut = m * mean
    return sum(1 for v in values if v > cut) / len(values)


def owl_outlier_ratios(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    activation_norms: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    m: float = 5.0,
) -> Dict[str, float]:
    """Layerwise outlier distribution ``D_l`` of the Wanda scores of each layer."""
    if not specs:
        raise ValueError("specs must not be empty")
    source, saliency = _resolve_saliency(activation_norms, contributions)
    out: Dict[str, float] = {}
    for spec in specs:
        if spec.name not in model:
            raise ValueError(f"model missing layer {spec.name!r}")
        scores = _layer_scores(spec, model[spec.name], source, saliency)
        out[spec.name] = layer_outlier_ratio(scores, m)
    return out


def owl_layer_densities(
    specs: Sequence[LayerSpec],
    outlier_ratios: Dict[str, float],
    density: float = 0.5,
    lam: float = 0.08,
    weighted: bool = False,
) -> Dict[str, float]:
    """Per-layer keep fractions from outlier ratios (OWL step 2).

    Ratios are min-max scaled to ``[0, 2 * lam]`` and shifted so the mean
    density (parameter-weighted when ``weighted``) equals ``density``;
    results are clipped to ``(0, 1]``.
    """
    density = _validate_density(density)
    lam = _check_lam(lam)
    if not specs:
        raise ValueError("specs must not be empty")
    names = [spec.name for spec in specs]
    missing = [n for n in names if n not in outlier_ratios]
    if missing:
        raise ValueError(f"outlier_ratios missing layer(s): {', '.join(missing)}")
    ratios = [float(outlier_ratios[n]) for n in names]
    lo, hi = min(ratios), max(ratios)
    if hi == lo:
        scaled = [0.0] * len(ratios)
    else:
        scaled = [(r - lo) / (hi - lo) * 2.0 * lam for r in ratios]
    if weighted:
        sizes = [layer_weight_count(spec) for spec in specs]
        centre = sum(s * w for s, w in zip(scaled, sizes)) / sum(sizes)
    else:
        centre = sum(scaled) / len(scaled)
    # Wanda needs density > 0, so a clipped layer keeps a vanishing fraction
    # (round(_MIN_DENSITY * N) == 0 weights), which is "pruned entirely".
    return {
        n: min(max(s - centre + density, _MIN_DENSITY), 1.0) for n, s in zip(names, scaled)
    }


def owl_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    activation_norms: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    m: float = 5.0,
    lam: float = 0.08,
    weighted: bool = False,
) -> Dict[str, object]:
    """Run OWL + Wanda and return the Wanda summary plus OWL diagnostics.

    Extra keys: ``method`` (``"owl"``), ``outlier_threshold``, ``lam``,
    ``weighted``, ``target_density`` and ``outlier_ratios``; each layer row
    gains ``outlier_ratio`` and ``owl_density``.
    """
    ratios = owl_outlier_ratios(
        specs, model, activation_norms, contributions=contributions, m=m
    )
    densities = owl_layer_densities(specs, ratios, density=density, lam=lam, weighted=weighted)
    summary = wanda_prune_summary(
        specs,
        model,
        activation_norms,
        contributions=contributions,
        density=density,
        scope="layer",
        per_layer=densities,
    )
    rows: List[Dict[str, object]] = summary["layers"]  # type: ignore[assignment]
    for row in rows:
        name = row["layer"]
        row["outlier_ratio"] = ratios[name]
        row["owl_density"] = densities[name]
    summary.update(
        method="owl",
        outlier_threshold=float(m),
        lam=float(lam),
        weighted=bool(weighted),
        target_density=float(density),
        outlier_ratios=ratios,
    )
    return summary


def owl_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    activation_norms: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    m: float = 5.0,
    lam: float = 0.08,
    weighted: bool = False,
) -> Dict[str, List[float]]:
    """One-shot OWL prune: Wanda with outlier-weighed per-layer densities."""
    summary = owl_prune_summary(
        specs, model, activation_norms, contributions=contributions,
        density=density, m=m, lam=lam, weighted=weighted,
    )
    return summary["pruned_weights"]  # type: ignore[return-value]
