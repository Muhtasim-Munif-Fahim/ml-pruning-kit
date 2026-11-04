"""Layer-wise pruning sensitivity analysis (Han et al., 2015; Distiller).

"Learning both Weights and Connections for Efficient Neural Networks"
prunes each layer *in isolation* to a sweep of densities and records how
much the network's quality degrades. Layers whose curves stay flat are
robust and can be pruned hard; layers whose curves fall off quickly are
sensitive and should keep more weights. Intel's Distiller popularised
this as a "sensitivity analysis" that then drives a per-layer density
schedule.

This module is framework-agnostic: the caller passes an ``evaluate``
callable mapping a ``{layer_name: flat weights}`` model to a scalar score
(accuracy, negative loss, ...). With ``higher_is_better=False`` the score
is treated as a loss.

* :func:`layer_sensitivity` -- sweep one layer, everything else dense.
* :func:`sensitivity_analysis` -- sweep every (or a subset of) layer(s).
* :func:`sensitivity_densities` -- pick, per layer, the lowest swept
  density whose degradation stays within ``tolerance`` (absolute or
  relative to the dense baseline).
* :func:`sensitivity_prune_model` -- analysis + allocation + magnitude
  pruning at the chosen per-layer densities, in one call.

By default the layer is magnitude-pruned (:func:`magnitude_prune_layer`),
but any ``prune_fn(weights, density) -> weights`` can be supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .prune import magnitude_prune_layer, model_density

__all__ = [
    "SensitivityCurve",
    "SensitivityAnalysis",
    "SensitivityPruneResult",
    "default_density_grid",
    "layer_sensitivity",
    "sensitivity_analysis",
    "sensitivity_densities",
    "sensitivity_prune_model",
]

Model = Dict[str, Sequence[float]]
Evaluate = Callable[[Dict[str, List[float]]], float]
PruneFn = Callable[[Sequence[float], float], List[float]]


def _default_prune(weights: Sequence[float], density: float) -> List[float]:
    return magnitude_prune_layer(weights, density=density)


def default_density_grid(steps: int = 10, min_density: float = 0.1) -> List[float]:
    """Evenly spaced densities from 1.0 down to ``min_density`` (inclusive)."""
    if steps < 2:
        raise ValueError("steps must be >= 2")
    if not 0.0 < min_density < 1.0:
        raise ValueError("min_density must be in (0, 1)")
    span = 1.0 - min_density
    return [round(1.0 - span * i / (steps - 1), 12) for i in range(steps)]


def _validate_grid(densities: Sequence[float]) -> List[float]:
    grid = sorted({float(d) for d in densities}, reverse=True)
    if not grid:
        raise ValueError("densities must not be empty")
    for d in grid:
        if not 0.0 < d <= 1.0:
            raise ValueError("every density must be in (0, 1]")
    return grid


def _copy(model: Model) -> Dict[str, List[float]]:
    if not model:
        raise ValueError("model must not be empty")
    return {name: [float(v) for v in w] for name, w in model.items()}


@dataclass(frozen=True)
class SensitivityCurve:
    """Score of the model when only ``layer`` is pruned, per density."""

    layer: str
    densities: Tuple[float, ...]
    scores: Tuple[float, ...]
    baseline: float
    higher_is_better: bool = True

    def degradation(self) -> Tuple[float, ...]:
        """Quality lost vs. the dense baseline (positive = worse)."""
        sign = 1.0 if self.higher_is_better else -1.0
        return tuple(sign * (self.baseline - s) for s in self.scores)

    def auc_degradation(self) -> float:
        """Trapezoidal area under the degradation-vs-sparsity curve.

        A single scalar to rank layers by sensitivity (larger = more
        sensitive). Densities are converted to sparsity ``1 - d``.
        """
        pts = sorted(zip((1.0 - d for d in self.densities), self.degradation()))
        if len(pts) < 2:
            return 0.0
        area = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            area += 0.5 * (y0 + y1) * (x1 - x0)
        return area


@dataclass(frozen=True)
class SensitivityAnalysis:
    baseline: float
    higher_is_better: bool
    curves: Dict[str, SensitivityCurve]

    def ranking(self) -> List[str]:
        """Layer names from most to least sensitive (by AUC degradation)."""
        return sorted(self.curves, key=lambda n: (-self.curves[n].auc_degradation(), n))


@dataclass
class SensitivityPruneResult:
    model: Dict[str, List[float]]
    densities: Dict[str, float]
    analysis: SensitivityAnalysis
    overall_density: float
    final_score: float
    notes: List[str] = field(default_factory=list)


def layer_sensitivity(
    model: Model,
    layer: str,
    evaluate: Evaluate,
    densities: Optional[Sequence[float]] = None,
    *,
    higher_is_better: bool = True,
    prune_fn: Optional[PruneFn] = None,
    baseline: Optional[float] = None,
) -> SensitivityCurve:
    """Prune only ``layer`` to each density and score the whole model."""
    base_model = _copy(model)
    if layer not in base_model:
        raise ValueError(f"unknown layer {layer!r}")
    grid = _validate_grid(densities if densities is not None else default_density_grid())
    prune = prune_fn or _default_prune
    if baseline is None:
        baseline = float(evaluate(_copy(base_model)))
    original = base_model[layer]
    scores: List[float] = []
    for d in grid:
        trial = dict(base_model)
        trial[layer] = list(original) if d >= 1.0 else [float(v) for v in prune(original, d)]
        if len(trial[layer]) != len(original):
            raise ValueError("prune_fn must preserve the layer length")
        scores.append(float(evaluate(trial)))
    return SensitivityCurve(
        layer=layer,
        densities=tuple(grid),
        scores=tuple(scores),
        baseline=float(baseline),
        higher_is_better=higher_is_better,
    )


def sensitivity_analysis(
    model: Model,
    evaluate: Evaluate,
    densities: Optional[Sequence[float]] = None,
    *,
    layers: Optional[Sequence[str]] = None,
    higher_is_better: bool = True,
    prune_fn: Optional[PruneFn] = None,
) -> SensitivityAnalysis:
    """Run :func:`layer_sensitivity` for every layer (or ``layers``)."""
    base_model = _copy(model)
    names = list(base_model) if layers is None else list(layers)
    if not names:
        raise ValueError("layers must not be empty")
    for name in names:
        if name not in base_model:
            raise ValueError(f"unknown layer {name!r}")
    baseline = float(evaluate(_copy(base_model)))
    curves = {
        name: layer_sensitivity(
            base_model,
            name,
            evaluate,
            densities,
            higher_is_better=higher_is_better,
            prune_fn=prune_fn,
            baseline=baseline,
        )
        for name in names
    }
    return SensitivityAnalysis(baseline=baseline, higher_is_better=higher_is_better, curves=curves)


def sensitivity_densities(
    analysis: SensitivityAnalysis,
    tolerance: float,
    *,
    relative: bool = False,
    monotone: bool = True,
) -> Dict[str, float]:
    """Lowest swept density per layer whose degradation is within ``tolerance``.

    ``relative=True`` interprets ``tolerance`` as a fraction of
    ``|baseline|``. With ``monotone=True`` (default) the sweep stops at
    the first density that exceeds the tolerance, so a noisy recovery at
    lower density is not trusted. A layer that fails even its highest
    swept density keeps density 1.0.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    limit = tolerance * abs(analysis.baseline) if relative else tolerance
    out: Dict[str, float] = {}
    for name, curve in analysis.curves.items():
        chosen = 1.0
        for d, deg in zip(curve.densities, curve.degradation()):
            if deg <= limit + 1e-12:
                chosen = min(chosen, d)
            elif monotone:
                break
        out[name] = chosen
    return out


def sensitivity_prune_model(
    model: Model,
    evaluate: Evaluate,
    tolerance: float,
    densities: Optional[Sequence[float]] = None,
    *,
    relative: bool = False,
    higher_is_better: bool = True,
    prune_fn: Optional[PruneFn] = None,
    monotone: bool = True,
) -> SensitivityPruneResult:
    """Sensitivity analysis -> per-layer densities -> prune every layer.

    Per-layer tolerances do not compose exactly when layers are pruned
    together, so the combined model is re-evaluated and its score is
    returned as ``final_score`` (with a note if it exceeds the tolerance).
    """
    base_model = _copy(model)
    analysis = sensitivity_analysis(
        base_model, evaluate, densities, higher_is_better=higher_is_better, prune_fn=prune_fn
    )
    chosen = sensitivity_densities(analysis, tolerance, relative=relative, monotone=monotone)
    prune = prune_fn or _default_prune
    pruned = {
        name: (list(w) if chosen[name] >= 1.0 else [float(v) for v in prune(w, chosen[name])])
        for name, w in base_model.items()
    }
    final = float(evaluate(pruned))
    sign = 1.0 if higher_is_better else -1.0
    limit = tolerance * abs(analysis.baseline) if relative else tolerance
    notes: List[str] = []
    if sign * (analysis.baseline - final) > limit + 1e-12:
        notes.append(
            "combined degradation exceeds the per-layer tolerance; "
            "consider a tighter tolerance or fine-tuning"
        )
    return SensitivityPruneResult(
        model=pruned,
        densities=chosen,
        analysis=analysis,
        overall_density=model_density(pruned),
        final_score=final,
        notes=notes,
    )
