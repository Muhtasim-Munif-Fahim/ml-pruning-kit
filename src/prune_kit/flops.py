"""Theoretical FLOPs / MAC compression reports for pruned layers.

Magnitude and structured pruning change parameter counts; they also change
the arithmetic cost of a forward pass once zeros are skipped (or channels
removed). This module estimates dense and sparse multiply-accumulate counts
for :class:`~prune_kit.layers.LayerSpec` models so a prune run can report
compression beyond "percent of weights kept".

Conventions (one forward example):

* dense ``(out, in)``: ``out * in`` MACs (bias ignored)
* conv ``(out, in, kh, kw)`` with spatial size ``H * W``: ``out * in * kh * kw * H * W``
* unstructured sparsity: scale dense MACs by the kept-weight fraction
* structured channel prune: use the kept out/in channel counts when provided

These are theoretical counts for comparison across prune methods, not
device-measured timings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .masks import mask_density

__all__ = [
    "LayerFlops",
    "FlopsReport",
    "dense_macs",
    "estimate_layer_macs",
    "flops_report",
]


@dataclass(frozen=True)
class LayerFlops:
    name: str
    kind: str
    params_dense: int
    params_kept: int
    density: float
    macs_dense: int
    macs_sparse: int

    @property
    def param_compression(self) -> float:
        return self.params_dense / self.params_kept if self.params_kept else float("inf")

    @property
    def mac_compression(self) -> float:
        return self.macs_dense / self.macs_sparse if self.macs_sparse else float("inf")


@dataclass(frozen=True)
class FlopsReport:
    layers: Tuple[LayerFlops, ...]
    params_dense: int
    params_kept: int
    macs_dense: int
    macs_sparse: int

    @property
    def density(self) -> float:
        return self.params_kept / self.params_dense if self.params_dense else 0.0

    @property
    def param_compression(self) -> float:
        return self.params_dense / self.params_kept if self.params_kept else float("inf")

    @property
    def mac_compression(self) -> float:
        return self.macs_dense / self.macs_sparse if self.macs_sparse else float("inf")

    def to_dict(self) -> Dict[str, object]:
        return {
            "params_dense": self.params_dense,
            "params_kept": self.params_kept,
            "density": self.density,
            "param_compression": self.param_compression,
            "macs_dense": self.macs_dense,
            "macs_sparse": self.macs_sparse,
            "mac_compression": self.mac_compression,
            "layers": [
                {
                    "name": layer.name,
                    "kind": layer.kind,
                    "params_dense": layer.params_dense,
                    "params_kept": layer.params_kept,
                    "density": layer.density,
                    "macs_dense": layer.macs_dense,
                    "macs_sparse": layer.macs_sparse,
                    "param_compression": layer.param_compression,
                    "mac_compression": layer.mac_compression,
                }
                for layer in self.layers
            ],
        }


def dense_macs(spec: LayerSpec, *, spatial_hw: Tuple[int, int] = (1, 1)) -> int:
    """Dense (unpruned) MAC count for one forward example on ``spec``."""
    if spec.kind == "dense":
        out_f, in_f = spec.shape
        return int(out_f * in_f)
    if spec.kind == "conv":
        out_c, in_c, kh, kw = spec.shape
        h, w = spatial_hw
        if h <= 0 or w <= 0:
            raise ValueError("spatial_hw must be positive")
        return int(out_c * in_c * kh * kw * h * w)
    raise ValueError(f"unsupported kind {spec.kind!r}")


def estimate_layer_macs(
    spec: LayerSpec,
    *,
    density: Optional[float] = None,
    mask: Optional[Sequence[int]] = None,
    kept_out: Optional[int] = None,
    kept_in: Optional[int] = None,
    spatial_hw: Tuple[int, int] = (1, 1),
) -> LayerFlops:
    """Estimate dense vs sparse MACs for one layer.

    Prefer ``mask`` (0/1 flat) when available; otherwise pass ``density``.
    For structured channel pruning, pass ``kept_out`` / ``kept_in`` to shrink
    the dense geometry (structured MACs replace unstructured scaling).
    """
    params = layer_weight_count(spec)
    macs_d = dense_macs(spec, spatial_hw=spatial_hw)

    structured = kept_out is not None or kept_in is not None
    if structured:
        if spec.kind == "dense":
            out_f, in_f = spec.shape
            out_k = out_f if kept_out is None else int(kept_out)
            in_k = in_f if kept_in is None else int(kept_in)
            if not (0 < out_k <= out_f and 0 < in_k <= in_f):
                raise ValueError("kept_out/kept_in out of range for dense layer")
            kept = int(out_k * in_k)
            macs_s = int(out_k * in_k)
        else:
            out_c, in_c, kh, kw = spec.shape
            h, w = spatial_hw
            out_k = out_c if kept_out is None else int(kept_out)
            in_k = in_c if kept_in is None else int(kept_in)
            if not (0 < out_k <= out_c and 0 < in_k <= in_c):
                raise ValueError("kept_out/kept_in out of range for conv layer")
            kept = int(out_k * in_k * kh * kw)
            macs_s = int(out_k * in_k * kh * kw * h * w)
        dens = kept / params if params else 0.0
    else:
        if mask is not None:
            if len(mask) != params:
                raise ValueError(f"mask length {len(mask)} != params {params}")
            dens = float(mask_density(mask))
            kept = int(sum(1 for m in mask if m))
        elif density is not None:
            if not 0.0 <= float(density) <= 1.0:
                raise ValueError("density must be in [0, 1]")
            dens = float(density)
            kept = int(round(dens * params))
        else:
            dens = 1.0
            kept = params
        macs_s = int(round(macs_d * dens))

    return LayerFlops(
        name=spec.name,
        kind=spec.kind,
        params_dense=params,
        params_kept=kept,
        density=dens,
        macs_dense=macs_d,
        macs_sparse=macs_s,
    )


def flops_report(
    specs: Sequence[LayerSpec],
    *,
    densities: Optional[Mapping[str, float]] = None,
    masks: Optional[Mapping[str, Sequence[int]]] = None,
    kept_channels: Optional[Mapping[str, Tuple[int, int]]] = None,
    spatial_hw: Mapping[str, Tuple[int, int]] | Tuple[int, int] = (1, 1),
) -> FlopsReport:
    """Aggregate FLOPs / parameter compression across a list of layers."""
    rows = []
    for spec in specs:
        hw = spatial_hw[spec.name] if isinstance(spatial_hw, Mapping) else spatial_hw
        ko = ki = None
        if kept_channels and spec.name in kept_channels:
            ko, ki = kept_channels[spec.name]
        mask = masks.get(spec.name) if masks else None
        dens = densities.get(spec.name) if densities else None
        rows.append(
            estimate_layer_macs(
                spec,
                density=dens,
                mask=mask,
                kept_out=ko,
                kept_in=ki,
                spatial_hw=hw,
            )
        )
    return FlopsReport(
        layers=tuple(rows),
        params_dense=sum(r.params_dense for r in rows),
        params_kept=sum(r.params_kept for r in rows),
        macs_dense=sum(r.macs_dense for r in rows),
        macs_sparse=sum(r.macs_sparse for r in rows),
    )
