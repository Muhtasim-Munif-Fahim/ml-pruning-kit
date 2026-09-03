"""Sparse mask helpers."""

from __future__ import annotations

from typing import Sequence


def dense_mask(
    shape: tuple[int, ...],
    density: float,
    *,
    seed: int | None = None,
) -> list[int]:
    """Generate a deterministic random 0/1 mask of the requested shape.

    ``density`` is the fraction of entries that should be 1 (kept).
    The mask is returned as a flat list; :func:`sparse_mask_to_dense`
    reshapes it for the caller. ``seed`` makes the layout reproducible.
    """
    if not shape:
        raise ValueError("shape must be non-empty")
    if any(dim <= 0 for dim in shape):
        raise ValueError("shape dimensions must be positive")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")
    import random

    rng = random.Random(seed)
    total = 1
    for dim in shape:
        total *= int(dim)
    keep = int(round(density * total))
    mask = [1] * keep + [0] * (total - keep)
    rng.shuffle(mask)
    return mask


def sparse_mask_to_dense(
    indices: Sequence[int],
    shape: tuple[int, ...],
) -> list[int]:
    """Convert an index list of 'kept' positions into a flat 0/1 mask."""
    total = 1
    for dim in shape:
        total *= int(dim)
    mask = [0] * total
    for index in indices:
        if not 0 <= int(index) < total:
            raise ValueError(f"index {index} out of range for shape {shape}")
        mask[int(index)] = 1
    return mask


def mask_density(mask: Sequence[int]) -> float:
    """Return the fraction of 1s in a flat 0/1 mask."""
    if not mask:
        raise ValueError("mask must not be empty")
    ones = sum(1 for value in mask if value)
    return ones / len(mask)


__all__ = ["dense_mask", "sparse_mask_to_dense", "mask_density"]