"""Tests for the masks module."""

from __future__ import annotations

import pytest

from prune_kit import dense_mask, mask_density, sparse_mask_to_dense


def test_dense_mask_full_density_returns_all_ones() -> None:
    mask = dense_mask((3, 4), density=1.0, seed=0)
    assert len(mask) == 12
    assert all(value == 1 for value in mask)


def test_dense_mask_zero_density_returns_all_zeros() -> None:
    mask = dense_mask((3, 4), density=0.0, seed=0)
    assert len(mask) == 12
    assert all(value == 0 for value in mask)


def test_dense_mask_half_density_keeps_correct_count() -> None:
    mask = dense_mask((2, 4), density=0.5, seed=42)
    assert sum(mask) == 4


def test_dense_mask_is_deterministic() -> None:
    a = dense_mask((10, 10), density=0.3, seed=7)
    b = dense_mask((10, 10), density=0.3, seed=7)
    assert a == b


def test_dense_mask_different_seeds_differ() -> None:
    a = dense_mask((10, 10), density=0.5, seed=1)
    b = dense_mask((10, 10), density=0.5, seed=2)
    assert a != b


def test_dense_mask_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        dense_mask((), density=0.5)
    with pytest.raises(ValueError, match="positive"):
        dense_mask((0, 4), density=0.5)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        dense_mask((4, 4), density=-0.1)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        dense_mask((4, 4), density=1.5)


def test_mask_density_zero() -> None:
    assert mask_density([0, 0, 0, 0]) == 0.0


def test_mask_density_one() -> None:
    assert mask_density([1, 1, 1]) == 1.0


def test_mask_density_partial() -> None:
    assert mask_density([1, 0, 1, 0, 0]) == pytest.approx(0.4)


def test_mask_density_rejects_empty() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        mask_density([])


def test_sparse_mask_to_dense_marks_indices() -> None:
    mask = sparse_mask_to_dense([0, 2, 4], (5,))
    assert mask == [1, 0, 1, 0, 1]


def test_sparse_mask_to_dense_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        sparse_mask_to_dense([0, 99], (5,))