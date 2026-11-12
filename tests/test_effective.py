"""Tests for effective sparsity (disconnected-weight removal)."""

from __future__ import annotations

import random

import pytest

from prune_kit import (
    conv_layer,
    dense_chain_connected,
    dense_layer,
    effective_masks,
    effective_sparsity_report,
    prune_disconnected,
)


def _matmul_bool(A, B):
    return [[any(A[i][k] and B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _brute_effective(specs, masks):
    """Weight j->i of layer l is effective iff input->j and i->output paths exist."""
    mats = []
    for s in specs:
        o, n = s.shape
        mats.append([[bool(masks[s.name][r * n + c]) for c in range(n)] for r in range(o)])
    out = {}
    for l, s in enumerate(specs):
        o, n = s.shape
        # reach_in[j]: some input reaches neuron j (input side of layer l)
        if l == 0:
            reach_in = [True] * n
        else:
            P = mats[0]
            for M in mats[1:l]:
                P = _matmul_bool(M, P)
            reach_in = [any(row) for row in P]
        if l == len(specs) - 1:
            reach_out = [True] * o
        else:
            Q = mats[l + 1]
            for M in mats[l + 2:]:
                Q = _matmul_bool(M, Q)
            reach_out = [any(Q[r][c] for r in range(len(Q))) for c in range(o)]
        out[s.name] = [
            1 if mats[l][r][c] and reach_in[c] and reach_out[r] else 0 for r in range(o) for c in range(n)
        ]
    return out


def test_hand_example():
    specs = [dense_layer("fc1", 2, 2), dense_layer("fc2", 2, 1)]
    # fc1: h0 <- x0 only; h1 has no inputs. fc2: y <- h0 and y <- h1.
    masks = {"fc1": [1, 0, 0, 0], "fc2": [1, 1]}
    eff = effective_masks(specs, masks)
    assert eff == {"fc1": [1, 0, 0, 0], "fc2": [1, 0]}
    rep = effective_sparsity_report(specs, masks)
    assert rep.active == 3 and rep.effective == 2
    assert rep.layers[1].dead_inputs == 1 and rep.layers[1].inactive_weights == 1
    assert rep.direct_sparsity == pytest.approx(1 - 3 / 6)
    assert rep.effective_sparsity == pytest.approx(1 - 2 / 6)
    assert "| fc2 |" in rep.table()


def test_dead_end_neuron_is_removed_upstream():
    specs = [dense_layer("fc1", 2, 2), dense_layer("fc2", 2, 1)]
    masks = {"fc1": [1, 1, 1, 1], "fc2": [1, 0]}  # h1 feeds nothing
    eff = effective_masks(specs, masks)
    assert eff["fc1"] == [1, 1, 0, 0]
    assert effective_sparsity_report(specs, masks).layers[0].dead_outputs == 1


@pytest.mark.parametrize("seed", range(8))
def test_matches_brute_force_paths_and_is_idempotent(seed):
    rng = random.Random(seed)
    sizes = [5, 6, 4, 5, 3]
    specs = [dense_layer(f"fc{i}", a, b) for i, (a, b) in enumerate(zip(sizes, sizes[1:]))]
    density = rng.choice([0.15, 0.25, 0.4])
    masks = {s.name: [1 if rng.random() < density else 0 for _ in range(s.shape[0] * s.shape[1])] for s in specs}
    eff = effective_masks(specs, masks)
    assert eff == _brute_effective(specs, masks)
    assert effective_masks(specs, eff) == eff
    rep = effective_sparsity_report(specs, masks)
    assert rep.connected == dense_chain_connected(specs, masks)
    assert rep.effective <= rep.active
    assert all(e <= m for s in specs for e, m in zip(eff[s.name], masks[s.name]))


def test_prune_disconnected_preserves_function():
    rng = random.Random(3)
    sizes = [4, 6, 3]
    specs = [dense_layer(f"fc{i}", a, b) for i, (a, b) in enumerate(zip(sizes, sizes[1:]))]
    model = {
        s.name: [rng.gauss(0, 1) if rng.random() < 0.3 else 0.0 for _ in range(s.shape[0] * s.shape[1])]
        for s in specs
    }

    def forward(m, x):
        for s in specs:
            o, n = s.shape
            x = [max(0.0, sum(m[s.name][r * n + c] * x[c] for c in range(n))) for r in range(o)]
        return x

    pruned = prune_disconnected(specs, model)
    for _ in range(20):
        x = [rng.gauss(0, 1) for _ in range(4)]
        assert forward(pruned, x) == pytest.approx(forward(model, x))
    assert sum(1 for s in specs for w in pruned[s.name] if w) <= sum(
        1 for s in specs for w in model[s.name] if w
    )


def test_input_output_subsets_and_collapse():
    specs = [dense_layer("fc1", 2, 2), dense_layer("fc2", 2, 2)]
    masks = {"fc1": [1, 0, 0, 1], "fc2": [1, 0, 0, 1]}  # two parallel paths
    eff = effective_masks(specs, masks, inputs_alive=[True, False])
    assert eff == {"fc1": [1, 0, 0, 0], "fc2": [1, 0, 0, 0]}
    eff = effective_masks(specs, masks, outputs_alive=[False, True])
    assert eff == {"fc1": [0, 0, 0, 1], "fc2": [0, 0, 0, 1]}
    collapsed = effective_sparsity_report(specs, {"fc1": [1, 0, 0, 0], "fc2": [0, 1, 0, 1]})
    assert not collapsed.connected and collapsed.effective == 0


def test_validation():
    specs = [dense_layer("fc1", 2, 2), dense_layer("fc2", 3, 1)]
    with pytest.raises(ValueError):
        effective_masks(specs, {"fc1": [1] * 4, "fc2": [1] * 3})
    with pytest.raises(ValueError):
        effective_masks([conv_layer("c", 2, 2, 3, 3)], {"c": [1] * 36})
    good = [dense_layer("fc1", 2, 2)]
    with pytest.raises(KeyError):
        effective_masks(good, {})
    with pytest.raises(ValueError):
        effective_masks(good, {"fc1": [1, 1]})
    with pytest.raises(ValueError):
        effective_masks(good, {"fc1": [1] * 4}, inputs_alive=[True])
