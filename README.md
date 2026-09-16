# ml-pruning-kit

A small, dependency-free Python toolkit for studying weight pruning in
neural networks. It implements magnitude pruning (single-step and
iterative), per-layer survival reporting, sparse-mask helpers, and a
tiny CLI. The codebase is intentionally framework-agnostic: every
routine works on flat Python lists of weights, so it composes with
PyTorch, TensorFlow, JAX, or a custom numpy implementation.

## Install

```bash
pip install -e .
```

## Library quick start

```python
from prune_kit import (
    dense_layer, conv_layer, magnitude_prune_model, model_survival_summary,
)

specs = [
    dense_layer("fc1", in_features=784, out_features=256),
    dense_layer("fc2", in_features=256, out_features=10),
    conv_layer("conv1", out_channels=16, in_channels=3, kernel_h=3, kernel_w=3),
]
weights = {
    "fc1": [0.01 * i for i in range(784 * 256)],
    "fc2": [0.005 * i for i in range(256 * 10)],
    "conv1": [0.001 * i for i in range(16 * 3 * 3 * 3)],
}
summary = model_survival_summary(specs, weights, density=0.5)
for row in summary["layers"]:
    print(row)
print(summary["overall_survival"])
```

## Iterative magnitude pruning (lottery ticket rewind)

Each round of iterative magnitude pruning removes `prune_fraction` of
the weights that are still non-zero, ranked by magnitude. When
`rewind=True`, surviving weights are reset to `initial_weights` after
every prune step — the lottery-ticket hypothesis reset. Ranking always
uses the pre-rewind (typically trained) magnitudes, so compounding
sparsity matches one-shot pruning to the product density.

```python
from prune_kit import iterative_magnitude_prune_model

trained = {"fc1": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 1.0]}
initial = {"fc1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}

result = iterative_magnitude_prune_model(
    trained,
    prune_fraction=0.2,
    rounds=3,
    rewind=True,
    initial_weights=initial,
)
print(result.weights)
print(result.density_curve())
```

The simulated training loop in `train_with_pruning` can do the same
reset between prune steps (`TrainingConfig(rewind=True)`), and can
compound sparsity with `prune_fraction` instead of an absolute
`prune_density`.

## CLI quick start

```bash
prune-kit survival --help
prune-kit imp --help
```

The `survival` command emits a Markdown table with per-layer survival,
kept weight counts, and the overall survival fraction. The `imp`
command runs iterative magnitude pruning (optionally with
`--rewind`) and prints the density after each round.

## Tests

```bash
pytest tests
```