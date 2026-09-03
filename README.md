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

## CLI quick start

```bash
prune-kit survival --help
```

The CLI emits a Markdown table with per-layer survival, kept weight
counts, and the overall survival fraction.

## Tests

```bash
pytest tests
```