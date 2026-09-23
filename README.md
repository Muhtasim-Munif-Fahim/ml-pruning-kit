# ml-pruning-kit

A small, dependency-free Python toolkit for studying weight pruning in
neural networks. It implements per-layer and global unstructured
magnitude pruning (single-step and iterative), structured channel/filter
pruning, first-order Taylor (soft-filter) channel pruning, per-layer
and per-channel survival reporting, sparse-mask helpers, and a tiny CLI.
The codebase is intentionally framework-agnostic: every routine works
on flat Python lists of weights, so it composes with PyTorch,
TensorFlow, JAX, or a custom numpy implementation.

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

## Structured channel / filter pruning

Conv kernels are stored C-contiguous as `(out_channels, in_channels,
kernel_h, kernel_w)`. Structured pruning zeros **whole output filters**
(`structure="filter"`) or **whole input channels** (`structure="channel"`),
ranked by L1 or L2 norm. Dense layers in a mixed model are copied
unchanged. `density` is the fraction of filters/channels to keep.

```python
from prune_kit import (
    conv_layer, dense_layer, structured_prune_model,
    model_channel_survival_summary,
)

specs = [
    dense_layer("fc1", in_features=8, out_features=4),
    conv_layer("conv1", out_channels=4, in_channels=2, kernel_h=3, kernel_w=3),
]
weights = {
    "fc1": [0.1 * i for i in range(4 * 8)],
    "conv1": [0.01 * (i + 1) for i in range(4 * 2 * 3 * 3)],
}

pruned = structured_prune_model(specs, weights, density=0.5, norm="l1")
summary = model_channel_survival_summary(
    specs, weights, density=0.5, norm="l2", structure="filter"
)
for row in summary["layers"]:
    print(row["layer"], row["kept_indices"], row["survival_fraction"])
```

L1 and L2 can rank filters differently: a sparse filter with one large
weight has a higher L2 than a dense filter of equal L1.

## Soft-filter first-order Taylor pruning

L1/L2 structured pruning ranks a conv filter by the norm of its
weights. First-order Taylor (FO) pruning ranks it by how much the loss
would move if that filter were removed. The per-weight saliency is
`|grad * weight|` (Molchanov et al.). A filter or input-channel score
is the **sum** of those terms (`reduction="mean"` divides by the group
size; within one layer the ranking is the same). `criterion="sq"` uses
`(grad * weight) ** 2` before the reduction. The lowest-scoring groups
are zeroed. The kernel shape does not change, so this is a soft,
shape-preserving filter prune: pruned filters stay in the tensor as
exact zeros. Dense layers in a mixed model are copied unchanged.
`density` is the fraction of filters or channels to keep. Ties break
toward the lower index.

When the saliency is already a per-weight first-order term — for
example activation-map products `(dL/dz) * z` laid out like the kernel
— pass those raw products as `contributions`. The same `abs` or `sq`
reduction is applied. Pass `grads` or `contributions`, not both.

```python
from prune_kit import conv_layer, dense_layer, taylor_prune_model, taylor_prune_summary

specs = [
    dense_layer("fc1", in_features=4, out_features=2),
    conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=2),
]
weights = {
    "fc1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "conv1": [10.0, 10.0, 0.1, 0.1],
}
grads = {
    "conv1": [0.0, 0.0, 5.0, 5.0],
}
pruned = taylor_prune_model(specs, weights, grads, density=0.5)
# Filter 0 is large but has zero gradient, so it is removed.
# Filter 1 is small but salient, so it is kept.
summary = taylor_prune_summary(specs, weights, grads, density=0.5)
print(summary["layers"][0]["kept_indices"])
```

FO and L1 can disagree on the same kernel: a large filter with a
near-zero gradient is pruned first, while a small filter with a large
gradient is kept. That is the complement to structured L1/L2 pruning
(weight norms only) and to global unstructured magnitude pruning
(individual weights, no channel groups).

## Global unstructured magnitude pruning

Per-layer magnitude pruning keeps the same fraction of weights inside
every layer. Global unstructured pruning ranks **every scalar weight
in the model together** and zeros the smallest magnitudes until a
target sparsity is reached, so a layer of small weights can be pruned
harder than a layer of large weights. `sparsity` is the fraction of
weights to remove (`0.0` keeps everything, `1.0` zeros everything).
The number kept is `round((1 - sparsity) * N)`, the same rounding as
per-layer pruning at `density = 1 - sparsity`. Ties break toward the
earlier layer in dict order, then the lower index.

An optional iterative schedule reaches that same final mask over
several rounds. Removals are spread evenly (earlier rounds take any
remainder). Pass `schedule` for an explicit cumulative sparsity after
each round. When `rewind=True`, surviving weights are reset to
`initial_weights` after every round. Ranking always uses the
pre-rewind magnitudes, so the kept positions match the one-shot global
prune. This complements per-layer lottery-ticket IMP (equal density
per layer) and structured filter/channel pruning.

```python
from prune_kit import (
    global_magnitude_prune_model,
    iterative_global_magnitude_prune_model,
)

trained = {
    "small": [0.1, 0.2, 0.3, 0.4],
    "large": [1.0, 2.0, 3.0, 4.0],
}
initial = {
    "small": [0.5, 0.5, 0.5, 0.5],
    "large": [0.5, 0.5, 0.5, 0.5],
}

one_shot = global_magnitude_prune_model(trained, sparsity=0.5)
# "small" is fully zeroed; "large" is kept.

result = iterative_global_magnitude_prune_model(
    trained,
    sparsity=0.5,
    rounds=2,
    rewind=True,
    initial_weights=initial,
)
print(result.weights)
print(result.density_curve())
print(result.schedule)
```

## CLI quick start

```bash
prune-kit survival --help
prune-kit imp --help
prune-kit structured --help
prune-kit taylor --help
prune-kit global --help
```

The `survival` command emits a Markdown table with per-layer survival,
kept weight counts, and the overall survival fraction. The `imp`
command runs iterative magnitude pruning (optionally with
`--rewind`) and prints the density after each round. The `structured`
command prunes whole conv filters or channels by `--norm l1|l2` and
prints a per-channel survival table (non-conv layers are skipped).
The `taylor` command scores conv filters or channels by
`|grad * weight|` (`--criterion abs|sq`, `--reduction sum|mean`) and
prints the same kind of survival table. `--grads` is a flat buffer
aligned with `--weights`. The `global` command prunes lowest-|w| weights to `--sparsity`
(optionally over `--rounds` or an explicit `--schedule`) and prints
the density after each round plus a per-layer kept-count table.
`--rewind` resets survivors to `--initial-weights`.

## Tests

```bash
pytest tests
```