# ml-pruning-kit

A small, dependency-free Python toolkit for studying weight pruning in
neural networks. It implements per-layer and global unstructured
magnitude pruning (single-step and iterative), SNIP single-shot
connection-sensitivity pruning, GraSP gradient-signal pruning, Wanda
activation-aware pruning, LAMP layer-adaptive magnitude pruning,
Movement pruning (|W_t - W_0|), SynFlow synaptic-flow pruning (|W| * |dR/dW|),
Optimal Brain Damage (OBD) second-order saliency ((1/2)*H_ii*w^2),
Optimal Brain Surgeon (OBS) inverse-Hessian saliency (w^2/(2*H^{-1}_ii)),
Zhu & Gupta gradual / polynomial magnitude pruning,
N:M semi-structured sparsity (e.g. 2:4 for sparse tensor cores),
structured channel/filter pruning,
first-order Taylor (soft-filter) channel pruning, per-layer and
per-channel survival reporting, sparse-mask helpers, and a tiny CLI.
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

## Gradual magnitude pruning (Zhu & Gupta)

Instead of pruning once, Zhu & Gupta's gradual schedule raises the target
sparsity smoothly during training with a cubic (default) polynomial:

```
s_t = s_f + (s_i - s_f) * (1 - (t - t0) / (t_f - t0))^3
```

`gradual_magnitude_prune_model` simulates that schedule (and optional
optimizer `update_fn` between events). Masks are monotone: once a weight is
pruned it stays zero. Without an `update_fn` the final mask equals one-shot
magnitude pruning at `density = 1 - final_sparsity`.

```python
from prune_kit import gradual_magnitude_prune_model, polynomial_sparsity_schedule

model = {"fc1": [0.1 * i for i in range(20)], "fc2": [0.05 * i for i in range(10)]}
result = gradual_magnitude_prune_model(
    model,
    final_sparsity=0.5,
    begin_step=0,
    end_step=100,
    frequency=20,
    scope="layer",
)
print(result.final_density(), result.target_curve())
print(polynomial_sparsity_schedule(final_sparsity=0.5, end_step=100, frequency=20)[:3])
```

CLI:

```bash
prune-kit gradual --specs 'fc1=dense:4x5,fc2=dense:2x5' \
  --weights $(python -c 'print(",".join(str(0.1*i) for i in range(30)))') \
  --final-sparsity 0.5 --end-step 100 --frequency 20 --json
```

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

## SNIP (connection sensitivity)

SNIP (Lee, Ajanthan, and Torr, ICLR 2019) prunes in one shot, before
training, by scoring each scalar connection with

```text
s = |weight * grad|
```

That is `|dL/dc|` at the auxiliary indicator `c = 1`, usually from a
single minibatch at initialization. The caller supplies the gradient
(or a precomputed `weight * grad` buffer as `contributions`). The
lowest scores are zeroed. `density` is the fraction of connections to
keep; the kept count is `round(density * N)`, so the target sparsity
is `1 - density` up to that rounding.

`scope="global"` (the paper, and the default) ranks every dense and
conv connection in the model together, so a layer of salient small
weights can be kept while a layer of large but insensitive weights is
removed. `scope="layer"` keeps `density` inside each layer, and
`per_layer` can override that density. Ties break toward the earlier
layer in `specs`, then the lower index. A weight that is already zero
stays zero.

Conv kernels use the same flat C-contiguous layout as the rest of the
kit. The summary's `masks` are 0/1 buffers from `sparse_mask_to_dense`;
`kept_weights` counts mask ones, which can include a connection whose
stored value was already zero.

```python
from prune_kit import dense_layer, conv_layer, snip_prune_model, snip_prune_summary

specs = [
    dense_layer("fc1", in_features=2, out_features=2),
    conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1),
]
weights = {
    "fc1": [10.0, 10.0, 0.0, 0.2],
    "conv1": [0.1, 0.1],
}
grads = {
    "fc1": [0.0, 0.0, 5.0, 1.0],
    "conv1": [4.0, 4.0],
}
pruned = snip_prune_model(specs, weights, grads, density=0.5)
summary = snip_prune_summary(specs, weights, grads, density=0.5)
print(summary["total_kept"], summary["sparsity"])
```


## GraSP (gradient signal preservation)

Lottery-ticket IMP, SNIP, and Taylor already cover magnitude rewind,
connection sensitivity, and first-order channel scores. GraSP (Wang,
Zhang, Grosse) is the Hessian-aware counterpart of SNIP: each connection
scores as ``-w * (Hg)`` where ``Hg`` is a Hessian-vector product with the
loss gradient, and the highest scores are kept so the pruned network
preserves gradient flow at initialization.

```python
from prune_kit import dense_layer, grasp_prune_model, grasp_scores

specs = [dense_layer("fc1", in_features=4, out_features=2)]
model = {"fc1": [0.5, -0.2, 0.1, 0.8, -0.4, 0.3, 0.9, -0.1]}
hg = {"fc1": [0.1, -0.3, 0.2, 0.05, -0.1, 0.4, -0.2, 0.15]}
print(grasp_scores(model["fc1"], hg["fc1"]))
pruned = grasp_prune_model(specs, model, hg, density=0.5, scope="global")
```

```bash
python -m prune_kit.cli grasp \
  --specs 'fc1=dense:2x4' \
  --weights '0.5,-0.2,0.1,0.8,-0.4,0.3,0.9,-0.1' \
  --hg '0.1,-0.3,0.2,0.05,-0.1,0.4,-0.2,0.15' \
  --density 0.5 --scope global
```


## Wanda (activation-aware pruning)

SNIP and GraSP score connections from gradients (and Hessians). Wanda
(Sun et al., *A Simple and Effective Pruning Approach for Large Language
Models*) needs only a short calibration set of **input activations**.
Each weight is scored by

```text
S_ij = |W_ij| * ||X_j||_2
```

where `||X_j||_2` is the L2 norm of input feature / channel `j` across
the calibration batch. The lowest scores are zeroed with the same
magnitude-style keep count used elsewhere (`round(density * N)`).
`density` is the fraction to keep. Default `scope="layer"` matches the
paper's per-layer sparsity; `scope="global"` ranks every connection
together. Optional `structure="filter"` / `"channel"` aggregates Wanda
scores per output filter or input channel and zeros whole groups
(structured Wanda).

Dense layers take column norms of length `in_features`. Conv kernels
take per-input-channel norms of length `in_channels` (broadcast across
`kh, kw`). Pass precomputed `|W| * ||X||` buffers as `contributions`.
`activation_column_norms` builds the L2 column norms from a flat
row-major calibration matrix.

```python
from prune_kit import (
    dense_layer, activation_column_norms,
    wanda_scores, wanda_prune_model, wanda_prune_summary,
)

specs = [dense_layer("fc1", in_features=4, out_features=2)]
model = {"fc1": [0.5, -0.2, 0.1, 0.8, -0.4, 0.3, 0.9, -0.1]}
# 3 calibration samples x 4 features
activations = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 2.0, 0.0, 0.0,
    0.0, 0.0, 3.0, 4.0,
]
norms = {"fc1": activation_column_norms(activations, 4)}
print(wanda_scores(model["fc1"], norms["fc1"], shape=(2, 4)))
pruned = wanda_prune_model(specs, model, norms, density=0.5, scope="layer")
summary = wanda_prune_summary(specs, model, norms, density=0.5)
print(summary["total_kept"], summary["sparsity"])
```

```bash
python -m prune_kit.cli wanda \
  --specs 'fc1=dense:2x4' \
  --weights '0.5,-0.2,0.1,0.8,-0.4,0.3,0.9,-0.1' \
  --activation-norms '1,2,3,4' \
  --density 0.5 --scope layer
```


## N:M semi-structured sparsity (2:4)

Sparse tensor cores (NVIDIA Ampere and later) accelerate layers whose
weights keep at most **N non-zeros in every group of M consecutive
weights** along the input dimension, most commonly 2:4. `nm_prune_*`
groups dense `(out, in)` weights by consecutive input columns and conv
`(out, in, kh, kw)` weights by consecutive input channels at each
`(out, kh, kw)` position, then keeps the N highest scores per group.
Scores default to `|w|`; pass any per-weight importance via `scores`
(e.g. Wanda scores for "Wanda 2:4"). `skip` leaves layers dense and
`is_nm_sparse` checks compliance.

```python
from prune_kit import dense_layer, is_nm_sparse, nm_prune_layer, nm_prune_summary

nm_prune_layer([0.1, -0.9, 0.3, 0.2, 5.0, 0.0, -4.0, 1.0], n=2, m=4)
# [0.0, -0.9, 0.3, 0.0, 5.0, 0.0, -4.0, 0.0]

specs = [dense_layer("fc1", in_features=8, out_features=2)]
summary = nm_prune_summary(specs, {"fc1": [float(i) for i in range(16)]}, n=2, m=4)
print(summary["overall_survival"])  # 0.5
```

CLI: `prune-kit nm --n 2 --m 4 --specs fc1=dense:8x2 --weights ... [--skip fc_out] [--json]`.

## ER / ERK layer-wise sparsity (RigL / SET)

Sparse-training methods such as SET and RigL (Evci et al., 2020) give
each layer its own density instead of a uniform one. **Erdős–Rényi**
scales a layer's density with `(n_out + n_in) / (n_out * n_in)`;
**Erdős–Rényi-Kernel** uses `sum(shape) / prod(shape)`, which also counts
conv kernel dims. One scale factor makes the parameter-weighted mean
density equal the requested global `density`. Any layer that would go
above 1 becomes fully dense and the leftover budget is spread over the
rest (the RigL procedure). `erk_power_scale=0` gives a uniform
allocation, `dense_layers` keeps named layers dense, and
`method="uniform"` is available as a baseline. `erk_prune_model` then
magnitude-prunes each layer to its allocated density.

```python
from prune_kit import conv_layer, dense_layer, erk_densities, erk_prune_summary

specs = [
    conv_layer("conv1", 16, 3, 3, 3),
    conv_layer("conv2", 32, 16, 3, 3),
    dense_layer("fc1", in_features=512, out_features=128),
    dense_layer("fc2", in_features=128, out_features=10),
]
print(erk_densities(specs, density=0.1))
# {'conv1': 0.4852, 'conv2': 0.0983, 'fc1': 0.0819, 'fc2': 0.904}  (rounded)
```

CLI: `prune-kit erk --specs conv1=conv:16x3x3x3,fc=dense:64x10 --weights ... --density 0.1 [--method er|uniform] [--dense conv1] [--json]`.

## LAMP (layer-adaptive magnitude pruning)

Uniform per-layer magnitude pruning forces every layer to the same
density. LAMP (Lee et al., *Layer-adaptive sparsity for the
Magnitude-based Pruning*) scores each weight with a **layer-local**
normalization so a single global ranking automatically gives different
layers different sparsities. The classic score sorts the layer by
ascending squared magnitude and sets

```text
score(W_[i]) = W_[i]^2 / sum_{j >= i} W_[j]^2
```

The practical alternative `mode="frobenius"` uses `|w| / ||W||_F`.
`density` is the fraction to keep (`round(density * N)`). Default
`scope="global"` matches the paper; `scope="layer"` keeps the same
fraction inside each layer. Dense and conv flat buffers are both
supported.

```python
from prune_kit import (
    dense_layer, lamp_scores, lamp_prune_model, lamp_prune_summary,
)

specs = [
    dense_layer("small", in_features=4, out_features=1),
    dense_layer("large", in_features=4, out_features=1),
]
model = {
    "small": [0.1, 0.2, 0.3, 0.4],
    "large": [1.0, 2.0, 3.0, 4.0],
}
print(lamp_scores(model["small"], mode="lamp"))
pruned = lamp_prune_model(specs, model, density=0.5, scope="global")
summary = lamp_prune_summary(specs, model, density=0.5, mode="frobenius")
print(summary["total_kept"], summary["mode"])
```

```bash
python -m prune_kit.cli lamp \
  --specs 'small=dense:1x4,large=dense:1x4' \
  --weights '0.1,0.2,0.3,0.4,1,2,3,4' \
  --density 0.5 --scope global --mode lamp
```

## Movement pruning (|W_t - W_0|)

Magnitude pruning ranks by |W|. Movement pruning (Sanh et al., *Movement
Pruning: Adaptive Sparsity by Fine-Tuning*) ranks by how far each weight
has moved from its initialization during fine-tuning:

```text
score_i = |W_t[i] - W_0[i]|
```

Connections with the **lowest** scores are pruned first (keep largest
movement). Pass `initial_weights` (`W_0`) or precomputed `movements`
(absolute cumulative movement). `density` is the fraction to keep
(`round(density * N)`). Default `scope="global"` ranks every score
together; `scope="layer"` keeps the same fraction inside each layer.
Dense and conv flat buffers are both supported.

```python
from prune_kit import (
    dense_layer, movement_scores, movement_prune_model, movement_prune_summary,
)

specs = [dense_layer("fc", in_features=4, out_features=1)]
model = {"fc": [1.0, 2.0, 3.0, 4.0]}
initial = {"fc": [0.5, 1.5, 2.5, 0.0]}
print(movement_scores(model["fc"], initial["fc"]))
pruned = movement_prune_model(specs, model, initial, density=0.5)
summary = movement_prune_summary(specs, model, initial, density=0.5)
print(summary["total_kept"], summary["overall_survival"])
```

```bash
python -m prune_kit.cli movement \
  --specs 'fc=dense:1x4' \
  --weights '1,2,3,4' \
  --initial-weights '0.5,1.5,2.5,0' \
  --density 0.5 --scope global
```


## SynFlow (synaptic flow)

SynFlow (Tanaka et al., *Pruning Neural Networks Without Any Data by
Iteratively Conserving Synaptic Flow*) is a **data-free** one-shot
pruner. After linearizing parameters (absolute values; identity
activations), each connection is scored by the product of its magnitude
and the gradient of the synaptic-flow objective ``R`` on unit inputs:

```text
score = |W| * |∂R/∂W|
```

Connections with the **lowest** scores are pruned first. Pass ``grads``
(``∂R/∂W``) or precomputed ``contributions``. When both are omitted, the
kit uses linearized multi-layer unit-input SynFlow when layer shapes
chain, otherwise the common single-layer **exponential** proxy
``|W_ij| * exp(∑_k |W_ik|)``. ``density`` is the fraction to keep
(``round(density * N)``). Default ``scope="global"``; dense and conv
flat buffers are both supported.

```python
from prune_kit import (
    dense_layer, synflow_scores, synflow_prune_model, synflow_prune_summary,
)

specs = [dense_layer("fc", in_features=4, out_features=1)]
model = {"fc": [1.0, 2.0, 3.0, 4.0]}
print(synflow_scores(model["fc"], shape=(1, 4)))  # exponential proxy
pruned = synflow_prune_model(specs, model, density=0.5)
summary = synflow_prune_summary(specs, model, density=0.5)
print(summary["total_kept"], summary["overall_survival"])
```

```bash
python -m prune_kit.cli synflow \
  --specs 'fc=dense:1x4' \
  --weights '1,2,3,4' \
  --density 0.5 --scope global
```


## Optimal Brain Damage (OBD)

Optimal Brain Damage (LeCun, Denker, Solla, *Optimal Brain Damage*) is a
classic **second-order** one-shot pruner. Each connection is scored by
how much the loss would change if that weight were set to zero, using
the diagonal of the Hessian:

```text
saliency_i ≈ (1/2) * H_ii * w_i^2
```

Connections with the **lowest** saliency are pruned first. Pass
`hess_diag` (precomputed diagonal Hessian `H_ii`) for the exact OBD
form, or `grads` to use the common **squared-gradient** diagonal proxy
`H_ii ≈ g_i^2` (so saliency becomes `(1/2) * g_i^2 * w_i^2`) when a
true Hessian is unavailable. Pass exactly one. `density` is the
fraction to keep (`round(density * N)`). Default `scope="global"`;
dense and conv flat buffers are both supported.

```python
from prune_kit import (
    dense_layer, obd_scores, obd_prune_model, obd_prune_summary,
)

specs = [dense_layer("fc", in_features=4, out_features=1)]
model = {"fc": [1.0, 2.0, 3.0, 4.0]}
hess = {"fc": [1.0, 1.0, 0.25, 0.25]}
print(obd_scores(model["fc"], hess["fc"]))
pruned = obd_prune_model(specs, model, hess, density=0.5)
summary = obd_prune_summary(specs, model, hess, density=0.5)
print(summary["total_kept"], summary["overall_survival"])
```

```bash
python -m prune_kit.cli obd \
  --specs 'fc=dense:1x4' \
  --weights '1,2,3,4' \
  --hess-diag '1,1,0.25,0.25' \
  --density 0.5 --scope global
```

## Optimal Brain Surgeon (OBS)

Optimal Brain Surgeon (Hassibi & Stork, *Optimal Brain Surgeon*) is the
classic **inverse-Hessian** follow-up to OBD. Each connection is scored by

```text
saliency_i ≈ w_i^2 / (2 * H^{-1}_ii)
```

Connections with the **lowest** saliency are pruned first. This kit
implements the saliency ranking / one-shot mask only (not the optimal
weight update of full OBS). Pass **exactly one** of:

* `hess_inv_diag` — diagonal of `H^{-1}` (**exact OBS** formula).
* `hess_diag` — diagonal of `H`, using the reciprocal proxy
  `H^{-1}_ii ≈ 1/(H_ii + eps)` (documented proxy; equals OBD when `H`
  is diagonal).
* `grads` — squared-gradient proxy `H_ii ≈ g_i^2` then
  `H^{-1}_ii ≈ 1/(g_i^2 + eps)` (practical stand-in, not exact OBS).

`density` is the fraction to keep (`round(density * N)`). Default
`scope="global"`; dense and conv flat buffers are both supported.

```python
from prune_kit import (
    dense_layer, obs_scores, obs_prune_model, obs_prune_summary,
)

specs = [dense_layer("fc", in_features=4, out_features=1)]
model = {"fc": [1.0, 2.0, 3.0, 4.0]}
# Exact OBS with inverse-Hessian diagonal (here H = I => H^{-1} = I)
hinv = {"fc": [1.0, 1.0, 1.0, 1.0]}
print(obs_scores(model["fc"], hinv["fc"]))
pruned = obs_prune_model(specs, model, hinv, density=0.5)
summary = obs_prune_summary(specs, model, hinv, density=0.5)
print(summary["total_kept"], summary["overall_survival"])
```

```bash
python -m prune_kit.cli obs \
  --specs 'fc=dense:1x4' \
  --weights '1,2,3,4' \
  --hess-inv-diag '1,1,1,1' \
  --density 0.5 --scope global
```

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
prune-kit snip --help
prune-kit grasp --help
prune-kit wanda --help
prune-kit lamp --help
prune-kit movement --help
prune-kit synflow --help
prune-kit obd --help
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
aligned with `--weights`. The `snip` command scores every dense and
conv connection by `|grad * weight|` and prints per-layer keep counts.
`--scope global` (default) ranks the whole model; `--scope layer`
keeps `--density` inside each layer. The `grasp` command scores connections by `-w*(Hg)`. The `wanda` command scores by `|W|*||X||_2` using `--activation-norms` (column norms for a single layer, or a per-weight buffer); optional `--structure filter|channel` aggregates scores into structured groups. The `lamp` command scores by classic LAMP (`W^2 / sum_{j>=i} W^2`) or `|w|/||W||_F` (`--mode lamp|frobenius`); `--scope global` (default) ranks every score together for layer-adaptive sparsity. The `movement` command scores by `|W_t - W_0|` using `--initial-weights` aligned with `--weights`; lowest movement is pruned first. The `synflow` command scores by `|W| * |dR/dW|` (optional `--grads`, else data-free unit-input / exponential proxy). The `obd` command scores by `(1/2)*H_ii*w^2` using `--hess-diag` or squared-gradient proxy via `--grads`. The `global` command prunes lowest-|w| weights to `--sparsity`
(optionally over `--rounds` or an explicit `--schedule`) and prints
the density after each round plus a per-layer kept-count table.
`--rewind` resets survivors to `--initial-weights`.

## Tests

```bash
pytest tests
```