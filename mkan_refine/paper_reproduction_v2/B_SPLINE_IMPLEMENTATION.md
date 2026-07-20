# B-spline KAN implementation

## Formula mapping

`BSplineKANLinear` implements the paper's Equations 6–10 as an edge-wise function:

`phi_ji(x_i) = w_base[j,i] * SiLU(x_i) + w_spline[j,i] * sum_k c[j,i,k] B_k(x_i)`

and sums over input edges to produce each output channel. The implementation contains two genuinely different branches:

- `base_weight [out,in]` operating on SiLU input;
- `spline_weight [out,in,grid_size+spline_order]` operating on recursively evaluated B-spline bases.

It does not implement the spline branch as another ordinary `nn.Linear` on SiLU input.

## Tensors and defaults

- input: `[..., in_features]`;
- normalized input: FP32 affine LayerNorm followed by a `tanh` map into the configured grid range by default;
- grid buffer: `[in_features, grid_size + 2*spline_order + 1]`;
- basis: `[..., in_features, grid_size+spline_order]`;
- base weight: `[out_features,in_features]`;
- spline coefficients: `[out_features,in_features,grid_size+spline_order]`;
- standalone spline scaler: `[out_features,in_features]`;
- output: `[...,out_features]`.

Baseline assumptions are `grid_size=5`, cubic `spline_order=3`, and grid range `[-1,1]`. The paper establishes B-spline functions and adaptive grids but the currently available text does not fix those hyperparameters; they are validation-search assumptions.

## Basis evaluation and numerical stability

- Degree-zero interval indicators are elevated recursively with the Cox–de Boor relation.
- Denominators are lower-bounded by FP32 epsilon.
- Right boundaries are moved inside the final half-open interval by one epsilon.
- The bounded normalization prevents large LayerNorm activations from falling outside every knot interval and silently zeroing the spline branch.
- KAN math is explicitly performed in FP32 even if upstream cached features are FP16.
- Basis, fitted coefficients, adaptive grids, and outputs are checked for NaN/Inf.
- Invalid dimensions, insufficient update samples, or non-monotonic grids fail loudly.

## Initialization

The base branch uses scaled Kaiming initialization. The spline branch initializes actual local spline coefficients with dimension-scaled small Gaussian noise. This is the stable direct-coefficient equivalent of fitting a small random curve and avoids a prohibitively large batched least-squares solve for the 1536→768 gate. `curve2coeff()` remains implemented and is used when an adaptive grid update must preserve an existing learned curve.

## Adaptive grid update

`update_grid()`:

1. records the existing per-edge spline curve on current activations;
2. creates quantile-based adaptive interior knots;
3. blends adaptive and uniform knots using `grid_eps`;
4. extends boundary knots for the configured spline order;
5. refits coefficients by least squares to preserve the pre-update curve;
6. verifies strict monotonicity and finite values.

Grid updates are optional and must be scheduled explicitly. The 30-epoch baseline uses no update until the static implementation and gradients have been validated; later frequencies are bounded validation-only search choices.

## Model integration

`MKANPaperHeadV2` uses B-spline KAN for:

- text-global context projection;
- visual-global context projection;
- visual token nonlinear energy scoring;
- symmetric text token nonlinear energy scoring;
- 1536→768 feature-level reliability gate;
- 768→512 and 512→5 classifier layers.

The gate is a 768-dimensional vector and fuses `(1-lambda)*vision + lambda*text`. Both text→vision and vision→text refinement paths are executed.

## Regularization

Each layer exposes an activation-magnitude plus entropy regularizer over effective spline edge weights. The model aggregates it through `regularization_loss()`. Its coefficient defaults to zero in the baseline unless the paper/code evidence or validation-only search justifies a nonzero value.

## Tests

- `test_kan_basis.py`: basis shape, non-negativity, partition behavior, and bounded normalization under extreme inputs;
- `test_kan_forward.py`: output shape, finite values, true spline tensor, nonlinear response;
- `test_kan_gradient.py`: finite nonzero spline gradients and coefficient update;
- `test_grid_update.py`: finite, monotonic, approximately curve-preserving adaptive update;
- `test_model_shapes.py`: full dual-stream attention, gate, feature, and logits shapes.
