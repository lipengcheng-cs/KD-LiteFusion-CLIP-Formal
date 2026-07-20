import torch
import torch.nn as nn

from mkan_refine.paper_reproduction_v2.kan import BSplineKANLinear


def test_forward_shape_finite_and_real_spline_parameters():
    layer = BSplineKANLinear(4, 3, grid_size=5, spline_order=3)
    x = torch.randn(2, 7, 4)
    output = layer(x)
    assert output.shape == (2, 7, 3)
    assert torch.isfinite(output).all()
    assert layer.spline_weight.ndim == 3
    assert layer.spline_weight.shape == (3, 4, 8)
    assert layer.spline_weight.numel() != nn.Linear(4, 3).weight.numel()


def test_spline_branch_has_non_affine_response():
    torch.manual_seed(7)
    layer = BSplineKANLinear(1, 1, grid_size=5, spline_order=3, normalize_input=False, bias=False)
    with torch.no_grad():
        layer.base_weight.zero_()
        layer.spline_weight.normal_(0.0, 0.5)
        layer.spline_scaler.fill_(1.0)
    x = torch.tensor([[-0.6], [0.0], [0.6]])
    y = layer(x).squeeze(-1)
    second_difference = y[0] - 2 * y[1] + y[2]
    assert second_difference.abs() > 1e-4

