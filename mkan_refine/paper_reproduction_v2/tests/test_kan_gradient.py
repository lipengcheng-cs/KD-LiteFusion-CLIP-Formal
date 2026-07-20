import torch

from mkan_refine.paper_reproduction_v2.kan import BSplineKANLinear


def test_backward_and_spline_coefficients_update():
    torch.manual_seed(11)
    layer = BSplineKANLinear(3, 2, grid_size=3, spline_order=2)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    before = layer.spline_weight.detach().clone()
    x = torch.randn(32, 3)
    target = torch.randn(32, 2)
    loss = torch.nn.functional.mse_loss(layer(x), target) + 1e-4 * layer.regularization_loss()
    loss.backward()
    assert layer.base_weight.grad is not None and torch.isfinite(layer.base_weight.grad).all()
    assert layer.spline_weight.grad is not None and torch.isfinite(layer.spline_weight.grad).all()
    assert layer.spline_weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.allclose(before, layer.spline_weight.detach())

