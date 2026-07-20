import torch

from mkan_refine.paper_reproduction_v2.kan import BSplineKANLinear


def test_grid_update_is_finite_monotonic_and_curve_preserving():
    torch.manual_seed(19)
    layer = BSplineKANLinear(3, 2, grid_size=5, spline_order=3, normalize_input=False)
    x = torch.randn(256, 3).clamp(-1.4, 1.4)
    with torch.no_grad():
        before = layer(x)
        old_grid = layer.grid.clone()
        report = layer.update_grid(x)
        after = layer(x)
    assert report["samples"] == 256
    assert not torch.allclose(old_grid, layer.grid)
    assert torch.all(layer.grid[:, 1:] > layer.grid[:, :-1])
    assert torch.isfinite(after).all()
    assert torch.mean(torch.abs(before - after)) < 0.05

