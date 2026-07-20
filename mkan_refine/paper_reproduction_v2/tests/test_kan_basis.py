import torch

from mkan_refine.paper_reproduction_v2.kan import BSplineKANLinear


def test_bspline_basis_shape_and_partition():
    layer = BSplineKANLinear(4, 3, grid_size=5, spline_order=3, normalize_input=False)
    x = torch.linspace(-0.95, 0.95, 41).unsqueeze(1).repeat(1, 4)
    basis = layer.b_splines(x)
    assert basis.shape == (41, 4, 8)
    assert torch.isfinite(basis).all()
    assert torch.allclose(basis.sum(dim=-1), torch.ones(41, 4), atol=2e-5, rtol=2e-5)
    assert (basis >= 0).all()


def test_normalized_inputs_remain_inside_spline_support():
    layer = BSplineKANLinear(4, 3, grid_size=5, spline_order=3, normalize_input=True)
    x = torch.tensor([[1e6, -1e6, 1e3, -1e3], [-9e5, 8e5, -7e4, 6e4]])
    basis = layer.b_splines(x)
    assert torch.isfinite(basis).all()
    assert torch.allclose(basis.sum(dim=-1), torch.ones(2, 4), atol=2e-5, rtol=2e-5)
