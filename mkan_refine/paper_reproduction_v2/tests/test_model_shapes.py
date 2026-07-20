import torch

from mkan_refine.paper_reproduction_v2.model import MKANPaperHeadV2, SplineConfig


def test_paper_head_shapes_and_dual_stream_outputs():
    torch.manual_seed(23)
    model = MKANPaperHeadV2(
        dim=8,
        num_classes=5,
        classifier_hidden=6,
        dropout=0.0,
        spline=SplineConfig(grid_size=3, spline_order=2),
    )
    output = model(
        vision_tokens=torch.randn(2, 10, 8),
        text_tokens=torch.randn(2, 7, 8),
        vision_global=torch.randn(2, 8),
        text_global=torch.randn(2, 8),
    )
    assert output["logits"].shape == (2, 5)
    assert output["feature"].shape == (2, 8)
    assert output["gate"].shape == (2, 8)
    assert output["vision_attention"].shape == (2, 10)
    assert output["text_attention"].shape == (2, 7)
    assert torch.allclose(output["vision_attention"].sum(1), torch.ones(2), atol=1e-5)
    assert torch.allclose(output["text_attention"].sum(1), torch.ones(2), atol=1e-5)
    assert all(torch.isfinite(value).all() for value in output.values())
