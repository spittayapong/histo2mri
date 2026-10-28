"""grid_sample_2d_det must match F.grid_sample (align_corners=True) and be deterministic on CUDA."""
import pytest
import torch
import torch.nn.functional as F

from histo2mri.sampling import grid_sample_2d_det

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("padding_mode", ["border", "zeros"])
@pytest.mark.parametrize("shape", [(37, 53, 40, 31), (120, 90, 64, 80)])
def test_matches_grid_sample(device, padding_mode, shape):
    H, W, Ho, Wo = shape
    gen = torch.Generator().manual_seed(0)
    inp = torch.randn(1, 2, H, W, generator=gen).to(device)
    grid = (torch.rand(1, Ho, Wo, 2, generator=gen) * 2.4 - 1.2).to(device)   # includes out-of-range
    weights = torch.randn(1, 2, Ho, Wo, generator=gen).to(device)

    results = []
    for fn in (lambda a, g: F.grid_sample(a, g, mode="bilinear", padding_mode=padding_mode, align_corners=True),
               lambda a, g: grid_sample_2d_det(a, g, padding_mode=padding_mode)):
        a = inp.clone().requires_grad_(True)
        g = grid.clone().requires_grad_(True)
        out = fn(a, g)
        (out * weights).sum().backward()
        results.append((out.detach(), a.grad, g.grad))

    (out_ref, ga_ref, gg_ref), (out_det, ga_det, gg_det) = results
    torch.testing.assert_close(out_det, out_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ga_det, ga_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gg_det, gg_ref, rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_deterministic_on_cuda():
    gen = torch.Generator().manual_seed(1)
    inp = torch.randn(1, 2, 300, 400, generator=gen).cuda()
    grid = (torch.rand(1, 300, 400, 2, generator=gen) * 2 - 1).cuda()
    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        grads = []
        for _ in range(3):
            a = inp.clone().requires_grad_(True)
            g = grid.clone().requires_grad_(True)
            (grid_sample_2d_det(a, g) ** 2).sum().backward()
            grads.append((a.grad.clone(), g.grad.clone()))
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
    for ga, gg in grads[1:]:
        assert torch.equal(ga, grads[0][0]) and torch.equal(gg, grads[0][1])
