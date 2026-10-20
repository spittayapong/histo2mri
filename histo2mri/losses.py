"""Image similarity losses (SSIM, NGF, MIND) and surface rigidity terms for the refinement."""
import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel(win, sigma, device, dtype):
    g = torch.arange(win, device=device, dtype=dtype) - win // 2
    g = torch.exp(-0.5 * (g / sigma) ** 2)
    g = (g / g.sum()).view(1, 1, win)
    k = g.transpose(1, 2) @ g
    return k


def _sobel_kernels(device, dtype):
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return sx, sy


def _norm01_t(x):
    return (x - x.amin(dim=(-2, -1), keepdim=True)) / (
        x.amax(dim=(-2, -1), keepdim=True) - x.amin(dim=(-2, -1), keepdim=True) + 1e-8
    )


def combined_masked_ssim_ngf_loss(warped, hist, mask, w_ssim=0.5, w_ngf=0.5, win=11, sigma=1.5, eps=1e-8):
    """
    SSIM+NGF loss, masked consistently: both terms' final reduction is
    restricted to mask==1 pixels. The previous version zeroed the images
    before a plain SSIM .mean(), which diluted the loss with trivial
    zero-zero background matches wherever the mask was 0 -- this computes
    SSIM on the real (unmasked) images and only restricts the final
    average, matching how the NGF term is already masked.
    """
    m = (mask > 0.1).float()
    w = _norm01_t(warped)
    h = _norm01_t(hist)

    N, C, H, W = w.shape
    k = _gaussian_kernel(win, sigma, w.device, w.dtype).repeat(C, 1, 1, 1)

    mu1 = F.conv2d(w, k, padding=win // 2, groups=C)
    mu2 = F.conv2d(h, k, padding=win // 2, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(w * w, k, padding=win // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(h * h, k, padding=win // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(w * h, k, padding=win // 2, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps
    )
    L_ssim = 1.0 - (ssim_map * m).sum() / (m.sum() + eps)

    sx, sy = _sobel_kernels(w.device, w.dtype)
    wgx = F.conv2d(w, sx, padding=1); wgy = F.conv2d(w, sy, padding=1)
    hgx = F.conv2d(h, sx, padding=1); hgy = F.conv2d(h, sy, padding=1)
    dot = wgx * hgx + wgy * hgy
    n1 = torch.sqrt(wgx * wgx + wgy * wgy + eps)
    n2 = torch.sqrt(hgx * hgx + hgy * hgy + eps)
    ngf2 = (dot * dot) / (n1 * n2 + eps)
    L_ngf = ((1.0 - ngf2) * m).sum() / (m.sum() + eps)

    loss = w_ssim * L_ssim + w_ngf * L_ngf
    return loss, {"ssim": float(L_ssim.detach().cpu()), "ngf": float(L_ngf.detach().cpu())}


# ------------------------------------------------------------
# SSIM + MIND alternative (MIND = Modality Independent Neighborhood
# Descriptor). Compares local self-similarity patterns rather than raw
# gradient direction, so it's less sensitive to histology/MRI contrast-
# polarity mismatches than NGF. Kept alongside combined_masked_ssim_ngf_loss
# (not replacing it) so either can be selected for testing.
# ------------------------------------------------------------
def _shift2d(x, dy, dx):
    """Shift x by (dy,dx) with replicate padding. x: (N,C,H,W)"""
    N, C, H, W = x.shape
    pad_t, pad_b = max(dy, 0), max(-dy, 0)
    pad_l, pad_r = max(dx, 0), max(-dx, 0)
    xpad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    y0 = pad_t - dy
    x0 = pad_l - dx
    return xpad[:, :, y0:y0 + H, x0:x0 + W]


def mind_descriptor_2d(img, radius=1, dilation=1, sigma=0.8, eps=1e-8):
    """
    MIND-SSC-like descriptor for 2D images. img: (N,1,H,W).
    Returns (N,8,H,W) -- one channel per 8-neighborhood offset.
    """
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    offs = [(dy * dilation, dx * dilation) for dy, dx in offs]

    ssd_list = []
    for dy, dx in offs:
        shifted = _shift2d(img, dy, dx)
        diff2 = (img - shifted).pow(2)
        k = 2 * radius + 1
        diff2 = F.avg_pool2d(diff2, kernel_size=k, stride=1, padding=radius)
        ssd_list.append(diff2)
    ssd = torch.cat(ssd_list, dim=1)  # (N,8,H,W)

    v = torch.min(ssd, dim=1, keepdim=True).values + eps
    desc = torch.exp(-(ssd / v))

    if sigma is not None and sigma > 0:
        win = int(max(3, 2 * round(3 * sigma) + 1))
        g = torch.arange(win, device=img.device, dtype=img.dtype) - win // 2
        g = torch.exp(-0.5 * (g / sigma) ** 2)
        g = (g / (g.sum() + eps)).view(1, 1, win, 1)
        desc = F.conv2d(desc, g.repeat(desc.shape[1], 1, 1, 1), padding=(win // 2, 0), groups=desc.shape[1])
        desc = F.conv2d(desc, g.transpose(2, 3).repeat(desc.shape[1], 1, 1, 1), padding=(0, win // 2), groups=desc.shape[1])

    desc = desc / (desc.mean(dim=1, keepdim=True) + eps)
    return desc


def masked_mind_loss(img1, img2, mask, radius=1, dilation=1, sigma=0.8, eps=1e-8):
    d1 = mind_descriptor_2d(img1, radius=radius, dilation=dilation, sigma=sigma, eps=eps)
    d2 = mind_descriptor_2d(img2, radius=radius, dilation=dilation, sigma=sigma, eps=eps)
    diff = (d1 - d2).abs().mean(dim=1, keepdim=True)  # (N,1,H,W)
    return (diff * mask).sum() / (mask.sum() + eps)


def combined_masked_ssim_mind_loss(warped, hist, mask, w_ssim=0.9, w_mind=0.1,
                                    mind_radius=1, mind_dilation=1, mind_sigma=0.8,
                                    win=11, sigma=1.5, eps=1e-8):
    m = (mask > 0.1).float()
    w = _norm01_t(warped)
    h = _norm01_t(hist)

    N, C, H, W = w.shape
    k = _gaussian_kernel(win, sigma, w.device, w.dtype).repeat(C, 1, 1, 1)

    mu1 = F.conv2d(w, k, padding=win // 2, groups=C)
    mu2 = F.conv2d(h, k, padding=win // 2, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(w * w, k, padding=win // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(h * h, k, padding=win // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(w * h, k, padding=win // 2, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps
    )
    L_ssim = 1.0 - (ssim_map * m).sum() / (m.sum() + eps)

    L_mind = masked_mind_loss(w, h, m, radius=mind_radius, dilation=mind_dilation, sigma=mind_sigma, eps=eps)

    loss = w_ssim * L_ssim + w_mind * L_mind
    return loss, {"ssim": float(L_ssim.detach().cpu()), "mind": float(L_mind.detach().cpu())}


def _central_diff_su_sv(S):
    """Central differences along image axes; one-sided at the borders.

    S: (H,W,C) -> (Su, Sv), each (H,W,C).  Su is d/dx (columns), Sv is d/dy (rows).
    Same stencil as the fold check, so the two agree by construction.
    """
    Su = torch.zeros_like(S)
    Sv = torch.zeros_like(S)
    Su[:, 1:-1] = (S[:, 2:] - S[:, :-2]) * 0.5
    Su[:, 0] = S[:, 1] - S[:, 0]
    Su[:, -1] = S[:, -1] - S[:, -2]
    Sv[1:-1, :] = (S[2:, :] - S[:-2, :]) * 0.5
    Sv[0, :] = S[1, :] - S[0, :]
    Sv[-1, :] = S[-1, :] - S[-2, :]
    return Su, Sv


def first_fundamental_form(xyz_norm, dims_t):
    """(E, F, G) of the surface, in VOXEL units.  xyz_norm: (H,W,3) in [0,1]^3."""
    S = xyz_norm * dims_t
    Su, Sv = _central_diff_su_sv(S)
    E = (Su * Su).sum(-1)
    Fq = (Su * Sv).sum(-1)
    G = (Sv * Sv).sum(-1)
    return E, Fq, G


def surface_area_ratio(xyz_norm, xyz_norm_base, dims, eps=1e-12):
    """Local area distortion of `xyz_norm` relative to `xyz_norm_base`.

    Returns a (H,W) numpy array.  1.0 == locally area-preserving.  This is the
    melting metric: it is finite and positive even where the sheet never folds.
    Spacing-independent for isotropic voxels (it is a ratio).
    """
    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    dims = np.asarray(dims, dtype=np.float32).reshape(1, 1, 3)
    a = _np(xyz_norm).astype(np.float32) * dims
    b = _np(xyz_norm_base).astype(np.float32) * dims

    def _area(S):
        Su = np.zeros_like(S); Sv = np.zeros_like(S)
        Su[:, 1:-1] = (S[:, 2:] - S[:, :-2]) * 0.5
        Su[:, 0] = S[:, 1] - S[:, 0]; Su[:, -1] = S[:, -1] - S[:, -2]
        Sv[1:-1, :] = (S[2:, :] - S[:-2, :]) * 0.5
        Sv[0, :] = S[1, :] - S[0, :]; Sv[-1, :] = S[-1, :] - S[-2, :]
        return np.linalg.norm(np.cross(Su, Sv, axis=-1), axis=-1)

    return _area(a) / np.maximum(_area(b), eps)


def _build_rigidity_weight(mode, mask_soft, conf, hist_t, edge_k=2.0, eps=1e-8):
    """Spatial rigidity coefficient w(x) in [0, ~1+edge_k].  All inputs detached.

    "mask" : uniform inside the (soft-dilated) tissue mask.  Start here.
    "conf" : rigid where the base surface has match support, free where it does
             not.  Complements rf_psi, which is gated by (1 - conf).
    "edge" : mask * (1 + edge_k * |grad(hist)| / max).  Pins the cortical ribbon
             and ventricle walls, lets homogeneous white matter move.  Closest
             analogue of "protect the bone" in the rigid-structure literature.
    "none" : uniform 1 everywhere.
    """
    m = mask_soft[0, 0].detach()
    if mode is None or mode == "none":
        return torch.ones_like(m)
    if mode == "mask":
        return m
    if mode == "conf":
        return (m * conf.detach()).clamp(0.0, 1.0)
    if mode == "edge":
        h = hist_t.detach()
        gx = torch.zeros_like(h); gy = torch.zeros_like(h)
        gx[..., :, 1:-1] = (h[..., :, 2:] - h[..., :, :-2]) * 0.5
        gy[..., 1:-1, :] = (h[..., 2:, :] - h[..., :-2, :]) * 0.5
        g = torch.sqrt(gx * gx + gy * gy)[0, 0]
        g = g / (g.max() + eps)
        return m * (1.0 + float(edge_k) * g)
    raise ValueError(f"unknown w_rigid_mode: {mode!r}")

