"""Image similarity losses and surface-geometry terms for the curved-sheet refinement.

Used by refine.pyramid_refinement. The losses compare the MRI sampled on the current sheet
with the histology on the common canvas: images are (N, C, H, W) tensors (in practice
N = C = 1) and every loss is averaged over mask pixels only. The geometry helpers work on
xyz_norm, an (H, W, 3) grid of sheet positions normalised to [0, 1] per axis in (x, y, z)
order; multiplying by (Xdim-1, Ydim-1, Zdim-1) gives MRI voxel-index coordinates.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel(win, sigma, device, dtype):
    """Return a normalised 2D Gaussian window of size (1, win, win)."""
    g = torch.arange(win, device=device, dtype=dtype) - win // 2
    g = torch.exp(-0.5 * (g / sigma) ** 2)
    g = (g / g.sum()).view(1, 1, win)
    k = g.transpose(1, 2) @ g
    return k


def _norm01_t(x):
    """Min-max normalise x to [0, 1] over its last two (H, W) axes, per sample and channel."""
    return (x - x.amin(dim=(-2, -1), keepdim=True)) / (
        x.amax(dim=(-2, -1), keepdim=True) - x.amin(dim=(-2, -1), keepdim=True) + 1e-8
    )


# ------------------------------------------------------------
# MIND (Modality Independent Neighbourhood Descriptor) compares local
# self-similarity patterns rather than raw intensities or gradient
# directions, so it is less sensitive to histology/MRI contrast inversions.
# ------------------------------------------------------------
def _shift2d(x, dy, dx):
    """Return x (N, C, H, W) shifted so that out[..., i, j] = x[..., i - dy, j - dx], replicating edges."""
    N, C, H, W = x.shape
    pad_t, pad_b = max(dy, 0), max(-dy, 0)
    pad_l, pad_r = max(dx, 0), max(-dx, 0)
    xpad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    y0 = pad_t - dy
    x0 = pad_l - dx
    return xpad[:, :, y0:y0 + H, x0:x0 + W]


def mind_descriptor_2d(img, radius=1, dilation=1, sigma=0.8, eps=1e-8):
    """Compute a MIND-style self-similarity descriptor for a 2D image.

    For each of the 8 neighbour offsets (scaled by `dilation`), the squared difference
    between the image and its shifted copy is box-filtered over a (2*radius+1)^2 patch.
    The patch distances are turned into exp(-d / v), where v is the per-pixel minimum
    distance over the 8 offsets, optionally smoothed with a separable Gaussian, and
    finally divided by their per-pixel mean over the 8 channels.

    Args:
        img: tensor (N, 1, H, W).
        radius: half-size of the box filter for the patch distance, in pixels.
        dilation: spacing of the neighbour offsets, in pixels.
        sigma: standard deviation of the Gaussian smoothing of the descriptor, in
            pixels; None or <= 0 disables it.
        eps: small constant for numerical stability.

    Returns:
        Tensor (N, 8, H, W), one channel per neighbour offset.
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
    """Compute the mask-weighted mean absolute difference between MIND descriptors.

    The absolute descriptor difference is averaged over the 8 channels, multiplied by
    `mask` (used as given, not binarised) and divided by the mask sum.

    Args:
        img1: tensor (N, 1, H, W).
        img2: tensor (N, 1, H, W).
        mask: tensor broadcastable to (N, 1, H, W), weights in [0, 1].
        radius: see mind_descriptor_2d.
        dilation: see mind_descriptor_2d.
        sigma: see mind_descriptor_2d.
        eps: small constant for numerical stability.

    Returns:
        Scalar tensor.
    """
    d1 = mind_descriptor_2d(img1, radius=radius, dilation=dilation, sigma=sigma, eps=eps)
    d2 = mind_descriptor_2d(img2, radius=radius, dilation=dilation, sigma=sigma, eps=eps)
    diff = (d1 - d2).abs().mean(dim=1, keepdim=True)  # (N,1,H,W)
    return (diff * mask).sum() / (mask.sum() + eps)


def combined_masked_ssim_mind_loss(warped, hist, mask, w_ssim=0.9, w_mind=0.1,
                                    mind_radius=1, mind_dilation=1, mind_sigma=0.8,
                                    win=11, sigma=1.5, eps=1e-8):
    """Compute a weighted SSIM + MIND dissimilarity, averaged over mask pixels.

    Both images are min-max normalised to [0, 1]. SSIM is computed on the full,
    unmasked images with a Gaussian window (constants C1 = 0.01^2, C2 = 0.03^2) and
    only the final average is restricted to the mask, so background pixels where both
    images are zero do not inflate the score. The MIND term is masked_mind_loss on the
    normalised images, with the binarised mask (mask > 0.1) as weight. The parameter
    profiles use SSIM only (w_mind = 0) or MIND only (w_ssim = 0).

    Args:
        warped: MRI sampled on the sheet, tensor (N, 1, H, W).
        hist: histology on the common canvas, tensor (N, 1, H, W).
        mask: tensor broadcastable to (N, 1, H, W); pixels with mask > 0.1 count.
        w_ssim: weight of the SSIM term.
        w_mind: weight of the MIND term.
        mind_radius: patch radius for the MIND descriptor, in pixels.
        mind_dilation: neighbour spacing for the MIND descriptor, in pixels.
        mind_sigma: Gaussian smoothing of the MIND descriptor, in pixels.
        win: SSIM window size in pixels.
        sigma: standard deviation of the SSIM Gaussian window, in pixels.
        eps: small constant for numerical stability.

    Returns:
        Tuple (loss, parts): loss is the scalar tensor
        w_ssim * (1 - masked mean SSIM) + w_mind * masked MIND distance;
        parts is a dict {"ssim": float, "mind": float} with the two unweighted terms.
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

    L_mind = masked_mind_loss(w, h, m, radius=mind_radius, dilation=mind_dilation, sigma=mind_sigma, eps=eps)

    loss = w_ssim * L_ssim + w_mind * L_mind
    return loss, {"ssim": float(L_ssim.detach().cpu()), "mind": float(L_mind.detach().cpu())}


def _central_diff_su_sv(S):
    """Return central differences (Su, Sv) of S (H, W, C), one-sided at the borders.

    Su is the derivative along columns (x), Sv along rows (y); both are (H, W, C). The
    stencil is the same as in surface_area_ratio and in the base-normal computation of
    refine.refine_curved_sheet.
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
    """Compute the first fundamental form (E, F, G) of a sheet in voxel units.

    The sheet is converted to voxel-index coordinates (xyz_norm * dims_t) and
    differentiated with central differences per canvas pixel, giving E = Su.Su,
    F = Su.Sv and G = Sv.Sv. sqrt(E*G - F^2) is the local area element.

    Args:
        xyz_norm: tensor (H, W, 3), normalised (x, y, z) positions in [0, 1].
        dims_t: tensor (3,), (Xdim-1, Ydim-1, Zdim-1) of the MRI volume.

    Returns:
        Tuple (E, F, G) of tensors, each (H, W).
    """
    S = xyz_norm * dims_t
    Su, Sv = _central_diff_su_sv(S)
    E = (Su * Su).sum(-1)
    Fq = (Su * Sv).sum(-1)
    G = (Sv * Sv).sum(-1)
    return E, Fq, G


def surface_area_ratio(xyz_norm, xyz_norm_base, dims, eps=1e-12):
    """Compute the per-pixel area ratio of a sheet relative to a base sheet.

    Both sheets are converted to voxel-index coordinates and the local area element
    |Su x Sv| is computed with central differences. A value of 1 means the area is
    locally preserved. Unlike a fold check (sign of the Jacobian), this measures
    stretching and compression, which can happen without folding. Because it is a
    ratio, the voxel spacing cancels for isotropic data.

    Args:
        xyz_norm: tensor or array (H, W, 3), normalised (x, y, z) positions in [0, 1].
        xyz_norm_base: reference sheet, same shape and convention.
        dims: sequence of 3, (Xdim-1, Ydim-1, Zdim-1) of the MRI volume.
        eps: lower bound on the base area to avoid division by zero.

    Returns:
        numpy array (H, W), area of xyz_norm divided by area of xyz_norm_base.
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
    """Return the (H, W) per-pixel weight of the rigidity terms, in [0, 1 + edge_k]; inputs are detached.

    Modes: "mask" uses the soft tissue mask; "conf" uses mask * conf, so the sheet is
    rigid where matches support the base surface and free elsewhere (the complement of
    the (1 - conf) gate on rf_psi/rf_beta); "edge" uses mask * (1 + edge_k * |grad hist|
    / max), stiffening high-contrast histology structures; "none" (or None) uses 1.
    mask_soft is (1, 1, H, W), conf is (H, W) and hist_t is (1, 1, H, W).
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

