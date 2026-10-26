"""Sample the MRI volume on a surface, plus intensity helpers for QC display.

Used by curved-sheet refinement (refine) and the pull-back to HR histology
(inverse). Volumes are torch tensors [1, 1, Z, Y, X] (sitk array order).
Surface positions are given in (x, y, z) order, either normalised to [0, 1]
per axis (sample_mri_on_sheet, align_corners=True, so 0 and 1 are the first
and last voxel centres) or as voxel indices (sample_mri_cubic_3d). The robust
normalisation and monotone quantile LUT helpers are used by the QC figures.
"""
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import numpy as np


def _prepare_data_tensor(sitk_img, device):
    """Convert a scalar sitk image to a float32 tensor with batch and channel axes on device."""
    arr = sitk.GetArrayFromImage(sitk_img)
    # Shape handling:
    #   (H, W)    -> (1, 1, H, W)
    #   (1, H, W) -> treated as a 2D slice -> (1, 1, H, W)
    #   (Z, Y, X) -> (1, 1, Z, Y, X)
    #   4D        -> only a batch axis is prepended.
    tensor = torch.as_tensor(arr, dtype=torch.float32, device=device)

    # If a 3D array has a singleton depth (1, H, W), squeeze it to 2D.
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)                # (H, W)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)              # (1, H, W)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)              # (1, Z, Y, X)
    elif tensor.ndim != 4:
        raise ValueError(f"Unexpected image dims {tensor.shape}; expected 2D or 3D scalar image.")

    return tensor.unsqueeze(0)                    # prepend batch -> (1, C, ...)


def sample_mri_on_sheet(mri_t, mri_mask_t, xyz_norm_sheet):
    """Sample the MRI trilinearly on a surface and mask it with the MRI mask.

    Uses torch grid_sample with align_corners=True, so the result is
    differentiable with respect to xyz_norm_sheet (used as the image term
    during curved-sheet refinement). The image is sampled with border padding;
    the mask is sampled with nearest interpolation and zero padding, binarised
    (> 0) and multiplied into the image.

    Args:
        mri_t: MRI volume tensor [1, C, Z, Y, X].
        mri_mask_t: MRI mask tensor [1, C, Z, Y, X].
        xyz_norm_sheet: [H, W, 3] tensor of surface positions, normalised
            (x, y, z) in [0, 1].

    Returns:
        Tensor [1, C, H, W]: masked MRI intensities on the surface.
    """
    xg = 2*xyz_norm_sheet[..., 0] - 1.0
    yg = 2*xyz_norm_sheet[..., 1] - 1.0
    zg = 2*xyz_norm_sheet[..., 2] - 1.0

    # grid_sample expects normalised (x, y, z) in [-1, 1]; shape (1, 1, H, W, 3).
    grid = torch.stack([xg, yg, zg], dim=-1).unsqueeze(0).unsqueeze(1)  # (1,1,H,W,3)

    warped3d = F.grid_sample(
        mri_t, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
        )
    warped3d_mask = F.grid_sample(
        mri_mask_t, grid,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
        )

    # Drop the singleton depth axis: (1, C, H, W)
    warped2d = warped3d[:, :, 0, :, :]
    warped2d_mask = warped3d_mask[:, :, 0, :, :]

    warped2d_mask = (warped2d_mask > 0).float()

    warped2d_masked = warped2d * warped2d_mask

    return warped2d_masked


def robust01(img, mask=None, pclip=(1, 99), eps=1e-8):
    """Rescale an image to [0, 1] between robust percentiles.

    The percentiles are computed over finite pixels inside mask; values are
    clipped to [0, 1] after rescaling. If the percentiles coincide, the
    min/max of the masked values are used; if no pixel is usable, the global
    nanmin/nanmax.

    Args:
        img: array of any shape.
        mask: optional bool array of the same shape; pixels used for the
            percentiles.
        pclip: (low, high) percentiles.
        eps: added to the denominator.

    Returns:
        Tuple (x01, (lo, hi)): float32 array in [0, 1] with the shape of img,
        and the intensity bounds that were mapped to 0 and 1.
    """
    x = np.asarray(img, np.float32)
    if mask is None:
        mask = np.isfinite(x)
    else:
        mask = mask & np.isfinite(x)
    if not np.any(mask):
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    else:
        vals = x[mask]
        lo, hi = np.percentile(vals, pclip)
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
    x01 = np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)
    return x01.astype(np.float32), (lo, hi)


def make_monotone_lut(
    src01,
    ref01,
    src_mask=None,
    ref_mask=None,
    n_quant=512,
    max_samples=1_000_000
):
    """Build a monotone intensity LUT that maps the src distribution onto ref.

    Quantile (histogram) matching: n_quant equally spaced quantiles of the
    source and reference values form the knots (xq, yq) of a piecewise-linear
    map. Duplicate source quantiles are dropped so xq is strictly increasing.
    Inputs larger than max_samples are randomly subsampled (unseeded, so the
    LUT can vary slightly between runs).

    Args:
        src01: source image, normally in [0, 1].
        ref01: reference image, normally in [0, 1].
        src_mask: optional bool mask selecting source pixels.
        ref_mask: optional bool mask selecting reference pixels.
        n_quant: number of quantile levels.
        max_samples: maximum number of pixels per image used for quantiles.

    Returns:
        Tuple (xq, yq) of float32 knot arrays for apply_lut_monotone(). If
        either input has no finite values, the identity (q, q); if the source
        is constant, a map to the reference median.
    """
    # Quantile levels in [0, 1]
    q = np.linspace(0.0, 1.0, n_quant, dtype=np.float32)

    # Select masked pixels
    svals = src01[src_mask] if src_mask is not None else src01.ravel()
    rvals = ref01[ref_mask] if ref_mask is not None else ref01.ravel()

    svals = np.asarray(svals, dtype=np.float32)
    rvals = np.asarray(rvals, dtype=np.float32)

    # Remove NaN / inf
    svals = svals[np.isfinite(svals)]
    rvals = rvals[np.isfinite(rvals)]

    if svals.size == 0 or rvals.size == 0:
        # Fallback: identity LUT
        return q, q

    # Subsample if too large
    if svals.size > max_samples:
        idx = np.random.default_rng(0).choice(svals.size, max_samples, replace=False)
        svals = svals[idx]

    if rvals.size > max_samples:
        idx = np.random.default_rng(1).choice(rvals.size, max_samples, replace=False)
        rvals = rvals[idx]

    # Quantiles
    xq = np.quantile(svals, q)
    yq = np.quantile(rvals, q)

    # np.interp needs strictly increasing xq: drop duplicate source quantiles
    xq_u, idx = np.unique(xq, return_index=True)
    yq_u = yq[idx]

    if xq_u.size < 2:
        # Degenerate: map to median
        med = np.median(rvals)
        return np.array([0.0, 1.0], np.float32), np.array([med, med], np.float32)

    return xq_u.astype(np.float32), yq_u.astype(np.float32)


def apply_lut_monotone(src01, xq, yq):
    """Apply a piecewise-linear LUT built by make_monotone_lut().

    Values outside [xq[0], xq[-1]] saturate to yq[0] or yq[-1].

    Args:
        src01: input array.
        xq: increasing LUT input knots.
        yq: LUT output knots.

    Returns:
        float32 array with the shape of src01.
    """
    out = np.interp(src01.ravel(), xq, yq).reshape(src01.shape).astype(np.float32)
    return out


# =========================
# Cubic kernels and 3D cubic sampler
# =========================
def cubic_weights_catmull_rom(t: torch.Tensor, a: float = -0.5):
    """Compute the 4 Keys cubic-convolution weights for a fractional offset.

    Keys kernel with parameter a; a = -0.5 gives Catmull-Rom, which is
    interpolating (passes through the samples) and slightly sharpening.

    Args:
        t: tensor of fractional offsets in [0, 1) from the floor sample.
        a: Keys kernel parameter.

    Returns:
        Tensor of shape t.shape + (4,): weights for the samples at offsets
        -1, 0, +1, +2 from the floor index.
    """
    x0 = t + 1.0
    x1 = t
    x2 = 1.0 - t
    x3 = 2.0 - t

    def phi(x):
        ax = torch.abs(x)
        ax2 = ax * ax
        ax3 = ax2 * ax
        return torch.where(
            ax <= 1,
            (a + 2) * ax3 - (a + 3) * ax2 + 1,
            torch.where(
                ax < 2,
                a * ax3 - 5*a * ax2 + 8*a * ax - 4*a,
                torch.zeros_like(ax)
            )
        )

    return torch.stack([phi(x0), phi(x1), phi(x2), phi(x3)], dim=-1)


def cubic_weights_bspline(t: torch.Tensor):
    """Compute the 4 uniform cubic B-spline weights for a fractional offset.

    The weights are applied directly to the samples without B-spline
    prefiltering, so the result is a smoothing approximation, not an
    interpolation.

    Args:
        t: tensor of fractional offsets in [0, 1) from the floor sample.

    Returns:
        Tensor of shape t.shape + (4,): weights for the samples at offsets
        -1, 0, +1, +2 from the floor index.
    """
    t2 = t * t
    t3 = t2 * t
    w0 = (1 - 3*t + 3*t2 - t3) / 6
    w1 = (4 - 6*t2 + 3*t3) / 6
    w2 = (1 + 3*t + 3*t2 - 3*t3) / 6
    w3 = t3 / 6
    return torch.stack([w0, w1, w2, w3], dim=-1)


def _reflect_indices(i: torch.Tensor, n: int) -> torch.Tensor:
    """Mirror integer indices into [0, n-1] without repeating the edge sample (period 2(n-1))."""
    if n <= 1:
        return torch.zeros_like(i)
    period = 2 * (n - 1)
    r = torch.remainder(i, period)
    r = torch.where(r < 0, r + period, r)
    return torch.where(r > (n - 1), period - r, r)


def _wrap_indices(i: torch.Tensor, n: int) -> torch.Tensor:
    """Wrap integer indices periodically into [0, n-1]."""
    if n <= 1:
        return torch.zeros_like(i)
    r = torch.remainder(i, n)
    return torch.where(r < 0, r + n, r)


def sample_mri_cubic_3d(
    vol_1_1_zyx: torch.Tensor,
    P_vox_xyz: torch.Tensor,
    kernel: str = "catmull-rom",
    padding_mode: str = "zeros",   # "zeros" | "reflect" | "wrap" | "border"
):
    """Sample a 3D volume with a separable cubic kernel at voxel positions.

    Each point uses the 4x4x4 neighbourhood around its floor index, weighted
    by the tensor product of 1D cubic weights. The result is differentiable
    with respect to P_vox_xyz through the weights. Out-of-range taps are
    handled by padding_mode:
        "zeros": taps outside the volume contribute 0 (weights are not
            renormalised, so values fade near the border);
        "reflect": mirrored indices (edge sample not repeated);
        "wrap": periodic indices;
        "border": indices clamped to the edge voxel.

    Args:
        vol_1_1_zyx: volume tensor [1, 1, Z, Y, X].
        P_vox_xyz: [H, W, 3] sample positions in voxel index units, in
            (x, y, z) order (x indexes X, z indexes Z); same device as the
            volume.
        kernel: "catmull-rom" (aliases "catmullrom", "keys") or "bspline"
            (alias "b-spline").
        padding_mode: "zeros", "reflect", "wrap" or "border".

    Returns:
        Tensor [1, 1, H, W] of sampled values.
    """
    assert vol_1_1_zyx.ndim == 5 and vol_1_1_zyx.shape[:2] == (1,1)
    vol = vol_1_1_zyx[0, 0]
    Z, Y, X = vol.shape

    H, W, _ = P_vox_xyz.shape
    pts = P_vox_xyz.reshape(-1, 3)
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]

    x0f = torch.floor(x); y0f = torch.floor(y); z0f = torch.floor(z)
    tx = x - x0f; ty = y - y0f; tz = z - z0f

    x0 = x0f.long(); y0 = y0f.long(); z0 = z0f.long()

    k = kernel.lower()
    if k in ["catmull-rom", "catmullrom", "keys"]:
        wx = cubic_weights_catmull_rom(tx)
        wy = cubic_weights_catmull_rom(ty)
        wz = cubic_weights_catmull_rom(tz)
    elif k in ["bspline", "b-spline"]:
        wx = cubic_weights_bspline(tx)
        wy = cubic_weights_bspline(ty)
        wz = cubic_weights_bspline(tz)
    else:
        raise ValueError("kernel must be 'catmull-rom' or 'bspline'")

    offs = torch.tensor([-1, 0, 1, 2], device=pts.device, dtype=torch.long)
    xi = x0[:, None] + offs[None, :]
    yi = y0[:, None] + offs[None, :]
    zi = z0[:, None] + offs[None, :]

    pm = padding_mode.lower()
    if pm == "reflect":
        xi_i = _reflect_indices(xi, X); yi_i = _reflect_indices(yi, Y); zi_i = _reflect_indices(zi, Z)
        mask64 = None
    elif pm == "wrap":
        xi_i = _wrap_indices(xi, X); yi_i = _wrap_indices(yi, Y); zi_i = _wrap_indices(zi, Z)
        mask64 = None
    elif pm == "border":
        xi_i = torch.clamp(xi, 0, X - 1)
        yi_i = torch.clamp(yi, 0, Y - 1)
        zi_i = torch.clamp(zi, 0, Z - 1)
        mask64 = None
    elif pm == "zeros":
        # Wrap only to keep the gather in bounds; out-of-range taps are zeroed by mask64.
        xi_i = _wrap_indices(xi, X); yi_i = _wrap_indices(yi, Y); zi_i = _wrap_indices(zi, Z)
        in_x = (xi >= 0) & (xi < X)
        in_y = (yi >= 0) & (yi < Y)
        in_z = (zi >= 0) & (zi < Z)
        mask_xyz = (in_z[:, :, None, None] & in_y[:, None, :, None] & in_x[:, None, None, :])
        mask64 = mask_xyz.reshape(-1, 64).to(vol.dtype)
    else:
        raise ValueError("padding_mode must be 'zeros', 'reflect', 'wrap', or 'border'")

    # Tensor-product 4x4x4 neighbourhood per point, flattened to 64 taps (z, y, x order).
    z3 = zi_i[:, :, None, None].expand(-1, 4, 4, 4)
    y3 = yi_i[:, None, :, None].expand(-1, 4, 4, 4)
    x3 = xi_i[:, None, None, :].expand(-1, 4, 4, 4)
    w3 = (wz[:, :, None, None] * wy[:, None, :, None] * wx[:, None, None, :])

    z64 = z3.reshape(-1, 64)
    y64 = y3.reshape(-1, 64)
    x64 = x3.reshape(-1, 64)
    w64 = w3.reshape(-1, 64)

    vol_flat = vol.reshape(-1)
    lin = (z64 * (Y * X) + y64 * X + x64).long()
    vals = torch.take(vol_flat, lin)
    if mask64 is not None:
        vals = vals * mask64

    out = (vals * w64).sum(dim=1).reshape(1, 1, H, W)
    return out

