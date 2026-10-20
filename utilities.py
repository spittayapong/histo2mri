import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import torch



def visualize_match(hist_img, mri_volume, match, patch_size, stride, z_window=1, origin=(0, 0)):
    """Visualize a matched histology–MRI patch pair.

    Expects:
      - match['hist_patch_idx'] = (i, j) where i is row (y), j is col (x)
      - match['mri_coords']     = (z_center, y_center, x_center)

    Coordinates are interpreted as patch centers for MRI and as grid indices for histology.
    """
    # Helpers to normalize ints/tuples to (h, w) and (y, x)
    def _to_hw_pair(val):
        if isinstance(val, int):
            return (val, val)
        return tuple(val[:2])

    patch_h, patch_w = _to_hw_pair(patch_size)
    stride_y, stride_x = _to_hw_pair(stride)
    origin_y, origin_x = _to_hw_pair(origin)

    # Unpack
    i, j = match['hist_patch_idx']
    zc, yc, xc = match['mri_coords']
    sim = match.get('correlation', None)

    # Histology patch: top-left from grid index (i, j)
    y0_h = int(round(origin_y + i * stride_y))
    x0_h = int(round(origin_x + j * stride_x))
    hist_patch = hist_img[y0_h : y0_h + patch_h, x0_h : x0_h + patch_w, ...]

    # MRI patch: convert center (yc, xc) to top-left, clamp to bounds
    Z, H, W = mri_volume.shape
    y0_m = int(round(yc - patch_h / 2.0))
    x0_m = int(round(xc - patch_w / 2.0))
    y0_m = max(0, min(H - patch_h, y0_m))
    x0_m = max(0, min(W - patch_w, x0_m))

    # z-window centered at zc (zc can be float; use nearest int center)
    zc_int = int(round(zc))
    half_z = z_window // 2
    z0 = max(0, zc_int - half_z)
    z1 = min(Z, zc_int + half_z + 1)

    mri_patch = mri_volume[z0:z1, y0_m : y0_m + patch_h, x0_m : x0_m + patch_w]
    mri_patch_mean = np.mean(mri_patch, axis=0)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    # Histology: support gray or RGB
    if hist_patch.ndim == 2:
        axs[0].imshow(hist_patch, cmap='gray')
    else:
        axs[0].imshow(hist_patch)
    axs[0].set_title(f"Histology (i={i}, j={j})")
    axs[0].axis('off')

    axs[1].imshow(mri_patch_mean, cmap='gray')
    if sim is not None:
        axs[1].set_title(f"MRI sim={sim:.2f}  z={zc_int}  y={yc:.1f}  x={xc:.1f}")
    else:
        axs[1].set_title(f"MRI  z={zc_int}  y={yc:.1f}  x={xc:.1f}")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def sitk_affine_to_hmat(tx):
    tx = sitk.AffineTransform(tx)  # ensures Affine
    M = np.array(tx.GetMatrix(), dtype=float).reshape(3, 3)
    t = np.array(tx.GetTranslation(), dtype=float)
    c = np.array(tx.GetCenter(), dtype=float)
    H = np.eye(4)
    H[:3, :3] = M
    H[:3, 3] = t + c - M @ c
    return H

def sitk_transform_to_hmat(tx: sitk.Transform) -> np.ndarray:
    origin = np.array(tx.TransformPoint((0., 0., 0.)))
    ex = np.array(tx.TransformPoint((1., 0., 0.)))
    ey = np.array(tx.TransformPoint((0., 1., 0.)))
    ez = np.array(tx.TransformPoint((0., 0., 1.)))
    H = np.eye(4)
    H[:3, :3] = np.column_stack([ex-origin, ey-origin, ez-origin])
    H[:3, 3]  = origin
    return H

def robust01(img, mask=None, pclip=(1, 99), eps=1e-8):
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
    n_quant=512,          # you can go back to 2048 once it's stable
    max_samples=1_000_000 # cap number of pixels used for quantiles
):
    # Quantile grid in [0,1]
    q = np.linspace(0.0, 1.0, n_quant, dtype=np.float32)

    # Apply masks
    svals = src01[src_mask] if src_mask is not None else src01.ravel()
    rvals = ref01[ref_mask] if ref_mask is not None else ref01.ravel()

    # To numpy float32 (saves memory vs float64)
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
        idx = np.random.choice(svals.size, max_samples, replace=False)
        svals = svals[idx]

    if rvals.size > max_samples:
        idx = np.random.choice(rvals.size, max_samples, replace=False)
        rvals = rvals[idx]

    # Quantiles
    xq = np.quantile(svals, q)
    yq = np.quantile(rvals, q)

    # Enforce monotonic xq for np.interp
    xq_u, idx = np.unique(xq, return_index=True)
    yq_u = yq[idx]

    if xq_u.size < 2:
        # Degenerate: map to median
        med = np.median(rvals)
        return np.array([0.0, 1.0], np.float32), np.array([med, med], np.float32)

    return xq_u.astype(np.float32), yq_u.astype(np.float32)

def apply_lut_monotone(src01, xq, yq):
    # Map src01 in [0,1] via piecewise linear, saturate outside
    out = np.interp(src01.ravel(), xq, yq).reshape(src01.shape).astype(np.float32)
    return out

# =========================
# Cubic kernels + 3D sampler (your clamp-free version)
# =========================
def cubic_weights_catmull_rom(t: torch.Tensor, a: float = -0.5):
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
    t2 = t * t
    t3 = t2 * t
    w0 = (1 - 3*t + 3*t2 - t3) / 6
    w1 = (4 - 6*t2 + 3*t3) / 6
    w2 = (1 + 3*t + 3*t2 - 3*t3) / 6
    w3 = t3 / 6
    return torch.stack([w0, w1, w2, w3], dim=-1)

def _reflect_indices(i: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 1:
        return torch.zeros_like(i)
    period = 2 * (n - 1)
    r = torch.remainder(i, period)
    r = torch.where(r < 0, r + period, r)
    return torch.where(r > (n - 1), period - r, r)

def _wrap_indices(i: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 1:
        return torch.zeros_like(i)
    r = torch.remainder(i, n)
    return torch.where(r < 0, r + n, r)

# def sample_mri_cubic_3d(
#     vol_1_1_zyx: torch.Tensor,
#     P_vox_xyz: torch.Tensor,
#     kernel: str = "catmull-rom",
#     padding_mode: str = "zeros",   # "zeros" | "reflect" | "wrap"
# ):
#     assert vol_1_1_zyx.ndim == 5 and vol_1_1_zyx.shape[:2] == (1,1)
#     vol = vol_1_1_zyx[0, 0]
#     Z, Y, X = vol.shape

#     H, W, _ = P_vox_xyz.shape
#     pts = P_vox_xyz.reshape(-1, 3)
#     x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]

#     x0f = torch.floor(x); y0f = torch.floor(y); z0f = torch.floor(z)
#     tx = x - x0f; ty = y - y0f; tz = z - z0f

#     x0 = x0f.long(); y0 = y0f.long(); z0 = z0f.long()

#     k = kernel.lower()
#     if k in ["catmull-rom", "catmullrom", "keys"]:
#         wx = cubic_weights_catmull_rom(tx)
#         wy = cubic_weights_catmull_rom(ty)
#         wz = cubic_weights_catmull_rom(tz)
#     elif k in ["bspline", "b-spline"]:
#         wx = cubic_weights_bspline(tx)
#         wy = cubic_weights_bspline(ty)
#         wz = cubic_weights_bspline(tz)
#     else:
#         raise ValueError("kernel must be 'catmull-rom' or 'bspline'")

#     offs = torch.tensor([-1, 0, 1, 2], device=pts.device, dtype=torch.long)
#     xi = x0[:, None] + offs[None, :]
#     yi = y0[:, None] + offs[None, :]
#     zi = z0[:, None] + offs[None, :]

#     pm = padding_mode.lower()
#     if pm == "reflect":
#         xi_i = _reflect_indices(xi, X); yi_i = _reflect_indices(yi, Y); zi_i = _reflect_indices(zi, Z)
#         mask64 = None
#     elif pm == "wrap":
#         xi_i = _wrap_indices(xi, X); yi_i = _wrap_indices(yi, Y); zi_i = _wrap_indices(zi, Z)
#         mask64 = None
#     elif pm == "zeros":
#         xi_i = _wrap_indices(xi, X); yi_i = _wrap_indices(yi, Y); zi_i = _wrap_indices(zi, Z)
#         in_x = (xi >= 0) & (xi < X)
#         in_y = (yi >= 0) & (yi < Y)
#         in_z = (zi >= 0) & (zi < Z)
#         mask_xyz = (in_z[:, :, None, None] & in_y[:, None, :, None] & in_x[:, None, None, :])
#         mask64 = mask_xyz.reshape(-1, 64).to(vol.dtype)
#     else:
#         raise ValueError("padding_mode must be 'zeros', 'reflect', or 'wrap'")

#     z3 = zi_i[:, :, None, None].expand(-1, 4, 4, 4)
#     y3 = yi_i[:, None, :, None].expand(-1, 4, 4, 4)
#     x3 = xi_i[:, None, None, :].expand(-1, 4, 4, 4)
#     w3 = (wz[:, :, None, None] * wy[:, None, :, None] * wx[:, None, None, :])

#     z64 = z3.reshape(-1, 64)
#     y64 = y3.reshape(-1, 64)
#     x64 = x3.reshape(-1, 64)
#     w64 = w3.reshape(-1, 64)

#     vol_flat = vol.reshape(-1)
#     lin = (z64 * (Y * X) + y64 * X + x64).long()
#     vals = torch.take(vol_flat, lin)

#     if mask64 is not None:
#         vals = vals * mask64

#     out = (vals * w64).sum(dim=1).reshape(1, 1, H, W)
#     return out


def sample_mri_cubic_3d(
    vol_1_1_zyx: torch.Tensor,
    P_vox_xyz: torch.Tensor,
    kernel: str = "catmull-rom",
    padding_mode: str = "zeros",   # "zeros" | "reflect" | "wrap" | "border"
):
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
        xi_i = _wrap_indices(xi, X); yi_i = _wrap_indices(yi, Y); zi_i = _wrap_indices(zi, Z)
        in_x = (xi >= 0) & (xi < X)
        in_y = (yi >= 0) & (yi < Y)
        in_z = (zi >= 0) & (zi < Z)
        mask_xyz = (in_z[:, :, None, None] & in_y[:, None, :, None] & in_x[:, None, None, :])
        mask64 = mask_xyz.reshape(-1, 64).to(vol.dtype)
    else:
        raise ValueError("padding_mode must be 'zeros', 'reflect', 'wrap', or 'border'")

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


@torch.no_grad()
def sample_mri_cubic_3d_tiled(
    vol_1_1_zyx: torch.Tensor,   # (1,1,Z,Y,X)
    P_vox_xyz: torch.Tensor,     # (H,W,3) float
    tile_hw: int = 512,
    kernel: str = "catmull-rom",
    padding_mode: str = "zeros",
):
    """
    Tile-wise wrapper around sample_mri_cubic_3d to avoid OOM.

    tile_hw: tile size in pixels (square). 256/384/512 are typical.
    """
    assert P_vox_xyz.ndim == 3 and P_vox_xyz.shape[-1] == 3
    H, W, _ = P_vox_xyz.shape
    device = P_vox_xyz.device
    out = torch.empty((1, 1, H, W), device=device, dtype=vol_1_1_zyx.dtype)

    for y0 in range(0, H, tile_hw):
        y1 = min(H, y0 + tile_hw)
        for x0 in range(0, W, tile_hw):
            x1 = min(W, x0 + tile_hw)

            P_tile = P_vox_xyz[y0:y1, x0:x1, :]   # (ht,wt,3)

            out[:, :, y0:y1, x0:x1] = sample_mri_cubic_3d(
                vol_1_1_zyx=vol_1_1_zyx,
                P_vox_xyz=P_tile,
                kernel=kernel,
                padding_mode=padding_mode
            )
    return out
