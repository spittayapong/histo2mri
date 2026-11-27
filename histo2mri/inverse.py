"""Map the refined MRI surface back into full-resolution (HR) histology space.

This is the last registration stage: after curved-sheet refinement, every
surface pixel carries an MRI position (xyz_norm_final) and a histology position
on the common canvas (xy_map_sym). The entry point, mri_on_histology_space(),
resamples the MRI onto the HR histology pixel grid in three stages:

  1. Compose the HR <-> common-canvas index affine from three sub-affines
     (HR -> proc, proc -> scaled, scaled -> common); build_hr_common_affines().
  2. Build the target-valid mask: the HR histology pixels where the surface can
     actually deliver a sample, restricted to the tissue mask;
     build_target_valid_mask().
  3. Pull the MRI through densely at HR resolution, tile by tile:
     HR pixel -> common canvas -> surface index -> xyz_norm_final -> MRI;
     resample_mri_to_hr().

All index affines are 4x4 homogeneous matrices acting on (x, y, z, 1) pixel
indices; z is 0 for the 2D section. A_common_to_hr is a product of
diagonal-plus-translation matrices, so its [0, 0] and [1, 1] entries are the
x and y scales (HR pixels per common-canvas pixel). build_target_valid_mask()
relies on this to size its smoothing kernels and prints a warning if the
affine ever contains a rotation (off-diagonal terms).
"""
import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter1d,
    minimum_filter1d,
    binary_fill_holes,
    map_coordinates,
)
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from .qc import plot_mri_on_histology
from .sampling import sample_mri_cubic_3d
import logging

logger = logging.getLogger(__name__)


def make_A_proc_to_scaled(proc_Ih, scaled_proc_Ih):
    """Build the proc -> scaled histology index affine.

    The x and y scales are the ratios of image sizes (scaled / proc).
    rescale_histology resizes with align_corners=False, which aligns pixel
    edges rather than pixel centres, so x_scaled = (x_proc + 0.5) * s - 0.5:
    a diagonal scaling plus a translation of 0.5 * (s - 1) pixels.

    Args:
        proc_Ih: prepared (padded) histology, sitk.Image.
        scaled_proc_Ih: the same image after rescaling by the best scale,
            sitk.Image.

    Returns:
        4x4 float64 array mapping homogeneous (x, y, z, 1) proc pixel indices
        to scaled pixel indices; z is left unchanged.
    """
    sx = scaled_proc_Ih.GetSize()[0] / proc_Ih.GetSize()[0]
    sy = scaled_proc_Ih.GetSize()[1] / proc_Ih.GetSize()[1]
    A = np.eye(4, dtype=np.float64)
    A[0, 0] = sx
    A[1, 1] = sy
    A[0, 3] = 0.5 * (sx - 1.0)
    A[1, 3] = 0.5 * (sy - 1.0)
    return A


def make_A_scaled_to_common(oy, ox):
    """Build the scaled-histology -> common-canvas index affine.

    The scaled histology sits on the common canvas at offset (oy, ox), so this
    is a pure translation.

    Args:
        oy: row offset of the histology on the common canvas (oy_h), pixels.
        ox: column offset of the histology on the common canvas (ox_h), pixels.

    Returns:
        4x4 float64 array mapping homogeneous (x, y, z, 1) scaled pixel indices
        to common-canvas pixel indices.
    """
    A = np.eye(4, dtype=np.float64)
    A[0, 3] = float(ox)  # x shift
    A[1, 3] = float(oy)  # y shift
    return A


@torch.no_grad()
def inverse_surface_image_to_hr(
    img_on_surface,
    xy_hist_map,
    A_common_to_hr,
    Ih_hr,
    valid_mask=None,
    target_valid_mask=None,
    is_mask=False,
    fill_holes=True,
    return_weight=False,
):
    """Splat an image defined on the surface grid into HR histology space.

    Each valid surface pixel has a common-canvas position (x, y) in
    xy_hist_map; it is mapped to HR pixel coordinates with A_common_to_hr and
    its value is scattered onto the HR grid. Images use a bilinear splat
    followed by division by the accumulated weight; masks (is_mask=True) use
    nearest-pixel assignment and are binarised (> 0). Because the surface is
    much coarser than HR histology the splat is sparse; with fill_holes=True
    (images only), HR pixels with no support inside target_valid_mask take the
    value of the nearest supported pixel. Pixels outside target_valid_mask are
    set to zero. Prints a one-line support summary.

    Args:
        img_on_surface: [Hs, Ws] values on the surface grid.
        xy_hist_map: [Hs, Ws, 2] common-canvas coordinates of each surface
            pixel, channel 0 = x (column), channel 1 = y (row), pixels.
        A_common_to_hr: 4x4 affine from homogeneous (x, y, z, 1) common-canvas
            indices to HR histology indices.
        Ih_hr: HR histology (sitk.Image or array, [1, Hhr, Whr] or
            [Hhr, Whr]); only its shape is used.
        valid_mask: optional [Hs, Ws] bool; surface pixels to splat. Default:
            all.
        target_valid_mask: optional [Hhr, Whr] bool; HR pixels allowed to
            receive values and to be hole-filled. Default: all.
        is_mask: if True, treat img_on_surface as a binary mask.
        fill_holes: if True, fill unsupported HR pixels inside
            target_valid_mask by nearest-neighbour copy (images only).
        return_weight: if True, also return the accumulated splat weights.

    Returns:
        out: [Hhr, Whr] float32 image in HR histology space; if return_weight
        is True, the tuple (out, wgt), where wgt is the [Hhr, Whr] float32
        accumulated splat weight (bilinear weights, or hit counts for masks),
        zeroed outside target_valid_mask.
    """
    import numpy as np
    import SimpleITK as sitk

    if isinstance(Ih_hr, sitk.Image):
        hr = sitk.GetArrayFromImage(Ih_hr)
    else:
        hr = np.asarray(Ih_hr)
    if hr.ndim == 3 and hr.shape[0] == 1:
        hr = hr[0]
    Hhr, Whr = hr.shape

    img = np.asarray(img_on_surface, dtype=np.float32)
    xy = np.asarray(xy_hist_map, dtype=np.float32)
    Hs, Ws = img.shape
    if xy.shape != (Hs, Ws, 2):
        raise ValueError(f"xy_hist_map must be {(Hs, Ws, 2)}, got {xy.shape}")

    if valid_mask is None:
        valid = np.ones((Hs, Ws), dtype=bool)
    else:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.shape != (Hs, Ws):
            raise ValueError(f"valid_mask must be {(Hs, Ws)}, got {valid.shape}")

    if target_valid_mask is None:
        target_valid = np.ones((Hhr, Whr), dtype=bool)
    else:
        target_valid = np.asarray(target_valid_mask).astype(bool)
        if target_valid.shape != (Hhr, Whr):
            raise ValueError(f"target_valid_mask must be {(Hhr, Whr)}, got {target_valid.shape}")

    ys, xs = np.where(valid)
    vals = img[ys, xs].astype(np.float32)
    x_common = xy[ys, xs, 0].astype(np.float64)
    y_common = xy[ys, xs, 1].astype(np.float64)

    A = np.asarray(A_common_to_hr, dtype=np.float64)
    if A.shape != (4, 4):
        raise ValueError(f"A_common_to_hr must be 4x4, got {A.shape}")

    p = np.stack([x_common, y_common, np.zeros_like(x_common), np.ones_like(x_common)], axis=1)
    q = (A @ p.T).T
    w = q[:, 3]

    ok = np.abs(w) > 1e-8
    xh = np.full_like(x_common, np.nan, dtype=np.float32)
    yh = np.full_like(y_common, np.nan, dtype=np.float32)
    xh[ok] = (q[ok, 0] / w[ok]).astype(np.float32)
    yh[ok] = (q[ok, 1] / w[ok]).astype(np.float32)

    keep = np.isfinite(xh) & np.isfinite(yh) & np.isfinite(vals)
    xh, yh, vals = xh[keep], yh[keep], vals[keep]

    inside = (xh >= 0) & (xh <= Whr - 1) & (yh >= 0) & (yh <= Hhr - 1)
    xh, yh, vals = xh[inside], yh[inside], vals[inside]

    out = np.zeros((Hhr, Whr), dtype=np.float32)
    wgt = np.zeros((Hhr, Whr), dtype=np.float32)

    if is_mask:
        # Masks: nearest-pixel assignment, so labels are never blended.
        xn = np.rint(xh).astype(np.int32)
        yn = np.rint(yh).astype(np.int32)
        m = (xn >= 0) & (xn < Whr) & (yn >= 0) & (yn < Hhr)
        np.add.at(out, (yn[m], xn[m]), vals[m])
        np.add.at(wgt, (yn[m], xn[m]), 1.0)
        out = (out > 0).astype(np.float32)
    else:
        # Images: bilinear splat, then normalise by the accumulated weight.
        x0 = np.floor(xh).astype(np.int32)
        y0 = np.floor(yh).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = xh - x0
        wy = yh - y0

        for xx, yy, ww in [
            (x0, y0, (1 - wx) * (1 - wy)),
            (x1, y0, wx * (1 - wy)),
            (x0, y1, (1 - wx) * wy),
            (x1, y1, wx * wy),
        ]:
            m = (xx >= 0) & (xx < Whr) & (yy >= 0) & (yy < Hhr) & (ww > 0)
            np.add.at(out, (yy[m], xx[m]), vals[m] * ww[m])
            np.add.at(wgt, (yy[m], xx[m]), ww[m])

        nz = wgt > 1e-6
        out[nz] /= wgt[nz]

        if fill_holes and np.any(nz):
            try:
                from scipy.ndimage import distance_transform_edt
                fillable = (~nz) & target_valid
                if np.any(fillable):
                    # EDT of ~nz returns, for every pixel, the index of the nearest supported pixel.
                    _, (ri, rj) = distance_transform_edt(~nz, return_indices=True)
                    out[fillable] = out[ri[fillable], rj[fillable]]
            except Exception:
                pass

    out[~target_valid] = 0
    wgt[~target_valid] = 0

    logger.debug(f'[inverse_surface_image_to_hr] src_valid={valid.sum()} proj_inside={inside.sum()} support={(wgt > 1e-06).sum()} support_frac={(wgt > 1e-06).mean():.8f}')

    if return_weight:
        return out, wgt
    return out


# ============================================================
# MRI on histology space
# ============================================================

def _hr_2d(Ih_hr):
    """Return HR histology as a 2D array; accepts sitk or numpy, [1, H, W] or [H, W]."""
    a = sitk.GetArrayFromImage(Ih_hr) if isinstance(Ih_hr, sitk.Image) else np.asarray(Ih_hr)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected 2D HR histology, got shape {a.shape}")
    return a


def _scale_from_affine(A):
    """Return the (x, y) scales of a diagonal-plus-translation affine, each clamped to >= 1.

    A[0, 0] and A[1, 1] are the scales only while the affine has no rotation;
    a warning is printed if off-diagonal terms appear.
    """
    A = np.asarray(A, dtype=np.float64)
    off = abs(A[0, 1]) + abs(A[1, 0])
    if off > 1e-6 * (abs(A[0, 0]) + abs(A[1, 1]) + 1e-12):
        logger.warning(f'A_common_to_hr has off-diagonal terms ({A[0, 1]:.3e}, {A[1, 0]:.3e}); |diag| is not the true scale.')
    return max(1.0, float(abs(A[0, 0]))), max(1.0, float(abs(A[1, 1])))


def build_hr_common_affines(proc_Ih, scaled_proc_Ih, A_hr_to_proc, oy_h, ox_h):
    """Compose the HR-histology <-> common-canvas index affines.

    A_hr_to_common = A_scaled_to_common @ A_proc_to_scaled @ A_hr_to_proc,
    and A_common_to_hr is its inverse. All factors are diagonal plus
    translation, so the result is too.

    Args:
        proc_Ih: prepared (padded) histology at MRI spacing, sitk.Image.
        scaled_proc_Ih: proc_Ih after rescaling by the best scale, sitk.Image.
        A_hr_to_proc: 4x4 affine from HR pixel indices to proc pixel indices
            (A_lr_to_proc @ A_hr_to_lr from preprocessing).
        oy_h: row offset of the scaled histology on the common canvas.
        ox_h: column offset of the scaled histology on the common canvas.

    Returns:
        Tuple (A_hr_to_common, A_common_to_hr) of 4x4 float64 arrays acting on
        homogeneous (x, y, z, 1) pixel indices.
    """
    A_proc_to_scaled = make_A_proc_to_scaled(proc_Ih, scaled_proc_Ih)
    A_scaled_to_common = make_A_scaled_to_common(oy_h, ox_h)
    A_hr_to_common = np.asarray(A_scaled_to_common @ A_proc_to_scaled @ A_hr_to_proc, dtype=np.float64)
    A_common_to_hr = np.linalg.inv(A_hr_to_common)
    return A_hr_to_common, A_common_to_hr


def sample_rgb_on_surface(Ih_rgb_hr, xy_map_sym, A_common_to_hr):
    """Colour each surface pixel with the HR RGB histology at its histology position.

    Surface pixel -> common-canvas (x, y) via xy_map_sym -> HR pixel via
    A_common_to_hr, read with nearest-neighbour lookup. Positions outside the HR
    image take the colour of the nearest border pixel; mask them with the surface's
    valid map.

    Args:
        Ih_rgb_hr: HR RGB histology, sitk.Image of size (W, H, 1) with 3 components,
            or an array [H, W, 3]; 0-255 or 0-1.
        xy_map_sym: [Hs, Ws, 2] common-canvas (x, y) of each surface pixel.
        A_common_to_hr: 4x4 index affine from the common canvas to HR pixels.

    Returns:
        float32 array [Hs, Ws, 3] in [0, 1].
    """
    if isinstance(Ih_rgb_hr, sitk.Image):
        rgb = sitk.GetArrayFromImage(Ih_rgb_hr)
    else:
        rgb = np.asarray(Ih_rgb_hr)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"Ih_rgb_hr must be [H, W, 3], got {rgb.shape}")
    rgb = rgb[..., :3]

    xy = xy_map_sym.detach().cpu().numpy() if torch.is_tensor(xy_map_sym) else np.asarray(xy_map_sym)
    Hs, Ws = xy.shape[:2]
    xy = xy.reshape(-1, 2).astype(np.float64)
    p = np.stack([xy[:, 0], xy[:, 1], np.zeros(len(xy)), np.ones(len(xy))], axis=1)
    q = p @ np.asarray(A_common_to_hr, dtype=np.float64).T
    w = np.where(np.abs(q[:, 3]) > 1e-8, q[:, 3], 1.0)
    col = np.clip(np.rint(np.nan_to_num(q[:, 0] / w)).astype(np.int64), 0, rgb.shape[1] - 1)
    row = np.clip(np.rint(np.nan_to_num(q[:, 1] / w)).astype(np.int64), 0, rgb.shape[0] - 1)

    out = rgb[row, col].astype(np.float32)
    if out.max(initial=0.0) > 1.0:
        out /= 255.0
    return out.reshape(Hs, Ws, 3)


def build_target_valid_mask(
    mri_on_final_sheet,
    xy_map_sym,
    A_common_to_hr,
    Ih_hr,
    Mh_hr,
    valid_display,
    occ_thresh=0.05,
    sigma_scale=0.35,
    verbose=True,
):
    """Build the mask of HR histology pixels the surface can deliver a sample to.

    The surface is at common-canvas resolution and HR histology is much finer,
    so splatting the surface support (an all-ones image over valid_display)
    into HR leaves a sparse stipple. Within the tissue bounding box (padded by
    two smoothing sigmas) the stipple is densified: Gaussian blur with
    sigma = max(2, sigma_scale * s) HR pixels per axis, where s is the HR
    pixels per common-canvas pixel taken from A_common_to_hr; threshold at
    occ_thresh times the blurred maximum; separable rectangular closing; hole
    filling. The result is intersected with the tissue mask so that nothing
    outside the section is claimed.

    Args:
        mri_on_final_sheet: [Hs, Ws] MRI sampled on the final surface; only
            its shape is used.
        xy_map_sym: [Hs, Ws, 2] common-canvas (x, y) of each surface pixel.
        A_common_to_hr: 4x4 common-canvas -> HR index affine (diagonal plus
            translation).
        Ih_hr: HR histology image (sitk.Image or array); defines the HR grid.
        Mh_hr: HR histology tissue mask (sitk.Image or array).
        valid_display: [Hs, Ws] bool; surface pixels that carry valid data.
        occ_thresh: threshold on the blurred support, as a fraction of its
            maximum.
        sigma_scale: smoothing sigma and closing half-width, as a fraction of
            the HR-per-common-pixel scale.
        verbose: print the scale, ROI and coverage.

    Returns:
        Tuple (target_valid_mask, support_w_hr): [Hhr, Whr] bool mask and
        [Hhr, Whr] float32 splat weight of the surface support. If the tissue
        mask is empty, the mask is all False.
    """
    support_img = np.ones_like(np.asarray(mri_on_final_sheet), dtype=np.float32)

    _, support_w_hr = inverse_surface_image_to_hr(
        support_img,
        xy_map_sym,
        A_common_to_hr,
        Ih_hr,
        valid_mask=valid_display,
        is_mask=False,
        fill_holes=False,
        return_weight=True,
    )
    support_w_hr = np.asarray(support_w_hr, dtype=np.float32)

    arr_Mh_hr = sitk.GetArrayFromImage(Mh_hr).astype(bool) if isinstance(Mh_hr, sitk.Image)\
        else np.asarray(Mh_hr).astype(bool)
    if arr_Mh_hr.ndim == 3 and arr_Mh_hr.shape[0] == 1:
        arr_Mh_hr = arr_Mh_hr[0]

    yy, xx = np.where(arr_Mh_hr)
    if yy.size == 0:
        if verbose:
            logger.warning('[target_valid_mask] histology mask is empty')
        return np.zeros_like(arr_Mh_hr, dtype=bool), support_w_hr

    sx, sy = _scale_from_affine(A_common_to_hr)

    pad_y = int(np.ceil(2.0 * max(2.0, sigma_scale * sy)))
    pad_x = int(np.ceil(2.0 * max(2.0, sigma_scale * sx)))

    hr_y0 = max(0, int(yy.min()) - pad_y)
    hr_y1 = min(arr_Mh_hr.shape[0], int(yy.max()) + pad_y + 1)
    hr_x0 = max(0, int(xx.min()) - pad_x)
    hr_x1 = min(arr_Mh_hr.shape[1], int(xx.max()) + pad_x + 1)

    support_bin = support_w_hr[hr_y0:hr_y1, hr_x0:hr_x1] > 1e-6
    tissue_roi = arr_Mh_hr[hr_y0:hr_y1, hr_x0:hr_x1]

    sig_y = max(2.0, sigma_scale * sy)
    sig_x = max(2.0, sigma_scale * sx)

    support_occ = gaussian_filter(
        support_bin.astype(np.float32), sigma=(sig_y, sig_x), mode="constant", cval=0.0
    )

    occ_max = float(support_occ.max())
    thr = occ_thresh * occ_max if occ_max > 0 else 0.0
    mask_roi = support_occ > thr

    # Rectangular closing: dilate then erode, separable per axis (cheap at HR size).
    close_y = max(5, int(np.ceil(sigma_scale * sy)) * 2 + 1)
    close_x = max(5, int(np.ceil(sigma_scale * sx)) * 2 + 1)

    mask_u8 = mask_roi.astype(np.uint8)
    mask_u8 = maximum_filter1d(mask_u8, size=close_x, axis=1, mode="constant", cval=0)
    mask_u8 = maximum_filter1d(mask_u8, size=close_y, axis=0, mode="constant", cval=0)
    mask_u8 = minimum_filter1d(mask_u8, size=close_x, axis=1, mode="constant", cval=0)
    mask_u8 = minimum_filter1d(mask_u8, size=close_y, axis=0, mode="constant", cval=0)

    mask_roi = binary_fill_holes(mask_u8.astype(bool))
    mask_roi &= tissue_roi

    target_valid_mask = np.zeros_like(arr_Mh_hr, dtype=bool)
    target_valid_mask[hr_y0:hr_y1, hr_x0:hr_x1] = mask_roi

    if verbose:
        logger.debug(f'[target_valid_mask] hr scale (sx,sy) = ({sx:.1f}, {sy:.1f})  roi = y[{hr_y0}:{hr_y1}] x[{hr_x0}:{hr_x1}]  coverage = {target_valid_mask.mean():.6f}')

    return target_valid_mask, support_w_hr


@torch.no_grad()
def resample_mri_to_hr(
    slab_Im,
    xyz_norm_final,
    xy_map_sym,
    valid_display,
    target_valid_mask,
    A_common_to_hr,
    Ih_hr,
    device="cuda:0",
    tile_hw=512,
    verbose=True,
):
    """Resample the MRI onto every HR histology pixel by pulling back through the surface.

    Chain: HR pixel -> common canvas -> surface index -> xyz_norm_final -> MRI.
    xy_map_sym maps surface index -> common-canvas xy; the inverse is obtained
    by scattered interpolation of (surface row, surface col) as two scalar
    fields over the common-canvas positions of the valid surface pixels.
    Interpolation is linear inside their convex hull; nearest-neighbour is used
    as a fallback only where target_valid_mask marks the pixel as valid, so the
    extrapolation cannot invent tissue. xyz_norm_final is then sampled
    bilinearly at the (row, col), scaled to MRI voxel indices and the MRI is
    sampled with a Catmull-Rom cubic kernel (border padding). Work is done
    tile by tile over the bounding box of target_valid_mask.

    Args:
        slab_Im: MRI slab, sitk.Image or array [Z, Y, X].
        xyz_norm_final: [Hs, Ws, 3] refined surface positions, normalised
            (x, y, z) in [0, 1]; tensor or array.
        xy_map_sym: [Hs, Ws, 2] common-canvas (x, y) of each surface pixel.
        valid_display: [Hs, Ws] bool; surface pixels used to build the inverse
            map.
        target_valid_mask: [Hhr, Whr] bool from build_target_valid_mask().
        A_common_to_hr: 4x4 common-canvas -> HR index affine.
        Ih_hr: HR histology image; defines the HR grid.
        device: torch device for the cubic MRI sampling.
        tile_hw: tile edge length in HR pixels.
        verbose: print sizes and coverage.

    Returns:
        Tuple (mri_on_hr, surface_valid_hr, final_valid_hr):
            mri_on_hr: [Hhr, Whr] float32 MRI intensities, 0 outside
                final_valid_hr.
            surface_valid_hr: [Hhr, Whr] bool, pixels that received a surface
                position.
            final_valid_hr: [Hhr, Whr] bool, surface_valid_hr & target_valid_mask.
    """
    _xyz = xyz_norm_final.detach().cpu().numpy() if torch.is_tensor(xyz_norm_final) else np.asarray(xyz_norm_final)
    _xy_map = xy_map_sym.detach().cpu().numpy() if torch.is_tensor(xy_map_sym) else np.asarray(xy_map_sym)
    _valid_display = np.asarray(valid_display, dtype=bool)
    _target_valid_mask = np.asarray(target_valid_mask, dtype=bool)

    Hhr, Whr = _hr_2d(Ih_hr).shape
    Hs, Ws = _xyz.shape[:2]
    if verbose:
        logger.debug(f'[resample_mri_to_hr] HR histology {Hhr} x {Whr}   surface {Hs} x {Ws}')

    surf_r, surf_c = np.where(_valid_display)
    if surf_r.size == 0:
        raise ValueError("valid_display is empty")

    common_x = _xy_map[surf_r, surf_c, 0].astype(np.float32)
    common_y = _xy_map[surf_r, surf_c, 1].astype(np.float32)
    pts_common = np.column_stack([common_y, common_x])   # queries are also (y, x)

    row_lin = LinearNDInterpolator(pts_common, surf_r.astype(np.float32), fill_value=np.nan)
    col_lin = LinearNDInterpolator(pts_common, surf_c.astype(np.float32), fill_value=np.nan)
    row_near = NearestNDInterpolator(pts_common, surf_r.astype(np.float32))
    col_near = NearestNDInterpolator(pts_common, surf_c.astype(np.float32))

    vol_arr = sitk.GetArrayFromImage(slab_Im).astype(np.float32) if isinstance(slab_Im, sitk.Image)\
        else np.asarray(slab_Im, dtype=np.float32)
    if vol_arr.ndim == 2:
        vol_arr = vol_arr[None]
    Zdim, Ydim, Xdim = vol_arr.shape
    vol_t = torch.from_numpy(vol_arr).to(device)[None, None]

    yy, xx = np.where(_target_valid_mask)
    if yy.size == 0:
        raise ValueError("target_valid_mask is empty")

    roi_y0, roi_y1 = int(yy.min()), int(yy.max()) + 1
    roi_x0, roi_x1 = int(xx.min()), int(xx.max()) + 1

    A_inv = np.linalg.inv(np.asarray(A_common_to_hr, dtype=np.float64))

    mri_on_hr = np.zeros((Hhr, Whr), dtype=np.float32)
    surface_valid_hr = np.zeros((Hhr, Whr), dtype=bool)

    for ys in range(roi_y0, roi_y1, tile_hw):
        ye = min(ys + tile_hw, roi_y1)
        for xs in range(roi_x0, roi_x1, tile_hw):
            xe = min(xs + tile_hw, roi_x1)

            tile_target = _target_valid_mask[ys:ye, xs:xe]
            if not np.any(tile_target):
                continue

            th, tw = tile_target.shape

            yy_tile, xx_tile = np.meshgrid(
                np.arange(ys, ye, dtype=np.float64),
                np.arange(xs, xe, dtype=np.float64),
                indexing="ij",
            )

            pts_flat = np.stack([
                xx_tile.ravel(),
                yy_tile.ravel(),
                np.zeros(th * tw, dtype=np.float64),
                np.ones(th * tw, dtype=np.float64),
            ], axis=1)

            q = (A_inv @ pts_flat.T).T
            w = q[:, 3]
            xc_tile = (q[:, 0] / w).astype(np.float32)
            yc_tile = (q[:, 1] / w).astype(np.float32)
            qpts = np.column_stack([yc_tile, xc_tile])

            surf_row = row_lin(qpts)
            surf_col = col_lin(qpts)
            valid = np.isfinite(surf_row) & np.isfinite(surf_col)

            fallback = (~valid) & tile_target.ravel()
            if np.any(fallback):
                surf_row[fallback] = row_near(qpts[fallback])
                surf_col[fallback] = col_near(qpts[fallback])

            valid = np.isfinite(surf_row) & np.isfinite(surf_col)

            xyz_flat = np.zeros((th * tw, 3), dtype=np.float32)
            if np.any(valid):
                row_clip = np.clip(surf_row[valid], 0, Hs - 1)
                col_clip = np.clip(surf_col[valid], 0, Ws - 1)
                sample_coords = np.vstack([row_clip, col_clip])
                for ch in range(3):
                    xyz_flat[valid, ch] = map_coordinates(
                        _xyz[..., ch].astype(np.float32),
                        sample_coords,
                        order=1, mode="nearest", prefilter=False,
                    ).astype(np.float32)

            xyz_tile = xyz_flat.reshape(th, tw, 3)

            P_tile = np.empty_like(xyz_tile, dtype=np.float32)
            P_tile[..., 0] = np.clip(xyz_tile[..., 0] * (Xdim - 1), 0, Xdim - 1)
            P_tile[..., 1] = np.clip(xyz_tile[..., 1] * (Ydim - 1), 0, Ydim - 1)
            P_tile[..., 2] = np.clip(xyz_tile[..., 2] * (Zdim - 1), 0, Zdim - 1)

            P_tile_t = torch.from_numpy(P_tile).to(device=device, dtype=torch.float32)
            mri_tile = sample_mri_cubic_3d(
                vol_t, P_tile_t, kernel="catmull-rom", padding_mode="border",
            )[0, 0].detach().cpu().numpy()

            valid_2d = valid.reshape(th, tw)
            mri_tile[~valid_2d] = 0.0

            mri_on_hr[ys:ye, xs:xe] = mri_tile
            surface_valid_hr[ys:ye, xs:xe] = valid_2d

    final_valid_hr = surface_valid_hr & _target_valid_mask
    mri_on_hr[~final_valid_hr] = 0.0

    if verbose:
        logger.info(f'[resample_mri_to_hr] surface_valid coverage = {surface_valid_hr.mean():.6f}   final_valid coverage = {final_valid_hr.mean():.6f}')

    return mri_on_hr, surface_valid_hr, final_valid_hr


def mri_on_histology_space(
    Ih_hr, Mh_hr,
    proc_Ih, scaled_proc_Ih, A_hr_to_proc,
    oy_h, ox_h,
    slab_Im,
    xyz_norm_final,
    xy_map_sym,
    valid_display,
    mri_on_final_sheet,
    device="cuda:0",
    tile_hw=512,
    occ_thresh=0.05,
    sigma_scale=0.35,
    show_proc_img=False,
    tile_size=348,
    save_file=None,
    save_file_single=None,
    verbose=True,
):
    """Resample the MRI on the refined surface into HR histology space.

    Entry point of this module. Runs build_hr_common_affines(),
    build_target_valid_mask() and resample_mri_to_hr() in order, prints its
    own progress, and optionally draws the QC overlay with
    qc.plot_mri_on_histology().

    Args:
        Ih_hr: HR histology image, sitk.Image [1, Hhr, Whr] or array.
        Mh_hr: HR histology tissue mask, same grid as Ih_hr.
        proc_Ih: prepared (padded) histology at MRI spacing, sitk.Image.
        scaled_proc_Ih: proc_Ih rescaled by the best scale, sitk.Image.
        A_hr_to_proc: 4x4 HR -> proc index affine.
        oy_h: row offset of the scaled histology on the common canvas.
        ox_h: column offset of the scaled histology on the common canvas.
        slab_Im: MRI slab, sitk.Image or array [Z, Y, X].
        xyz_norm_final: [Hs, Ws, 3] refined surface positions, normalised
            (x, y, z) in [0, 1].
        xy_map_sym: [Hs, Ws, 2] common-canvas (x, y) of each surface pixel.
        valid_display: [Hs, Ws] bool; valid surface pixels.
        mri_on_final_sheet: [Hs, Ws] MRI sampled on the final surface; only its
            shape is used.
        device: torch device for MRI sampling.
        tile_hw: tile edge length for the HR resampling, pixels.
        occ_thresh: support threshold for build_target_valid_mask().
        sigma_scale: smoothing scale for build_target_valid_mask().
        show_proc_img: if True, draw the QC overlay.
        tile_size: checkerboard tile size for the QC overlay, HR pixels.
        save_file: optional output path for the multi-panel QC figure.
        save_file_single: optional output path for a single image of the MRI
            in histology space.
        verbose: print progress from the sub-steps.

    Returns:
        Tuple (mri_on_hr, final_valid_hr, A_hr_to_common, A_common_to_hr, info):
            mri_on_hr: [Hhr, Whr] float32 MRI resampled onto HR histology
                pixels, 0 outside final_valid_hr.
            final_valid_hr: [Hhr, Whr] bool validity mask.
            A_hr_to_common: 4x4 HR -> common-canvas index affine.
            A_common_to_hr: 4x4 common-canvas -> HR index affine.
            info: dict with target_valid_mask, surface_valid_hr, support_w_hr,
                Hhr and Whr.
    """
    logger.debug('-------- MRI on Histology Space --------')

    A_hr_to_common, A_common_to_hr = build_hr_common_affines(
        proc_Ih, scaled_proc_Ih, A_hr_to_proc, oy_h, ox_h
    )

    target_valid_mask, support_w_hr = build_target_valid_mask(
        mri_on_final_sheet=mri_on_final_sheet,
        xy_map_sym=xy_map_sym,
        A_common_to_hr=A_common_to_hr,
        Ih_hr=Ih_hr,
        Mh_hr=Mh_hr,
        valid_display=valid_display,
        occ_thresh=occ_thresh,
        sigma_scale=sigma_scale,
        verbose=verbose,
    )

    mri_on_hr, surface_valid_hr, final_valid_hr = resample_mri_to_hr(
        slab_Im=slab_Im,
        xyz_norm_final=xyz_norm_final,
        xy_map_sym=xy_map_sym,
        valid_display=valid_display,
        target_valid_mask=target_valid_mask,
        A_common_to_hr=A_common_to_hr,
        Ih_hr=Ih_hr,
        device=device,
        tile_hw=tile_hw,
        verbose=verbose,
    )

    if show_proc_img:
        plot_mri_on_histology(
            mri_on_hr, Ih_hr, Mh_hr,
            tile_size=tile_size,
            save_file=save_file,
            save_file_single=save_file_single,
        )

    Hhr, Whr = mri_on_hr.shape
    info = dict(
        target_valid_mask=target_valid_mask,
        surface_valid_hr=surface_valid_hr,
        support_w_hr=support_w_hr,
        Hhr=Hhr, Whr=Whr,
    )
    return mri_on_hr, final_valid_hr, A_hr_to_common, A_common_to_hr, info

