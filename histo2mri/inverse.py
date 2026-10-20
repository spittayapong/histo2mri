"""Map the refined surface back into high-resolution histology space."""
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


def make_A_proc_to_scaled(proc_Ih, scaled_proc_Ih):
    sx = scaled_proc_Ih.GetSize()[0] / proc_Ih.GetSize()[0]
    sy = scaled_proc_Ih.GetSize()[1] / proc_Ih.GetSize()[1]
    A = np.eye(4, dtype=np.float64)
    A[0, 0] = sx
    A[1, 1] = sy
    return A


def make_A_scaled_to_common(oy, ox):
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
        # Nearest assignment for true binary-mask backprojection only.
        xn = np.rint(xh).astype(np.int32)
        yn = np.rint(yh).astype(np.int32)
        m = (xn >= 0) & (xn < Whr) & (yn >= 0) & (yn < Hhr)
        np.add.at(out, (yn[m], xn[m]), vals[m])
        np.add.at(wgt, (yn[m], xn[m]), 1.0)
        out = (out > 0).astype(np.float32)
    else:
        # Bilinear splat for image backprojection.
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
                    _, (ri, rj) = distance_transform_edt(~nz, return_indices=True)
                    out[fillable] = out[ri[fillable], rj[fillable]]
            except Exception:
                pass

    out[~target_valid] = 0
    wgt[~target_valid] = 0

    print(
        f"[inverse_surface_image_to_hr] "
        f"src_valid={valid.sum()} "
        f"proj_inside={inside.sum()} "
        f"support={(wgt > 1e-6).sum()} "
        f"support_frac={(wgt > 1e-6).mean():.8f}"
    )

    if return_weight:
        return out, wgt
    return out


# ============================================================
# MRI on Histology Space  (lifted from pipeline notebook)
# ============================================================

def _hr_2d(Ih_hr):
    """HR histology as a plain 2D float array, accepting sitk or numpy, [1,H,W] or [H,W]."""
    a = sitk.GetArrayFromImage(Ih_hr) if isinstance(Ih_hr, sitk.Image) else np.asarray(Ih_hr)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected 2D HR histology, got shape {a.shape}")
    return a


def _scale_from_affine(A):
    """x/y scale of a diagonal-plus-translation index affine, with a sanity check.

    The chain A_scaled_to_common @ A_proc_to_scaled @ A_hr_to_proc is built from
    diagonal + translation matrices only, so A[0,0] and A[1,1] ARE the scales.
    If a rotation ever enters the chain this assumption breaks silently, hence
    the warning.
    """
    A = np.asarray(A, dtype=np.float64)
    off = abs(A[0, 1]) + abs(A[1, 0])
    if off > 1e-6 * (abs(A[0, 0]) + abs(A[1, 1]) + 1e-12):
        print(f"[inverse_to_histo_v12] WARNING: A_common_to_hr has off-diagonal terms "
              f"({A[0, 1]:.3e}, {A[1, 0]:.3e}); |diag| is not the true scale.")
    return max(1.0, float(abs(A[0, 0]))), max(1.0, float(abs(A[1, 1])))


def build_hr_common_affines(proc_Ih, scaled_proc_Ih, A_hr_to_proc, oy_h, ox_h):
    """Compose the HR-histology <-> common-canvas index affines.

    Returns (A_hr_to_common, A_common_to_hr), both 4x4 float64.
    """
    A_proc_to_scaled = make_A_proc_to_scaled(proc_Ih, scaled_proc_Ih)
    A_scaled_to_common = make_A_scaled_to_common(oy_h, ox_h)
    A_hr_to_common = np.asarray(A_scaled_to_common @ A_proc_to_scaled @ A_hr_to_proc, dtype=np.float64)
    A_common_to_hr = np.linalg.inv(A_hr_to_common)
    return A_hr_to_common, A_common_to_hr


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
    """Where in HR histology space can the surface actually deliver a sample?

    The surface is coarse (common-canvas resolution) and HR histology is ~100x
    finer, so splatting the surface's support into HR leaves a sparse stipple.
    We densify it: blur the splat footprint, threshold, close the gaps, fill
    holes, and finally intersect with the real tissue mask so nothing outside
    the section is ever claimed.

    Returns (target_valid_mask [Hhr,Whr] bool, support_w_hr [Hhr,Whr] float32).
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
            print("[target_valid_mask] histology mask is empty")
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
        print(f"[target_valid_mask] hr scale (sx,sy) = ({sx:.1f}, {sy:.1f})  "
              f"roi = y[{hr_y0}:{hr_y1}] x[{hr_x0}:{hr_x1}]  "
              f"coverage = {target_valid_mask.mean():.6f}")

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
    """Dense HR pull-back:  HR pixel -> common -> surface index -> xyz -> MRI.

    xy_map_sym sends surface index -> common-canvas xy.  We need the inverse, so
    we scatter-interpolate (surface row, surface col) as two scalar fields over
    the common-space point cloud, then query it at every HR pixel mapped back
    into common space.  Linear inside the convex hull; nearest as a fallback,
    but ONLY where target_valid_mask already says the pixel is legitimate, so
    the nearest-neighbour extrapolation can never invent tissue.

    Returns (mri_on_hr [Hhr,Whr] float32,
             surface_valid_hr [Hhr,Whr] bool,
             final_valid_hr [Hhr,Whr] bool).
    """
    _xyz = xyz_norm_final.detach().cpu().numpy() if torch.is_tensor(xyz_norm_final) else np.asarray(xyz_norm_final)
    _xy_map = xy_map_sym.detach().cpu().numpy() if torch.is_tensor(xy_map_sym) else np.asarray(xy_map_sym)
    _valid_display = np.asarray(valid_display, dtype=bool)
    _target_valid_mask = np.asarray(target_valid_mask, dtype=bool)

    Hhr, Whr = _hr_2d(Ih_hr).shape
    Hs, Ws = _xyz.shape[:2]
    if verbose:
        print(f"[resample_mri_to_hr] HR histology {Hhr} x {Whr}   surface {Hs} x {Ws}")

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
        print(f"[resample_mri_to_hr] surface_valid coverage = {surface_valid_hr.mean():.6f}   "
              f"final_valid coverage = {final_valid_hr.mean():.6f}")

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
    """Entry point: take the refined sheet all the way to HR histology space.

    Mirrors refine_curved_sheet() in ddf_estimation_v12 -- one call, prints its
    own progress, returns the arrays the notebook needs.

    Returns
    -------
    mri_on_hr        : [Hhr,Whr] float32, MRI resampled onto HR histology pixels
    final_valid_hr   : [Hhr,Whr] bool
    A_hr_to_common   : 4x4
    A_common_to_hr   : 4x4
    info             : dict(target_valid_mask, surface_valid_hr, support_w_hr, Hhr, Whr)
    """
    print("\n-------- MRI on Histology Space --------")

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

