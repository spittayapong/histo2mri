import numpy as np
import torch
import SimpleITK as sitk
from matplotlib import pyplot as plt
import os

from utilities import sample_mri_cubic_3d, robust01, make_monotone_lut, apply_lut_monotone

from scipy.ndimage import (
    gaussian_filter,
    maximum_filter1d,
    minimum_filter1d,
    binary_fill_holes,
    map_coordinates,
)
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# ============================================================
# v12 = v7 + the "MRI on Histology Space" stage lifted out of the notebook.
#
# v7 already held the low-level pieces (affine builders, the scatter/splat
# backprojection, the visual checks).  What lived inline in the notebook was the
# stage that ties them together:
#
#   1. compose the HR <-> common-canvas affine from the three sub-affines
#   2. build target_valid_mask: where in HR histology space is the surface
#      actually able to deliver a sample
#   3. pull the MRI through, densely, at HR resolution, tile by tile:
#         HR pixel -> common space -> surface index -> xyz_norm_final -> MRI
#
# Those are now build_A_hr_common(), build_target_valid_mask() and
# resample_mri_to_hr(), with mri_on_histology_space() as the single entry point
# -- the same shape as refine_curved_sheet() in ddf_estimation_v12.
#
# This is a pure refactor.  Every array it produces is bit-identical to what the
# notebook cells produced; the extraction is verified against a literal copy of
# those cells.  No parameter was retuned.
#
# NOTE on the affine: A_common_to_hr is built from a chain of diagonal-plus-
# translation matrices, so its [0,0] and [1,1] entries are the x/y scales.
# build_target_valid_mask() relies on that to size its smoothing kernels.  It
# warns (rather than silently misbehaving) if a rotation ever appears there.
# ============================================================


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
# def inverse_surface_image_to_hr(
#     img_on_surface,          # 2D array: hist_on_init_tps or mri_on_final_sheet
#     xy_hist_map,             # [H,W,2] in COMMON hist frame (x,y)
#     A_common_to_hr,          # 4x4 index affine: [i_hr] = A_common_to_hr @ [i_common]
#     Ih_hr,                   # HR reference image (sitk or 2D np)
#     valid_mask=None,         # [H,W] bool, e.g. valid_surf
#     is_mask=False,           # True for mask backprojection
#     fill_holes=True,         # fill zero-weight holes by nearest neighbor
#     return_weight=False,
# ):
#     import numpy as np
#     import SimpleITK as sitk

#     # target HR size
#     if isinstance(Ih_hr, sitk.Image):
#         hr = sitk.GetArrayFromImage(Ih_hr)
#     else:
#         hr = np.asarray(Ih_hr)
#     if hr.ndim == 3 and hr.shape[0] == 1:
#         hr = hr[0]
#     Hhr, Whr = hr.shape

#     img = np.asarray(img_on_surface, dtype=np.float32)
#     xy = np.asarray(xy_hist_map, dtype=np.float32)
#     Hs, Ws = img.shape
#     if xy.shape != (Hs, Ws, 2):
#         raise ValueError(f"xy_hist_map must be {(Hs, Ws, 2)}, got {xy.shape}")

#     if valid_mask is None:
#         valid = np.ones((Hs, Ws), dtype=bool)
#     else:
#         valid = np.asarray(valid_mask).astype(bool)
#         if valid.shape != (Hs, Ws):
#             raise ValueError(f"valid_mask must be {(Hs, Ws)}, got {valid.shape}")

#     ys, xs = np.where(valid)
#     vals = img[ys, xs].astype(np.float32)
#     x_common = xy[ys, xs, 0].astype(np.float64)
#     y_common = xy[ys, xs, 1].astype(np.float64)

#     A = np.asarray(A_common_to_hr, dtype=np.float64)
#     if A.shape != (4, 4):
#         raise ValueError(f"A_common_to_hr must be 4x4, got {A.shape}")

#     p = np.stack([x_common, y_common, np.zeros_like(x_common), np.ones_like(x_common)], axis=1)  # [N,4]
#     q = (A @ p.T).T
#     w = q[:, 3]
#     ok = np.abs(w) > 1e-8

#     xh = np.full_like(x_common, np.nan, dtype=np.float32)
#     yh = np.full_like(y_common, np.nan, dtype=np.float32)
#     xh[ok] = (q[ok, 0] / w[ok]).astype(np.float32)
#     yh[ok] = (q[ok, 1] / w[ok]).astype(np.float32)

#     keep = np.isfinite(xh) & np.isfinite(yh)
#     xh, yh, vals = xh[keep], yh[keep], vals[keep]

#     # keep points projected inside HR bounds
#     inside = (xh >= 0) & (xh <= Whr - 1) & (yh >= 0) & (yh <= Hhr - 1)
#     xh, yh, vals = xh[inside], yh[inside], vals[inside]

#     out = np.zeros((Hhr, Whr), dtype=np.float32)
#     wgt = np.zeros((Hhr, Whr), dtype=np.float32)

#     if is_mask:
#         xn = np.rint(xh).astype(np.int32)
#         yn = np.rint(yh).astype(np.int32)
#         m = (xn >= 0) & (xn < Whr) & (yn >= 0) & (yn < Hhr)
#         np.add.at(out, (yn[m], xn[m]), vals[m])
#         np.add.at(wgt, (yn[m], xn[m]), 1.0)
#         out = (out > 0).astype(np.float32)
#     else:
#         x0 = np.floor(xh).astype(np.int32); y0 = np.floor(yh).astype(np.int32)
#         x1 = x0 + 1; y1 = y0 + 1
#         wx = xh - x0; wy = yh - y0

#         for xx, yy, ww in [
#             (x0, y0, (1 - wx) * (1 - wy)),
#             (x1, y0, wx * (1 - wy)),
#             (x0, y1, (1 - wx) * wy),
#             (x1, y1, wx * wy),
#         ]:
#             m = (xx >= 0) & (xx < Whr) & (yy >= 0) & (yy < Hhr) & (ww > 0)
#             np.add.at(out, (yy[m], xx[m]), vals[m] * ww[m])
#             np.add.at(wgt, (yy[m], xx[m]), ww[m])

#         nz = wgt > 1e-6
#         out[nz] /= wgt[nz]

#         if fill_holes and np.any(~nz) and np.any(nz):
#             try:
#                 from scipy.ndimage import distance_transform_edt
#                 _, (ri, rj) = distance_transform_edt(~nz, return_indices=True)
#                 out[~nz] = out[ri[~nz], rj[~nz]]
#             except Exception:
#                 pass  # if scipy not available, keep holes as 0

#     print(f"[inverse_surface_image_to_hr] coverage={float((wgt > 1e-6).mean()):.4f}")

#     if return_weight:
#         return out, wgt
#     return out

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

def build_surface_to_hr_map(
    xy_hist_map,          # [H,W,2] in COMMON hist frame
    A_common_to_hr,       # 4x4 index affine
    Ih_hr,                # target HR image
    valid_mask=None,      # [H,W] bool (e.g., valid_surf)
):
    import numpy as np
    import SimpleITK as sitk

    xy = np.asarray(xy_hist_map, np.float32)
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError(f"xy_hist_map must be [H,W,2], got {xy.shape}")
    Hs, Ws = xy.shape[:2]

    hr = sitk.GetArrayFromImage(Ih_hr) if isinstance(Ih_hr, sitk.Image) else np.asarray(Ih_hr)
    if hr.ndim == 3 and hr.shape[0] == 1:
        hr = hr[0]
    Hhr, Whr = hr.shape

    if valid_mask is None:
        valid = np.ones((Hs, Ws), dtype=bool)
    else:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.shape != (Hs, Ws):
            raise ValueError(f"valid_mask shape {valid.shape} != {(Hs, Ws)}")

    ys, xs = np.where(valid)
    xc = xy[ys, xs, 0].astype(np.float64)
    yc = xy[ys, xs, 1].astype(np.float64)

    A = np.asarray(A_common_to_hr, np.float64)
    if A.shape != (4, 4):
        raise ValueError(f"A_common_to_hr must be 4x4, got {A.shape}")

    p = np.stack([xc, yc, np.zeros_like(xc), np.ones_like(xc)], axis=1)  # [N,4]
    q = (A @ p.T).T
    w = q[:, 3]
    ok = np.abs(w) > 1e-8

    xh = (q[ok, 0] / w[ok]).astype(np.float32)
    yh = (q[ok, 1] / w[ok]).astype(np.float32)
    ys = ys[ok]
    xs = xs[ok]

    inside = (xh >= 0) & (xh <= Whr - 1) & (yh >= 0) & (yh <= Hhr - 1)
    xh, yh, ys, xs = xh[inside], yh[inside], ys[inside], xs[inside]

    x0 = np.floor(xh).astype(np.int32); y0 = np.floor(yh).astype(np.int32)
    x1 = x0 + 1; y1 = y0 + 1
    wx = xh - x0; wy = yh - y0

    w00 = (1 - wx) * (1 - wy)
    w10 = wx * (1 - wy)
    w01 = (1 - wx) * wy
    w11 = wx * wy

    return {
        "Hs": Hs, "Ws": Ws, "Hhr": Hhr, "Whr": Whr,
        "src_y": ys, "src_x": xs,
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
        "w00": w00.astype(np.float32),
        "w10": w10.astype(np.float32),
        "w01": w01.astype(np.float32),
        "w11": w11.astype(np.float32),
    }


def resample_mri_to_hr_with_precomputed_map(
    mri_vol,              # slab_Im or slab_raw_Im
    xyz_norm_sheet,       # [H,W,3] final sheet
    surf2hr_map,          # output of build_surface_to_hr_map
    device="cuda:0",
    kernel="catmull-rom",
    padding_mode="border",
    return_weight=False,
    splat_radius=2,
    splat_sigma=1.0,
    min_weight=1e-6,
    fill_holes=True,
    reconstruct_sigma=None,
):
    vol = sitk.GetArrayFromImage(mri_vol).astype(np.float32) if isinstance(mri_vol, sitk.Image) else np.asarray(mri_vol, np.float32)
    if vol.ndim == 2:
        vol = vol[None, ...]
    Z, Y, X = vol.shape

    Hs, Ws = surf2hr_map["Hs"], surf2hr_map["Ws"]
    if torch.is_tensor(xyz_norm_sheet):
        xyz = xyz_norm_sheet.detach().to("cpu", dtype=torch.float32).numpy()
    else:
        xyz = np.asarray(xyz_norm_sheet, dtype=np.float32)

    if xyz.shape[:2] != (Hs, Ws):
        raise ValueError(f"xyz_norm_sheet shape {xyz.shape[:2]} != map shape {(Hs, Ws)}")

    P = np.empty_like(xyz, dtype=np.float32)
    P[..., 0] = xyz[..., 0] * (X - 1)
    P[..., 1] = xyz[..., 1] * (Y - 1)
    P[..., 2] = xyz[..., 2] * (Z - 1)

    inside_mri = (
        (P[..., 0] >= 0) & (P[..., 0] <= X - 1) &
        (P[..., 1] >= 0) & (P[..., 1] <= Y - 1) &
        (P[..., 2] >= 0) & (P[..., 2] <= Z - 1)
    )

    vol_t = torch.from_numpy(vol).to(device)[None, None]
    P_t = torch.from_numpy(P).to(device=device, dtype=torch.float32)
    mri_sheet = sample_mri_cubic_3d(vol_t, P_t, kernel=kernel, padding_mode=padding_mode)[0, 0].cpu().numpy()

    ys = surf2hr_map["src_y"]
    xs = surf2hr_map["src_x"]
    vals = mri_sheet[ys, xs]

    ok = inside_mri[ys, xs] & np.isfinite(vals)
    vals = vals[ok]

    x0 = surf2hr_map["x0"][ok]
    x1 = surf2hr_map["x1"][ok]
    y0 = surf2hr_map["y0"][ok]
    y1 = surf2hr_map["y1"][ok]
    w00 = surf2hr_map["w00"][ok]
    w10 = surf2hr_map["w10"][ok]
    w01 = surf2hr_map["w01"][ok]
    w11 = surf2hr_map["w11"][ok]

    Hhr, Whr = surf2hr_map["Hhr"], surf2hr_map["Whr"]
    out_num = np.zeros((Hhr, Whr), np.float32)
    raw_wgt = np.zeros((Hhr, Whr), np.float32)

    def _acc(xx, yy, ww, vv):
        m = (xx >= 0) & (xx < Whr) & (yy >= 0) & (yy < Hhr) & (ww > 0) & np.isfinite(vv)
        np.add.at(out_num, (yy[m], xx[m]), vv[m] * ww[m])
        np.add.at(raw_wgt, (yy[m], xx[m]), ww[m])

    if splat_radius <= 0:
        _acc(x0, y0, w00, vals)
        _acc(x1, y0, w10, vals)
        _acc(x0, y1, w01, vals)
        _acc(x1, y1, w11, vals)
    else:
        offs = range(-int(splat_radius), int(splat_radius) + 1)
        two_sig2 = 2.0 * float(splat_sigma) * float(splat_sigma) + 1e-12

        for dx in offs:
            for dy in offs:
                g = np.float32(np.exp(-(dx * dx + dy * dy) / two_sig2))
                _acc(x0 + dx, y0 + dy, w00 * g, vals)
                _acc(x1 + dx, y0 + dy, w10 * g, vals)
                _acc(x0 + dx, y1 + dy, w01 * g, vals)
                _acc(x1 + dx, y1 + dy, w11 * g, vals)

    def _estimate_axis_spacing(a0, a1):
        pts = np.unique(np.concatenate([a0, a1]).astype(np.int32))
        if pts.size < 2:
            return 0.0
        diffs = np.diff(np.sort(pts))
        diffs = diffs[diffs > 1]
        if diffs.size == 0:
            diffs = np.diff(np.sort(pts))
            diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return 0.0
        return float(np.median(diffs))

    if reconstruct_sigma is None:
        sx = _estimate_axis_spacing(x0, x1)
        sy = _estimate_axis_spacing(y0, y1)
        reconstruct_sigma = (
            float(np.clip(0.35 * sy, 1.0, 32.0)),
            float(np.clip(0.35 * sx, 1.0, 32.0)),
        )

    acc = out_num
    wgt = raw_wgt
    raw_support = raw_wgt > float(min_weight)

    if reconstruct_sigma is not None:
        try:
            from scipy.ndimage import gaussian_filter

            if np.isscalar(reconstruct_sigma):
                sig_y = sig_x = float(reconstruct_sigma)
            else:
                sig_y, sig_x = [float(v) for v in reconstruct_sigma]

            if sig_y > 0 or sig_x > 0:
                acc = gaussian_filter(acc, sigma=(sig_y, sig_x), mode="constant", cval=0.0)
                wgt = gaussian_filter(wgt, sigma=(sig_y, sig_x), mode="constant", cval=0.0)
                support = gaussian_filter(raw_support.astype(np.float32), sigma=(sig_y, sig_x), mode="constant", cval=0.0)
                support_thr = 0.05 * float(support.max()) if support.size and support.max() > 0 else 0.0
                nz = support > max(float(min_weight), support_thr)
            else:
                nz = raw_support
        except Exception:
            nz = raw_support
    else:
        nz = raw_support

    out = np.zeros((Hhr, Whr), np.float32)
    valid = nz & (wgt > float(min_weight))
    out[valid] = acc[valid] / np.maximum(wgt[valid], float(min_weight))
    wgt = np.where(nz, wgt, 0.0).astype(np.float32)

    if fill_holes and np.any(~valid) and np.any(valid):
        try:
            from scipy.ndimage import distance_transform_edt
            _, (ri, rj) = distance_transform_edt(~valid, return_indices=True)
            out[~valid] = out[ri[~valid], rj[~valid]]
        except Exception:
            pass

    print(
        "[resample_mri_to_hr_with_precomputed_map] "
        f"inside-mri pts={ok.mean():.3f}, "
        f"raw coverage={(raw_wgt > min_weight).mean():.3f}, "
        f"recon coverage={(wgt > min_weight).mean():.3f}, "
        f"reconstruct_sigma={reconstruct_sigma}"
    )

    if return_weight:
        return out, wgt, mri_sheet
    return out


def normalize_image(img, eps=1e-8):
    """Normalize image to [0, 1]."""
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()
    return (img - min_val) / (max_val - min_val + eps)

def make_checkerboard(a, b, tile_size=32):
    """
    Create a checkerboard composite of two normalized images.
    Both must have the same shape.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Normalize both images
    a_norm = normalize_image(a)
    b_norm = normalize_image(b)

    h, w = a.shape

    yy, xx = np.indices((h, w))
    mask = ((yy // tile_size + xx // tile_size) % 2).astype(bool)

    out = np.where(mask, a_norm, b_norm)
    return out

# Precheck: visualize sampled mri on histology space
def check_mri_on_histo(mri_on_histo, Ih_hr, Mh_hr, tile_size=384, save_file=None, save_file_single=None):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()

    arr_Ih_hr = sitk.GetArrayFromImage(Ih_hr)[0]
    arr_Mh_hr = sitk.GetArrayFromImage(Mh_hr)[0]

    hist_disp = np.flipud(arr_Ih_hr).astype(np.float32)
    mri_disp  = np.flipud(mri_on_histo).astype(np.float32)
    mask_disp = np.flipud(arr_Mh_hr).astype(np.float32)

    # keep same behavior as final-check style, but in histology support
    hist_disp = hist_disp * mask_disp
    mri_disp  = mri_disp * mask_disp

    mask_src = None
    mask_ref = None
    h01, _ = robust01(hist_disp, mask=mask_src, pclip=(1, 99))
    mri_lut, _ = robust01(mri_disp, mask=mask_ref, pclip=(1, 99))

    xq, yq = make_monotone_lut(
        src01=h01, ref01=mri_lut,
        src_mask=mask_src, ref_mask=mask_ref,
        n_quant=2048
    )
    histo_lut = apply_lut_monotone(h01, xq, yq)

    # hist_norm = normalize_image(hist_disp)
    # mri_norm  = normalize_image(mri_disp)
    hist_norm = hist_disp
    mri_norm  = mri_disp

    axes[0].imshow(hist_norm, cmap='gray')
    axes[0].set_title("Histology on histology space")

    axes[1].imshow(mri_norm, cmap='gray')
    axes[1].set_title("MRI on histology space")

    checker = make_checkerboard(hist_norm, mri_norm, tile_size=tile_size)
    axes[2].imshow(checker, cmap='gray')
    axes[2].set_title(f"Checkerboard (tile={tile_size})")

    axes[3].imshow(histo_lut, cmap='gray')
    axes[3].set_title("Histology on histology space")

    axes[4].imshow(mri_lut, cmap='gray')
    axes[4].set_title("MRI on histology space")

    checker_lut = make_checkerboard(histo_lut, mri_lut, tile_size=tile_size)
    axes[5].imshow(checker_lut, cmap='gray')
    axes[5].set_title(f"Checkerboard (tile={tile_size})")

    plt.tight_layout()

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        fig.savefig(save_file, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_file}")

    plt.show()

    if save_file_single is not None:
        os.makedirs(os.path.dirname(save_file_single), exist_ok=True)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(mri_disp, cmap='gray')
        ax2.set_title("MRI on histology space")
        ax2.axis("on")
        fig2.tight_layout()
        fig2.savefig(save_file_single, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved image to: {save_file_single}")

def check_mri_on_histo_rgb(
    arr_Ih_hr,
    arr_Ih_rgb_hr,
    mri_clean,
    arr_Mh_hr=None,
    tile=384,
    pclip=(1, 99),
    n_quant=2048,
    save_file=None,
    flip_display=False,
):
    def _to_np(x):
        if isinstance(x, sitk.Image):
            return sitk.GetArrayFromImage(x)
        return np.asarray(x)

    def to_gray2d(a, name):
        a = _to_np(a)
        if a.ndim == 2:
            return a.astype(np.float32)
        if a.ndim == 3 and a.shape[0] == 1:
            return a[0].astype(np.float32)
        if a.ndim == 3 and a.shape[-1] == 1:
            return a[..., 0].astype(np.float32)
        raise ValueError(f"{name} must be 2D grayscale (or singleton 3D), got {a.shape}")

    def to_rgb3(a, name):
        a = _to_np(a)
        if a.ndim == 3 and a.shape[-1] == 3:
            return a.astype(np.float32)
        raise ValueError(f"{name} must be RGB with shape (H,W,3), got {a.shape}")

    hist_gray = to_gray2d(arr_Ih_hr, "arr_Ih_hr")
    hist_rgb  = to_rgb3(arr_Ih_rgb_hr, "arr_Ih_rgb_hr")
    mri_gray  = to_gray2d(mri_clean, "mri_clean")

    if hist_gray.shape != mri_gray.shape:
        raise ValueError(f"Shape mismatch: hist_gray {hist_gray.shape} vs mri_gray {mri_gray.shape}")
    if hist_rgb.shape[:2] != hist_gray.shape:
        raise ValueError(f"Shape mismatch: hist_rgb {hist_rgb.shape} vs hist_gray {hist_gray.shape}")

    mask = None
    if arr_Mh_hr is not None:
        mask = (to_gray2d(arr_Mh_hr, "arr_Mh_hr") > 0)
        if mask.shape != hist_gray.shape:
            raise ValueError(f"Mask shape mismatch: {mask.shape} vs {hist_gray.shape}")

    if mask is not None:
        hist_gray = hist_gray * mask
        mri_gray  = mri_gray * mask
        hist_rgb  = hist_rgb * mask[..., None]

    h01, _ = robust01(hist_gray, mask=mask, pclip=pclip)
    m01, _ = robust01(mri_gray,  mask=mask, pclip=pclip)

    xq, yq = make_monotone_lut(
        src01=h01, ref01=m01,
        src_mask=mask, ref_mask=mask,
        n_quant=n_quant
    )
    histo_lut = apply_lut_monotone(h01, xq, yq)

    rgb = hist_rgb.copy()
    if rgb.max() > 1.0:
        rgb /= 255.0

    mri_rgb = np.repeat(m01[..., None], 3, axis=-1)

    yy, xx = np.indices(h01.shape)
    cb = ((yy // int(tile) + xx // int(tile)) % 2 == 0)

    checker_lut = np.where(cb, histo_lut, m01)
    checker_rgb = np.where(cb[..., None], rgb, mri_rgb)

    if mask is not None:
        histo_lut_disp   = np.where(mask, histo_lut, 1.0)
        m01_disp         = np.where(mask, m01, 1.0)
        checker_lut_disp = np.where(mask, checker_lut, 1.0)
        rgb_disp         = np.where(mask[..., None], rgb, 1.0)
        checker_rgb_disp = np.where(mask[..., None], checker_rgb, 1.0)
        mask_disp        = mask.astype(np.float32)
    else:
        histo_lut_disp   = histo_lut
        m01_disp         = m01
        checker_lut_disp = checker_lut
        rgb_disp         = rgb
        checker_rgb_disp = checker_rgb
        mask_disp        = np.ones_like(h01, dtype=np.float32)

    if flip_display:
        histo_lut_disp   = np.flipud(histo_lut_disp)
        m01_disp         = np.flipud(m01_disp)
        checker_lut_disp = np.flipud(checker_lut_disp)
        rgb_disp         = np.flipud(rgb_disp)
        checker_rgb_disp = np.flipud(checker_rgb_disp)
        mask_disp        = np.flipud(mask_disp)

    fig, ax = plt.subplots(2, 3, figsize=(20, 14))

    ax[0, 0].imshow(histo_lut_disp, cmap="gray", origin="lower")
    ax[0, 0].set_title("Histology (LUT -> MRI)")
    ax[0, 1].imshow(m01_disp, cmap="gray", origin="lower")
    ax[0, 1].set_title("MRI (robust01)")
    ax[0, 2].imshow(checker_lut_disp, cmap="gray", origin="lower")
    ax[0, 2].set_title(f"Checkerboard LUT (tile={tile})")

    ax[1, 0].imshow(rgb_disp, origin="lower")
    ax[1, 0].set_title("RGB Histo")
    ax[1, 1].imshow(m01_disp, cmap="gray", origin="lower")
    ax[1, 1].set_title("MRI on Histo")
    ax[1, 2].imshow(checker_rgb_disp, origin="lower")
    ax[1, 2].set_title(f"Checkerboard RGB (tile={tile})")

    for a in ax.ravel():
        a.axis("off")

    fig.tight_layout()

    if save_file is not None:
        d = os.path.dirname(save_file)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_file}")

    plt.show()
    plt.close(fig)


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


def build_A_hr_common(proc_Ih, scaled_proc_Ih, A_hr_to_proc, oy_h, ox_h):
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

    arr_Mh_hr = sitk.GetArrayFromImage(Mh_hr).astype(bool) if isinstance(Mh_hr, sitk.Image) \
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

    vol_arr = sitk.GetArrayFromImage(slab_Im).astype(np.float32) if isinstance(slab_Im, sitk.Image) \
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

    A_hr_to_common, A_common_to_hr = build_A_hr_common(
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
        check_mri_on_histo(
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
