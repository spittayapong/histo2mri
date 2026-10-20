import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import os
import imageio.v2 as imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import sample_mri_cubic_3d, sample_mri_cubic_3d_tiled
from pose_estimation_v6 import ThinPlateSplineSurface3D, _tps_residuals, _fit_tps_on_subset
from refinement_v7 import _prepare_data_tensor, BSplineResidualSurface3D, combined_masked_ssim_ngf_loss, _upsample_ctrl, _copy_ctrl, ScalarNormalOffsetField, bending_energy, sheet_inside_volume_penalty, _plot_surface, sample_mri_on_sheet, BSplineFFD2D, bending_energy_2d

# ============================================================
# v12 = v11 + LOCAL RIGIDITY.
#
# Why: phi is a stationary velocity field integrated by scaling-and-squaring,
# so det(J) > 0 is guaranteed -- the sheet never folds. But non-folding is not
# non-melting. The only restraint on phi in v11 was bending_energy_2d(phi.ctrl),
# a squared-SECOND-difference penalty on the velocity control points, and second
# derivatives vanish identically on any affine field. A field that expands local
# area by 3.5x or compresses it by 2.6x therefore costs *exactly zero* bending
# energy. Measured on a real section: area ratio final/base had
# p1..p99 = 0.378 .. 3.471 with 0% folded.
#
# In the rigidity decomposition of Staring/Klein/Pluim (2007), rigidity of
# J = I + grad(u) is three conditions:
#     affinity      : d2u = 0        <- v11's bending energy already does this
#     orthonormality: J^T J = I      <- MISSING in v11 (no shear/scale penalty)
#     properness    : det J = 1      <- MISSING in v11 (no area penalty)
# v11 had exactly the one condition that does not prevent melting.
#
# v12 adds the other two, but expressed on the SURFACE rather than on phi's 2D
# Jacobian: we penalize drift of the first fundamental form (E,F,G) of
# xyz_norm away from that of xyz_norm_base. This is the Riemannian form of the
# same two conditions, and it has three practical advantages:
#   - reg_fn already receives xyz_norm, so ONE term simultaneously constrains
#     phi, rf_psi, alpha AND rf_beta (alpha has the same affine blindness: a
#     linear ramp tilts the sheet for free);
#   - it is measured relative to the TPS base surface, so it never fights the
#     genuine sheet curvature the matches already established -- it only charges
#     what the refinement itself adds;
#   - properness becomes log(area/area0)^2, which is symmetric: a 2x expansion
#     and a 2x compression cost the same. (area/area0 - 1)^2 would under-charge
#     compression, and compression is the direction that melts.
#
# Defaults: lam_area on, lam_shape OFF. Melting is predominantly area exchange;
# leaving shear free lets the genuine sectioning deformation through, which is
# physically real. Turn lam_shape up only if area preservation alone is not
# enough.
#
# NOTE on units: xyz_norm is normalized per-axis by (dim-1), so multiplying by
# dims recovers voxel-index coords. For ISOTROPIC spacing that is a uniform
# scaling of physical space, and since every rigidity term here is a RATIO
# against the base surface, the spacing factor cancels exactly. No GetSpacing()
# needed. (For anisotropic data this would need a spacing weight.)
# ============================================================

# ============================================================
# v11 refinement design (see conversation notes): four fields, alternated
# per pyramid level (see pyramid_refinement docstring below for detail).
#   - phi     : 2D stationary velocity field, integrated via scaling-and-
#               squaring into a diffeomorphic displacement, RESAMPLES the
#               fixed matches-warm-started base surface tangentially.
#               Unconditional (not gated), NOT a warp of the histology image.
#   - alpha   : scalar offset along the fixed surface normal. Unconditional.
#   - rf_psi  : a second, independent velocity field (same mechanism as phi),
#               gated by (1 - conf), composed on top of phi's resample.
#   - rf_beta : a second, independent scalar normal offset (same mechanism
#               as alpha), gated by (1 - conf).
#   rf_psi + rf_beta replace v10's single free 3-vector residual_free field
#   (tangential + normal = any 3D direction, but each half now goes through
#   the same diffeomorphic/smooth treatment as phi/alpha instead of being
#   an unconstrained free vector that could fold the surface).
# ============================================================

def _decompose_A2(A2):
    U, svals, Vt = np.linalg.svd(A2)
    R2 = U @ Vt
    if np.linalg.det(R2) < 0:
        U[:, -1] *= -1
        R2 = U @ Vt
    S = Vt.T @ np.diag(svals) @ Vt
    s_uni = float(np.sqrt(max(np.linalg.det(S), 1e-12)))
    A_residual = S / max(s_uni, 1e-8)
    return s_uni, R2, A_residual

def fit_G_plane_canvas_from_matches(
    clean_matches,
    x_range_native,              # (x0,x1) in MRI native plane coords
    y_range_native,              # (y0,y1)
    plane_step=1.0,
    mri_key="mri_center_zyx",    # (z,y,x)
    hist_key="hist_center_yx",   # (y,x)
    mri_hw=None,                 # (H,W) for hard filtering
    hist_hw=None,                # (H,W) for hard filtering
    robust=True,
    ransac_thr_px=10.0,
    ransac_iters=2000,
    trim_thr_px=20.0,
    min_inliers=4,
    rng_seed=0,
):
    def _fit_affine_ls(P, U):
        P1 = np.column_stack([P, np.ones(len(P), dtype=np.float32)]).astype(np.float32)
        cx, *_ = np.linalg.lstsq(P1, U[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(P1, U[:, 1], rcond=None)
        A2 = np.array([[cx[0], cx[1]], [cy[0], cy[1]]], dtype=np.float32)
        t = np.array([cx[2], cy[2]], dtype=np.float32)  # (tx,ty)
        return A2, t

    def _fit_similarity(P, U):
        # Best similarity U ~= s*R*P + t (Umeyama). 4 DOF: uniform scale, rotation,
        # translation -- no independent per-axis scale, so it cannot collapse/smear
        # the way a full affine does on few/noisy matches.
        P = np.asarray(P, np.float64); U = np.asarray(U, np.float64)
        muP = P.mean(0); muU = U.mean(0)
        Pc = P - muP; Uc = U - muU
        C = (Uc.T @ Pc) / len(P)                          # 2x2 cross-covariance
        Uu, D, Vt = np.linalg.svd(C)
        S = np.eye(2, dtype=np.float64)
        if np.linalg.det(Uu) * np.linalg.det(Vt) < 0:     # guard against reflection
            S[-1, -1] = -1.0
        R = Uu @ S @ Vt
        varP = (Pc ** 2).sum() / len(P)
        s = float((D * np.diag(S)).sum() / max(varP, 1e-12))
        A2 = (s * R).astype(np.float32)
        t = (muU - A2 @ muP).astype(np.float32)
        return A2, t

    def _residuals(A2, t, P, U):
        U_hat = P @ A2.T + t
        return np.linalg.norm(U_hat - U, axis=1)

    # collect
    X_zyx, U_yx = [], []
    for m in clean_matches:
        if mri_key not in m or hist_key not in m:
            continue
        z, y, x = m[mri_key]
        if (z, y, x) == (-1, -1, -1):
            continue
        hy, hx = m[hist_key]
        X_zyx.append((float(z), float(y), float(x)))
        U_yx.append((float(hy), float(hx)))

    X_zyx = np.asarray(X_zyx, np.float32)
    U_yx = np.asarray(U_yx, np.float32)
    if len(X_zyx) < 3:
        raise ValueError("Need >= 3 valid matches.")

    z = X_zyx[:, 0]
    y = X_zyx[:, 1]
    x = X_zyx[:, 2]
    hy = U_yx[:, 0]
    hx = U_yx[:, 1]

    # hard bounds filter (critical)
    keep = np.ones(len(X_zyx), dtype=bool)
    if mri_hw is not None:
        Hmri, Wmri = mri_hw
        keep &= (x >= 0) & (x < Wmri) & (y >= 0) & (y < Hmri)
    if hist_hw is not None:
        Hh, Wh = hist_hw
        keep &= (hx >= 0) & (hx < Wh) & (hy >= 0) & (hy < Hh)

    X_zyx = X_zyx[keep]
    U_yx = U_yx[keep]
    if len(X_zyx) < 3:
        raise ValueError("Too few matches after bounds filtering.")

    z = X_zyx[:, 0]
    y = X_zyx[:, 1]
    x = X_zyx[:, 2]

    # native plane: z = a*x + b*y + c_native
    A = np.c_[x, y, np.ones_like(x)]
    a, b, c_native = np.linalg.lstsq(A, z, rcond=None)[0]

    # encode origin into G
    x0, x1 = map(float, x_range_native)
    y0, y1 = map(float, y_range_native)
    step = float(plane_step)

    xs = np.arange(x0, x1 + 1e-6, step, dtype=np.float32)
    ys = np.arange(y0, y1 + 1e-6, step, dtype=np.float32)
    Wm, Hm = len(xs), len(ys)

    # local plane coords u,v where x=x0+u, y=y0+v
    c_local = float(c_native + a * x0 + b * y0)

    P_local = np.column_stack([x - x0, y - y0]).astype(np.float32)        # (u,v)
    U_xy = np.column_stack([U_yx[:, 1], U_yx[:, 0]]).astype(np.float32)   # (x,y)

    # robust affine fit
    N = len(P_local)
    if robust and N >= 4:
        rng = np.random.default_rng(rng_seed)
        best_inl = None
        best_count = -1
        for _ in range(ransac_iters):
            s = rng.choice(N, 3, replace=False)
            try:
                A2_c, t_c = _fit_similarity(P_local[s], U_xy[s])
            except np.linalg.LinAlgError:
                continue
            r_c = _residuals(A2_c, t_c, P_local, U_xy)
            inl = r_c <= float(ransac_thr_px)
            c = int(inl.sum())
            if c > best_count:
                best_count = c
                best_inl = inl

        if best_inl is None or best_inl.sum() < min_inliers:
            best_inl = np.ones(N, dtype=bool)

        A2, t_xy = _fit_similarity(P_local[best_inl], U_xy[best_inl])

        # one trim-refit pass
        r = _residuals(A2, t_xy, P_local, U_xy)
        inl2 = r <= float(trim_thr_px)
        if inl2.sum() >= min_inliers:
            A2, t_xy = _fit_similarity(P_local[inl2], U_xy[inl2])
            inlier_mask = inl2
        else:
            inlier_mask = best_inl
    else:
        # A2, t_xy = _affine_2d(P_local, U_xy)
        A2, t_xy = _fit_similarity(P_local, U_xy)
        inlier_mask = np.ones(N, dtype=bool)

    A2_invT = np.linalg.inv(A2).T.astype(np.float32)
    s_uni, R2, A_residual = _decompose_A2(A2.astype(np.float64))

    # 3D embedding
    X0 = np.array([x0, y0, c_local], dtype=np.float32)
    e1 = np.array([1.0, 0.0, float(a)], dtype=np.float32)
    e2 = np.array([0.0, 1.0, float(b)], dtype=np.float32)
    n = np.cross(e1, e2)
    n = (n / (np.linalg.norm(n) + 1e-12)).astype(np.float32)

    U_hat = P_local @ A2.T + t_xy
    err = np.linalg.norm(U_hat - U_xy, axis=1)
    err_in = err[inlier_mask] if inlier_mask.any() else err

    G = {
        "X0": X0, "e1": e1, "e2": e2, "n": n,
        "A2": A2.astype(np.float32),
        "t_xy": t_xy.astype(np.float32),
        "A2_invT": A2_invT,
        "s": np.float32(s_uni),
        "R2": R2.astype(np.float32),
        "A_residual": A_residual.astype(np.float32),

        "plane_abc": (float(a), float(b), float(c_local)),
        "origin_xy_native": (x0, y0),

        "x_range": (0.0, float((Wm - 1) * step)),
        "y_range": (0.0, float((Hm - 1) * step)),
        "plane_step": step,

        "x_range_native": (x0, x1),
        "y_range_native": (y0, y1),

        "reprojection": {
            "mean_px": float(err_in.mean()),
            "median_px": float(np.median(err_in)),
            "p95_px": float(np.percentile(err_in, 95)),
            "max_px": float(err_in.max()),
            "num": int(len(err_in)),
            "num_all": int(len(err)),
        },
    }
    return G

@torch.no_grad()
def resample_mri_on_G(
    mri_vol,
    G,
    device="cuda:0",
    kernel="catmull-rom",
    padding_mode="zeros",
    use_tiled=False,
    tile_hw=512,
    canvas_shape=None,          # None -> return local plane image; (Hc,Wc) -> place on canvas
    common_shift_yx=None,       # used only when canvas_shape is given
):
    vol = sitk.GetArrayFromImage(mri_vol).astype(np.float32) if isinstance(mri_vol, sitk.Image) else np.asarray(mri_vol, np.float32)
    if vol.ndim == 2:
        vol = vol[None, ...]
    Z, Y, X = vol.shape

    X0 = np.asarray(G["X0"], np.float32)
    e1 = np.asarray(G["e1"], np.float32)
    e2 = np.asarray(G["e2"], np.float32)
    x0, x1 = map(float, G["x_range"])
    y0, y1 = map(float, G["y_range"])
    step = float(G.get("plane_step", 1.0))

    xs = np.arange(x0, x1 + 1e-6, step, dtype=np.float32)
    ys = np.arange(y0, y1 + 1e-6, step, dtype=np.float32)
    Wm, Hm = len(xs), len(ys)

    if canvas_shape is None:
        # local plane image
        uu, vv = np.meshgrid(xs, ys)  # local plane coords
        in_plane = np.ones_like(uu, dtype=bool)
    else:
        # place local plane on common canvas
        Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])
        if common_shift_yx is None:
            oy = (Hc - Hm) // 2
            ox = (Wc - Wm) // 2
        else:
            oy, ox = int(common_shift_yx[0]), int(common_shift_yx[1])

        yy, xx = np.meshgrid(np.arange(Hc, dtype=np.float32), np.arange(Wc, dtype=np.float32), indexing="ij")
        uu = (xx - ox) * step + x0
        vv = (yy - oy) * step + y0
        in_plane = (uu >= x0) & (uu <= x1) & (vv >= y0) & (vv <= y1)

    # IMPORTANT: no extra x0/y0 shift here (already encoded in X0)
    XYZ = X0[None, None, :] + uu[..., None] * e1[None, None, :] + vv[..., None] * e2[None, None, :]

    Xv, Yv, Zv = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    valid = in_plane & (Xv >= 0) & (Xv <= X - 1) & (Yv >= 0) & (Yv <= Y - 1) & (Zv >= 0) & (Zv <= Z - 1)

    vol_t = torch.from_numpy(vol).to(device)[None, None]
    P_t = torch.from_numpy(XYZ.astype(np.float32)).to(device)

    if use_tiled:
        out = sample_mri_cubic_3d_tiled(vol_t, P_t, tile_hw=tile_hw, kernel=kernel, padding_mode=padding_mode)
    else:
        out = sample_mri_cubic_3d(vol_t, P_t, kernel=kernel, padding_mode=padding_mode)

    mri = out[0, 0].detach().cpu().numpy()
    if padding_mode.lower() == "zeros":
        mri[~valid] = 0.0
    return mri

def find_clean_matches_common(clean_matches, oy_h, ox_h):
    clean_matches_common = []
    for m in clean_matches:
        if "mri_center_zyx" not in m or "hist_center_yx" not in m:
            continue
        z, y, x = m["mri_center_zyx"]
        if (z, y, x) == (-1, -1, -1):
            continue
        hy, hx = m["hist_center_yx"]

        # hist points are local -> move to common frame
        hyc = hy + oy_h
        hxc = hx + ox_h

        mm = dict(m)
        mm["hist_center_yx"] = (float(hyc), float(hxc))
        # leave mri_center_zyx as-is (already in matcher frame)
        clean_matches_common.append(mm)

    return clean_matches_common

@torch.no_grad()
def resample_histology_on_G(
    hist_img,                   # 2D np or sitk (1,H,W)/(H,W)
    G,
    device="cuda:0",
    canvas_shape=None,          # None -> local uv image, (Hc,Wc) -> common canvas
    common_shift_yx=None,       # optional; if None and canvas, use G["canvas_offset_yx"] if exists
    fill_invalid=0.0,
    return_extra=False,         # True -> (hist_out, valid, uv_hw2, xy_hw2)
):
    # hist -> 2D
    if isinstance(hist_img, sitk.Image):
        h = sitk.GetArrayFromImage(hist_img)
        if h.ndim == 3 and h.shape[0] == 1:
            h = h[0]
    else:
        h = np.asarray(hist_img)
    if h.ndim != 2:
        raise ValueError(f"hist_img must be 2D, got {h.shape}")
    h = h.astype(np.float32)
    Hh, Wh = h.shape

    # uv range from G
    u0, u1 = map(float, G["x_range"])
    v0, v1 = map(float, G["y_range"])
    step = float(G.get("plane_step", 1.0))

    us = np.arange(u0, u1 + 1e-6, step, dtype=np.float32)
    vs = np.arange(v0, v1 + 1e-6, step, dtype=np.float32)
    Wm, Hm = len(us), len(vs)

    # build uv grid
    if canvas_shape is None:
        uu, vv = np.meshgrid(us, vs)  # local
        in_uv = np.ones_like(uu, dtype=bool)
    else:
        Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])
        if common_shift_yx is None:
            if "canvas_offset_yx" in G:
                oy, ox = map(int, G["canvas_offset_yx"])
            else:
                oy = (Hc - Hm) // 2
                ox = (Wc - Wm) // 2
        else:
            oy, ox = int(common_shift_yx[0]), int(common_shift_yx[1])

        yy, xx = np.meshgrid(
            np.arange(Hc, dtype=np.float32),
            np.arange(Wc, dtype=np.float32),
            indexing="ij"
        )
        uu = (xx - ox) * step + u0
        vv = (yy - oy) * step + v0
        in_uv = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)

    Hout, Wout = uu.shape
    uv_flat = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    # uv -> hist(x,y) via G forward map
    t = np.asarray(G["t_xy"], dtype=np.float32)
    if "A2" in G:
        A2 = np.asarray(G["A2"], dtype=np.float32)
        xy = uv_flat @ A2.T + t[None, :]
    elif "A2_invT" in G:
        A2 = np.linalg.inv(np.asarray(G["A2_invT"], dtype=np.float32)).T
        xy = uv_flat @ A2.T + t[None, :]
    elif ("s" in G) and ("R2" in G):
        s = float(G["s"])
        R2 = np.asarray(G["R2"], dtype=np.float32)
        xy = s * (uv_flat @ R2.T) + t[None, :]
    else:
        raise KeyError("G needs A2 (or A2_invT) or (s,R2), plus t_xy.")

    xh = xy[:, 0].reshape(Hout, Wout)
    yh = xy[:, 1].reshape(Hout, Wout)

    valid_hist = (xh >= 0) & (xh <= Wh - 1) & (yh >= 0) & (yh <= Hh - 1)
    valid = in_uv & valid_hist

    # bilinear sample hist
    hist_t = torch.from_numpy(h).to(device=device)[None, None]  # (1,1,H,W)
    xn = 2.0 * (xh / max(Wh - 1, 1)) - 1.0
    yn = 2.0 * (yh / max(Hh - 1, 1)) - 1.0
    grid = np.stack([xn, yn], axis=-1).astype(np.float32)
    grid_t = torch.from_numpy(grid).to(device=device)[None]      # (1,Hout,Wout,2)

    out = F.grid_sample(hist_t, grid_t, mode="bilinear", padding_mode="zeros", align_corners=True)
    hist_out = out[0, 0].detach().cpu().numpy()

    if fill_invalid is not None:
        hist_out = hist_out.copy()
        hist_out[~valid] = float(fill_invalid)

    if return_extra:
        return hist_out, valid, uv_flat.reshape(Hout, Wout, 2), xy.reshape(Hout, Wout, 2)
    return hist_out

def check_mri_on_init_G_common_canvas(
    slab_Im=None,
    proc_Ih=None,
    G=None,
    mri_on_init_G=None,           # precomputed MRI-on-G (optional)
    hist_on_init_G=None,          # precomputed Hist-on-G (optional)
    img_size=None,                # backward compatibility
    save_file=None,
    device="cuda:0",
    canvas_shape=None,            # (H,W)
    common_shift_yx=None,
    kernel="catmull-rom",
    padding_mode="zeros",
    flip_display=False,
    show_checker=True,
    checker_tile=24,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    def _to_2d(img):
        if isinstance(img, sitk.Image):
            arr = sitk.GetArrayFromImage(img)
        else:
            arr = np.asarray(img)
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape={arr.shape}")
        return arr.astype(np.float32)

    def _norm01(a, eps=1e-8):
        a = a.astype(np.float32)
        mn, mx = a.min(), a.max()
        if mx <= mn + eps:
            return np.zeros_like(a, dtype=np.float32)
        return (a - mn) / (mx - mn + eps)

    def _checker(a, b, valid, tile=24):
        a = _norm01(a)
        b = _norm01(b)
        h, w = a.shape
        yy, xx = np.indices((h, w))
        m = ((yy // tile + xx // tile) % 2).astype(bool)
        out = np.where(m, a, b)
        out[~valid] = 0.0
        return out

    # infer target shape if needed
    target_shape = None
    if canvas_shape is not None:
        target_shape = (int(canvas_shape[0]), int(canvas_shape[1]))
    elif mri_on_init_G is not None:
        target_shape = _to_2d(mri_on_init_G).shape
    elif hist_on_init_G is not None:
        target_shape = _to_2d(hist_on_init_G).shape
    elif img_size is not None:
        target_shape = (int(img_size[0]), int(img_size[1]))

    # 1) MRI on G
    if mri_on_init_G is not None:
        mri_np = _to_2d(mri_on_init_G)
        valid_mri = np.ones_like(mri_np, dtype=bool)
    else:
        if slab_Im is None or G is None:
            raise ValueError("Need either precomputed mri_on_init_G OR both slab_Im and G.")
        out = resample_mri_on_G(
            mri_vol=slab_Im,
            G=G,
            device=device,
            kernel=kernel,
            padding_mode=padding_mode,
            canvas_shape=target_shape,
            common_shift_yx=common_shift_yx,
        )
        if isinstance(out, tuple):
            mri_np = out[0]
            valid_mri = out[1] if len(out) > 1 and out[1] is not None else np.ones_like(mri_np, dtype=bool)
        else:
            mri_np = out
            valid_mri = np.ones_like(mri_np, dtype=bool)

    # 2) Hist on G
    if hist_on_init_G is not None:
        hist_np = _to_2d(hist_on_init_G)
        valid_hist = np.ones_like(hist_np, dtype=bool)
    else:
        if proc_Ih is None or G is None:
            raise ValueError("Need either precomputed hist_on_init_G OR both proc_Ih and G.")
        h_out = resample_histology_on_G(
            hist_img=proc_Ih,
            G=G,
            device=device,
            canvas_shape=mri_np.shape,
            common_shift_yx=common_shift_yx,
            fill_invalid=0.0,
            return_extra=True,
        )
        hist_np, valid_hist = h_out[0], h_out[1]

    if hist_np.shape != mri_np.shape:
        raise ValueError(f"Shape mismatch: hist {hist_np.shape} vs mri {mri_np.shape}")

    valid = valid_mri & valid_hist

    hist_disp = hist_np.copy()
    mri_disp = mri_np.copy()
    hist_disp[~valid] = 0.0
    mri_disp[~valid] = 0.0

    if flip_display:
        hist_disp = np.flipud(hist_disp)
        mri_disp = np.flipud(mri_disp)
        valid = np.flipud(valid)

    ncols = 3 if show_checker else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    axes[0].imshow(hist_disp, cmap="gray", origin="lower")
    axes[0].set_title("Histology on Init G")
    axes[1].imshow(mri_disp, cmap="gray", origin="lower")
    axes[1].set_title("MRI on Init G")
    if show_checker:
        axes[2].imshow(_checker(hist_disp, mri_disp, valid, tile=checker_tile), cmap="gray", origin="lower")
        axes[2].set_title("Checkerboard")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        # ax2.imshow(_checker(hist_disp, mri_disp, valid, tile=checker_tile), cmap="gray", origin="lower")
        ax2.imshow(mri_disp, cmap="gray", origin="lower")
        ax2.set_title("Checkerboard: Hist vs MRI on Init G")
        ax2.axis("off")
        fig2.tight_layout()
        fig2.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved image to: {save_file}")

    return mri_disp, hist_disp, valid


# Init DDF

# 2D TPS Helper
# reuse the same kernel as pose_estimation_v6.py
def _tps_kernel_r2logr(r2, eps=1e-9):
    r = torch.sqrt(torch.clamp(r2, min=eps))
    return r2 * torch.log(r + eps)


class ThinPlateSplineMap2D(nn.Module):
    """
    Vector TPS map M(u,v) = [x(u,v), y(u,v)].
    Fits from uv -> xy.
    """
    def __init__(self, device=None, dtype=torch.float32):
        super().__init__()
        self._device = device
        self._dtype = dtype

        self.register_buffer("ctrl_uv", None, persistent=False)
        self.register_buffer("mean_uv", None, persistent=False)
        self.register_buffer("scale_uv", None, persistent=False)

        self.register_buffer("w_x", None, persistent=False)
        self.register_buffer("a_x", None, persistent=False)
        self.register_buffer("w_y", None, persistent=False)
        self.register_buffer("a_y", None, persistent=False)

        self.lam = 0.0

    def fit_from_uv_xy(self, ctrl_uv, targets_xy, lam=0.0, normalize_uv=False):
        ctrl_uv = torch.as_tensor(ctrl_uv, dtype=self._dtype, device=self._device)
        targets_xy = torch.as_tensor(targets_xy, dtype=self._dtype, device=self._device)

        if ctrl_uv.ndim != 2 or ctrl_uv.shape[1] != 2:
            raise ValueError("ctrl_uv must be (N,2)")
        if targets_xy.ndim != 2 or targets_xy.shape[1] != 2:
            raise ValueError("targets_xy must be (N,2)")
        if ctrl_uv.shape[0] != targets_xy.shape[0]:
            raise ValueError("ctrl_uv and targets_xy must have same N")

        device = ctrl_uv.device
        dtype = ctrl_uv.dtype
        N = ctrl_uv.shape[0]
        if N < 3:
            raise ValueError("Need at least 3 control points for TPS2D")

        qtol = 1e-5
        key = torch.round(ctrl_uv / qtol).to(torch.int64)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        if uniq.shape[0] < N:
            Nu = uniq.shape[0]
            uv_sum = torch.zeros((Nu, 2), device=device, dtype=dtype)
            xy_sum = torch.zeros((Nu, 2), device=device, dtype=dtype)
            cnt = torch.zeros((Nu, 1), device=device, dtype=dtype)

            uv_sum.index_add_(0, inv, ctrl_uv)
            xy_sum.index_add_(0, inv, targets_xy)
            cnt.index_add_(0, inv, torch.ones((N, 1), device=device, dtype=dtype))

            ctrl_uv = uv_sum / cnt
            targets_xy = xy_sum / cnt
            N = ctrl_uv.shape[0]

        if normalize_uv:
            mean_uv = ctrl_uv.mean(0, keepdim=True)
            scale_uv = ctrl_uv.std(0, keepdim=True)
            scale_uv = torch.where(scale_uv == 0, torch.ones_like(scale_uv), scale_uv)
            uv_n = (ctrl_uv - mean_uv) / scale_uv
        else:
            mean_uv = torch.zeros((1, 2), device=device, dtype=dtype)
            scale_uv = torch.ones((1, 2), device=device, dtype=dtype)
            uv_n = ctrl_uv

        diff = uv_n[:, None, :] - uv_n[None, :, :]
        r2 = (diff ** 2).sum(-1)
        K = _tps_kernel_r2logr(r2)

        ones = torch.ones((N, 1), device=device, dtype=dtype)
        Pm = torch.cat([ones, uv_n], dim=1)

        B = torch.cat([targets_xy, torch.zeros((3, 2), device=device, dtype=dtype)], dim=0)

        I_N = torch.eye(N, device=device, dtype=dtype)
        Z_33 = torch.zeros((3, 3), device=device, dtype=dtype)

        lam_base = max(float(lam), 1e-6)
        sol = None
        lam_used = lam_base

        for fac in [1.0, 10.0, 100.0, 1000.0]:
            lam_try = lam_base * fac
            L_top = torch.cat([K + lam_try * I_N, Pm], dim=1)
            L_bot = torch.cat([Pm.T, Z_33], dim=1)
            L = torch.cat([L_top, L_bot], dim=0)
            try:
                sol = torch.linalg.solve(L, B)
                lam_used = lam_try
                break
            except RuntimeError:
                sol = None

        if sol is None:
            L_top = torch.cat([K + (lam_base * 1000.0) * I_N, Pm], dim=1)
            L_bot = torch.cat([Pm.T, Z_33], dim=1)
            L = torch.cat([L_top, L_bot], dim=0)
            sol = torch.linalg.lstsq(L, B).solution
            lam_used = lam_base * 1000.0

        w = sol[:N, :]
        a = sol[N:, :]

        self.ctrl_uv = ctrl_uv
        self.mean_uv = mean_uv
        self.scale_uv = scale_uv
        self.lam = float(lam_used)

        self.w_x, self.w_y = w[:, 0], w[:, 1]
        self.a_x, self.a_y = a[:, 0], a[:, 1]
        return self

    def _normalize_uv(self, uv):
        return (uv - self.mean_uv) / self.scale_uv

    def _ctrl_uv_norm(self):
        return (self.ctrl_uv - self.mean_uv) / self.scale_uv

    def _eval_scalar(self, uv_query, w, a):
        uv_query = torch.as_tensor(uv_query, dtype=self._dtype, device=self.ctrl_uv.device)
        uv_n = self._normalize_uv(uv_query)
        ctrl_n = self._ctrl_uv_norm()
        diff = uv_n[:, None, :] - ctrl_n[None, :, :]
        r2 = (diff ** 2).sum(-1)
        Uker = _tps_kernel_r2logr(r2)
        u_n = uv_n[:, 0]
        v_n = uv_n[:, 1]
        return a[0] + a[1] * u_n + a[2] * v_n + Uker @ w

    def forward(self, uv_query):
        if uv_query.ndim != 2 or uv_query.shape[1] != 2:
            raise ValueError("uv_query must be (M,2)")
        x = self._eval_scalar(uv_query, self.w_x, self.a_x)
        y = self._eval_scalar(uv_query, self.w_y, self.a_y)
        return torch.stack([x, y], dim=1)

def xyz_from_G_uv(uv, G):
    uv = uv.detach().cpu().numpy() if torch.is_tensor(uv) else np.asarray(uv, np.float32)
    X0 = np.asarray(G["X0"], np.float32)
    e1 = np.asarray(G["e1"], np.float32)
    e2 = np.asarray(G["e2"], np.float32)
    return X0[None, :] + uv[:, 0:1] * e1[None, :] + uv[:, 1:2] * e2[None, :]

def hist_xy_from_G_uv(uv, G):
    uv = uv.detach().cpu().numpy() if torch.is_tensor(uv) else np.asarray(uv, np.float32)
    A2 = np.asarray(G["A2"], np.float32)
    t = np.asarray(G["t_xy"], np.float32)
    return uv @ A2.T + t[None, :]

def uv_support_confidence(uv_flat, ctrl_uv, Hout, Wout, sigma_uv=20.0, min_conf=0.25, hard_mult=2.0):
    cuv = ctrl_uv.detach().cpu().numpy() if torch.is_tensor(ctrl_uv) else np.asarray(ctrl_uv, np.float32)
    tree = cKDTree(cuv.astype(np.float32))
    d, _ = tree.query(np.asarray(uv_flat, np.float32), k=1)
    d = d.reshape(Hout, Wout)
    conf = np.exp(-(d * d) / (2.0 * float(sigma_uv) ** 2))
    conf = np.clip((conf - float(min_conf)) / (1.0 - float(min_conf) + 1e-8), 0.0, 1.0)
    conf = conf ** 2
    conf[d > float(hard_mult) * float(sigma_uv)] = 0.0
    return conf.astype(np.float32), d

@torch.no_grad()
def fit_residual_tps3d_with_G_anchors(
    uv_ctrl,
    xyz_ctrl,
    G,
    device="cuda:0",
    lam=2e-2,
    normalize_uv=False,
    anchor_step=24.0,
    far_thr=35.0,
    max_anchors=1200,
    seed=42,
):
    uv_np = uv_ctrl.detach().cpu().numpy() if torch.is_tensor(uv_ctrl) else np.asarray(uv_ctrl, np.float32)
    xyz_np = xyz_ctrl.detach().cpu().numpy() if torch.is_tensor(xyz_ctrl) else np.asarray(xyz_ctrl, np.float32)

    xyz_g_ctrl = xyz_from_G_uv(uv_np, G)
    dxyz_ctrl = xyz_np - xyz_g_ctrl

    uv_anchor, _ = build_G_anchors_uv_xyz(
        uv_ctrl=uv_np, G=G,
        anchor_step=anchor_step, far_thr=far_thr,
        max_anchors=max_anchors, seed=seed
    )
    dxyz_anchor = np.zeros((len(uv_anchor), 3), np.float32)

    uv_all = np.concatenate([uv_np, uv_anchor], axis=0).astype(np.float32)
    dxyz_all = np.concatenate([dxyz_ctrl, dxyz_anchor], axis=0).astype(np.float32)

    tps_res = ThinPlateSplineSurface3D(device=torch.device(device), dtype=torch.float32)
    tps_res.fit_from_uv_xyz(
        ctrl_uv=torch.from_numpy(uv_all).to(device),
        targets_xyz=torch.from_numpy(dxyz_all).to(device),
        lam=float(lam),
        normalize_uv=normalize_uv,
    ).to(device)
    return tps_res, uv_all, dxyz_all, uv_anchor


@torch.no_grad()
def fit_residual_tps2d_with_G_anchors(
    uv_ctrl,
    hist_xy_ctrl,
    G,
    device="cuda:0",
    lam=2e-2,
    normalize_uv=False,
    anchor_step=24.0,
    far_thr=35.0,
    max_anchors=1200,
    seed=42,
):
    uv_np = uv_ctrl.detach().cpu().numpy() if torch.is_tensor(uv_ctrl) else np.asarray(uv_ctrl, np.float32)
    hist_xy_np = hist_xy_ctrl.detach().cpu().numpy() if torch.is_tensor(hist_xy_ctrl) else np.asarray(hist_xy_ctrl, np.float32)

    hist_xy_g_ctrl = hist_xy_from_G_uv(uv_np, G)
    dxy_ctrl = hist_xy_np - hist_xy_g_ctrl

    uv_anchor, _ = build_G_anchors_uv_xyz(
        uv_ctrl=uv_np, G=G,
        anchor_step=anchor_step, far_thr=far_thr,
        max_anchors=max_anchors, seed=seed
    )
    dxy_anchor = np.zeros((len(uv_anchor), 2), np.float32)

    uv_all = np.concatenate([uv_np, uv_anchor], axis=0).astype(np.float32)
    dxy_all = np.concatenate([dxy_ctrl, dxy_anchor], axis=0).astype(np.float32)

    tps2d_res = ThinPlateSplineMap2D(device=torch.device(device), dtype=torch.float32)
    tps2d_res.fit_from_uv_xy(
        ctrl_uv=torch.from_numpy(uv_all).to(device),
        targets_xy=torch.from_numpy(dxy_all).to(device),
        lam=float(lam),
        normalize_uv=normalize_uv,
    ).to(device)
    return tps2d_res, uv_all, dxy_all, uv_anchor

@torch.no_grad()
def resample_mri_on_G_residual_tps(
    mri_vol,
    G,
    tps3d_residual,
    ctrl_uv,
    device="cuda:0",
    kernel="catmull-rom",
    padding_mode="zeros",
    canvas_shape=None,
    common_shift_yx=None,
    fill_invalid=0.0,
    sigma_uv=20.0,
    min_conf=0.25,
    hard_mult=2.0,
    max_residual_vox=16.0,
    return_extra=False,
):
    vol = sitk.GetArrayFromImage(mri_vol).astype(np.float32) if isinstance(mri_vol, sitk.Image) else np.asarray(mri_vol, np.float32)
    if vol.ndim == 2:
        vol = vol[None, ...]
    Z, Y, X = vol.shape

    u0, u1 = map(float, G["x_range"])
    v0, v1 = map(float, G["y_range"])
    step = float(G.get("plane_step", 1.0))

    us = np.arange(u0, u1 + 1e-6, step, dtype=np.float32)
    vs = np.arange(v0, v1 + 1e-6, step, dtype=np.float32)
    Wm, Hm = len(us), len(vs)

    if canvas_shape is None:
        uu, vv = np.meshgrid(us, vs)
        in_uv = np.ones_like(uu, dtype=bool)
    else:
        Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])
        if common_shift_yx is None:
            oy = (Hc - Hm) // 2
            ox = (Wc - Wm) // 2
        else:
            oy, ox = int(common_shift_yx[0]), int(common_shift_yx[1])
        yy, xx = np.meshgrid(np.arange(Hc, dtype=np.float32), np.arange(Wc, dtype=np.float32), indexing="ij")
        uu = (xx - ox) * step + u0
        vv = (yy - oy) * step + v0
        in_uv = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)

    Hout, Wout = uu.shape
    uv_flat = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    xyz_g = xyz_from_G_uv(uv_flat, G).reshape(Hout, Wout, 3)
    dxyz = tps3d_residual(torch.from_numpy(uv_flat).to(device)).detach().cpu().numpy().reshape(Hout, Wout, 3)

    dn = np.linalg.norm(dxyz, axis=-1, keepdims=True) + 1e-8
    dxyz = dxyz * np.minimum(1.0, float(max_residual_vox) / dn)

    conf, dmin = uv_support_confidence(uv_flat, ctrl_uv, Hout, Wout, sigma_uv=sigma_uv, min_conf=min_conf, hard_mult=hard_mult)
    xyz = xyz_g + conf[..., None] * dxyz

    Xv, Yv, Zv = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    valid = in_uv & (Xv >= 0) & (Xv <= X - 1) & (Yv >= 0) & (Yv <= Y - 1) & (Zv >= 0) & (Zv <= Z - 1)

    vol_t = torch.from_numpy(vol).to(device)[None, None]
    P_t = torch.from_numpy(xyz.astype(np.float32)).to(device)
    out = sample_mri_cubic_3d(vol_t, P_t, kernel=kernel, padding_mode=padding_mode)
    mri_out = out[0, 0].detach().cpu().numpy()

    if fill_invalid is not None:
        mri_out = mri_out.copy()
        mri_out[~valid] = float(fill_invalid)

    if return_extra:
        return mri_out, valid, xyz, conf, dmin
    return mri_out


@torch.no_grad()
def resample_histology_on_G_residual_tps(
    hist_img,
    G,
    tps2d_residual,
    ctrl_uv,
    device="cuda:0",
    canvas_shape=None,
    common_shift_yx=None,
    fill_invalid=0.0,
    is_mask=False,
    sigma_uv=20.0,
    min_conf=0.25,
    hard_mult=2.0,
    max_residual_px=20.0,
    return_extra=False,
):
    if isinstance(hist_img, sitk.Image):
        h = sitk.GetArrayFromImage(hist_img)
    else:
        h = np.asarray(hist_img)
    if h.ndim == 3 and h.shape[0] == 1:
        h = h[0]
    h = h.astype(np.float32)
    Hh, Wh = h.shape

    u0, u1 = map(float, G["x_range"])
    v0, v1 = map(float, G["y_range"])
    step = float(G.get("plane_step", 1.0))

    us = np.arange(u0, u1 + 1e-6, step, dtype=np.float32)
    vs = np.arange(v0, v1 + 1e-6, step, dtype=np.float32)
    Wm, Hm = len(us), len(vs)

    if canvas_shape is None:
        uu, vv = np.meshgrid(us, vs)
        in_uv = np.ones_like(uu, dtype=bool)
    else:
        Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])
        if common_shift_yx is None:
            oy = (Hc - Hm) // 2
            ox = (Wc - Wm) // 2
        else:
            oy, ox = int(common_shift_yx[0]), int(common_shift_yx[1])
        yy, xx = np.meshgrid(np.arange(Hc, dtype=np.float32), np.arange(Wc, dtype=np.float32), indexing="ij")
        uu = (xx - ox) * step + u0
        vv = (yy - oy) * step + v0
        in_uv = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)

    Hout, Wout = uu.shape
    uv_flat = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    xy_g = hist_xy_from_G_uv(uv_flat, G).reshape(Hout, Wout, 2)
    dxy = tps2d_residual(torch.from_numpy(uv_flat).to(device)).detach().cpu().numpy().reshape(Hout, Wout, 2)

    dn = np.linalg.norm(dxy, axis=-1, keepdims=True) + 1e-8
    dxy = dxy * np.minimum(1.0, float(max_residual_px) / dn)

    conf, dmin = uv_support_confidence(uv_flat, ctrl_uv, Hout, Wout, sigma_uv=sigma_uv, min_conf=min_conf, hard_mult=hard_mult)
    xy = xy_g + conf[..., None] * dxy

    xh = xy[..., 0]
    yh = xy[..., 1]
    valid = in_uv & (xh >= 0) & (xh <= (Wh - 1)) & (yh >= 0) & (yh <= (Hh - 1))

    hist_t = torch.from_numpy(h).to(device=device)[None, None]
    xn = 2.0 * (xh / max(Wh - 1, 1)) - 1.0
    yn = 2.0 * (yh / max(Hh - 1, 1)) - 1.0
    grid = np.stack([xn, yn], axis=-1).astype(np.float32)
    grid_t = torch.from_numpy(grid).to(device=device)[None]

    mode = "nearest" if is_mask else "bilinear"
    out = F.grid_sample(hist_t, grid_t, mode=mode, padding_mode="zeros", align_corners=True)
    hist_out = out[0, 0].detach().cpu().numpy()

    if is_mask:
        hist_out = (hist_out > 0.5).astype(np.float32)

    if fill_invalid is not None:
        hist_out = hist_out.copy()
        hist_out[~valid] = float(fill_invalid)

    if return_extra:
        return hist_out, valid, uv_flat.reshape(Hout, Wout, 2), xy, conf, dmin
    return hist_out


@torch.no_grad()
def robust_fit_tps_ransac_mad(
    matches_filtered,
    G,
    device="cuda",
    # key controls:
    hist_key: str = "hist_center_yx",
    mri_key: str = "mri_center_zyx",
    # Robustness selection:
    method: str = "ransac",      # "ransac" | "mad" | "all"
    # RANSAC controls:
    n_iters: int = 200,
    sample_size: int = 30,
    inlier_thresh: float = 5.0,
    min_inliers: int = 20,
    score_type: str = "inliers",
    truncated_cap: float = 10.0,
    seed: int = 0,
    # MAD controls:
    mad_thresh: float = 3.5,
    mad_iterations: int = 1,
    # TPS controls:
    lam_hyp: float = 1e-3,
    lam_final: float = 5e-3,
    normalize_uv: bool = False,
):
    print(f"\n-------- Estimate Initial Curved Sheet (method={method}) --------")
    device = torch.device(device)

    # build controls from matches + G
    pts_hist_xy, pts_xyz = [], []  # hist (x,y), MRI (x,y,z)
    for m in matches_filtered:
        if hist_key not in m or mri_key not in m:
            continue
        z, y, x = m[mri_key]
        if (z, y, x) == (-1, -1, -1):
            continue
        cy, cx = m[hist_key]
        pts_hist_xy.append((float(cx), float(cy)))
        pts_xyz.append((float(x), float(y), float(z)))

    if len(pts_hist_xy) < 3:
        raise ValueError(f"Not enough valid matches to fit TPS: N={len(pts_hist_xy)}")

    U = torch.as_tensor(np.asarray(pts_hist_xy, dtype=np.float32), device=device)   # (N,2)
    xyz_ctrl = torch.as_tensor(np.asarray(pts_xyz, dtype=np.float32), device=device) # (N,3)

    t = torch.as_tensor(G["t_xy"], dtype=torch.float32, device=device)
    if "A2_invT" in G:
        A2_invT = torch.as_tensor(G["A2_invT"], dtype=torch.float32, device=device)
        uv_ctrl = (U - t) @ A2_invT
    elif "A2" in G:
        A2 = torch.as_tensor(G["A2"], dtype=torch.float32, device=device)
        uv_ctrl = (U - t) @ torch.linalg.inv(A2).T
    elif ("s" in G) and ("R2" in G):
        s = torch.as_tensor(G["s"], dtype=torch.float32, device=device)
        R2 = torch.as_tensor(G["R2"], dtype=torch.float32, device=device)
        uv_ctrl = ((U - t) @ R2) / s
    else:
        raise KeyError("G must contain one of: A2_invT, A2, or (s and R2), plus t_xy.")

    N = uv_ctrl.shape[0]

    def _final_refit(inlier_mask, lam):
        uv_in = uv_ctrl[inlier_mask]
        xyz_in = xyz_ctrl[inlier_mask]
        tps = ThinPlateSplineSurface3D(device=device, dtype=torch.float32)
        tps.fit_from_uv_xyz(
            ctrl_uv=uv_in, targets_xyz=xyz_in, lam=lam, normalize_uv=normalize_uv
        ).to(device)
        residuals_final = _tps_residuals(tps, uv_ctrl, xyz_ctrl)
        return tps, uv_in, xyz_in, residuals_final

    if method == "all":
        inlier_mask = torch.ones(N, device=device, dtype=torch.bool)
        tps_final, uv_in, xyz_in, residuals_final = _final_refit(inlier_mask, lam_final)
        best_score = int(inlier_mask.sum())
        return tps_final, uv_in, xyz_in, inlier_mask, residuals_final, best_score

    if method == "mad":
        tps = ThinPlateSplineSurface3D(device=device, dtype=torch.float32)
        tps.fit_from_uv_xyz(
            ctrl_uv=uv_ctrl, targets_xyz=xyz_ctrl, lam=lam_hyp, normalize_uv=normalize_uv
        ).to(device)
        inlier_mask = torch.ones(N, device=device, dtype=torch.bool)
        residuals = _tps_residuals(tps, uv_ctrl, xyz_ctrl)

        for _ in range(max(1, mad_iterations)):
            median = residuals.median()
            mad = (residuals - median).abs().median()
            scale = 1.4826 * mad + 1e-8
            thresh = median + mad_thresh * scale
            new_mask = residuals <= thresh
            if int(new_mask.sum()) < 3 or torch.equal(new_mask, inlier_mask):
                break
            inlier_mask = new_mask
            tps = ThinPlateSplineSurface3D(device=device, dtype=torch.float32)
            tps.fit_from_uv_xyz(
                ctrl_uv=uv_ctrl[inlier_mask],
                targets_xyz=xyz_ctrl[inlier_mask],
                lam=lam_hyp,
                normalize_uv=normalize_uv,
            ).to(device)
            residuals = _tps_residuals(tps, uv_ctrl, xyz_ctrl)

        tps_final, uv_in, xyz_in, residuals_final = _final_refit(inlier_mask, lam_final)
        best_score = int(inlier_mask.sum())
        return tps_final, uv_in, xyz_in, inlier_mask, residuals_final, best_score

    if method != "ransac":
        raise ValueError("method must be one of {'ransac', 'mad', 'all'}")
    if N < max(3, sample_size):
        raise ValueError(f"Not enough points for RANSAC TPS: N={N}, sample_size={sample_size}")

    dtype = uv_ctrl.dtype
    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(seed)
    best_inlier_mask = None
    best_score = None
    all_idx = torch.arange(N, device=device)

    for _ in range(n_iters):
        perm = torch.randperm(N, generator=g_cpu, device="cpu")[:sample_size]
        idx = all_idx[perm.to(device)]
        try:
            tps_h = _fit_tps_on_subset(
                uv_ctrl, xyz_ctrl, idx,
                device=device, dtype=dtype,
                lam=lam_hyp, normalize_uv=normalize_uv
            )
        except RuntimeError:
            continue

        residuals = _tps_residuals(tps_h, uv_ctrl, xyz_ctrl)
        inlier_mask = residuals <= inlier_thresh
        num_inliers = int(inlier_mask.sum())
        if num_inliers < min_inliers:
            continue

        if score_type == "inliers":
            score = num_inliers
        elif score_type == "truncated":
            cost = torch.clamp(residuals, max=truncated_cap).sum().item()
            score = -cost
        else:
            raise ValueError("score_type must be 'inliers' or 'truncated'")

        if (best_score is None) or (score > best_score):
            best_score = score
            best_inlier_mask = inlier_mask.clone()
            if num_inliers > 0.9 * N:
                break

    if best_inlier_mask is None:
        print("RANSAC TPS: no valid hypothesis found; falling back to all points.")
        best_inlier_mask = torch.ones(N, device=device, dtype=torch.bool)

    tps_final, uv_in, xyz_in, residuals_final = _final_refit(best_inlier_mask, lam_final)
    return tps_final, uv_in, xyz_in, best_inlier_mask, residuals_final, best_score

class ResidualToGSurface3D(torch.nn.Module):
    def __init__(self, G, residual_tps, device="cuda:0"):
        super().__init__()
        self.G = G
        self.residual_tps = residual_tps
        self.device = torch.device(device)

    def forward(self, uv_query):
        uv = uv_query.detach().cpu().numpy().astype(np.float32)
        xyz_g = xyz_from_G_uv(uv, self.G)
        dxyz = self.residual_tps(uv_query).detach().cpu().numpy().astype(np.float32)
        xyz = xyz_g + dxyz
        return torch.from_numpy(xyz).to(device=uv_query.device, dtype=torch.float32)


# --- refit TPS with G-plane anchors (prevents swirl/extrapolation) ---
def build_G_anchors_uv_xyz(
    uv_ctrl,                 # [N,2] uv used by TPS fit
    G,
    anchor_step=24.0,        # uv grid spacing
    far_thr=35.0,            # keep anchors only if far from real controls
    max_anchors=1200,
    seed=42,
):
    uv = uv_ctrl.detach().cpu().numpy() if torch.is_tensor(uv_ctrl) else np.asarray(uv_ctrl, np.float32)
    uv = uv.astype(np.float32)

    u0, u1 = map(float, G["x_range"])
    v0, v1 = map(float, G["y_range"])
    us = np.arange(u0, u1 + 1e-6, float(anchor_step), dtype=np.float32)
    vs = np.arange(v0, v1 + 1e-6, float(anchor_step), dtype=np.float32)
    UU, VV = np.meshgrid(us, vs)
    uv_grid = np.stack([UU.ravel(), VV.ravel()], axis=1)  # [M,2]

    if len(uv) > 0:
        tree = cKDTree(uv)
        d, _ = tree.query(uv_grid, k=1)
        keep = d >= float(far_thr)
        uv_anchor = uv_grid[keep]
    else:
        uv_anchor = uv_grid

    if len(uv_anchor) > int(max_anchors):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(uv_anchor), size=int(max_anchors), replace=False)
        uv_anchor = uv_anchor[idx]

    # G plane: xyz = X0 + u*e1 + v*e2
    X0 = np.asarray(G["X0"], np.float32)
    e1 = np.asarray(G["e1"], np.float32)
    e2 = np.asarray(G["e2"], np.float32)
    xyz_anchor = X0[None, :] + uv_anchor[:, 0:1] * e1[None, :] + uv_anchor[:, 1:2] * e2[None, :]

    return uv_anchor.astype(np.float32), xyz_anchor.astype(np.float32)


@torch.no_grad()
def resample_mri_on_G_tps(
    mri_vol,
    G,
    tps3d_module,                 # fitted TPS: uv -> xyz(voxel)
    device="cuda:0",
    kernel="catmull-rom",
    padding_mode="zeros",
    use_tiled=False,
    tile_hw=512,
    canvas_shape=None,            # None -> local uv image, (Hc,Wc) -> common canvas image
    common_shift_yx=None,         # (oy,ox), used only when canvas_shape is not None
    fill_invalid=0.0,
    return_extra=False,           # True -> return (mri, valid, xyz, uu, vv)
):
    """
    Curved-sheet MRI resampling on TPS surface using G's uv range/step.
    Requires G keys: x_range, y_range, plane_step.
    """
    # --- volume to [Z,Y,X] numpy ---
    if isinstance(mri_vol, torch.Tensor):
        v = mri_vol.detach().cpu()
        if v.ndim == 5:      # (1,1,Z,Y,X)
            vol = v[0, 0].numpy().astype(np.float32)
        elif v.ndim == 3:    # (Z,Y,X)
            vol = v.numpy().astype(np.float32)
        else:
            raise ValueError(f"Unsupported torch mri_vol shape: {tuple(v.shape)}")
    else:
        vol = sitk.GetArrayFromImage(mri_vol).astype(np.float32) if isinstance(mri_vol, sitk.Image) else np.asarray(mri_vol, dtype=np.float32)
        if vol.ndim == 2:
            vol = vol[None, ...]

    Z, Y, X = vol.shape

    # --- uv range from G ---
    u0, u1 = map(float, G["x_range"])
    v0, v1 = map(float, G["y_range"])
    step = float(G.get("plane_step", 1.0))

    us = np.arange(u0, u1 + 1e-6, step, dtype=np.float32)
    vs = np.arange(v0, v1 + 1e-6, step, dtype=np.float32)
    Wm, Hm = len(us), len(vs)

    # --- build uv grid (local or canvas-mapped) ---
    if canvas_shape is None:
        uu, vv = np.meshgrid(us, vs)  # local uv image
        in_uv = np.ones_like(uu, dtype=bool)
    else:
        Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])

        if common_shift_yx is None:
            oy = (Hc - Hm) // 2
            ox = (Wc - Wm) // 2
        else:
            oy, ox = int(common_shift_yx[0]), int(common_shift_yx[1])

        yy_c, xx_c = np.meshgrid(
            np.arange(Hc, dtype=np.float32),
            np.arange(Wc, dtype=np.float32),
            indexing="ij"
        )

        # canvas pixel -> uv coords
        uu = (xx_c - ox) * step + u0
        vv = (yy_c - oy) * step + v0
        in_uv = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)

    Hout, Wout = uu.shape

    # --- TPS eval: uv -> xyz(voxel) ---
    uv_flat = np.stack([uu, vv], axis=-1).reshape(-1, 2)
    uv_t = torch.as_tensor(uv_flat, dtype=torch.float32, device=device)
    xyz_t = tps3d_module(uv_t).reshape(Hout, Wout, 3).to(torch.float32)  # xyz voxel
    xyz_np = xyz_t.detach().cpu().numpy()

    Xv, Yv, Zv = xyz_np[..., 0], xyz_np[..., 1], xyz_np[..., 2]
    valid = in_uv & (Xv >= 0) & (Xv <= X - 1) & (Yv >= 0) & (Yv <= Y - 1) & (Zv >= 0) & (Zv <= Z - 1)

    # --- cubic sampling ---
    vol_t = torch.from_numpy(vol).to(device=device)[None, None]  # (1,1,Z,Y,X)

    if use_tiled:
        out = sample_mri_cubic_3d_tiled(
            vol_1_1_zyx=vol_t,
            P_vox_xyz=xyz_t,
            tile_hw=tile_hw,
            kernel=kernel,
            padding_mode=padding_mode,
        )
    else:
        out = sample_mri_cubic_3d(
            vol_1_1_zyx=vol_t,
            P_vox_xyz=xyz_t,
            kernel=kernel,
            padding_mode=padding_mode,
        )

    mri_out = out[0, 0].detach().cpu().numpy()

    if fill_invalid is not None:
        mri_out = mri_out.copy()
        mri_out[~valid] = float(fill_invalid)

    if return_extra:
        return mri_out, valid, xyz_np, uu, vv
    return mri_out

def check_mri_on_init_curved_sheet(
    mri_on_init_tps,
    hist_on_init_tps,
    valid=None,                  # optional bool mask, same shape
    save_file=None,
    flip_display=False,
    show_checker=True,
    checker_tile=24,
):
    def _to_2d(x):
        if isinstance(x, sitk.Image):
            a = sitk.GetArrayFromImage(x)
        else:
            a = np.asarray(x)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2:
            raise ValueError(f"Expected 2D, got {a.shape}")
        return a.astype(np.float32)

    def _norm01(a, eps=1e-8):
        mn, mx = float(a.min()), float(a.max())
        return np.zeros_like(a) if mx <= mn + eps else (a - mn) / (mx - mn + eps)

    def _checker(a, b, m, tile):
        aa, bb = _norm01(a), _norm01(b)
        yy, xx = np.indices(a.shape)
        c = ((yy // tile + xx // tile) % 2).astype(bool)
        out = np.where(c, aa, bb)
        out[~m] = 0.0
        return out

    mri = _to_2d(mri_on_init_tps)
    hist = _to_2d(hist_on_init_tps)
    if mri.shape != hist.shape:
        raise ValueError(f"Shape mismatch: mri {mri.shape} vs hist {hist.shape}")

    if valid is None:
        valid = np.ones_like(mri, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != mri.shape:
            raise ValueError(f"valid shape {valid.shape} != image shape {mri.shape}")


    mri_disp = mri.copy();   mri_disp[~valid] = 0.0
    hist_disp = hist.copy(); hist_disp[~valid] = 0.0

    if flip_display:
        mri_disp = np.flipud(mri_disp)
        hist_disp = np.flipud(hist_disp)
        valid = np.flipud(valid)

    ncols = 3 if show_checker else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    axes[0].imshow(hist_disp, cmap="gray", origin="lower"); axes[0].set_title("Hist on Init TPS"); axes[0].axis("off")
    axes[1].imshow(mri_disp, cmap="gray", origin="lower");  axes[1].set_title("MRI on Init TPS");  axes[1].axis("off")
    if show_checker:
        axes[2].imshow(_checker(hist_disp, mri_disp, valid, checker_tile), cmap="gray", origin="lower")
        axes[2].set_title("Checkerboard"); axes[2].axis("off")
    plt.tight_layout(); plt.show()

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        # ax2.imshow(_checker(hist_disp, mri_disp, valid, tile=checker_tile), cmap="gray", origin="lower")
        ax2.imshow(mri_disp, cmap="gray", origin="lower")
        ax2.set_title("Checkerboard: Hist vs MRI on Init TPS")
        ax2.axis("off")
        fig2.tight_layout()
        fig2.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved image to: {save_file}")

    return mri_disp, hist_disp, valid


# Refinement
def warm_start_residual_ctrl_from_matches(
    residual, init_tps, init_G, matches, H, W, Xdim, Ydim, Zdim,
    device="cuda:0", n_iter=300, lr=1e-2, lam_bend=1e-4,
    hist_key="hist_center_yx", mri_key="mri_center_zyx",
):
    dev = torch.device(device)

    # build match tensors
    hist_xy = []
    xyz = []
    for m in matches:
        if hist_key not in m or mri_key not in m:
            continue
        z, y, x = m[mri_key]
        if (z, y, x) == (-1, -1, -1):
            continue
        hy, hx = m[hist_key]
        hist_xy.append((float(hx), float(hy)))   # x,y
        xyz.append((float(x), float(y), float(z)))  # x,y,z

    U = torch.tensor(np.asarray(hist_xy, np.float32), device=dev)   # (N,2) hist x,y
    X = torch.tensor(np.asarray(xyz, np.float32), device=dev)       # (N,3)

    # inverse G: hist(x,y) -> uv for TPS eval
    t = torch.as_tensor(init_G["t_xy"], dtype=torch.float32, device=dev)
    A2_invT = torch.as_tensor(init_G["A2_invT"], dtype=torch.float32, device=dev)
    uv_ctrl = (U - t) @ A2_invT

    with torch.no_grad():
        pred = init_tps(uv_ctrl)            # (N,3) xyz
        dxyz = X - pred                     # vector residual in voxel
        target = torch.stack([
            dxyz[:, 0] / max(Xdim - 1, 1),
            dxyz[:, 1] / max(Ydim - 1, 1),
            dxyz[:, 2] / max(Zdim - 1, 1),
        ], dim=1)                           # normalized residual vector

    # residual model domain is normalized hist pixel coords [0,1]
    uv_norm = torch.stack([
        U[:, 0] / max(W - 1, 1),
        U[:, 1] / max(H - 1, 1),
    ], dim=1)

    opt = torch.optim.Adam(residual.parameters(), lr=lr)
    for _ in range(n_iter):
        pred_d = residual(uv_norm)  # (N,3), H=None path
        loss = F.smooth_l1_loss(pred_d, target) + lam_bend * bending_energy(residual.ctrl)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return residual

def _to_t_1_1_hw(x, device):
    if isinstance(x, sitk.Image):
        a = sitk.GetArrayFromImage(x)
    else:
        a = np.asarray(x)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"Expected 2D or [1,H,W], got {a.shape}")
    return torch.as_tensor(a, dtype=torch.float32, device=device)[None, None]

def make_soft_dilated_mask(
    mask_t,                 # [1,1,H,W], bool/0-1
    dilate_px=10,           # dilation radius in pixels
    blur_ks=15,             # odd kernel size for soft edge
    blur_passes=1,
    min_val=0.0,
    max_val=1.0,
):
    m = (mask_t > 0.5).float()

    # 1) dilate
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=dilate_px)

    # 2) soften boundary
    if blur_ks > 1:
        if blur_ks % 2 == 0:
            blur_ks += 1
        for _ in range(max(1, int(blur_passes))):
            m = F.avg_pool2d(m, kernel_size=blur_ks, stride=1, padding=blur_ks // 2)

    # 3) clamp
    m = m.clamp(min_val, max_val)
    return m

def prepare_coords_from_G_canvas(G, H, W, device, common_shift_yx=None):
    # UV from same definition as resample_mri_on_G_tps
    u0, u1 = map(float, G["x_range"])
    v0, v1 = map(float, G["y_range"])
    step = float(G.get("plane_step", 1.0))

    us = np.arange(u0, u1 + 1e-6, step, dtype=np.float32)
    vs = np.arange(v0, v1 + 1e-6, step, dtype=np.float32)
    Wm, Hm = len(us), len(vs)

    if common_shift_yx is None:
        if "canvas_offset_yx" in G:
            oy, ox = map(int, G["canvas_offset_yx"])
        else:
            oy = (H - Hm) // 2
            ox = (W - Wm) // 2
    else:
        oy, ox = map(int, common_shift_yx)

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    uu = (xx - ox) * step + u0
    vv = (yy - oy) * step + v0

    uv_plane_flat = torch.from_numpy(np.stack([uu, vv], axis=-1).reshape(-1, 2)).to(device=device, dtype=torch.float32)

    uv_norm_pix = torch.empty((H, W, 2), device=device, dtype=torch.float32)
    uv_norm_pix[..., 0] = torch.arange(W, device=device, dtype=torch.float32)[None, :] / max(W - 1, 1)
    uv_norm_pix[..., 1] = torch.arange(H, device=device, dtype=torch.float32)[:, None] / max(H - 1, 1)

    valid_uv = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)
    valid_uv = torch.from_numpy(valid_uv).to(device=device)

    return uv_plane_flat, uv_norm_pix, valid_uv


def _sheet_frame_like_tps(
    xyz_norm, Xdim, Ydim, Zdim,
    matches_xyz=None, scores=None,
    elev=24, azim=36, alpha=0.9, n_contours=12, cmap="viridis"
):
    xyz = xyz_norm.detach().cpu().numpy() if torch.is_tensor(xyz_norm) else np.asarray(xyz_norm)
    Xs = xyz[..., 0] * (Xdim - 1)
    Ys = xyz[..., 1] * (Ydim - 1)
    Zs = xyz[..., 2] * (Zdim - 1)

    valid = (Xs >= 0) & (Xs <= Xdim - 1) & (Ys >= 0) & (Ys <= Ydim - 1) & (Zs >= 0) & (Zs <= Zdim - 1)
    Xp = np.where(valid, Xs, np.nan)
    Yp = np.where(valid, Ys, np.nan)
    Zp = np.where(valid, Zs, np.nan)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    zmin, zmax = np.nanmin(Zp), np.nanmax(Zp)
    norm = plt.Normalize(zmin, zmax)
    facecolors = plt.cm.get_cmap(cmap)(norm(Zp))
    ax.plot_surface(Xp, Yp, Zp, facecolors=facecolors, linewidth=0, antialiased=True, shade=False, alpha=alpha)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label("Depth (Z)")

    try:
        ax.contour(Xp, Yp, Zp, levels=n_contours, colors="k", linewidths=0.4, alpha=0.5)
    except Exception:
        pass

    ax.set_xlim(0, Xdim - 1); ax.set_ylim(0, Ydim - 1); ax.set_zlim(0, Zdim - 1)
    ax.set_box_aspect((Xdim, Ydim, Zdim))
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()  # RGB
    plt.close(fig)
    return frame

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


def pyramid_refinement(
    mri_t,
    mri_mask_t,
    hist_t,
    hist_mask_t,
    xyz_norm_base,       # [H,W,3] FIXED base surface (tps + frozen matches-warm-started residual)
    n0_base,              # [H,W,3] FIXED unit normal field of xyz_norm_base
    conf,                 # [H,W] match-support confidence in [0,1] (from _support_conf_from_ctrl_uv)
    uv_norm_pix,
    grid_id_2d,           # identity sampling grid in [-1,1] units, (1,H,W,2)
    H, W,
    levels,
    device,
    save_file=None,
    frame_cb=None,        # callback(xyz_norm) -> save gif frame
    frame_every=20,
    patience=10,
    base_supersample=32,   # upsample xyz_norm_base/n0_base this many times before phi resamples them
    phi_n_steps=6,          # scaling-and-squaring integration steps for phi's velocity field
    rf_psi_n_steps=6,       # scaling-and-squaring integration steps for rf_psi's velocity field
    # ---- v12: local rigidity -------------------------------------------------
    lam_area=1.0,           # properness  : weight on log(area/area0)^2
    lam_shape=0.0,          # orthonormality: weight on ||[[E,F],[F,G]] - base||^2 (OFF by default)
    w_rigid_mode="mask",    # "mask" | "conf" | "edge" | "none"
    rigid_edge_k=2.0,       # only used by w_rigid_mode="edge"
):
    """
    v11 refinement: four fields, alternated per pyramid level.

      phi     : 2D STATIONARY VELOCITY field, integrated via scaling-and-
                squaring into a diffeomorphic (non-folding by construction)
                displacement, then used to RESAMPLE xyz_norm_base
                tangentially. UNCONDITIONAL -- active everywhere, not gated
                by confidence.
      alpha   : scalar offset along the (resampled) surface normal.
                UNCONDITIONAL, same as phi.
      rf_psi  : a SECOND, independent 2D stationary velocity field (same
                mechanism as phi), gated by (1 - conf) and composed ON TOP
                of phi's already-integrated resample (its own separate
                scaling-and-squaring integration, then applied via a
                warp-of-a-warp lookup) -- an extra tangential correction
                that only matters where phi/alpha's "trust the base
                surface" assumption breaks down (no local match support,
                TPS fell back to the flat G-plane there). (NOTE: an
                earlier diagnostic pass suggested this composition
                reintroduced ~26% folding vs phi alone -- that turned out
                to be a measurement bug (torch.autograd.grad through the
                scaling-and-squaring recursion accumulates numerical error
                that grows with more steps); verified correct with
                finite-differences on the converged, no_grad output: 0%
                folded. Composition is fine.)
      rf_beta : a SECOND, independent scalar normal offset (same mechanism
                as alpha), also gated by (1 - conf) -- the extra "lift off
                the surface" correction in the same low-confidence regions.

      rf_psi + rf_beta together replace v10's single free 3-vector
      residual_free field: any 3D direction is exactly tangential + normal,
      so no expressiveness is lost -- but the tangential part now goes
      through the same diffeomorphic-integration treatment as phi, so it
      can no longer fold the surface the way an unconstrained free vector
      could.
    """
    similarity_metric = "mind"   # "ngf" or "mind" -- switch here to test MIND instead of NGF
    w_ssim, w_ngf, w_mind = 1.0, 0.0, 0.0     # For Others
    # w_ssim, w_ngf, w_mind = 0.0, 0.0, 1.0       # For H&E C-Block coz IGL is the darkest and molecule is the brightest, brighter than WM, so that intensity comparison doesn't work then MIND is used instead.
    hist_mask_fixed = (hist_mask_t > 0.5).float().detach()
    mask_soft = make_soft_dilated_mask(hist_mask_fixed, dilate_px=8, blur_ks=11, blur_passes=1).detach()

    gate = (1.0 - conf).clamp(0.0, 1.0).detach()          # [H,W]
    gate3 = gate.unsqueeze(-1)                             # [H,W,1] for broadcasting onto 3-vectors

    # ---- v12: local rigidity setup -------------------------------------------
    # dims recovers voxel-index coords from xyz_norm. Isotropic spacing => the
    # physical scale cancels in every ratio below, so no GetSpacing() is needed.
    Zdim_r, Ydim_r, Xdim_r = mri_t.shape[-3:]
    dims_t = torch.tensor(
        [float(Xdim_r - 1), float(Ydim_r - 1), float(Zdim_r - 1)],
        device=device, dtype=torch.float32,
    )

    with torch.no_grad():
        E0, F0, G0 = first_fundamental_form(xyz_norm_base.detach(), dims_t)
        area0_sq = (E0 * G0 - F0 * F0).clamp_min(1e-12)
        area0 = area0_sq.sqrt()
        E0, F0, G0, area0 = E0.detach(), F0.detach(), G0.detach(), area0.detach()

        w_rigid = _build_rigidity_weight(
            w_rigid_mode, mask_soft, conf, hist_t, edge_k=rigid_edge_k
        ).detach()
        w_sum = w_rigid.sum().clamp_min(1e-8)

    rigid_on = (float(lam_area) > 0.0) or (float(lam_shape) > 0.0)
    print(f"[v12 rigidity] mode={w_rigid_mode} lam_area={lam_area} lam_shape={lam_shape} "
          f"active_px={int((w_rigid > 0).sum())}/{H*W}")

    def _rigidity_terms(xyz_norm):
        """(L_area, L_shape): w-weighted means. Both zero on a perfectly rigid refinement."""
        if not rigid_on:
            z = torch.zeros((), device=device, dtype=torch.float32)
            return z, z
        E, Fq, G = first_fundamental_form(xyz_norm, dims_t)
        area = (E * G - Fq * Fq).clamp_min(1e-12).sqrt()
        # properness: det J = 1, in symmetric log form
        L_area = (w_rigid * torch.log(area / area0).pow(2)).sum() / w_sum
        if float(lam_shape) > 0.0:
            # orthonormality: J^T J = I, i.e. the metric tensor must not drift
            shape = ((E - E0).pow(2) + 2.0 * (Fq - F0).pow(2) + (G - G0).pow(2)) / area0_sq
            L_shape = (w_rigid * shape).sum() / w_sum
        else:
            L_shape = torch.zeros((), device=device, dtype=torch.float32)
        return L_area, L_shape

    def _rigidity_loss(xyz_norm, cfg):
        L_area, L_shape = _rigidity_terms(xyz_norm)
        lam = float(cfg.get("lamRigid", 1.0))
        return (L_area, L_shape), lam * (float(lam_area) * L_area + float(lam_shape) * L_shape)

    def _report_area(tag, xyz_norm):
        with torch.no_grad():
            r = surface_area_ratio(xyz_norm, xyz_norm_base, dims_t.detach().cpu().numpy())
            m = (w_rigid > 0).detach().cpu().numpy()
            if m.sum() == 0:
                return
            p = np.percentile(r[m], [1, 5, 50, 95, 99])
            print(f"    [{tag}] area ratio final/base p1,p5,p50,p95,p99 = "
                  f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f} {p[4]:.3f}")

    xyz_base_img = xyz_norm_base.permute(2, 0, 1).unsqueeze(0).detach()   # [1,3,H,W]
    n0_base_img = n0_base.permute(2, 0, 1).unsqueeze(0).detach()          # [1,3,H,W]

    if base_supersample and base_supersample > 1:
        Hs, Ws = H * int(base_supersample), W * int(base_supersample)
        xyz_base_img = F.interpolate(xyz_base_img, size=(Hs, Ws), mode="bicubic", align_corners=True)
        n0_base_img = F.interpolate(n0_base_img, size=(Hs, Ws), mode="bicubic", align_corners=True)
        n0_base_img = n0_base_img / (n0_base_img.norm(dim=1, keepdim=True) + 1e-8)

    def _resample_surface(grid2d):
        xyz = F.grid_sample(xyz_base_img, grid2d, mode="bilinear", padding_mode="border", align_corners=True)
        n0s = F.grid_sample(n0_base_img, grid2d, mode="bilinear", padding_mode="border", align_corners=True)
        xyz = xyz[0].permute(1, 2, 0)                       # (H,W,3)
        n0s = n0s[0].permute(1, 2, 0)
        n0s = n0s / (n0s.norm(dim=-1, keepdim=True) + 1e-8)
        return xyz, n0s

    def _scaling_and_squaring_2d(v, n_steps=6):
        # v: (H,W,2) stationary velocity field, same normalized [-1,1] units as grid_id_2d.
        # Integrates the flow ODE d(phi)/dt = v(phi(t)) from t=0 (identity) to t=1 by
        # repeated self-composition of a tiny initial step -- this is the standard
        # diffeomorphic-registration trick (Christensen fluid / VoxelMorph-diff / DARTEL):
        # a smooth field composed with itself can't introduce folding, unlike one large jump.
        d = v / (2 ** n_steps)
        for _ in range(n_steps):
            d_img = d.permute(2, 0, 1).unsqueeze(0)                     # (1,2,H,W)
            query = grid_id_2d + d.unsqueeze(0)                         # (1,H,W,2)
            d_warped = F.grid_sample(d_img, query, mode="bilinear", padding_mode="border", align_corners=True)
            d = d + d_warped[0].permute(1, 2, 0)                        # (H,W,2)
        return d

    def _compose_disp_2d(base_grid2d, extra_disp):
        # base_grid2d: (1,H,W,2) sampling grid (e.g. phi's identity + integrated displacement).
        # extra_disp:  (H,W,2) an ADDITIONAL displacement (e.g. rf_psi's gated, integrated
        # velocity), evaluated AT base_grid2d's positions and added on top -- i.e. this
        # composes "extra" AFTER "base" (extra_map ∘ base_map), which stays diffeomorphic
        # if both individually are, without needing to touch base's own computation at all.
        extra_img = extra_disp.permute(2, 0, 1).unsqueeze(0)                # (1,2,H,W)
        extra_at_base = F.grid_sample(extra_img, base_grid2d, mode="bilinear", padding_mode="border", align_corners=True)
        return base_grid2d + extra_at_base.permute(0, 2, 3, 1)              # (1,H,W,2)

    def _run_block(name, params, lr, n_iter, build_xyz_fn, reg_fn, print_extra_fn):
        opt = torch.optim.Adam([{"params": params, "lr": lr}])
        best_loss, no_improve = float("inf"), 0
        for it in range(n_iter):
            xyz_norm, out_val = build_xyz_fn()

            if frame_cb is not None and (it % max(1, int(frame_every)) == 0 or it == n_iter - 1):
                frame_cb(xyz_norm)

            mri2d = sample_mri_on_sheet(mri_t, mri_mask_t, xyz_norm)
            if similarity_metric == "mind":
                loss_img, comp = combined_masked_ssim_mind_loss(mri2d, hist_t, mask_soft, w_ssim=w_ssim, w_mind=w_mind)
            else:
                loss_img, comp = combined_masked_ssim_ngf_loss(mri2d, hist_t, mask_soft, w_ssim=w_ssim, w_ngf=w_ngf)
            L_reg_terms, loss_extra = reg_fn(xyz_norm, out_val)
            loss = loss_img + loss_extra

            opt.zero_grad()
            loss.backward()
            opt.step()

            if it % 50 == 0 or it == n_iter - 1:
                metric_name = "ngf" if "ngf" in comp else "mind"
                print(f"[{name}] it={it:04d}/{n_iter} loss={loss.item():.6f} img={loss_img.item():.6f} "
                      f"{print_extra_fn(L_reg_terms)} ssim={comp['ssim']:.4f} {metric_name}={comp[metric_name]:.4f}")

            if loss.item() < best_loss:
                best_loss, no_improve = loss.item(), 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"[{name}] Early stopping at it={it} (no improvement for {patience} iters)")
                break

    phi = None
    alpha = None
    rf_psi = None
    rf_beta = None

    print("===== Pyramid refinement v11: phi (velocity) + alpha (normal) + rf_psi (gated velocity) + rf_beta (gated normal) =====")
    for li, cfg in enumerate(levels):
        gh, gw = cfg["gh"], cfg["gw"]

        if phi is None:
            phi = BSplineFFD2D(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(phi.ctrl.detach(), gh, gw)
            phi = BSplineFFD2D(gh, gw, device=device).to(device)
            _copy_ctrl(phi.ctrl, ctrl_up)

        if alpha is None:
            alpha = ScalarNormalOffsetField(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(alpha.ctrl.detach(), gh, gw)
            alpha = ScalarNormalOffsetField(gh, gw, device=device).to(device)
            _copy_ctrl(alpha.ctrl, ctrl_up)

        if rf_psi is None:
            rf_psi = BSplineFFD2D(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(rf_psi.ctrl.detach(), gh, gw)
            rf_psi = BSplineFFD2D(gh, gw, device=device).to(device)
            _copy_ctrl(rf_psi.ctrl, ctrl_up)

        if rf_beta is None:
            rf_beta = ScalarNormalOffsetField(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(rf_beta.ctrl.detach(), gh, gw)
            rf_beta = ScalarNormalOffsetField(gh, gw, device=device).to(device)
            _copy_ctrl(rf_beta.ctrl, ctrl_up)

        # ---- Step Phi: freeze alpha + rf_psi + rf_beta, optimize tangential resample ----
        with torch.no_grad():
            a_fix = alpha(uv_norm_pix, H=H, W=W)
            disp_psi_fix = _scaling_and_squaring_2d(gate3 * rf_psi(uv_norm_pix, H=H, W=W), n_steps=rf_psi_n_steps)
            rf_beta_fix = gate3 * rf_beta(uv_norm_pix, H=H, W=W)

        print(f"\n--- Level {li} ({gh}x{gw}) Step Phi ---")

        def _build_phi():
            v = phi(uv_norm_pix, H=H, W=W)
            disp = _scaling_and_squaring_2d(v, n_steps=phi_n_steps)
            grid2d = grid_id_2d + disp.unsqueeze(0)
            grid2d = _compose_disp_2d(grid2d, disp_psi_fix)
            xyz_resampled, n0_resampled = _resample_surface(grid2d)
            xyz_norm = xyz_resampled + a_fix * n0_resampled + rf_beta_fix * n0_resampled
            return xyz_norm, None

        def _reg_phi(xyz_norm, _):
            L_bend = bending_energy_2d(phi.ctrl)
            L_sheet = sheet_inside_volume_penalty(xyz_norm)
            (L_area, L_shape), L_rigid = _rigidity_loss(xyz_norm, cfg)
            return (L_bend, L_sheet, L_area, L_shape), \
                   cfg["lamA"] * L_bend + cfg["lamSheet"] * L_sheet + L_rigid

        _run_block(
            f"L{li} Phi", phi.parameters(), cfg["lrA"], cfg["nA"],
            _build_phi, _reg_phi,
            lambda t: f"bend={t[0].item():.2e} sheet={t[1].item():.2e} area={t[2].item():.2e} shape={t[3].item():.2e}"
        )

        # ---- Step Alpha: freeze phi + rf_psi + rf_beta, optimize normal offset ----
        with torch.no_grad():
            disp_fix = _scaling_and_squaring_2d(phi(uv_norm_pix, H=H, W=W), n_steps=phi_n_steps)
            grid2d_fix = grid_id_2d + disp_fix.unsqueeze(0)
            disp_psi_fix = _scaling_and_squaring_2d(gate3 * rf_psi(uv_norm_pix, H=H, W=W), n_steps=rf_psi_n_steps)
            grid2d_fix = _compose_disp_2d(grid2d_fix, disp_psi_fix)
            xyz_resampled_fix, n0_resampled_fix = _resample_surface(grid2d_fix)
            rf_beta_fix = gate3 * rf_beta(uv_norm_pix, H=H, W=W)

        print(f"--- Level {li} ({gh}x{gw}) Step Alpha ---")

        def _build_alpha():
            a = alpha(uv_norm_pix, H=H, W=W)
            xyz_norm = xyz_resampled_fix + a * n0_resampled_fix + rf_beta_fix * n0_resampled_fix
            return xyz_norm, a

        def _reg_alpha(xyz_norm, a):
            L_bend = bending_energy(alpha.ctrl)
            L_sheet = sheet_inside_volume_penalty(xyz_norm)
            L_out = (a[..., 0].pow(2) * (1.0 - mask_soft[0, 0])).mean()
            # alpha has the same affine blindness as phi: a linear ramp in alpha
            # tilts the sheet at zero bending cost, which stretches local area.
            (L_area, L_shape), L_rigid = _rigidity_loss(xyz_norm, cfg)
            return (L_bend, L_sheet, L_out, L_area, L_shape), \
                   cfg["lamB"] * L_bend + cfg["lamSheet"] * L_sheet + cfg["lamOut"] * L_out + L_rigid

        _run_block(
            f"L{li} Alpha", alpha.parameters(), cfg["lrB"], cfg["nB"],
            _build_alpha, _reg_alpha,
            lambda t: f"bend={t[0].item():.2e} sheet={t[1].item():.2e} out={t[2].item():.2e} area={t[3].item():.2e} shape={t[4].item():.2e}"
        )

        # ---- Step ResidualFree: freeze phi + alpha, jointly optimize rf_psi (tangential,
        # composed on top of phi) + rf_beta (normal), both gated by 1-conf ----
        with torch.no_grad():
            disp_fix = _scaling_and_squaring_2d(phi(uv_norm_pix, H=H, W=W), n_steps=phi_n_steps)
            grid2d_phi_fix = grid_id_2d + disp_fix.unsqueeze(0)
            a_fix = alpha(uv_norm_pix, H=H, W=W)

        print(f"--- Level {li} ({gh}x{gw}) Step ResidualFree (rf_psi tangential + rf_beta normal, gated by 1-conf) ---")

        def _build_rf():
            v_psi = rf_psi(uv_norm_pix, H=H, W=W)
            disp_psi = _scaling_and_squaring_2d(gate3 * v_psi, n_steps=rf_psi_n_steps)
            grid2d = _compose_disp_2d(grid2d_phi_fix, disp_psi)
            xyz_resampled, n0_resampled = _resample_surface(grid2d)
            rf_beta_gated = gate3 * rf_beta(uv_norm_pix, H=H, W=W)
            xyz_norm = xyz_resampled + a_fix * n0_resampled + rf_beta_gated * n0_resampled
            return xyz_norm, rf_beta_gated

        def _reg_rf(xyz_norm, rf_beta_gated):
            L_bend_psi = bending_energy_2d(rf_psi.ctrl)
            L_bend_beta = bending_energy(rf_beta.ctrl)
            L_sheet = sheet_inside_volume_penalty(xyz_norm)
            L_out = (rf_beta_gated[..., 0].pow(2) * (1.0 - mask_soft[0, 0])).mean()
            (L_area, L_shape), L_rigid = _rigidity_loss(xyz_norm, cfg)
            return (L_bend_psi, L_bend_beta, L_sheet, L_out, L_area, L_shape), \
                   cfg["lamF"] * (L_bend_psi + L_bend_beta) + cfg["lamSheet"] * L_sheet \
                   + cfg["lamOut"] * L_out + L_rigid

        _run_block(
            f"L{li} ResidualFree", list(rf_psi.parameters()) + list(rf_beta.parameters()), cfg["lrF"], cfg["nF"],
            _build_rf, _reg_rf,
            lambda t: f"bend_psi={t[0].item():.2e} bend_beta={t[1].item():.2e} sheet={t[2].item():.2e} "
                      f"out={t[3].item():.2e} area={t[4].item():.2e} shape={t[5].item():.2e}"
        )

        with torch.no_grad():
            _dv = _scaling_and_squaring_2d(phi(uv_norm_pix, H=H, W=W), n_steps=phi_n_steps)
            _g2 = _compose_disp_2d(
                grid_id_2d + _dv.unsqueeze(0),
                _scaling_and_squaring_2d(gate3 * rf_psi(uv_norm_pix, H=H, W=W), n_steps=rf_psi_n_steps),
            )
            _xyz_l, _n0_l = _resample_surface(_g2)
            _report_area(
                f"L{li} end",
                _xyz_l + alpha(uv_norm_pix, H=H, W=W) * _n0_l
                       + gate3 * rf_beta(uv_norm_pix, H=H, W=W) * _n0_l,
            )

    with torch.no_grad():
        disp_final = _scaling_and_squaring_2d(phi(uv_norm_pix, H=H, W=W), n_steps=phi_n_steps)
        grid2d_phi_final = grid_id_2d + disp_final.unsqueeze(0)
        disp_psi_final = _scaling_and_squaring_2d(gate3 * rf_psi(uv_norm_pix, H=H, W=W), n_steps=rf_psi_n_steps)
        grid2d_final = _compose_disp_2d(grid2d_phi_final, disp_psi_final)
        xyz_resampled_final, n0_resampled_final = _resample_surface(grid2d_final)
        a_final = alpha(uv_norm_pix, H=H, W=W)
        rf_beta_final = gate3 * rf_beta(uv_norm_pix, H=H, W=W)
        xyz_norm_final = xyz_resampled_final + a_final * n0_resampled_final + rf_beta_final * n0_resampled_final
        _report_area("FINAL", xyz_norm_final)
        if frame_cb is not None:
            frame_cb(xyz_norm_final)
        if save_file is not None:
            _plot_surface(
                xyz_norm_final[..., 2],
                "Pyramid refinement v11 – final surface (phi+alpha+gated rf_psi/rf_beta)",
                save_file=f"{save_file}_final.png",
            )

    return phi, alpha, rf_psi, rf_beta, xyz_norm_final

def refine_curved_sheet(
    slab_Im, slab_Mm,
    init_G, init_tps, matches_for_warmstart,
    device,
    hist_on_init_tps=None,
    hist_mask_on_init_tps=None,
    ctrl_uv_support=None,
    tps_sigma_uv=20.0,
    tps_min_conf=0.25,
    tps_max_disp_vox=16.0,
    save_file=None,
    gif_file=None,         # e.g. ".../refine_sheet.gif"
    gif_every=20,          # capture every N iters
    gif_fps=10,
    # ---- v12: local rigidity (see module header) -----------------------------
    lam_area=1.0,
    lam_shape=0.0,
    w_rigid_mode="mask",
    rigid_edge_k=2.0,
    lam_rigid_levels=(1.0, 1.0, 1.0),   # per-pyramid-level multiplier on the rigidity loss
):
    print("\n-------- Refine Curved Sheet (v12: + local rigidity) --------")

    def _xyz_from_G_uv(uv_plane_flat, G, H, W, device):
        uv = uv_plane_flat.detach().cpu().numpy().reshape(H, W, 2).astype(np.float32)
        uu, vv = uv[..., 0], uv[..., 1]
        X0 = np.asarray(G["X0"], np.float32)
        e1 = np.asarray(G["e1"], np.float32)
        e2 = np.asarray(G["e2"], np.float32)
        xyz_g = X0[None, None, :] + uu[..., None] * e1[None, None, :] + vv[..., None] * e2[None, None, :]
        return torch.from_numpy(xyz_g).to(device=device, dtype=torch.float32)

    def _support_conf_from_ctrl_uv(uv_plane_flat, ctrl_uv, H, W, device, sigma_uv=20.0, min_conf=0.25):
        if ctrl_uv is None:
            return torch.ones((H, W), device=device, dtype=torch.float32)
        cuv = ctrl_uv.detach().cpu().numpy() if torch.is_tensor(ctrl_uv) else np.asarray(ctrl_uv, np.float32)
        if cuv.ndim != 2 or cuv.shape[0] == 0 or cuv.shape[1] != 2:
            return torch.ones((H, W), device=device, dtype=torch.float32)

        uv = uv_plane_flat.detach().cpu().numpy().astype(np.float32)
        d, _ = cKDTree(cuv.astype(np.float32)).query(uv, k=1)
        conf = np.exp(-(d * d) / (2.0 * float(sigma_uv) ** 2))
        conf = np.clip((conf - float(min_conf)) / (1.0 - float(min_conf) + 1e-8), 0.0, 1.0)
        conf = conf ** 2
        conf[d > 2.0 * float(sigma_uv)] = 0.0
        return torch.from_numpy(conf.reshape(H, W)).to(device=device, dtype=torch.float32)

    def _build_xyz_norm_tps_safe(init_tps, G, uv_plane_flat, H, W, Xdim, Ydim, Zdim, device,
                                 ctrl_uv=None, sigma_uv=20.0, min_conf=0.25, max_disp_vox=16.0):
        with torch.no_grad():
            xyz_tps = init_tps(uv_plane_flat.detach()).view(H, W, 3)  # voxel xyz
            xyz_g = _xyz_from_G_uv(uv_plane_flat, G, H, W, device)

            disp = xyz_tps - xyz_g
            dn = torch.linalg.norm(disp, dim=-1, keepdim=True).clamp_min(1e-6)
            disp = disp * torch.clamp(float(max_disp_vox) / dn, max=1.0)

            conf = _support_conf_from_ctrl_uv(
                uv_plane_flat, ctrl_uv, H, W, device,
                sigma_uv=sigma_uv, min_conf=min_conf
            )
            xyz_safe = xyz_g + conf[..., None] * disp

            xyz_norm = torch.stack([
                xyz_safe[..., 0] / max(Xdim - 1, 1),
                xyz_safe[..., 1] / max(Ydim - 1, 1),
                xyz_safe[..., 2] / max(Zdim - 1, 1),
            ], dim=-1)
        return xyz_norm

    mri_t = _prepare_data_tensor(slab_Im, device)
    mri_mask_t = _prepare_data_tensor(slab_Mm, device)
    Zdim, Ydim, Xdim = mri_t.shape[-3:]

    if hist_on_init_tps is None:
        raise ValueError("hist_on_init_tps is required for this mode.")

    hist_t = _to_t_1_1_hw(hist_on_init_tps, device)
    if hist_mask_on_init_tps is None:
        hist_mask_t = torch.ones_like(hist_t)
    else:
        hist_mask_t = (_to_t_1_1_hw(hist_mask_on_init_tps, device) > 0.5).float()

    _, _, H, W = hist_t.shape

    matches_xyz = []
    matches_scores = []
    has_scores = True
    for m in matches_for_warmstart:
        if "mri_center_zyx" not in m:
            continue
        z, y, x = m["mri_center_zyx"]
        if (z, y, x) == (-1, -1, -1):
            continue
        matches_xyz.append((float(x), float(y), float(z)))
        if "score" in m:
            matches_scores.append(float(m["score"]))
        else:
            has_scores = False

    matches_xyz = np.asarray(matches_xyz, dtype=np.float32)
    matches_scores = np.asarray(matches_scores, dtype=np.float32) if (has_scores and len(matches_scores) == len(matches_xyz)) else None

    writer = None
    if gif_file is not None:
        gif_dir = os.path.dirname(gif_file)
        if gif_dir:
            os.makedirs(gif_dir, exist_ok=True)
        writer = imageio.get_writer(gif_file, mode="I", fps=max(1, int(gif_fps)))

    def _frame_cb(xyz_norm):
        if writer is None:
            return
        frame = _sheet_frame_like_tps(
            xyz_norm=xyz_norm,
            Xdim=Xdim, Ydim=Ydim, Zdim=Zdim,
            matches_xyz=matches_xyz,
            scores=matches_scores,
        )
        writer.append_data(frame)

    try:
        uv_plane_flat_init, uv_norm_pix, _ = prepare_coords_from_G_canvas(
            init_G, H, W, device, common_shift_yx=init_G.get("canvas_offset_yx", None)
        )

        grid_h, grid_w = 12, 12
        residual_warm = BSplineResidualSurface3D(grid_h, grid_w, device=device).to(device)
        residual_warm = warm_start_residual_ctrl_from_matches(
            residual=residual_warm,
            init_tps=init_tps,
            init_G=init_G,
            matches=matches_for_warmstart,
            H=H, W=W, Xdim=Xdim, Ydim=Ydim, Zdim=Zdim,
            device=device
        )

        with torch.no_grad():
            xyz_norm_tps = _build_xyz_norm_tps_safe(
                init_tps=init_tps,
                G=init_G,
                uv_plane_flat=uv_plane_flat_init,
                H=H, W=W,
                Xdim=Xdim, Ydim=Ydim, Zdim=Zdim,
                device=device,
                ctrl_uv=ctrl_uv_support,
                sigma_uv=tps_sigma_uv,
                min_conf=tps_min_conf,
                max_disp_vox=tps_max_disp_vox,
            )
            _frame_cb(xyz_norm_tps)

            delta_norm_init = residual_warm(uv_norm_pix, H=H, W=W)
            xyz_norm_base = (xyz_norm_tps + delta_norm_init).detach()

            S = xyz_norm_base.clamp(0, 1)
            Su = torch.zeros_like(S)
            Sv = torch.zeros_like(S)
            Su[:, 1:-1] = (S[:, 2:] - S[:, :-2]) * 0.5
            Su[:, 0] = S[:, 1] - S[:, 0]
            Su[:, -1] = S[:, -1] - S[:, -2]
            Sv[1:-1, :] = (S[2:, :] - S[:-2, :]) * 0.5
            Sv[0, :] = S[1, :] - S[0, :]
            Sv[-1, :] = S[-1, :] - S[-2, :]
            n0 = torch.cross(Su, Sv, dim=-1)
            n0 = (n0 / (n0.norm(dim=-1, keepdim=True) + 1e-8)).detach()

            # Same confidence field used to blend TPS vs G-plane fallback above --
            # reused here to gate residual_free's freedom (low conf = untrustworthy
            # base surface = residual_free gets real freedom there).
            conf = _support_conf_from_ctrl_uv(
                uv_plane_flat_init, ctrl_uv_support, H, W, device,
                sigma_uv=tps_sigma_uv, min_conf=tps_min_conf
            ).detach()

        grid_id_2d = torch.stack([
            2 * uv_norm_pix[..., 0] - 1.0,
            2 * uv_norm_pix[..., 1] - 1.0,
        ], dim=-1).unsqueeze(0)   # (1,H,W,2)

        # levels = [
        #     dict(gh=12, gw=12, nA=500, lrA=2e-3, lamA=8e-4,
        #                     nB=500, lrB=2e-3, lamB=2e-3,
        #                     nF=300, lrF=2e-3, lamF=2e-3,
        #                     lamSheet=2e-3, lamOut=1e-2),
        #     dict(gh=24, gw=24, nA=600, lrA=1e-3, lamA=6e-4,
        #                     nB=600, lrB=1e-3, lamB=1.5e-3,
        #                     nF=150, lrF=1e-3, lamF=1.5e-3,
        #                     lamSheet=1.5e-3, lamOut=1e-2),
        #     dict(gh=32, gw=32, nA=700, lrA=7e-4, lamA=4e-4,
        #                     nB=700, lrB=7e-4, lamB=1e-3,
        #                     nF=200, lrF=7e-4, lamF=1e-3,
        #                     lamSheet=1e-3, lamOut=1e-2),
        # ]
        _lr = list(lam_rigid_levels) + [lam_rigid_levels[-1]] * 3
        levels = [
            dict(gh=12, gw=12, nA=100, lrA=2e-3, lamA=8e-4,
                            nB=100, lrB=5e-4, lamB=1e-2,
                            nF=60, lrF=2e-3, lamF=8e-3,
                            lamSheet=2e-3, lamOut=1e-2, lamRigid=_lr[0]),
            dict(gh=24, gw=24, nA=120, lrA=1e-3, lamA=6e-4,
                            nB=120, lrB=2.5e-4, lamB=7.5e-3,
                            nF=30, lrF=1e-3, lamF=6e-3,
                            lamSheet=1.5e-3, lamOut=1e-2, lamRigid=_lr[1]),
            dict(gh=32, gw=32, nA=140, lrA=7e-4, lamA=4e-4,
                            nB=140, lrB=1.75e-4, lamB=5e-3,
                            nF=40, lrF=7e-4, lamF=4e-3,
                            lamSheet=1e-3, lamOut=1e-2, lamRigid=_lr[2]),
        ]

        phi_refined, alpha_refined, rf_psi_refined, rf_beta_refined, xyz_norm_final = pyramid_refinement(
            mri_t=mri_t,
            mri_mask_t=mri_mask_t,
            hist_t=hist_t,
            hist_mask_t=hist_mask_t,
            xyz_norm_base=xyz_norm_base,
            n0_base=n0,
            conf=conf,
            uv_norm_pix=uv_norm_pix,
            grid_id_2d=grid_id_2d,
            H=H, W=W,
            levels=levels,
            device=device,
            save_file=save_file,
            frame_cb=_frame_cb if writer is not None else None,
            frame_every=gif_every,
            lam_area=lam_area,
            lam_shape=lam_shape,
            w_rigid_mode=w_rigid_mode,
            rigid_edge_k=rigid_edge_k,
        )

        with torch.no_grad():
            P_vox_xyz = torch.stack([
                xyz_norm_final[..., 0] * (Xdim - 1),
                xyz_norm_final[..., 1] * (Ydim - 1),
                xyz_norm_final[..., 2] * (Zdim - 1),
            ], dim=-1)

            mri_on_final_sheet = sample_mri_cubic_3d(
                mri_t, P_vox_xyz, kernel="catmull-rom", padding_mode="border"
            )[0, 0].detach().cpu().numpy()
        
        if torch.is_tensor(xyz_norm_final):
            valid_mri = (
                torch.isfinite(xyz_norm_final).all(dim=-1) &
                (xyz_norm_final[..., 0] >= 0) & (xyz_norm_final[..., 0] <= 1) &
                (xyz_norm_final[..., 1] >= 0) & (xyz_norm_final[..., 1] <= 1) &
                (xyz_norm_final[..., 2] >= 0) & (xyz_norm_final[..., 2] <= 1)
            )
        else:
            xyz_norm_final = np.asarray(xyz_norm_final)
            valid_mri = (
                np.isfinite(xyz_norm_final).all(axis=-1) &
                (xyz_norm_final[..., 0] >= 0) & (xyz_norm_final[..., 0] <= 1) &
                (xyz_norm_final[..., 1] >= 0) & (xyz_norm_final[..., 1] <= 1) &
                (xyz_norm_final[..., 2] >= 0) & (xyz_norm_final[..., 2] <= 1)
            )
        
        valid_out = valid_mri.detach().cpu().numpy()

        return xyz_norm_final, mri_on_final_sheet, valid_out, xyz_norm_base, conf, phi_refined, n0, alpha_refined, rf_psi_refined, rf_beta_refined
    finally:
        if writer is not None:
            writer.close()
            print(f"Saved GIF to: {gif_file}")

def check_mri_on_final_sheet(
    hist_on_init_tps,
    valid_display,
    mri_on_final_sheet,
    save_file=None,
    save_file_single=None,
    show_checker=True,
    checker_tile=24,
    flip_display=False,
):

    def _to_numpy(x):
        if isinstance(x, sitk.Image):
            return sitk.GetArrayFromImage(x)
        if torch is not None and torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _to_2d(x, dtype=np.float32):
        a = _to_numpy(x)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2:
            raise ValueError(f"Expected 2D or [1,H,W], got {a.shape}")
        return a.astype(dtype, copy=False)

    def _norm01(a, eps=1e-8):
        mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
        if mx <= mn + eps:
            return np.zeros_like(a, dtype=np.float32)
        return (a - mn) / (mx - mn + eps)

    def _checker(a, b, tile=24):
        aa, bb = _norm01(a), _norm01(b)
        yy, xx = np.indices(aa.shape)
        m = ((yy // tile + xx // tile) % 2).astype(bool)
        return np.where(m, aa, bb)

    h = _to_2d(hist_on_init_tps, dtype=np.float32)
    v = _to_2d(valid_display, dtype=np.float32) > 0.5
    r = _to_2d(mri_on_final_sheet, dtype=np.float32)

    if h.shape != r.shape:
        raise ValueError(f"Shape mismatch hist {h.shape} vs mri {r.shape}")
    if v.shape != h.shape:
        raise ValueError(f"valid_display shape mismatch {v.shape} vs hist {h.shape}")

    h_disp = h.copy()
    r_disp = r.copy()
    h_disp[~v] = 0.0
    r_disp[~v] = 0.0

    if flip_display:
        h_disp = np.flipud(h_disp)
        r_disp = np.flipud(r_disp)
        v = np.flipud(v)

    ncols = 4 if show_checker else 3
    fig, ax = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    ax[0].imshow(h_disp, cmap="gray", origin="lower")
    ax[0].set_title("Target Histology on Sheet")

    ax[1].imshow(v.astype(np.float32), cmap="gray", origin="lower")
    ax[1].set_title("Valid Display Region")

    ax[2].imshow(r_disp, cmap="gray", origin="lower")
    ax[2].set_title("MRI on Final Sheet")

    if show_checker:
        ax[3].imshow(_checker(h_disp, r_disp, tile=checker_tile), cmap="gray", origin="lower")
        ax[3].set_title(f"Checkerboard (tile={checker_tile})")

    for a in ax:
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

    if save_file_single is not None:
        d = os.path.dirname(save_file_single)
        if d:
            os.makedirs(d, exist_ok=True)
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.imshow(r_disp, cmap="gray", origin="lower")
        ax3.axis("off")
        fig3.tight_layout(pad=0)
        fig3.savefig(save_file_single, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig3)
        print(f"Saved mri_on_final_sheet to: {save_file_single}")

    return h_disp, v.astype(np.float32), r_disp, v
