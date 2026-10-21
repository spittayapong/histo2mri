"""Global in-plane transform G: a plane in MRI voxel space fitted to the patch matches.

G is fitted after the common canvas is built (canvas) and is the base of every later
surface model (tps, refine). It maps plane parameters (u, v), which are MRI x/y offsets
in voxels from the canvas origin, to MRI (x, y, z) on the plane, and to histology (x, y)
in common-canvas pixels through a similarity transform. Matches store MRI points as
(z, y, x) and histology points as (y, x).
"""
import numpy as np
import torch


def _decompose_A2(A2):
    """Polar-decompose a 2x2 matrix A2 = R2 @ S; return (uniform scale, R2, S / scale)."""
    U, svals, Vt = np.linalg.svd(A2)
    R2 = U @ Vt
    if np.linalg.det(R2) < 0:
        U[:, -1] *= -1
        R2 = U @ Vt
    S = Vt.T @ np.diag(svals) @ Vt
    s_uni = float(np.sqrt(max(np.linalg.det(S), 1e-12)))
    A_residual = S / max(s_uni, 1e-8)
    return s_uni, R2, A_residual


def fit_global_plane(
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
    """Fit the global in-plane transform G from histology-MRI patch matches.

    The out-of-plane part is a least-squares plane z = a*x + b*y + c through the MRI match
    centres. The in-plane part is a 2D similarity (uniform scale, rotation without
    reflection, translation) from plane coordinates (u, v) = (x - x0, y - y0) to histology
    (x, y). With ``robust=True`` and at least 4 matches the similarity is estimated by RANSAC
    on 3-point samples (inlier if the error is <= ``ransac_thr_px``), refitted on the best
    inlier set (or on all points if it has fewer than ``min_inliers``), and refitted once
    more on the points within ``trim_thr_px`` of that model. Only the in-plane fit is
    robust; the z-plane uses every match that passes the bounds filter. A similarity is
    used rather than a full affine because it cannot collapse or shear when the matches are
    few or noisy.

    Args:
        clean_matches: list of match dicts; must contain ``mri_key`` and ``hist_key``.
            Matches whose MRI point is (-1, -1, -1) are skipped. Histology points should be
            in common-canvas pixels (see ``matches_to_common_canvas``).
        x_range_native: (x0, x1) MRI x range, in voxels, covered by the common canvas
            columns.
        y_range_native: (y0, y1) MRI y range, in voxels, covered by the common canvas rows.
        plane_step: spacing of the plane grid in voxels (one canvas pixel per step).
        mri_key: match key of the MRI point, (z, y, x) in MRI slab voxels.
        hist_key: match key of the histology point, (y, x) in pixels.
        mri_hw: optional (H, W); drop matches whose MRI (y, x) lies outside it.
        hist_hw: optional (H, W); drop matches whose histology (y, x) lies outside it.
        robust: use RANSAC plus one trim-refit pass for the in-plane similarity.
        ransac_thr_px: RANSAC inlier threshold in histology pixels.
        ransac_iters: number of RANSAC samples.
        trim_thr_px: inlier threshold of the trim-refit pass, in histology pixels.
        min_inliers: minimum inlier count for the RANSAC set and the trim set to be used.
        rng_seed: seed of the RANSAC sampler.

    Returns:
        G, a dict with:
            X0: (3,) float32, plane origin in MRI (x, y, z) voxels, i.e. the plane point at
                (x0, y0).
            e1, e2: (3,) float32 plane tangents (1, 0, a) and (0, 1, b). They are not unit
                length, so that xyz = X0 + u*e1 + v*e2 has x = x0 + u and y = y0 + v.
            n: (3,) float32 unit normal, normalised e1 x e2.
            A2: (2, 2) float32 linear part (s * R2) of the map uv -> histology xy.
            t_xy: (2,) float32 translation of that map: hist_xy = uv @ A2.T + t_xy.
            A2_invT: (2, 2) float32 inv(A2).T, so that uv = (hist_xy - t_xy) @ A2_invT.
            s: float32 uniform scale of A2 (sqrt of the determinant of its symmetric part).
            R2: (2, 2) float32 rotation of A2 from the polar decomposition.
            A_residual: (2, 2) float32 symmetric part divided by s (about identity for a
                similarity).
            plane_abc: (a, b, c_local), with z = a*u + b*v + c_local in plane coordinates.
            origin_xy_native: (x0, y0).
            x_range, y_range: (0, (W-1)*step) and (0, (H-1)*step), the u and v ranges of
                the plane grid.
            plane_step: grid spacing in voxels.
            x_range_native, y_range_native: the input MRI ranges.
            reprojection: in-plane errors of the final similarity, in histology pixels,
                over the inlier set: mean_px, median_px, p95_px, max_px, num (inliers) and
                num_all (all matches after bounds filtering).

    Raises:
        ValueError: if fewer than 3 valid matches remain before or after bounds filtering.
    """
    def _fit_affine_ls(P, U):
        P1 = np.column_stack([P, np.ones(len(P), dtype=np.float32)]).astype(np.float32)
        cx, *_ = np.linalg.lstsq(P1, U[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(P1, U[:, 1], rcond=None)
        A2 = np.array([[cx[0], cx[1]], [cy[0], cy[1]]], dtype=np.float32)
        t = np.array([cx[2], cy[2]], dtype=np.float32)  # (tx,ty)
        return A2, t

    def _fit_similarity(P, U):
        # Least-squares similarity U ~= s*R*P + t (Umeyama). 4 DOF: uniform scale, rotation,
        # translation. Without independent per-axis scale it cannot collapse or smear
        # the way a full affine can on few or noisy matches.
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

    # Drop matches outside the given image bounds before any fit.
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

    # Plane grid over the canvas; (x0, y0) becomes the origin of the plane coordinates.
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

    # In-plane similarity uv -> histology xy (RANSAC + one trim-refit pass if robust).
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
        A2, t_xy = _fit_similarity(P_local, U_xy)
        inlier_mask = np.ones(N, dtype=bool)

    A2_invT = np.linalg.inv(A2).T.astype(np.float32)
    s_uni, R2, A_residual = _decompose_A2(A2.astype(np.float64))

    # 3D embedding of the plane: xyz = X0 + u*e1 + v*e2.
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


def matches_to_common_canvas(clean_matches, oy_h, ox_h):
    """Shift histology match centres from the histology image into the common canvas.

    Matches without ``mri_center_zyx`` or ``hist_center_yx``, or whose MRI point is
    (-1, -1, -1), are dropped. Each kept match is copied and only its ``hist_center_yx`` is
    shifted; all other keys, including ``hist_topleft_yx`` and the MRI coordinates, are
    copied unchanged.

    Args:
        clean_matches: list of match dicts with ``hist_center_yx`` in histology pixels.
        oy_h: row offset of the histology image in the common canvas.
        ox_h: column offset of the histology image in the common canvas.

    Returns:
        New list of match dicts with ``hist_center_yx`` = (y + oy_h, x + ox_h) as floats.
    """
    clean_matches_common = []
    for m in clean_matches:
        if "mri_center_zyx" not in m or "hist_center_yx" not in m:
            continue
        z, y, x = m["mri_center_zyx"]
        if (z, y, x) == (-1, -1, -1):
            continue
        hy, hx = m["hist_center_yx"]

        # Histology image pixels -> common-canvas pixels.
        hyc = hy + oy_h
        hxc = hx + ox_h

        mm = dict(m)
        mm["hist_center_yx"] = (float(hyc), float(hxc))
        # mri_center_zyx stays in MRI slab voxels.
        clean_matches_common.append(mm)

    return clean_matches_common


def xyz_from_G_uv(uv, G):
    """Map plane coordinates to MRI positions on the plane of G.

    Args:
        uv: (N, 2) plane coordinates (u, v) in voxels, numpy array or torch tensor.
        G: global plane dict from ``fit_global_plane`` (uses X0, e1, e2).

    Returns:
        (N, 3) float32 numpy array of MRI (x, y, z) voxel positions, X0 + u*e1 + v*e2.
    """
    uv = uv.detach().cpu().numpy() if torch.is_tensor(uv) else np.asarray(uv, np.float32)
    X0 = np.asarray(G["X0"], np.float32)
    e1 = np.asarray(G["e1"], np.float32)
    e2 = np.asarray(G["e2"], np.float32)
    return X0[None, :] + uv[:, 0:1] * e1[None, :] + uv[:, 1:2] * e2[None, :]


def hist_xy_from_G_uv(uv, G):
    """Map plane coordinates to histology pixels with the in-plane similarity of G.

    Args:
        uv: (N, 2) plane coordinates (u, v) in voxels, numpy array or torch tensor.
        G: global plane dict from ``fit_global_plane`` (uses A2, t_xy).

    Returns:
        (N, 2) float32 numpy array of histology (x, y) in common-canvas pixels,
        uv @ A2.T + t_xy.
    """
    uv = uv.detach().cpu().numpy() if torch.is_tensor(uv) else np.asarray(uv, np.float32)
    A2 = np.asarray(G["A2"], np.float32)
    t = np.asarray(G["t_xy"], np.float32)
    return uv @ A2.T + t[None, :]

