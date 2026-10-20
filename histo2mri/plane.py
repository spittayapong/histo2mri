"""Global in-plane transform G: a plane in MRI space fitted to the patch matches."""
import numpy as np
import torch


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


def matches_to_common_canvas(clean_matches, oy_h, ox_h):
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

