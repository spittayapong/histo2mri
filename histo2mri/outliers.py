"""Robust outlier rejection for histology-to-MRI patch matches."""
import numpy as np
from scipy.spatial import cKDTree


def ransac_plane_fit_z_from_matches_robust(
    matches, score_key="score", z_min=None, z_max=None,
    dist_thresh=8.0, n_iter=2000, seed=0,
    knn_k=3, knn_mad_k=3.5,    # spatial isolation filter
):
    pts_zyx = np.array([m["mri_center_zyx"] for m in matches], dtype=np.float64)
    sc = np.array([m.get(score_key, 0.0) for m in matches], dtype=np.float64)

    valid = np.isfinite(sc) & np.isfinite(pts_zyx).all(axis=1) & (pts_zyx[:, 0] >= 0)
    if z_min is not None: valid &= (pts_zyx[:, 0] >= z_min)
    if z_max is not None: valid &= (pts_zyx[:, 0] <= z_max)

    valid_idx = np.where(valid)[0]
    pts = pts_zyx[valid]
    if len(pts) < 3:
        return dict(ok=False, reason="too_few_points")

    # ---------- Stage 1: spatial isolation in XY ----------
    # reject points with bad (x,y) whose (x,y) are too isolated from neighbors based on kNN distance + MAD threshold
    xy = np.stack([pts[:, 2], pts[:, 1]], axis=1)  # [N,2]
    N = xy.shape[0]
    k = min(knn_k, max(1, N - 1))

    if N > 3 and k >= 1:
        D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)
        dk = np.partition(D, kth=k-1, axis=1)[:, k-1]  # k-th nearest distance

        med = np.median(dk)
        mad = np.median(np.abs(dk - med)) + 1e-8
        thr = med + knn_mad_k * 1.4826 * mad
        spatial_keep = dk <= thr
    else:
        spatial_keep = np.ones(N, dtype=bool)

    pts2 = pts[spatial_keep]
    idx2 = valid_idx[spatial_keep]
    if len(pts2) < 3:
        return dict(ok=False, reason="too_few_after_spatial_filter")

    xyz = np.stack([pts2[:, 2], pts2[:, 1], pts2[:, 0]], axis=1)  # (x,y,z)
    N2 = xyz.shape[0]
    rng = np.random.default_rng(seed)

    # ---------- Stage 2: RANSAC plane ----------
    # reject points with bad z relative to the fitted plane
    best_inliers = None
    best_cnt = -1

    for _ in range(n_iter):
        s = rng.choice(N2, size=3, replace=False)
        p1, p2, p3 = xyz[s]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n = n / nn
        d = -np.dot(n, p1)

        dist = np.abs(xyz @ n + d)
        inl = dist <= dist_thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inl

    if best_inliers is None or best_cnt < 3:
        return dict(ok=False, reason="no_model")

    in_xyz = xyz[best_inliers]
    A = np.stack([in_xyz[:, 0], in_xyz[:, 1], np.ones(len(in_xyz))], axis=1)
    z = in_xyz[:, 2]
    a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]

    # map inliers back to original matches
    inlier_match_idx = idx2[best_inliers]

    return dict(
        ok=True,
        a=float(a), b=float(b), c=float(c),
        inlier_idx=inlier_match_idx,                 # indices in original matches
        inlier_count=int(len(inlier_match_idx)),
        total_count=int(len(valid_idx)),
        inlier_ratio=float(len(inlier_match_idx) / max(1, len(valid_idx))),
    )


def _extract_match_points(
    matches,
    score_key="score",
    z_min=None,
    z_max=None,
    return_idx=False,
    dtype=np.float32,
):
    """
    Extract valid MRI match points and scores from match dicts.
    Returns:
      - return_idx=False: (pts, sc)
      - return_idx=True : (pts, sc, idx_valid)
    where:
      pts: [N,3] as (z,y,x)
      sc : [N]
      idx_valid: indices in original `matches`
    """
    if len(matches) == 0:
        pts = np.empty((0, 3), dtype=dtype)
        sc = np.empty((0,), dtype=dtype)
        if return_idx:
            return pts, sc, np.empty((0,), dtype=np.int64)
        return pts, sc

    pts = np.array([m["mri_center_zyx"] for m in matches], dtype=dtype)  # [N,3] (z,y,x)
    sc = np.array([m.get(score_key, 0.0) for m in matches], dtype=dtype)

    valid = np.isfinite(sc) & np.isfinite(pts).all(axis=1) & (pts[:, 0] >= 0)
    if z_min is not None:
        valid &= (pts[:, 0] >= z_min)
    if z_max is not None:
        valid &= (pts[:, 0] <= z_max)

    pts = pts[valid]
    sc = sc[valid]

    if return_idx:
        idx_valid = np.flatnonzero(valid).astype(np.int64)
        return pts, sc, idx_valid
    return pts, sc


def knn_outlier_mask_plane_xy_iqr(
    pts_zyx,
    k=8,
    plane_mad_k=2.5,      # stricter: 2.0
    tukey_k=1.5,          # stricter: 1.0
    cc_radius_scale=2.5,  # stricter: 2.0
    min_component_size=6,
):
    """
    Robust outlier mask for pts [z,y,x].
    Returns:
      keep2 : (N,) bool
      r     : (N,) local in-plane kNN residual (larger = more outlier)
      fence : (lo, hi) Tukey fence for r on plane-inliers

    Steps:
      1) remove points far from fitted 3D plane (MAD)
      2) in-plane local kNN-IQR outlier removal
      3) keep largest connected component on plane
    """
    pts = np.asarray(pts_zyx, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=np.float64), (0.0, 0.0)

    # zyx -> xyz
    X = pts[:, [2, 1, 0]]

    # ---- Step 1: plane MAD filter ----
    X0 = X.mean(axis=0)
    Xc = X - X0
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    e1, e2, nrm = Vt[0], Vt[1], Vt[2]
    nrm = nrm / (np.linalg.norm(nrm) + 1e-12)

    d_plane = np.abs((X - X0) @ nrm)
    med = np.median(d_plane)
    mad = np.median(np.abs(d_plane - med))
    sigma = 1.4826 * mad + 1e-8
    thr_plane = med + plane_mad_k * sigma
    keep_plane = d_plane <= thr_plane

    if keep_plane.sum() < 3:
        return keep_plane.copy(), np.full(n, np.inf), (0.0, 0.0)

    # ---- Project to plane uv ----
    P = np.column_stack([(X - X0) @ e1, (X - X0) @ e2])

    # ---- Step 2: local kNN-IQR in plane ----
    idx_plane = np.where(keep_plane)[0]
    Pp = P[idx_plane]

    tree = cKDTree(Pp)
    kk = min(k + 1, len(Pp))
    dists, _ = tree.query(Pp, k=kk)
    if kk <= 1:
        r_plane = np.zeros(len(Pp), dtype=np.float64)
    else:
        r_plane = np.median(dists[:, 1:], axis=1)

    q1, q3 = np.percentile(r_plane, [25, 75])
    iqr = max(q3 - q1, 1e-8)
    lo = max(0.0, q1 - tukey_k * iqr)
    hi = q3 + tukey_k * iqr

    keep_knn_plane = (r_plane >= lo) & (r_plane <= hi)

    # map r back to full array
    r = np.full(n, np.inf, dtype=np.float64)
    r[idx_plane] = r_plane

    keep2 = np.zeros(n, dtype=bool)
    keep2[idx_plane[keep_knn_plane]] = True

    # ---- Step 3: keep largest connected component on plane ----
    idx_keep = np.where(keep2)[0]
    if len(idx_keep) >= max(3, min_component_size):
        Pk = P[idx_keep]

        # auto radius from current local scale
        r_keep = r[idx_keep]
        med_r = np.median(r_keep[np.isfinite(r_keep)]) if np.any(np.isfinite(r_keep)) else 0.0
        radius = max(1e-8, cc_radius_scale * med_r)

        tree_k = cKDTree(Pk)
        nbrs = tree_k.query_ball_point(Pk, r=radius)

        visited = np.zeros(len(Pk), dtype=bool)
        comps = []
        for i in range(len(Pk)):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in nbrs[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(comp)

        largest = max(comps, key=len)
        if len(largest) >= min_component_size:
            keep_cc = np.zeros(len(Pk), dtype=bool)
            keep_cc[np.array(largest, dtype=int)] = True

            keep2_new = np.zeros(n, dtype=bool)
            keep2_new[idx_keep[keep_cc]] = True
            keep2 = keep2_new

    return keep2, r, (lo, hi)


def filter_outliers(
    all_matches,
    z_min=None,
    z_max=None,
):
    # Coarse pre-filter outliers.
    coarse_idx = ransac_plane_fit_z_from_matches_robust(
        all_matches,
        z_min=z_min,
        z_max=z_max,
        # dist_thresh=5.0,      # For H&E NB_C-Block
        dist_thresh=20.0,       # For others
        knn_k=3,
        knn_mad_k=3.5,
    )


    coarse_matches = [all_matches[i] for i in coarse_idx["inlier_idx"]]

    pts, sc, idx_valid = _extract_match_points(
        all_matches, z_min=z_min, z_max=z_max, return_idx=True
    )


    inlier_set = set(coarse_idx["inlier_idx"])
    inlier_mask = np.array([i in inlier_set for i in idx_valid], dtype=bool)

    pts_in = pts[inlier_mask]
    sc_in  = sc[inlier_mask]
    idx_in = idx_valid[inlier_mask]

    # Stricter outlier cleanup
    clean_idx, _, _ = knn_outlier_mask_plane_xy_iqr(
        pts_in, k=8, plane_mad_k=3.0, tukey_k=3.0, cc_radius_scale=3.0        # For others
        # pts_in, k=3, plane_mad_k=2.0, tukey_k=1.5, cc_radius_scale=2.0          # For H&E NB_C-Block
    )
    pts_clean = pts_in[clean_idx]
    sc_clean  = sc_in[clean_idx]
    idx_clean = idx_in[clean_idx]
    print("removed:", (~clean_idx).sum(), "kept:", clean_idx.sum())

    clean_matches = [all_matches[i] for i in idx_clean]

    return clean_matches, coarse_matches, pts_clean, sc_clean

