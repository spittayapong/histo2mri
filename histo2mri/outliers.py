"""Robust outlier rejection for histology-to-MRI patch matches.

Runs after patch matching and before the ROI plane / common canvas step. A
histology section is assumed to lie close to one plane in the MRI volume, so
matches whose MRI centers fall far from that plane, or that are spatially
isolated, are discarded. Match points are handled as MRI voxel indices in
(z, y, x) order (``mri_center_zyx``) and converted to (x, y, z) for plane fits.
"""
import numpy as np
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


def ransac_plane_fit_z_from_matches_robust(
    matches, score_key="score", z_min=None, z_max=None,
    dist_thresh=8.0, n_iter=2000, seed=0,
    knn_k=3, knn_mad_k=3.5,    # spatial isolation filter
):
    """Fit a plane z = a*x + b*y + c to match centers with a kNN pre-filter and RANSAC.

    Matches are first restricted to valid ones (finite score and center,
    z >= 0, optionally z inside [z_min, z_max]). Stage 1 removes points that are
    isolated in (x, y): the distance to the k-th nearest neighbor is compared
    with a median + knn_mad_k * 1.4826 * MAD threshold (skipped for 3 or fewer
    points). Stage 2 runs RANSAC on the remaining (x, y, z) points: each
    iteration fits a plane through 3 random points and counts points whose
    orthogonal distance to that plane is at most ``dist_thresh``. The largest
    inlier set is kept and z = a*x + b*y + c is refitted to it by least squares.
    Scores are only used to reject non-finite entries; they do not weight the fit.

    Args:
        matches: List of match dicts; uses ``mri_center_zyx`` and ``score_key``.
        score_key: Key of the match score in each dict.
        z_min: Optional lower bound on the MRI z index (inclusive).
        z_max: Optional upper bound on the MRI z index (inclusive).
        dist_thresh: RANSAC inlier threshold, orthogonal point-to-plane
            distance in MRI voxels.
        n_iter: Number of RANSAC iterations.
        seed: Seed for the RANSAC random generator.
        knn_k: Neighbour rank k for the (x, y) isolation filter.
        knn_mad_k: Number of robust standard deviations (1.4826 * MAD) above
            the median k-th neighbor distance at which a point is rejected.

    Returns:
        dict. On success: ``ok=True``, plane coefficients ``a``, ``b``, ``c``
        (MRI voxel units, z = a*x + b*y + c), ``inlier_idx`` (indices into
        ``matches``), ``inlier_count``, ``total_count`` (number of valid matches
        before both stages) and ``inlier_ratio``. On failure: ``ok=False`` and
        ``reason`` ("too_few_points", "too_few_after_spatial_filter" or
        "no_model").
    """
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
    # Reject points whose k-th nearest neighbor in (x, y) is unusually far away
    # (median + MAD threshold on the k-th neighbor distance).
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
    # Reject points far from the best 3-point plane (orthogonal distance).
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

    # Least-squares refit of z = a*x + b*y + c on the RANSAC inliers.
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
    """Return the valid MRI match centers (z, y, x) and scores, optionally with their indices.

    Valid means finite score and center, z >= 0 and z inside [z_min, z_max] if given.
    Returns (pts [N, 3] as (z, y, x), sc [N]) or, with ``return_idx=True``,
    (pts, sc, idx_valid) where idx_valid [N] holds indices into ``matches``.
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
    plane_mad_k=2.5,      # smaller = stricter
    tukey_k=1.5,          # smaller = stricter
    cc_radius_scale=2.5,  # smaller = stricter
    min_component_size=6,
):
    """Compute a keep-mask for match centers using a plane, kNN-IQR and connectivity test.

    Three steps on points converted to (x, y, z):

    1. Plane test: fit a total-least-squares plane (SVD of the centered points)
       and drop points whose orthogonal distance exceeds
       median + plane_mad_k * 1.4826 * MAD of the distances.
    2. Local density test: project the remaining points onto the plane's two
       in-plane axes, take for each point the median distance to its k nearest
       neighbors (``r``), and keep points with ``r`` inside the Tukey fences
       [max(0, Q1 - tukey_k*IQR), Q3 + tukey_k*IQR]. The lower fence can be
       above zero, so points in unusually dense clusters can be dropped too.
    3. Connectivity: link kept points closer than cc_radius_scale * median(r)
       in the plane and keep only the largest connected component, provided
       it has at least ``min_component_size`` points; otherwise the mask from
       step 2 is returned unchanged.

    With fewer than 4 points all points are kept.

    Args:
        pts_zyx: Match centers, array [N, 3] in MRI voxel (z, y, x).
        k: Number of neighbors for the in-plane kNN distance.
        plane_mad_k: MAD multiplier for the plane-distance test.
        tukey_k: Tukey fence multiplier for the kNN distance test.
        cc_radius_scale: Linking radius for the connected-component step,
            as a multiple of the median kNN distance of the kept points.
        min_component_size: Minimum size of the largest component for the
            connectivity step to be applied.

    Returns:
        Tuple (keep, r, fence):
            keep: bool array [N], True for points that are kept.
            r: float array [N], in-plane median kNN distance in voxels
                (larger means more isolated); inf for points removed in step 1.
            fence: (lo, hi) Tukey fences applied to ``r``.
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

        # Linking radius scales with the typical kNN spacing of the kept points.
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
    plane_dist_thresh=20.0,
    knn_params=None,
):
    """Remove outlier matches in two stages: coarse RANSAC plane, then kNN/IQR cleanup.

    Stage 1 (coarse) calls ``ransac_plane_fit_z_from_matches_robust``: an
    (x, y) isolation filter (k=3, MAD multiplier 3.5) followed by a RANSAC
    plane fit for z = a*x + b*y + c with inlier distance ``plane_dist_thresh``.
    Stage 2 (strict) runs ``knn_outlier_mask_plane_xy_iqr`` on the stage-1
    inliers: SVD plane distance test, in-plane kNN-IQR test and largest
    connected component. Both thresholds come from the parameter profile
    (``outlier_plane_dist`` and ``outlier_knn`` in ``pipeline.PROFILES``).

    Args:
        all_matches: List of match dicts from ``match_patches`` (uses
            ``mri_center_zyx`` and ``score``).
        z_min: Optional lower bound on the MRI z index (inclusive).
        z_max: Optional upper bound on the MRI z index (inclusive).
        plane_dist_thresh: RANSAC inlier threshold in MRI voxels (orthogonal
            distance to the plane).
        knn_params: Keyword arguments for ``knn_outlier_mask_plane_xy_iqr``
            (k, plane_mad_k, tukey_k, cc_radius_scale, optionally
            min_component_size). Defaults to the "default" profile values.

    Returns:
        Tuple (clean_matches, coarse_matches, pts_clean, sc_clean):
            clean_matches: Match dicts kept after both stages.
            coarse_matches: Match dicts kept after stage 1 only.
            pts_clean: float32 array [M, 3], MRI centers (z, y, x) of
                ``clean_matches``.
            sc_clean: float32 array [M], their scores.
    """
    if knn_params is None:
        knn_params = dict(k=8, plane_mad_k=3.0, tukey_k=3.0, cc_radius_scale=3.0)

    # Stage 1: coarse RANSAC plane filter.
    coarse_idx = ransac_plane_fit_z_from_matches_robust(
        all_matches,
        z_min=z_min,
        z_max=z_max,
        dist_thresh=plane_dist_thresh,
        knn_k=3,
        knn_mad_k=3.5,
    )
    if not coarse_idx.get("ok", False):
        raise ValueError(f"Coarse outlier filter failed ({coarse_idx.get('reason')}): "
                         f"too few usable matches ({len(all_matches)} candidates)")

    coarse_matches = [all_matches[i] for i in coarse_idx["inlier_idx"]]

    pts, sc, idx_valid = _extract_match_points(
        all_matches, z_min=z_min, z_max=z_max, return_idx=True
    )


    # Restrict the valid points to the stage-1 inliers, keeping their match indices.
    inlier_set = set(coarse_idx["inlier_idx"])
    inlier_mask = np.array([i in inlier_set for i in idx_valid], dtype=bool)

    pts_in = pts[inlier_mask]
    sc_in  = sc[inlier_mask]
    idx_in = idx_valid[inlier_mask]

    # Stage 2: stricter plane / kNN-IQR / connectivity cleanup.
    clean_idx, _, _ = knn_outlier_mask_plane_xy_iqr(pts_in, **knn_params)
    pts_clean = pts_in[clean_idx]
    sc_clean  = sc_in[clean_idx]
    idx_clean = idx_in[clean_idx]
    logger.info('Outlier filter: removed %s, kept %s matches', (~clean_idx).sum(), clean_idx.sum())

    clean_matches = [all_matches[i] for i in idx_clean]

    return clean_matches, coarse_matches, pts_clean, sc_clean

