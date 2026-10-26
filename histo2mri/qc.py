"""Quality-control figures for each stage of the registration.

Covers the prepared images (preprocessing), patch matches and the ROI plane
(matching / canvas), the histology and MRI on the fitted sheet (refinement)
and the MRI mapped back onto full-resolution histology (inverse). Volumes are
indexed (z, y, x); match points are MRI voxel indices (z, y, x). Figures are
shown with matplotlib and saved only when a path is given.
"""
import numpy as np
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
import math
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from .canvas import _to_zyx_volume, fit_plane_z_ax_by_c
from .outliers import _extract_match_points
from .sampling import apply_lut_monotone, make_monotone_lut, robust01
import logging

logger = logging.getLogger(__name__)


def _show():
    """Show the current figures when the Matplotlib backend can display them (not under Agg)."""
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


def show_proc_image(img, mask, title, idx=0, save_file=None):
    """Show one slice of a prepared image and its mask.

    The image is set to 0 outside the mask. The figure has two panels, the
    masked image and the mask, both flipped vertically for display. If
    ``save_file`` is given, only the masked image panel is saved, as a
    separate figure at 300 dpi.

    Args:
        img: Prepared image (SimpleITK image, 3D).
        mask: Its mask (SimpleITK image, same size).
        title: Label used in the panel titles (e.g. "Histo", "MRI").
        idx: Index of the slice along the first array axis (z).
        save_file: Optional output path for the image panel; its directory is
            created if needed.
    """
    arr_img = sitk.GetArrayFromImage(img)[idx]
    arr_mask = sitk.GetArrayFromImage(mask)[idx]

    arr_img[arr_mask==0.0] = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    axes[0].imshow(np.flipud(arr_img), cmap='gray')
    axes[0].set_title(f"Processed {title}")

    axes[1].imshow(np.flipud(arr_mask), cmap='gray')
    axes[1].set_title(f"Processed {title} Mask")

    plt.tight_layout()
    _show()
    plt.close(fig)

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(np.flipud(arr_img), cmap='gray')
        ax2.set_title(f"Processed {title}")
        ax2.axis("on")

        fig2.tight_layout()
        fig2.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close(fig2)

        logger.debug(f'Saved histology image to: {save_file}')


def show_matched_pairs(
    matches,
    hist_img,
    mri_img,
    patch=64,
    ncols=3,
    max_pairs=None,
    sort_by_score=True,
    save_file=None,
    hist_mask=None,
    mri_mask=None,
):
    """Plot a grid of matched histology/MRI patch pairs.

    Only matches with a valid MRI position (z >= 0) are shown, by default
    sorted by descending score. Each pair has a histology patch (title:
    histology index and center), the MRI patch from slice z (title: MRI
    center and score) and, if both masks are given, the overlap of the two
    patch masks (title: fraction of overlapping pixels). Patches are flipped
    vertically for display. The figure is saved at 200 dpi if ``save_file``
    is given.

    Args:
        matches: List of match dicts (uses ``hist_topleft_yx``,
            ``mri_topleft_zyx``, ``hist_center_yx``, ``mri_center_zyx``,
            ``hist_index`` and ``score``).
        hist_img: Histology array [H, W] used for matching.
        mri_img: MRI slab array [Z, Y, X] used for matching.
        patch: Patch size in pixels (same for histology and MRI).
        ncols: Number of pairs per row.
        max_pairs: Optional maximum number of pairs to show.
        sort_by_score: Sort pairs by descending score.
        save_file: Optional output path.
        hist_mask: Optional histology mask [H, W].
        mri_mask: Optional MRI mask [Z, Y, X].
    """
    vals = [m for m in matches if m["mri_center_zyx"][0] >= 0]
    if len(vals) == 0:
        logger.warning('No valid matches.')
        return

    if sort_by_score:
        vals = sorted(vals, key=lambda d: d["score"], reverse=True)
    if max_pairs is not None:
        vals = vals[:max_pairs]

    show_overlap = (hist_mask is not None) and (mri_mask is not None)
    panels_per_pair = 3 if show_overlap else 2

    n = len(vals)
    rows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        rows,
        ncols * panels_per_pair,
        figsize=(3 * panels_per_pair * ncols, 3 * rows)
    )
    axes = np.array(axes).reshape(rows, ncols * panels_per_pair)

    for r in range(rows):
        for c in range(ncols * panels_per_pair):
            axes[r, c].axis("off")

    for k, m in enumerate(vals):
        r = k // ncols
        g = k % ncols

        c0 = panels_per_pair * g
        c1 = c0 + 1
        c2 = c0 + 2 if show_overlap else None

        hy0, hx0 = m["hist_topleft_yx"]
        mz, my0, mx0 = m["mri_topleft_zyx"]

        hp = hist_img[hy0:hy0 + patch, hx0:hx0 + patch]
        mp = mri_img[mz, my0:my0 + patch, mx0:mx0 + patch]

        axes[r, c0].imshow(np.flipud(hp), cmap="gray")
        axes[r, c0].set_title(
            f"Hist idx={m['hist_index']} center={m['hist_center_yx']}",
            fontsize=9
        )
        axes[r, c0].axis("off")

        axes[r, c1].imshow(np.flipud(mp), cmap="gray")
        axes[r, c1].set_title(
            f"MRI {m['mri_center_zyx']} score={m['score']:.3f}",
            fontsize=9
        )
        axes[r, c1].axis("off")

        if show_overlap:
            hm = hist_mask[hy0:hy0 + patch, hx0:hx0 + patch] > 0
            mm = mri_mask[mz, my0:my0 + patch, mx0:mx0 + patch] > 0

            overlap = hm & mm
            overlap_frac = overlap.mean()

            axes[r, c2].imshow(np.flipud(overlap.astype(np.uint8) * 255), cmap="gray", vmin=0, vmax=255)
            axes[r, c2].set_title(f"Overlap frac={overlap_frac:.3f}", fontsize=9)
            axes[r, c2].axis("off")

    plt.tight_layout()

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
        fig.savefig(save_file, dpi=200, bbox_inches="tight")
        logger.debug(f'Saved figure to: {save_file}')

    _show()
    plt.close(fig)


def _make_xz_canvas(vol_zyx, z_min, z_max, mode="mip"):
    """Project the z band [z_min, z_max] over y (max or mean) into a [Z, X] image (0 elsewhere)."""
    Z, Y, X = vol_zyx.shape
    z_min = int(max(0, z_min))
    z_max = int(min(Z - 1, z_max))
    band = vol_zyx[z_min:z_max + 1]  # [zb,Y,X]

    if mode == "mean":
        proj = band.mean(axis=1)  # [zb,X]
    else:
        proj = band.max(axis=1)   # [zb,X] mip over Y

    canvas = np.zeros((Z, X), dtype=np.float32)  # full Z range
    canvas[z_min:z_max + 1, :] = proj
    return canvas


def _make_yz_canvas(vol_zyx, z_min, z_max, mode="mip"):
    """Project the z band [z_min, z_max] over x (max or mean) into a [Z, Y] image (0 elsewhere)."""
    Z, Y, X = vol_zyx.shape
    z_min = int(max(0, z_min))
    z_max = int(min(Z - 1, z_max))
    band = vol_zyx[z_min:z_max + 1]  # [zb,Y,X]

    if mode == "mean":
        proj = band.mean(axis=2)  # [zb,Y]
    else:
        proj = band.max(axis=2)   # [zb,Y] mip over X

    canvas = np.zeros((Z, Y), dtype=np.float32)  # full Z range
    canvas[z_min:z_max + 1, :] = proj
    return canvas


def visualize_roi_from_matches(
    matches,
    mri_vol,
    z_min=None,
    z_max=None,
    score_key="score",
    max_points=50000,
    plane_stride=16,
    mode="mip",           # "mip" or "mean" for XZ/YZ background
    elev=20,
    azim=35,
    use_score_weight=True,
    figsize=(18, 5),
    save_file=None,
):
    """Plot match centers in the MRI volume with a fitted plane and XZ/YZ projections.

    Valid match centers inside [z_min, z_max] are collected (randomly
    subsampled to ``max_points`` if needed) and a plane z = a*x + b*y + c is
    fitted to them, weighted by min-max normalized score if
    ``use_score_weight``. This plane is for display only. One figure with
    three panels, points colored by score:

    - 3D scatter of the centers with the fitted plane (clipped to [0, Z-1]).
    - XZ view: centers over the projection of the z band along y.
    - YZ view: centers over the projection of the z band along x.

    Projections are maximum ("mip") or mean ("mean") intensity, rescaled to
    the 1st-99th percentile. The figure is saved at 200 dpi if ``save_file``
    is given, and the plane equation is printed.

    Args:
        matches: List of match dicts (uses ``mri_center_zyx`` and ``score_key``).
        mri_vol: MRI volume, SimpleITK image or array [Z, Y, X].
        z_min: Lower z index of the band (default 0).
        z_max: Upper z index of the band (default Z-1).
        score_key: Key of the match score.
        max_points: Maximum number of points plotted.
        plane_stride: Grid spacing in voxels of the plotted plane mesh.
        mode: Projection type for the XZ/YZ backgrounds, "mip" or "mean".
        elev: Elevation of the 3D view in degrees.
        azim: Azimuth of the 3D view in degrees.
        use_score_weight: Weight the plane fit by normalized score.
        figsize: Figure size in inches.
        save_file: Optional output path.

    Returns:
        dict with plane coefficients ``a``, ``b``, ``c`` and ``n_points``, or
        None if fewer than 3 valid points are available.
    """
    vol = _to_zyx_volume(mri_vol)  # [Z,Y,X]
    Z, Y, X = vol.shape

    if z_min is None:
        z_min = 0
    if z_max is None:
        z_max = Z - 1
    z_min = int(max(0, z_min))
    z_max = int(min(Z - 1, z_max))

    pts, sc = _extract_match_points(matches, score_key=score_key, z_min=z_min, z_max=z_max)
    if len(pts) < 3:
        logger.warning('Not enough valid points for visualization.')
        return None

    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, sc = pts[idx], sc[idx]

    # Score-weighted plane fit (weights = scores min-max normalized to [0, 1]).
    w = None
    if use_score_weight:
        s = sc - np.min(sc)
        w = s / (np.max(s) + 1e-8)
    a, b, c = fit_plane_z_ax_by_c(pts, weights=w)

    # plane mesh for 3D panel
    xx = np.arange(0, X, plane_stride, dtype=np.float32)
    yy = np.arange(0, Y, plane_stride, dtype=np.float32)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.clip(a * XX + b * YY + c, 0, Z - 1)

    # XZ/YZ canvases
    def _norm01_img(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)

    canvas_xz = _norm01_img(_make_xz_canvas(vol, z_min, z_max, mode=mode))
    canvas_yz = _norm01_img(_make_yz_canvas(vol, z_min, z_max, mode=mode))

    z, y, x = pts[:, 0], pts[:, 1], pts[:, 2]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0])

    # (1) 3D
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax0.scatter(x, y, z, c=sc, s=8, cmap="turbo", alpha=0.9)
    ax0.plot_surface(XX, YY, ZZ, alpha=0.28, color="cyan", linewidth=0, antialiased=False)
    ax0.set_xlim(0, X - 1); ax0.set_ylim(0, Y - 1); ax0.set_zlim(0, Z - 1)
    ax0.set_xlabel("X"); ax0.set_ylabel("Y"); ax0.set_zlabel("Z")
    ax0.set_title("3D points + fitted plane")
    ax0.view_init(elev=elev, azim=azim)

    # (2) XZ
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(canvas_xz, cmap="gray", origin="lower", extent=[0, X - 1, 0, Z - 1], aspect="auto")
    ax1.scatter(x, z, c=sc, s=10, cmap="turbo", alpha=0.9)
    ax1.set_xlim(0, X - 1); ax1.set_ylim(0, Z - 1)
    ax1.set_xlabel("X"); ax1.set_ylabel("Z")
    ax1.set_title(f"XZ overlay ({mode})")

    # (3) YZ
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(canvas_yz, cmap="gray", origin="lower", extent=[0, Y - 1, 0, Z - 1], aspect="auto")
    p2 = ax2.scatter(y, z, c=sc, s=10, cmap="turbo", alpha=0.9)
    ax2.set_xlim(0, Y - 1); ax2.set_ylim(0, Z - 1)
    ax2.set_xlabel("Y"); ax2.set_ylabel("Z")
    ax2.set_title(f"YZ overlay ({mode})")

    # One colorbar shared by all three panels.
    fig.colorbar(p2, ax=[ax0, ax1, ax2], shrink=0.8, label=score_key)

    if save_file is not None:
        import os
        save_dir = os.path.dirname(save_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_file, dpi=200, bbox_inches="tight")
        logger.debug(f'Saved figure to: {save_file}')

    logger.debug(f'Fitted plane: z = {a:.5f}*x + {b:.5f}*y + {c:.5f}')
    _show()
    plt.close(fig)

    return {"a": float(a), "b": float(b), "c": float(c), "n_points": int(len(pts))}


def plot_roi_support(clean_matches, scaled_proc_Ih, plane_img, x0, y0, step=1.0, save_file=None):
    """Plot the convex-hull support of the clean matches on the histology and on the ROI plane.

    Left panel: scaled histology with the match centers (``hist_center_yx``) and their
    convex hull, titled with the fraction of image pixels inside it. Right panel: the
    MRI plane image from canvas.extract_roi_plane with the MRI match centers converted to
    plane-image pixels, their hull, and the hull coverage over all pixels and over
    non-NaN pixels. The coverages are logged at debug level.

    Args:
        clean_matches: Outlier-filtered match dicts (``hist_center_yx``, ``mri_center_zyx``).
        scaled_proc_Ih: Scaled, prepared histology (SimpleITK image or array; a 3D array
            uses slice 0).
        plane_img: MRI plane image [H, W] from canvas.extract_roi_plane.
        x0: MRI x voxel index of plane-image column 0.
        y0: MRI y voxel index of plane-image row 0.
        step: Plane grid spacing in voxels.
        save_file: Optional path for the figure (PNG, 200 dpi).

    Raises:
        ValueError: If fewer than 3 points are available for either hull.
    """
    h = sitk.GetArrayFromImage(scaled_proc_Ih) if hasattr(scaled_proc_Ih, "GetSize") else np.asarray(scaled_proc_Ih)
    if h.ndim == 3:
        h = h[0]

    # Histology support
    pts_hist = np.array([[m["hist_center_yx"][1], m["hist_center_yx"][0]] for m in clean_matches], dtype=np.float32)  # (x,y)
    if len(pts_hist) < 3:
        raise ValueError("Need >=3 clean_matches for histology hull.")
    poly_h = pts_hist[ConvexHull(pts_hist).vertices]
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    hist_cov = float(Path(poly_h).contains_points(np.c_[xx.ravel(), yy.ravel()]).mean())

    # Plane support: MRI (x, y) voxel positions converted to plane-image pixels.
    pts_xy = np.array(
        [[m["mri_center_zyx"][2], m["mri_center_zyx"][1]] for m in clean_matches if m["mri_center_zyx"][0] >= 0],
        dtype=np.float32,
    )
    Hp, Wp = plane_img.shape
    px = (pts_xy[:, 0] - float(x0)) / step
    py = (pts_xy[:, 1] - float(y0)) / step
    in_img = (px >= 0) & (px < Wp) & (py >= 0) & (py < Hp)
    pts_px_in = np.stack([px, py], axis=1)[in_img]
    if len(pts_px_in) < 3:
        raise ValueError("Need >=3 MRI points inside the plane image for the plane hull.")
    poly_p = pts_px_in[ConvexHull(pts_px_in).vertices]
    yyg, xxg = np.mgrid[0:Hp, 0:Wp]
    inside_p = Path(poly_p).contains_points(np.c_[xxg.ravel(), yyg.ravel()]).reshape(Hp, Wp)
    plane_cov_all = float(inside_p.mean())
    finite = np.isfinite(plane_img)
    plane_cov_valid = float((inside_p & finite).sum() / max(finite.sum(), 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(h, cmap="gray", origin="lower")
    axes[0].scatter(pts_hist[:, 0], pts_hist[:, 1], s=8, c="lime")
    axes[0].plot(np.r_[poly_h[:, 0], poly_h[0, 0]], np.r_[poly_h[:, 1], poly_h[0, 1]], c="cyan", lw=1.2)
    axes[0].set_title(f"Histology support (hull cov={hist_cov:.3f})")
    axes[0].axis("off")
    axes[1].imshow(np.nan_to_num(plane_img, nan=0.0), cmap="gray", origin="lower")
    axes[1].scatter(pts_px_in[:, 0], pts_px_in[:, 1], s=8, c="lime")
    axes[1].plot(np.r_[poly_p[:, 0], poly_p[0, 0]], np.r_[poly_p[:, 1], poly_p[0, 1]], c="cyan", lw=1.2)
    axes[1].set_title(f"Plane support (all={plane_cov_all:.3f}, valid={plane_cov_valid:.3f})")
    axes[1].axis("off")
    plt.tight_layout()

    if save_file is not None:
        save_dir = os.path.dirname(save_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_file, dpi=200, bbox_inches="tight")
        logger.debug(f"Saved figure to: {save_file}")
    _show()
    plt.close(fig)

    logger.debug(f"hull coverage: hist={hist_cov:.6f} plane_all={plane_cov_all:.6f} plane_valid={plane_cov_valid:.6f}")


def plot_mri_on_sheet(
    hist_on_init_tps,
    valid_display,
    mri_on_final_sheet,
    save_file=None,
    save_file_single=None,
    show_checker=True,
    checker_tile=24,
    flip_display=False,
):
    """Plot the histology and the MRI on the refined sheet side by side.

    Both images live on the sheet (common-canvas) grid. They are set to 0
    outside ``valid_display`` and optionally flipped vertically. Panels:
    histology on the sheet, the valid display mask, the MRI sampled on the
    final sheet, and (if ``show_checker``) a checkerboard of the two images,
    each min-max normalized. The figure is saved at 300 dpi to ``save_file``;
    ``save_file_single`` saves the MRI panel alone.

    Args:
        hist_on_init_tps: Histology resampled on the initial surface (G plus
            residual TPS), array/tensor/SimpleITK image [H, W] or [1, H, W].
        valid_display: Display mask of the same shape (> 0.5 is valid).
        mri_on_final_sheet: MRI sampled on the refined sheet, same shape.
        save_file: Optional output path for the full figure.
        save_file_single: Optional output path for the MRI-only image.
        show_checker: Add the checkerboard panel.
        checker_tile: Checkerboard tile size in pixels.
        flip_display: Flip all images vertically before display.

    Returns:
        Tuple (h_disp, v_float, r_disp, v): masked (and possibly flipped)
        histology [H, W], the valid mask as float32 [H, W], the masked MRI
        [H, W], and the valid mask as bool [H, W].
    """

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
        logger.debug(f'Saved: {save_file}')

    _show()
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
        logger.debug(f'Saved mri_on_final_sheet to: {save_file_single}')

    return h_disp, v.astype(np.float32), r_disp, v


def normalize_image(img, eps=1e-8):
    """Min-max normalize an image to [0, 1].

    Args:
        img: Image array of any shape.
        eps: Small constant added to the denominator.

    Returns:
        float32 array of the same shape.
    """
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()
    return (img - min_val) / (max_val - min_val + eps)


def make_checkerboard(a, b, tile_size=32):
    """Build a checkerboard composite of two images after min-max normalization.

    Args:
        a: 2D image [H, W]; fills the tiles with odd (row + column) tile index.
        b: 2D image of the same shape; fills the other tiles, including the
            tile at array index (0, 0).
        tile_size: Tile size in pixels.

    Returns:
        float32 array [H, W] in [0, 1].

    Raises:
        ValueError: If the shapes differ.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_norm = normalize_image(a)
    b_norm = normalize_image(b)

    h, w = a.shape

    yy, xx = np.indices((h, w))
    mask = ((yy // tile_size + xx // tile_size) % 2).astype(bool)

    out = np.where(mask, a_norm, b_norm)
    return out


def plot_mri_on_histology(mri_on_histo, Ih_hr, Mh_hr, tile_size=384, save_file=None, save_file_single=None):
    """Plot the MRI resampled onto the HR histology grid next to the histology.

    All images are flipped vertically and multiplied by the histology mask.
    The 2x3 figure shows:

    - Top row: masked histology, masked MRI, and their checkerboard (each
      min-max normalized).
    - Bottom row: histology intensity-mapped to the MRI by a monotone
      quantile LUT, MRI rescaled to [0, 1] by its 1st-99th percentile, and
      their checkerboard. The percentiles and the LUT are computed over the
      whole image, including the zeroed background.

    The figure is saved at 300 dpi if ``save_file`` is given;
    ``save_file_single`` saves the masked MRI panel alone.

    Args:
        mri_on_histo: MRI resampled on the HR histology grid, array [H, W].
        Ih_hr: HR grayscale histology (SimpleITK image, single slice).
        Mh_hr: HR histology mask (SimpleITK image, single slice).
        tile_size: Checkerboard tile size in HR pixels.
        save_file: Optional output path for the full figure.
        save_file_single: Optional output path for the MRI-only image.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()

    arr_Ih_hr = sitk.GetArrayFromImage(Ih_hr)[0]
    arr_Mh_hr = sitk.GetArrayFromImage(Mh_hr)[0]

    hist_disp = np.flipud(arr_Ih_hr).astype(np.float32)
    mri_disp  = np.flipud(mri_on_histo).astype(np.float32)
    mask_disp = np.flipud(arr_Mh_hr).astype(np.float32)

    # Restrict both images to the histology mask.
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
        os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
        fig.savefig(save_file, dpi=300, bbox_inches="tight")
        logger.debug(f'Saved figure to: {save_file}')

    _show()
    plt.close(fig)

    if save_file_single is not None:
        os.makedirs(os.path.dirname(save_file_single) or '.', exist_ok=True)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(mri_disp, cmap='gray')
        ax2.set_title("MRI on histology space")
        ax2.axis("on")
        fig2.tight_layout()
        fig2.savefig(save_file_single, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        logger.debug(f'Saved image to: {save_file_single}')


def plot_mri_on_histology_rgb(
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
    """Plot the MRI on the HR histology grid against grayscale and RGB histology.

    Inside the optional mask, histology and MRI are rescaled to [0, 1] by
    their ``pclip`` percentiles, and the grayscale histology is mapped to the
    MRI intensity distribution with a monotone quantile LUT. Pixels outside
    the mask are shown white. The 2x3 figure shows:

    - Top row: LUT-mapped histology, rescaled MRI, and their checkerboard.
    - Bottom row: RGB histology, rescaled MRI, and a checkerboard of RGB
      histology and MRI.

    In both checkerboards the tile at array index (0, 0) shows histology.
    Images are drawn with ``origin="lower"`` and optionally flipped
    vertically first.
    The figure is saved at 300 dpi if ``save_file`` is given.

    Args:
        arr_Ih_hr: HR grayscale histology, [H, W] or singleton 3D, array or
            SimpleITK image.
        arr_Ih_rgb_hr: HR RGB histology, array [H, W, 3] (0-255 or 0-1).
        mri_clean: MRI resampled on the HR histology grid, [H, W].
        arr_Mh_hr: Optional HR histology mask, [H, W] (> 0 is tissue).
        tile: Checkerboard tile size in HR pixels.
        pclip: Lower/upper percentiles for intensity rescaling.
        n_quant: Number of quantiles for the LUT.
        save_file: Optional output path.
        flip_display: Flip all images vertically before display.

    Raises:
        ValueError: If the shapes of the inputs do not agree.
    """
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
        logger.debug(f'Saved: {save_file}')

    _show()
    plt.close(fig)


def _plot_surface(z_tensor, title, save_file=None):
    """Plot a 2D tensor [H, W] as a 3D height surface; save at 300 dpi if ``save_file`` is given."""
    z = z_tensor.detach().cpu().numpy()          # (H, W)
    ys, xs = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Translucent red fill with a thin red wireframe to show the shape.
    ax.plot_surface(
        xs, ys, z,
        color=(0.95, 0.45, 0.45, 0.35),  # RGBA (soft red, 35% opaque)
        edgecolor=(0.82, 0.20, 0.20, 0.6),
        linewidth=0.3,
        antialiased=True
    )

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    plt.tight_layout()

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
        fig.savefig(save_file, dpi=300, bbox_inches="tight")
        logger.debug(f'Saved figure to: {save_file}')

    _show()
    plt.close(fig)

