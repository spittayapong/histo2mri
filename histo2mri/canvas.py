"""MRI plane at the matched ROI, and the common canvas shared by histology and MRI.

Runs after outlier filtering and before the global-plane fit. A plane
z = a*x + b*y + c (MRI voxel indices) is fitted to the clean match centers and
the MRI slab is sampled on it, giving a 2D plane image whose row/column 0
corresponds to MRI voxel (y0, x0). The histology and the plane image are then
centered on a common canvas with offsets (oy_h, ox_h) and (oy_m, ox_m).
"""
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
import torch
import logging

logger = logging.getLogger(__name__)


def fit_plane_z_ax_by_c(pts_zyx, weights=None):
    """Fit the plane z = a*x + b*y + c to points by (weighted) least squares.

    Args:
        pts_zyx: Array [N, 3] of points as (z, y, x), e.g. MRI voxel indices.
        weights: Optional non-negative weights [N]; the residuals are weighted
            by sqrt(weight), with weights clipped below at 1e-8.

    Returns:
        Tuple (a, b, c) of floats.
    """
    z = pts_zyx[:, 0].astype(np.float64)
    y = pts_zyx[:, 1].astype(np.float64)
    x = pts_zyx[:, 2].astype(np.float64)
    A = np.stack([x, y, np.ones_like(x)], axis=1)

    if weights is not None:
        w = np.sqrt(np.clip(weights, 1e-8, None))
        Aw = A * w[:, None]
        zw = z * w
        a, b, c = np.linalg.lstsq(Aw, zw, rcond=None)[0]
    else:
        a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]

    return float(a), float(b), float(c)


def _to_zyx_volume(vol):
    """Return a SimpleITK image or array as a float32 array [Z, Y, X] (2D input gets Z=1)."""
    if isinstance(vol, sitk.Image):
        arr = sitk.GetArrayFromImage(vol)  # [Z,Y,X]
    else:
        arr = np.asarray(vol)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume [Z,Y,X], got shape={arr.shape}")
    return arr.astype(np.float32)


@torch.no_grad()
def resample_mri_on_plane_gpu(
    mri_vol, a, b, c,               # plane: z = a*x + b*y + c
    x_range=None, y_range=None,     # (min,max), in voxel index
    step=1.0,
    device="cuda:0",
    padding_mode="zeros"
):
    """Sample an MRI volume on the plane z = a*x + b*y + c with trilinear interpolation.

    A regular (x, y) grid is built over ``x_range`` x ``y_range`` with spacing
    ``step``; z is computed from the plane and the volume is sampled with
    ``torch.nn.functional.grid_sample`` (align_corners=True). Pixels whose z
    falls outside [0, Z-1] are set to NaN. Pixels with x or y outside the
    volume are not set to NaN; they get the ``padding_mode`` value.

    Args:
        mri_vol: MRI volume, SimpleITK image or array [Z, Y, X] (a 2D array is
            treated as a single slice).
        a: Plane coefficient for x (MRI voxels).
        b: Plane coefficient for y (MRI voxels).
        c: Plane offset (MRI voxels).
        x_range: (min, max) x voxel index, inclusive; defaults to the full width.
        y_range: (min, max) y voxel index, inclusive; defaults to the full height.
        step: Grid spacing in voxels.
        device: Torch device used for sampling.
        padding_mode: ``grid_sample`` padding mode for out-of-volume samples.

    Returns:
        Tuple (plane_img, XX, YY, ZZ, valid), all arrays [H, W] with row i at
        y = y_range[0] + i*step and column j at x = x_range[0] + j*step:
            plane_img: Sampled intensities (float32), NaN where z is out of range.
            XX, YY, ZZ: MRI voxel coordinates of each sample.
            valid: bool mask, True where z is inside the volume.
    """
    vol = _to_zyx_volume(mri_vol)
    Z, Y, X = vol.shape

    if x_range is None: x_range = (0, X - 1)
    if y_range is None: y_range = (0, Y - 1)

    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])

    xs = np.arange(x0, x1 + 1e-6, step, dtype=np.float32)
    ys = np.arange(y0, y1 + 1e-6, step, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)  # [H,W]
    ZZ = a * XX + b * YY + c

    valid = (ZZ >= 0) & (ZZ <= (Z - 1))

    # normalize to [-1, 1] for grid_sample (align_corners=True)
    xn = 2.0 * (XX / max(X - 1, 1)) - 1.0
    yn = 2.0 * (YY / max(Y - 1, 1)) - 1.0
    zn = 2.0 * (ZZ / max(Z - 1, 1)) - 1.0

    grid = np.stack([xn, yn, zn], axis=-1)              # [H,W,3] (x,y,z)
    grid_t = torch.from_numpy(grid).to(device)[None, None]  # [1,1,H,W,3]

    vol_t = torch.from_numpy(vol).to(device)[None, None]    # [1,1,Z,Y,X]

    samp = F.grid_sample(
        vol_t, grid_t, mode="bilinear",
        padding_mode=padding_mode, align_corners=True
    )  # [1,1,1,H,W]

    plane_img = samp[0, 0, 0].detach().cpu().numpy()
    plane_img[~valid] = np.nan
    return plane_img, XX, YY, ZZ, valid


def extract_roi_plane(clean_matches, pts_clean, slab_Im, device, margin_xy=12, pad_xy=32, step=1.0):
    """Fit a plane through the clean MRI match centers and sample the MRI slab on it.

    An unweighted least-squares plane z = a*x + b*y + c is fitted to ``pts_clean`` and
    the slab is sampled on it (``resample_mri_on_plane_gpu``) over the (x, y) bounding
    box of the points, enlarged by ``margin_xy`` voxels (lower bounds clipped at 0) and
    then by ``pad_xy`` voxels (not clipped, so x0/y0 can be negative). The plane image
    sets the size of the common canvas, and its origin (x0, y0) sets the (u, v) frame of
    the global plane G. The QC figures for this step are qc.show_matched_pairs,
    qc.visualize_roi_from_matches and qc.plot_roi_support.

    Args:
        clean_matches: Outlier-filtered match dicts; at least 3 are required.
        pts_clean: float array [N, 3], MRI centers of ``clean_matches`` as (z, y, x)
            voxel indices (from outliers.filter_outliers).
        slab_Im: Prepared MRI slab (SimpleITK image or array [Z, Y, X]).
        device: Torch device for plane sampling.
        margin_xy: Margin in voxels around the match bounding box.
        pad_xy: Extra padding in voxels added after the margin.
        step: Plane grid spacing in voxels.

    Returns:
        Tuple (plane_img, x0, x1, y0, y1): the MRI plane image [H, W] (NaN where the
        plane leaves the slab in z) and the integer MRI voxel bounds of its grid (row 0
        is y0, column 0 is x0).

    Raises:
        ValueError: If fewer than 3 clean matches are given.
    """
    if len(clean_matches) < 3:
        raise ValueError("Need >=3 clean_matches to fit the ROI plane.")

    a, b, c = fit_plane_z_ax_by_c(pts_clean)

    x_min = max(0, int(np.floor(pts_clean[:, 2].min() - margin_xy)))
    x_max = int(np.ceil(pts_clean[:, 2].max() + margin_xy))
    y_min = max(0, int(np.floor(pts_clean[:, 1].min() - margin_xy)))
    y_max = int(np.ceil(pts_clean[:, 1].max() + margin_xy))

    x0, x1 = x_min - pad_xy, x_max + pad_xy
    y0, y1 = y_min - pad_xy, y_max + pad_xy

    plane_img, _, _, _, _ = resample_mri_on_plane_gpu(
        mri_vol=slab_Im,
        a=a, b=b, c=c,
        x_range=(x0, x1),
        y_range=(y0, y1),
        step=step,
        device=device,
    )
    logger.debug(f"ROI plane: z = {a:.6f}*x + {b:.6f}*y + {c:.6f}, x {x0}..{x1}, y {y0}..{y1}")
    return plane_img, int(x0), int(x1), int(y0), int(y1)


def pad2d_to_canvas(img2d, Hc, Wc, oy, ox, fill=0.0):
    """Place a 2D image on a larger canvas with its top-left corner at (oy, ox).

    Args:
        img2d: Image array [h, w].
        Hc: Canvas height in pixels.
        Wc: Canvas width in pixels.
        oy: Row offset of the image on the canvas.
        ox: Column offset of the image on the canvas.
        fill: Value for canvas pixels not covered by the image.

    Returns:
        Array [Hc, Wc] with the dtype of ``img2d``. The image must fit inside
        the canvas at the given offset.
    """
    out = np.full((Hc, Wc), fill, dtype=img2d.dtype)
    h, w = img2d.shape
    out[oy:oy+h, ox:ox+w] = img2d
    return out


def create_common_canvas(scaled_proc_Ih, scaled_proc_Mh, plane_img):
    """Center the scaled histology and the MRI plane image on a shared 2D canvas.

    The canvas is the element-wise maximum of the two image sizes, and each
    image is centered on it. Canvas pixel (r, c) therefore corresponds to
    histology pixel (r - oy_h, c - ox_h) and to plane-image pixel
    (r - oy_m, c - ox_m), i.e. MRI voxel x = x0 + (c - ox_m), y = y0 + (r - oy_m)
    for a plane grid with step 1.

    Args:
        scaled_proc_Ih: Scaled, prepared histology (SimpleITK image, 2D or
            single-slice 3D).
        scaled_proc_Mh: Matching histology mask (SimpleITK image).
        plane_img: MRI plane image [Hm, Wm] from ``extract_roi_plane``.

    Returns:
        Tuple (Hc, Wc, oy_h, ox_h, oy_m, ox_m, hist_common, mri_common, mask_common):
            Hc, Wc: Canvas height and width in pixels.
            oy_h, ox_h: Row/column offset of the histology on the canvas.
            oy_m, ox_m: Row/column offset of the MRI plane image on the canvas.
            hist_common: float32 [Hc, Wc] histology on the canvas (0 outside).
            mri_common: float32 [Hc, Wc] MRI plane image on the canvas (0
                outside; NaNs of ``plane_img`` are kept).
            mask_common: float32 [Hc, Wc] histology mask on the canvas.
    """
    arr_h = sitk.GetArrayFromImage(scaled_proc_Ih).astype(np.float32)
    hist2d = arr_h[0] if arr_h.ndim == 3 else arr_h

    arr_mh = sitk.GetArrayFromImage(scaled_proc_Mh).astype(np.float32)
    mask2d = arr_mh[0] if arr_mh.ndim == 3 else arr_mh

    mri2d = plane_img.astype(np.float32)
    Hh, Wh = hist2d.shape
    Hm, Wm = mri2d.shape
    Hc = max(Hh, Hm)
    Wc = max(Wh, Wm)
    # Center both images on the canvas.
    oy_h = (Hc - Hh) // 2
    ox_h = (Wc - Wh) // 2
    oy_m = (Hc - Hm) // 2
    ox_m = (Wc - Wm) // 2
    hist_common = pad2d_to_canvas(hist2d, Hc, Wc, oy_h, ox_h, fill=0.0)
    mask_common = pad2d_to_canvas(mask2d, Hc, Wc, oy_h, ox_h, fill=0.0)
    mri_common  = pad2d_to_canvas(mri2d, Hc, Wc, oy_m, ox_m, fill=0.0)

    logger.debug('hist original size: %s', (Hh, Wh))
    logger.debug('mri original size: %s', (Hm, Wm))
    logger.debug('common canvas: %s', (Hc, Wc))
    logger.debug('hist offset (oy,ox): %s', (oy_h, ox_h))
    logger.debug('mri offset  (oy,ox): %s', (oy_m, ox_m))

    return Hc, Wc, oy_h, ox_h, oy_m, ox_m, hist_common, mri_common, mask_common

