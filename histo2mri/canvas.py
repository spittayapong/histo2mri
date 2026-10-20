"""MRI plane at the matched ROI, and the common canvas shared by histology and MRI."""
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from .qc import _to_zyx_volume, fit_plane_z_ax_by_c, show_matched_pairs, visualize_roi_from_matches


@torch.no_grad()
def resample_mri_on_plane_gpu(
    mri_vol, a, b, c,               # plane: z = a*x + b*y + c
    x_range=None, y_range=None,     # (min,max), in voxel index
    step=1.0,
    device="cuda:0",
    padding_mode="zeros"
):
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


def visualize_matches_on_image(
    clean_matches,
    scaled_proc_Ih,
    pts_clean,        # [N,3] as (z,y,x)
    slab_Im,
    device,
    margin_xy=12,
    pad_xy=32,
    step=1.0,
    save_file=None,
):
    # ---------- Left panel: histology support ----------
    h = sitk.GetArrayFromImage(scaled_proc_Ih) if hasattr(scaled_proc_Ih, "GetSize") else np.asarray(scaled_proc_Ih)
    if h.ndim == 3:
        h = h[0]

    pts_hist = np.array([[m["hist_center_yx"][1], m["hist_center_yx"][0]] for m in clean_matches], dtype=np.float32)  # (x,y)
    if len(pts_hist) < 3:
        raise ValueError("Need >=3 clean_matches for histology hull.")

    hull_h = ConvexHull(pts_hist)
    poly_h = pts_hist[hull_h.vertices]
    path_h = Path(poly_h)

    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    inside_h = path_h.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(H, W)
    hist_cov = float(inside_h.mean())

    # ---------- Build plane image ----------
    a, b, c = fit_plane_z_ax_by_c(pts_clean)

    x_min = max(0, int(np.floor(pts_clean[:, 2].min() - margin_xy)))
    x_max = int(np.ceil(pts_clean[:, 2].max() + margin_xy))
    y_min = max(0, int(np.floor(pts_clean[:, 1].min() - margin_xy)))
    y_max = int(np.ceil(pts_clean[:, 1].max() + margin_xy))

    x0, x1 = x_min - pad_xy, x_max + pad_xy
    y0, y1 = y_min - pad_xy, y_max + pad_xy

    plane_img, XX, YY, ZZ, valid = resample_mri_on_plane_gpu(
        mri_vol=slab_Im,
        a=a, b=b, c=c,
        x_range=(x0, x1),
        y_range=(y0, y1),
        step=step,
        device=device,
    )

    # ---------- Right panel: plane support ----------
    pts_xy = np.array(
        [[m["mri_center_zyx"][2], m["mri_center_zyx"][1]] for m in clean_matches if m["mri_center_zyx"][0] >= 0],
        dtype=np.float32
    )
    if len(pts_xy) < 3:
        raise ValueError("Need >=3 valid MRI points for plane hull.")

    Hp, Wp = plane_img.shape
    px0, py0 = float(XX[0, 0]), float(YY[0, 0])
    sx = float(XX[0, 1] - XX[0, 0]) if Wp > 1 else 1.0
    sy = float(YY[1, 0] - YY[0, 0]) if Hp > 1 else 1.0

    px = (pts_xy[:, 0] - px0) / sx
    py = (pts_xy[:, 1] - py0) / sy
    pts_px = np.stack([px, py], axis=1)

    in_img = (px >= 0) & (px < Wp) & (py >= 0) & (py < Hp)
    pts_px_in = pts_px[in_img]
    if len(pts_px_in) < 3:
        raise ValueError("Too few points inside plane_img.")

    hull_p = ConvexHull(pts_px_in)
    poly_p = pts_px_in[hull_p.vertices]
    path_p = Path(poly_p)

    yyg, xxg = np.mgrid[0:Hp, 0:Wp]
    inside_p = path_p.contains_points(np.c_[xxg.ravel(), yyg.ravel()]).reshape(Hp, Wp)

    plane_cov_all = float(inside_p.mean())
    plane_cov_valid = float((inside_p & np.isfinite(plane_img)).sum() / max(np.isfinite(plane_img).sum(), 1))

    # ---------- One figure, 2 columns ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Histology panel
    axes[0].imshow(h, cmap="gray", origin="lower")
    axes[0].scatter(pts_hist[:, 0], pts_hist[:, 1], s=8, c="lime")
    axes[0].plot(np.r_[poly_h[:, 0], poly_h[0, 0]], np.r_[poly_h[:, 1], poly_h[0, 1]], c="cyan", lw=1.2)
    axes[0].set_title(f"Histology support (hull cov={hist_cov:.3f})")
    axes[0].axis("off")

    # Plane panel
    axes[1].imshow(np.nan_to_num(plane_img, nan=0.0), cmap="gray", origin="lower")
    axes[1].scatter(pts_px_in[:, 0], pts_px_in[:, 1], s=8, c="lime")
    axes[1].plot(np.r_[poly_p[:, 0], poly_p[0, 0]], np.r_[poly_p[:, 1], poly_p[0, 1]], c="cyan", lw=1.2)
    axes[1].set_title(f"Plane support (all={plane_cov_all:.3f}, valid={plane_cov_valid:.3f})")
    axes[1].axis("off")

    plt.tight_layout()

    if save_file is not None:
        import os
        save_dir = os.path.dirname(save_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_file, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_file}")

    plt.show()

    print(f"hull coverage (hist): {hist_cov:.6f}")
    print(f"hull coverage (plane all pixels): {plane_cov_all:.6f}")
    print(f"hull coverage (plane valid pixels): {plane_cov_valid:.6f}")
    print(f"plane: z = {a:.6f}*x + {b:.6f}*y + {c:.6f}")

    return {
        "plane_img": plane_img,
        "x0": int(x0),
        "x1": int(x1),
        "y0": int(y0),
        "y1": int(y1)
    }


def extract_roi_plane(
    slab_Im,
    scaled_proc_Ih,
    pts_clean,
    sc_clean,
    all_matches,
    best_patch,
    coarse_matches,
    clean_matches,
    z_min,
    z_max,
    save_file=None,
    save_file2=None,
    save_file3=None,
    device="cuda:0",
):
    show_matched_pairs(
        clean_matches,
        all_matches["hist_img"],
        all_matches["mri_img"],
        patch=best_patch,
        ncols=2,
        save_file=save_file,
        hist_mask=all_matches["hist_mask"],
        mri_mask=all_matches["mri_mask"],
    )


    viz = visualize_roi_from_matches(
        matches=clean_matches,   # or inlier_matches
        mri_vol=slab_Im,
        z_min=z_min,
        z_max=z_max,
        mode="mip",
        save_file=save_file2,
    )

    out = visualize_matches_on_image(
        clean_matches=clean_matches,
        scaled_proc_Ih=scaled_proc_Ih,
        pts_clean=pts_clean,
        slab_Im=slab_Im,
        device=device,
        save_file=save_file3,
    )
    plane_img = out["plane_img"]
    x0, x1, y0, y1 = out["x0"], out["x1"], out["y0"], out["y1"]

    return plane_img, x0, x1, y0, y1


def pad2d_to_canvas(img2d, Hc, Wc, oy, ox, fill=0.0):
    out = np.full((Hc, Wc), fill, dtype=img2d.dtype)
    h, w = img2d.shape
    out[oy:oy+h, ox:ox+w] = img2d
    return out


def create_common_canvas(scaled_proc_Ih, scaled_proc_Mh, plane_img):
    arr_h = sitk.GetArrayFromImage(scaled_proc_Ih).astype(np.float32)
    hist2d = arr_h[0] if arr_h.ndim == 3 else arr_h

    arr_mh = sitk.GetArrayFromImage(scaled_proc_Mh).astype(np.float32)
    mask2d = arr_mh[0] if arr_mh.ndim == 3 else arr_mh

    mri2d = plane_img.astype(np.float32)
    Hh, Wh = hist2d.shape
    Hm, Wm = mri2d.shape
    Hc = max(Hh, Hm)
    Wc = max(Wh, Wm)
    # choose placement (centered here)
    oy_h = (Hc - Hh) // 2
    ox_h = (Wc - Wh) // 2
    oy_m = (Hc - Hm) // 2
    ox_m = (Wc - Wm) // 2
    hist_common = pad2d_to_canvas(hist2d, Hc, Wc, oy_h, ox_h, fill=0.0)
    mask_common = pad2d_to_canvas(mask2d, Hc, Wc, oy_h, ox_h, fill=0.0)
    mri_common  = pad2d_to_canvas(mri2d, Hc, Wc, oy_m, ox_m, fill=0.0)

    print("hist original size:", (Hh, Wh))
    print("mri original size:", (Hm, Wm))
    print("common canvas:", (Hc, Wc))
    print("hist offset (oy,ox):", (oy_h, ox_h))
    print("mri offset  (oy,ox):", (oy_m, ox_m))

    return Hc, Wc, oy_h, ox_h, oy_m, ox_m, hist_common, mri_common, mask_common

