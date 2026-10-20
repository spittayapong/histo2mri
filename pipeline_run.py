"""Batch-run the full histo<->MRI registration pipeline (pipeline_v12.ipynb, ported
to a script) over multiple sections listed in a sections-csv (same format as annotations_run.py).

Modules: roi_extract_v8, ddf_estimation_v12 (curved-sheet refinement + local rigidity),
inverse_to_histo_v12 (MRI -> HR histology space).

For each section this produces, in <save-dir>/<subject_id>/<histo_type>/<num_str>/:
  - <subject>_<num_str>_transform_grid.npz: everything needed to later map a histology landmark
    (in scaled_proc_Ih pixel space) to MRI space for quantitative evaluation (A_common_to_hr,
    xyz_norm_final, xy_hist_map, valid_surf, ox_h, oy_h, G_e1, G_e2, mri_spacing, mri_size,
    subject_id, num_str).
  - <subject>_<num_str>_process.log
  - <subject>_<num_str>_histo_in_mri_space[_all].png: histology + MRI resampled onto the common
    curved-sheet canvas (MRI's native pixel scale), with checkerboard.
  - <subject>_<num_str>_mri_on_histo[_all|_rgb].png: MRI resampled into full HR histology-pixel
    space, with checkerboard (plus an RGB-histology overlay variant).

All other intermediate prechecks and the refine_curved_sheet GIF from the source notebook are
intentionally not ported here to keep runs fast; only the outputs above are produced.
"""
import argparse
import contextlib
import csv
import logging
import os
import shutil
import time
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

import section_io
from image_prep_v7 import resample_histo_to_mri, prepare_histo_regis, prepare_mri_slab
from roi_extract_v8 import (
    coarse_scale_search_hog_mind_cached,
    coarse_patch_search_hog_mind,
    filter_outliers,
    scale_histo_with_best_scale,
    run_patch_matching_hog_mind_fusion_gpu_with_border,
    check_roi,
    create_common_canvas,
)
from ddf_estimation_v12 import (
    find_clean_matches_common,
    fit_G_plane_canvas_from_matches,
    robust_fit_tps_ransac_mad,
    refine_curved_sheet,
    check_mri_on_final_sheet,
    fit_residual_tps3d_with_G_anchors,
    fit_residual_tps2d_with_G_anchors,
    resample_histology_on_G_residual_tps,
    ResidualToGSurface3D,
)
from inverse_to_histo_v12 import (
    build_A_hr_common,
    mri_on_histology_space,
    check_mri_on_histo_rgb,
)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Batch-run the full registration pipeline over sections")
    p.add_argument("--sections-csv", required=True,
                    help="Same format as annotations_run.py: internal: subject_id,slice_id,orientation,spacing. "
                         "NB: subject_id,block_id,slice_id,orientation,stain,spacing. "
                         "spacing is a quoted 'x,y,z' string. Optional 'flip' column: letters x/y/z.")
    p.add_argument("--ds-type", default="internal", choices=["internal", "NB"])
    p.add_argument("--histo-type", required=True,
                    help="Suffix in data/<subject_id>/histo_<histo_type>/, e.g. myelin/iron/he.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--save-dir", default="./evaluations/")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--pclip", nargs=2, type=float, default=(1, 99), metavar=("LOW", "HIGH"))
    p.add_argument("--n-slices-offset-divisor", type=int, default=2,
                    help="n_slices = Im depth // this (only affects QC display slice index)")
    p.add_argument("--scale-max", type=float, default=1.30, help="Upper bound (inclusive-ish) for best-scale search")
    p.add_argument("--scale-step", type=float, default=0.05)
    p.add_argument("--device", default=None, help="cuda:0 / cpu; auto-detect if omitted")
    p.add_argument("--mri-on-histo-tile-size", type=int, default=348,
                    help="Checkerboard tile size (px) for the mri_on_histo overlay. "
                         "Use e.g. 2048 for NB sections.")
    p.add_argument("--lam-area", type=float, default=2.0,
                    help="Local rigidity: weight on log(area/area0)^2 (properness). 0 disables rigidity.")
    p.add_argument("--lam-shape", type=float, default=0.0,
                    help="Local rigidity: weight on first-fundamental-form drift (orthonormality).")
    p.add_argument("--w-rigid-mode", default="mask", choices=["mask", "conf", "edge", "none"],
                    help="Spatial rigidity weight map used by the refinement.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N rows (dry-run)")
    p.add_argument("--log-csv", default=None, help="Summary log path; defaults to <save-dir>/run_log_<timestamp>.csv")
    p.add_argument("--verbose", action="store_true",
                    help="Show print()/tqdm output from the underlying modules. Off by default.")
    return p


@contextlib.contextmanager
def suppress_output(enabled=True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _reset_logging(log_path):
    logging.getLogger().handlers.clear()
    open(log_path, "w").close()
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def process_one_section(subject_id, num_str, histo_type, spacing, orientation, flip,
                         args, device, common_kwargs, log_dir):
    log_path = os.path.join(log_dir, f"{subject_id}_{num_str}_process.log")
    _reset_logging(log_path)
    logging.info("Process started")
    section_start = time.perf_counter()

    # ---- Load ----
    vols = section_io.load_section(subject_id, num_str, histo_type, spacing, orientation, flip, args.data_root)
    Ih_hr, Mh_hr, Im, Mm = vols["Ih_hr"], vols["Mh_hr"], vols["Im"], vols["Mm"]
    Ih_rgb_hr = vols.get("Ih_rgb_hr")
    logging.info(f"Loaded section {subject_id}/{num_str}")

    # ---- Resample histology to MRI spacing ----
    Ih, Mh, A_hr_to_lr = resample_histo_to_mri(Ih_hr=Ih_hr, Mh_hr=Mh_hr, Im=Im)

    # ---- Prepare histology ----
    scale_ratio = (1, 1)
    proc_Ih, proc_Mh, A_lr_to_proc = prepare_histo_regis(
        Ih=Ih, Mh=Mh, scale_ratio=scale_ratio, patch_size=args.patch_size,
        show_proc_img=False,
    )
    A_hr_to_proc = A_lr_to_proc @ A_hr_to_lr
    _ = np.linalg.inv(A_hr_to_proc)  # A_proc_to_hr: unused, kept for parity with the source notebook
    logging.info(f"Prepared histology - proc_Ih size:{proc_Ih.GetSize()}")

    # ---- Prepare MRI ----
    n_slices = Im.GetSize()[2] // args.n_slices_offset_divisor
    _, slab_Im, slab_Mm = prepare_mri_slab(
        Im=Im, Mm=Mm, n_slices=n_slices, show_proc_img=False, pclip=tuple(args.pclip),
    )
    logging.info(f"Prepared MRI slab - slab_Im size:{slab_Im.GetSize()}")

    # ---- Find ROI: best scale ----
    scales = np.arange(1.00, args.scale_max + 1e-6, args.scale_step)
    best_scale_info, _ = coarse_scale_search_hog_mind_cached(
        proc_Ih=proc_Ih, histo_mask=proc_Mh, slab_Im=slab_Im, slab_Mm=slab_Mm,
        scales=scales, common_kwargs=common_kwargs,
    )
    best_scale = best_scale_info["scale"]
    logging.info(f"Best scale: {best_scale} score:{best_scale_info['score']}")

    # ---- Find ROI: best patch ----
    patch_matrix = [
        {"patch": 96, "hist_stride": 64, "mri_stride": 8},
        {"patch": 64, "hist_stride": 32, "mri_stride": 4},
    ]
    best_patch, best_hist_stride, best_mri_stride, z_center, z_min, z_max = coarse_patch_search_hog_mind(
        proc_Ih=proc_Ih, proc_Mh=proc_Mh, slab_Im=slab_Im, slab_Mm=slab_Mm,
        best_scale=best_scale, patch_matrix=patch_matrix,
        z_min=None, z_max=None, common_kwargs=common_kwargs,
    )
    logging.info(f"Best patch:{best_patch} hist_stride:{best_hist_stride} mri_stride:{best_mri_stride} "
                 f"z_center:{z_center}")

    # ---- Fine matching ----
    scaled_proc_Ih, scaled_proc_Mh = scale_histo_with_best_scale(proc_Ih, proc_Mh, best_scale)

    size = slab_Im.GetSize()
    Z = int(size[2]) if slab_Im.GetDimension() == 3 else 1
    z_min = max(0, int(z_center - 60))
    z_max = min(Z - 1, int(z_center + 60))

    all_matches = run_patch_matching_hog_mind_fusion_gpu_with_border(
        proc_Ih=scaled_proc_Ih, histo_mask=scaled_proc_Mh,
        slab_Im=slab_Im, slab_Mm=slab_Mm,
        z_min=z_min, z_max=z_max,
        hist_patch=best_patch, hist_stride=best_hist_stride,
        mri_patch=best_patch, mri_stride=best_mri_stride,
        **common_kwargs,
    )
    clean_matches, coarse_matches, pts_clean, sc_clean = filter_outliers(
        all_matches=all_matches["matches"], z_min=z_min, z_max=z_max,
    )
    logging.info(f"Fine matching - clean_matches:{len(clean_matches)}")

    # check_roi is functionally required (not just QC): plane_img/x0/x1/y0/y1 feed the G fit below.
    plane_img, x0, x1, y0, y1 = check_roi(
        slab_Im, scaled_proc_Ih, pts_clean, sc_clean, all_matches, best_patch,
        coarse_matches, clean_matches, z_min, z_max, device=device,
    )

    # ---- Common canvas ----
    Hc, Wc, oy_h, ox_h, oy_m, ox_m, hist_common, mri_common, mask_common = create_common_canvas(
        scaled_proc_Ih, scaled_proc_Mh, plane_img
    )
    logging.info(f"Common canvas - hist_common:{hist_common.shape}")

    # ---- Global in-plane transform G ----
    step = 1.0
    x_range_final = (x0 - ox_m * step, x0 + (Wc - 1 - ox_m) * step)
    y_range_final = (y0 - oy_m * step, y0 + (Hc - 1 - oy_m) * step)

    clean_matches_common = find_clean_matches_common(clean_matches=clean_matches, oy_h=oy_h, ox_h=ox_h)
    yy = np.array([m["mri_center_zyx"][1] for m in clean_matches_common], np.float32)
    xx = np.array([m["mri_center_zyx"][2] for m in clean_matches_common], np.float32)
    Hm_fit = int(np.ceil(yy.max())) + 1
    Wm_fit = int(np.ceil(xx.max())) + 1

    G = fit_G_plane_canvas_from_matches(
        clean_matches_common, x_range_final, y_range_final,
        mri_hw=(Hm_fit, Wm_fit), hist_hw=(Hc, Wc), robust=True,
    )
    logging.info("Fitted initial in-plane G")

    proc_Ih_common_arr = hist_common.astype(np.float32)[None, ...]
    proc_Ih_common_sitk = sitk.GetImageFromArray(proc_Ih_common_arr)
    proc_Mh_common_arr = mask_common.astype(np.float32)[None, ...]
    proc_Mh_common_sitk = sitk.GetImageFromArray(proc_Mh_common_arr)

    # ---- Init DDF: RANSAC then MAD ----
    # Adaptive inlier_thresh: robust MAD fit on all matches, then med + 3*MAD (clipped).
    _tps_diag, _, _, _, _resid_diag, _ = robust_fit_tps_ransac_mad(
        clean_matches_common, G, method="mad",
        mad_thresh=2.5, mad_iterations=3, lam_hyp=1e-2, lam_final=1e-2,
        device=device, normalize_uv=False,
    )
    _r = _resid_diag.cpu().numpy()
    _med = float(np.median(_r))
    _scale = 1.4826 * float(np.median(np.abs(_r - _med))) + 1e-8
    adaptive_thresh = float(np.clip(_med + 3.0 * _scale, 3.0, 20.0))

    # NB sections use a smaller minimal set; all others use 5.
    ransac_sample_size = 3 if args.ds_type == "NB" else 5

    # G was fitted from clean_matches_common, and robust_fit_tps_ransac_mad recovers uv by
    # inverting G on the histology coords it is given. They must therefore be in the same
    # common-canvas frame; passing clean_matches (local) offsets every uv by (ox_h, oy_h).
    tps_ransac, uv_in, xyz_in, inlier_mask, _, score = robust_fit_tps_ransac_mad(
        clean_matches_common, G, method="ransac",
        n_iters=300, sample_size=ransac_sample_size, inlier_thresh=adaptive_thresh, min_inliers=30,
        device=device, normalize_uv=False, lam_hyp=1e-2, lam_final=1e-2,
    )
    logging.info(f"Init DDF - adaptive inlier_thresh={adaptive_thresh:.2f}, sample_size={ransac_sample_size}")
    matches_inliers = [m for m, keep in zip(clean_matches_common, inlier_mask.cpu().tolist()) if keep]

    tps_init_final, uv_in2, xyz_in2, inlier_mask2, _, score2 = robust_fit_tps_ransac_mad(
        matches_inliers, G, method="mad",
        mad_thresh=2.5, mad_iterations=2, lam_hyp=1e-2, lam_final=1e-2,
        device=device, normalize_uv=False,
    )
    matches_inliers_final = [m for m, keep in zip(matches_inliers, inlier_mask2.cpu().tolist()) if keep]

    hist_xy_in2 = np.asarray(
        [[float(m["hist_center_yx"][1]), float(m["hist_center_yx"][0])] for m in matches_inliers_final],
        dtype=np.float32,
    )

    # In-plane residual against the rigid affine G is a clean bad-match detector: the flexible
    # TPS absorbs these (they pass RANSAC/MAD), G does not. Drop matches far in-plane from G.
    _X0 = np.asarray(G["X0"], np.float32); _e1 = np.asarray(G["e1"], np.float32)
    _e2 = np.asarray(G["e2"], np.float32); _n = np.asarray(G["n"], np.float32)
    _uvn = uv_in2.detach().cpu().numpy() if torch.is_tensor(uv_in2) else np.asarray(uv_in2)
    _xyzn = xyz_in2.detach().cpu().numpy() if torch.is_tensor(xyz_in2) else np.asarray(xyz_in2)
    _res = _xyzn - (_X0[None] + _uvn[:, 0:1]*_e1[None] + _uvn[:, 1:2]*_e2[None])
    _inp = np.linalg.norm(_res - (_res @ _n)[:, None]*_n[None, :], axis=1)

    INPLANE_MAX = 30.0   # vox
    _keep = _inp <= INPLANE_MAX
    _km = torch.as_tensor(_keep, device=uv_in2.device) if torch.is_tensor(uv_in2) else _keep
    uv_in2 = uv_in2[_km]
    xyz_in2 = xyz_in2[_km]
    hist_xy_in2 = np.asarray(hist_xy_in2)[_keep]
    matches_inliers_final = [m for m, k in zip(matches_inliers_final, _keep.tolist()) if k]
    logging.info(f"Init DDF - in-plane filter: kept {int(_keep.sum())}/{len(_keep)} (max in-plane {_inp.max():.1f} vox)")

    tps3d_res, uv_all_3d, dxyz_all, uv_anchor_3d = fit_residual_tps3d_with_G_anchors(
        uv_ctrl=uv_in2, xyz_ctrl=xyz_in2, G=G, device=device, lam=1e-1,
        normalize_uv=False, anchor_step=24.0, far_thr=35.0, max_anchors=1200, seed=42,
    )
    tps2d_res, uv_all_2d, dxy_all, uv_anchor_2d = fit_residual_tps2d_with_G_anchors(
        uv_ctrl=uv_in2, hist_xy_ctrl=hist_xy_in2, G=G, device=device, lam=2e-1,
        normalize_uv=False, anchor_step=24.0, far_thr=35.0, max_anchors=1200, seed=42,
    )

    # Per-section clamp: cover the kept matches' curvature (p99, robust to a single extreme), with margin.
    _uvn2 = uv_in2.detach().cpu().numpy() if torch.is_tensor(uv_in2) else np.asarray(uv_in2)
    _xyzn2 = xyz_in2.detach().cpu().numpy() if torch.is_tensor(xyz_in2) else np.asarray(xyz_in2)
    _mag = np.linalg.norm(_xyzn2 - (_X0[None] + _uvn2[:, 0:1]*_e1[None] + _uvn2[:, 1:2]*_e2[None]), axis=1)
    clamp_vox = float(np.clip(1.25 * np.percentile(_mag, 99), 8.0, 60.0))
    logging.info(f"Init DDF - per-section clamp_vox={clamp_vox:.2f} (kept |dxyz| max={_mag.max():.1f} vox)")

    hist_on_init_tps, valid_hist, uv_map_sym, xy_map_sym, conf_hist, dmin_hist = resample_histology_on_G_residual_tps(
        hist_img=proc_Ih_common_sitk, G=G, tps2d_residual=tps2d_res, ctrl_uv=uv_in2,
        canvas_shape=(Hc, Wc), device=device, is_mask=False, sigma_uv=20.0, min_conf=0.25,
        hard_mult=2.0, max_residual_px=20.0, return_extra=True,
    )
    hist_mask_on_init_tps = resample_histology_on_G_residual_tps(
        hist_img=proc_Mh_common_sitk, G=G, tps2d_residual=tps2d_res, ctrl_uv=uv_in2,
        canvas_shape=(Hc, Wc), device=device, is_mask=True, sigma_uv=20.0, min_conf=0.25,
        hard_mult=2.0, max_residual_px=20.0, return_extra=False,
    )
    hist_mask_on_init_tps = (hist_mask_on_init_tps > 0.5)
    logging.info("Initial TPS on top of G fitted")

    # ---- Refinement ----
    init_tps_abs = ResidualToGSurface3D(G=G, residual_tps=tps3d_res, device=device).to(device)
    (xyz_norm_final, mri_on_final_sheet, valid_mri,
     _xyz_norm_base, _conf, _phi, _n0, _alpha, _rf_psi, _rf_beta) = refine_curved_sheet(
        slab_Im=slab_Im, slab_Mm=slab_Mm, init_G=G, init_tps=init_tps_abs,
        matches_for_warmstart=matches_inliers_final, device=device,
        hist_on_init_tps=hist_on_init_tps, hist_mask_on_init_tps=None,
        ctrl_uv_support=uv_in2, tps_sigma_uv=20.0, tps_min_conf=0.25, tps_max_disp_vox=clamp_vox,
        lam_area=args.lam_area, lam_shape=args.lam_shape, w_rigid_mode=args.w_rigid_mode,
        gif_file=None,
    )
    logging.info("Refined curved sheet")

    valid_display = valid_mri & valid_hist & hist_mask_on_init_tps
    check_mri_on_final_sheet(
        hist_on_init_tps=hist_on_init_tps, valid_display=valid_display,
        mri_on_final_sheet=mri_on_final_sheet,
        save_file=os.path.join(log_dir, f"{subject_id}_{num_str}_histo_in_mri_space_all.png"),
    )
    # check_mri_on_final_sheet's save_file_single only saves the MRI-on-sheet panel, not
    # histology, so save the "Target Histology on Sheet" panel ourselves under this name.
    _save_target_histo_on_sheet(
        subject_id=subject_id, num_str=num_str, log_dir=log_dir,
        hist_on_init_tps=hist_on_init_tps, valid_display=valid_display,
    )

    # ---- MRI on histology space (affine chain) ----
    A_hr_to_common, A_common_to_hr = build_A_hr_common(
        proc_Ih, scaled_proc_Ih, A_hr_to_proc, oy_h, ox_h
    )
    logging.info("Composed A_common_to_hr")

    _save_mri_on_histo_space(
        subject_id=subject_id, num_str=num_str, histo_type=histo_type, flip=flip,
        args=args, log_dir=log_dir, Ih_hr=Ih_hr, Mh_hr=Mh_hr,
        proc_Ih=proc_Ih, scaled_proc_Ih=scaled_proc_Ih, A_hr_to_proc=A_hr_to_proc,
        oy_h=oy_h, ox_h=ox_h,
        xyz_norm_final=xyz_norm_final, xy_map_sym=xy_map_sym,
        valid_display=valid_display, mri_on_final_sheet=mri_on_final_sheet,
        slab_Im=slab_Im, device=device, Ih_rgb_hr=Ih_rgb_hr,
    )

    # ---- Save transform for later evaluation ----
    save_path = os.path.join(log_dir, f"{subject_id}_{num_str}_transform_grid.npz")
    np.savez_compressed(
        save_path,
        A_common_to_hr=_to_np(A_common_to_hr).astype(np.float64),
        xyz_norm_final=_to_np(xyz_norm_final).astype(np.float32),
        xy_hist_map=_to_np(xy_map_sym).astype(np.float32),
        valid_surf=_to_np(valid_display).astype(np.uint8),
        ox_h=np.float64(ox_h),
        oy_h=np.float64(oy_h),
        G_e1=_to_np(G["e1"]).astype(np.float64),
        G_e2=_to_np(G["e2"]).astype(np.float64),
        mri_spacing=np.asarray(slab_Im.GetSpacing(), dtype=np.float64),
        mri_size=np.asarray(slab_Im.GetSize(), dtype=np.int64),
        subject_id=np.array(subject_id),
        num_str=np.array(num_str),
        version=np.array([2], dtype=np.int32),
    )
    logging.info(f"Saved transform grid -> {save_path}")

    del all_matches, clean_matches, coarse_matches, matches_inliers, matches_inliers_final
    del tps_ransac, tps_init_final, tps3d_res, tps2d_res, init_tps_abs
    torch.cuda.empty_cache()

    elapsed_sec = time.perf_counter() - section_start
    logging.info(f"Process finished in {elapsed_sec:.2f}s")

    return {
        "output_npz": save_path,
        "n_clean_matches": len(clean_matches_common),
        "best_scale": best_scale,
        "elapsed_sec": elapsed_sec,
    }


def _save_target_histo_on_sheet(subject_id, num_str, log_dir, hist_on_init_tps, valid_display):
    """Single-panel save of the "Target Histology on Sheet" content (same masking as the
    corresponding panel in check_mri_on_final_sheet's composite figure)."""
    h_disp = _to_np(hist_on_init_tps).astype(np.float32)
    if h_disp.ndim == 3 and h_disp.shape[0] == 1:
        h_disp = h_disp[0]
    h_disp = h_disp.copy()
    h_disp[~np.asarray(valid_display, dtype=bool)] = 0.0

    save_path = os.path.join(log_dir, f"{subject_id}_{num_str}_histo_in_mri_space.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(h_disp, cmap="gray", origin="lower")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_mri_on_histo_space(subject_id, num_str, histo_type, flip, args, log_dir, Ih_hr, Mh_hr,
                              proc_Ih, scaled_proc_Ih, A_hr_to_proc, oy_h, ox_h,
                              xyz_norm_final, xy_map_sym, valid_display, mri_on_final_sheet,
                              slab_Im, device, Ih_rgb_hr=None):
    """Dense HR-resolution overlay of the transform on the histology image: MRI resampled into
    full HR histology-pixel space, with checkerboard (grayscale + RGB-histology variants).

    The pull-back itself now lives in inverse_to_histo_v12.mri_on_histology_space(); this wrapper
    only handles the two things that are specific to batch running: skipping a section gracefully
    when it has no usable support, and the optional RGB variant."""
    tile_size = args.mri_on_histo_tile_size
    save_file = os.path.join(log_dir, f"{subject_id}_{num_str}_mri_on_histo_all.png")
    save_file_single = os.path.join(log_dir, f"{subject_id}_{num_str}_mri_on_histo.png")

    try:
        mri_on_hr_updated, final_valid_hr, _, _, _ = mri_on_histology_space(
            Ih_hr=Ih_hr, Mh_hr=Mh_hr,
            proc_Ih=proc_Ih, scaled_proc_Ih=scaled_proc_Ih, A_hr_to_proc=A_hr_to_proc,
            oy_h=oy_h, ox_h=ox_h,
            slab_Im=slab_Im,
            xyz_norm_final=xyz_norm_final, xy_map_sym=xy_map_sym,
            valid_display=valid_display, mri_on_final_sheet=mri_on_final_sheet,
            device=device, tile_hw=512,
            show_proc_img=True, tile_size=tile_size,
            save_file=save_file, save_file_single=save_file_single,
            verbose=args.verbose,
        )
    except ValueError as e:
        # empty histology mask, or empty valid_display: nothing to project
        logging.info(f"Dense HR QC skipped: {e}")
        return

    logging.info(f"Dense HR QC coverage: {final_valid_hr.mean():.6f}")

    if Ih_rgb_hr is None:
        logging.info("Dense HR QC: no RGB histology found, skipping check_mri_on_histo_rgb")
        return

    arr_Ih_hr = sitk.GetArrayFromImage(Ih_hr)
    arr_Ih_hr_2d = arr_Ih_hr[0] if arr_Ih_hr.ndim == 3 and arr_Ih_hr.shape[0] == 1 else arr_Ih_hr
    arr_Ih_rgb_hr = sitk.GetArrayFromImage(Ih_rgb_hr)[0]

    save_file_rgb = os.path.join(log_dir, f"{subject_id}_{num_str}_mri_on_histo_rgb.png")
    check_mri_on_histo_rgb(arr_Ih_hr_2d, arr_Ih_rgb_hr, mri_on_hr_updated,
                            tile=tile_size, save_file=save_file_rgb)


def main():
    args = build_arg_parser().parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    common_kwargs = dict(
        device=device,
        cell_size=8, bins=18,
        min_histo_mask_frac=0.2,
        min_mri_mask_frac=0.50,
        min_overlap_frac=0.35,
        topk_hog=64, w_hog=0.6, w_mind=0.4,
        mind_radius=2, mind_dilation=1,
        hist_batch=128,
        mri_chunk=16384,
        keep_mri_bank_on_gpu=True,
    )

    with open(args.sections_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    os.makedirs(args.save_dir, exist_ok=True)
    log_csv = args.log_csv or os.path.join(args.save_dir, f"run_log_{datetime.now():%Y%m%d_%H%M%S}.csv")

    total = len(rows)
    results = []
    for i, row in enumerate(rows, start=1):
        subject_id = row["subject_id"]
        histo_type = args.histo_type
        entry = {"subject_id": subject_id, "histo_type": histo_type}
        try:
            num_str = section_io.build_num_str(row, args.ds_type)
        except KeyError as e:
            entry.update(status="error", error=f"Missing column {e} required for --ds-type {args.ds_type}")
            print(f"[{i}/{total}] {subject_id} ... FAILED ({entry['error']})", flush=True)
            results.append(entry)
            continue

        entry["num_str"] = num_str
        print(f"[{i}/{total}] {subject_id} {histo_type} {num_str} ... RUNNING", flush=True)
        row_start = time.perf_counter()
        try:
            spacing = section_io.parse_spacing(row["spacing"])
            orientation = section_io.parse_orientation(row["orientation"])
            flip = section_io.parse_flip(row.get("flip", ""))

            log_dir = os.path.join(args.save_dir, subject_id, histo_type, num_str)
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)

            with suppress_output(enabled=not args.verbose):
                info = process_one_section(
                    subject_id, num_str, histo_type, spacing, orientation, flip,
                    args, device, common_kwargs, log_dir,
                )
            entry.update(status="ok", **info)
            print(f"[{i}/{total}] {subject_id} {histo_type} {num_str} ... DONE "
                  f"(n_matches={info['n_clean_matches']}, best_scale={info['best_scale']:.3f}, "
                  f"elapsed={info['elapsed_sec']:.1f}s)", flush=True)
        except Exception as e:
            entry.update(status="error", error=f"{type(e).__name__}: {e}",
                          elapsed_sec=time.perf_counter() - row_start)
            print(f"[{i}/{total}] {subject_id} {histo_type} {num_str} ... FAILED "
                  f"({type(e).__name__}: {e}, elapsed={entry['elapsed_sec']:.1f}s)", flush=True)
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
        results.append(entry)

    pd.DataFrame(results).to_csv(log_csv, index=False)
    print(f"\nDone. Summary log -> {log_csv}")
    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"{n_ok}/{len(results)} sections succeeded")


if __name__ == "__main__":
    main()
