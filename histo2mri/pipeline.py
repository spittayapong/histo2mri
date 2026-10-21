"""Register one 2D histology section to a 3D MRI volume, running all pipeline steps in order.

register_section is the per-section driver called by the command-line tool (histo2mri.cli).
It ends with a curved sheet in MRI voxel space (xyz_norm, normalised (x, y, z) per axis) sampled
on the common canvas, plus the index affine from the common canvas back to the full-resolution
(HR) histology, and saves both to <subject>_<section>_transform_grid.npz. PROFILES holds named
parameter sets for stains/regions that need different settings.
"""
import logging
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from .canvas import create_common_canvas, extract_roi_plane
from .inverse import build_hr_common_affines, mri_on_histology_space
from .io import load_section
from .matching import match_patches, rescale_histology, search_patch_config, search_scale
from .outliers import filter_outliers
from .plane import fit_global_plane, matches_to_common_canvas
from .preprocessing import prepare_histology, prepare_mri_slab, resample_histo_to_mri
from .qc import plot_mri_on_histology_rgb, plot_mri_on_sheet
from .refine import refine_curved_sheet
from .tps import ResidualToGSurface3D, fit_residual_tps2d, fit_residual_tps3d, fit_tps_robust, resample_histology_on_surface


# Named settings for stains/regions that need different parameters. "default" is used for
# everything unless a section asks for another profile (--profile, or a "profile" column in
# the sections CSV). Keys:
#   outlier_plane_dist: distance threshold (MRI voxels) of the coarse RANSAC plane pre-filter
#     in filter_outliers.
#   outlier_knn: parameters of the stricter kNN outlier filter that follows it.
#   refine_similarity_weights: (w_ssim, w_ngf, w_mind) similarity weights used by
#     refine_curved_sheet.
# Profiles other than "default":
#   he-cblock: H&E sections of the cerebellum blocks. The internal granular layer is the
#     darkest structure there and the molecular layer the brightest, so their contrast does
#     not follow MRI intensity order: the refinement uses MIND (structure) instead of SSIM
#     (intensity), and the stricter kNN outlier filter suits the sparse matches.
PROFILES = {
    "default": dict(
        outlier_plane_dist=20.0,
        outlier_knn=dict(k=8, plane_mad_k=3.0, tukey_k=3.0, cc_radius_scale=3.0),
        refine_similarity_weights=(1.0, 0.0, 0.0),   # (w_ssim, w_ngf, w_mind)
    ),
    "he-cblock": dict(
        outlier_plane_dist=20.0,
        outlier_knn=dict(k=3, plane_mad_k=2.0, tukey_k=1.5, cc_radius_scale=2.0),
        refine_similarity_weights=(0.0, 0.0, 1.0),
    ),
}


def _to_np(x):
    """Return x as a numpy array (torch tensors are detached and moved to the CPU)."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _reset_logging(log_path):
    """Truncate log_path and route the root logger (INFO level) to it, dropping previous handlers."""
    logging.getLogger().handlers.clear()
    open(log_path, "w").close()
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def register_section(subject_id, num_str, histo_type, spacing, orientation, flip,
                         args, device, common_kwargs, log_dir, profile="default"):
    """Register one histology section to its MRI volume and save the transform.

    Steps: load the section (io.load_section); resample the histology to MRI spacing and pad
    it to a multiple of the patch size; normalise the MRI intensities; search the histology
    scale and the patch/stride configuration, then match histology patches to MRI patches
    within +/-60 slices of the estimated z centre; remove outlier matches; sample the MRI on
    a plane through the matches (ROI plane) and place it and the histology on a common
    canvas; fit the global plane G; fit a TPS surface robustly (RANSAC with an adaptive
    threshold from a MAD fit, then a MAD refit), drop matches far in-plane from G, and fit
    3D and 2D residual TPS on top of G; resample the histology onto the canvas through G and
    the 2D residual; refine the curved sheet in the MRI so that the MRI sampled on it
    matches that histology; pull the MRI back to the HR histology for QC; and save the
    transform. Errors are not caught here; they propagate to the caller.

    Args:
        subject_id: Subject folder name under args.data_root; also the file-name prefix.
        num_str: Section identifier used in file names (see io.build_num_str).
        histo_type: Histology type; selects the folder data_root/<subject_id>/histo_<histo_type>.
        spacing: (x, y, z) voxel spacing applied to images loaded from .npz files (see
            io.load_section); .mnc files use their own header spacing.
        orientation: "axial", "coronal" or "sagittal"; selects the axis permutation applied
            to the MRI (io.ORIENTATION_PERMUTE).
        flip: (flip_x, flip_y, flip_z) booleans; axes along which the histology is flipped
            on load.
        args: argparse.Namespace from cli.build_arg_parser. Fields read here: data_root,
            patch_size (pad the histology to a multiple of it), n_slices_offset_divisor
            (MRI slice index for the preparation display; the display is off, so it has no
            effect on the result), pclip (low/high percentiles for MRI intensity
            normalisation), scale_max and scale_step (scale grid 1.00..scale_max for the
            scale search), ds_type ("NB" uses a RANSAC sample size of 3, otherwise 5),
            lam_area, lam_shape and w_rigid_mode (local-rigidity settings of the sheet
            refinement), mri_on_histo_tile_size (checkerboard tile of the MRI-on-histology
            figures) and verbose (progress output of the MRI-on-histology step).
        device: torch.device used for matching, TPS fitting and refinement.
        common_kwargs: Patch-matcher keyword arguments (descriptor, mask-fraction,
            fusion-weight and batching settings, plus device) passed unchanged to
            search_scale, search_patch_config and match_patches.
        log_dir: Existing output directory for this section; all files below are written
            here.
        profile: Key of PROFILES selecting the outlier-filter and refinement settings.

    Files written to log_dir (prefix <subject_id>_<num_str>):
        _process.log: Log of this section (overwritten on each call).
        _histo_in_mri_space_all.png: Histology resampled onto the canvas, valid region,
            MRI on the final sheet and their checkerboard.
        _histo_in_mri_space.png: The histology-on-sheet panel alone.
        _mri_on_histo_all.png, _mri_on_histo.png: MRI resampled into HR histology pixels
            (composite with checkerboards, and the MRI alone). Skipped when there is no
            usable support.
        _mri_on_histo_rgb.png: Overlay on the RGB histology; only when an RGB image exists
            and the previous step was not skipped.
        _transform_grid.npz: The transform, with keys
            A_common_to_hr: (4, 4) float64 index affine mapping a homogeneous
                common-canvas histology pixel (x, y, z, 1) to an HR histology pixel index
                (x, y, z) of the loaded (already flipped) histology image. It composes
                HR -> LR (spacing ratio), LR -> proc (padding), proc -> scaled (best scale)
                and scaled -> common (offset ox_h, oy_h), and inverts the result.
            xyz_norm_final: (Hc, Wc, 3) float32 refined sheet: for each common-canvas
                pixel, the MRI position (x, y, z) divided by (X-1, Y-1, Z-1) of the
                orientation-permuted MRI volume. Values outside [0, 1] are outside the
                volume.
            xy_hist_map: (Hc, Wc, 2) float32 histology position (x, y), in common-canvas
                histology pixels, that is mapped to each canvas pixel (G plus the
                confidence-weighted 2D residual TPS). Apply A_common_to_hr to reach HR
                pixels.
            valid_surf: (Hc, Wc) uint8, 1 where the sheet is inside the MRI volume, the
                histology lookup is inside the histology image and inside its mask.
            ox_h, oy_h: float64 histology offset (pixels) in the common canvas; already
                included in A_common_to_hr.
            G_e1, G_e2: (3,) float64 in-plane axes of G in MRI voxel (x, y, z); not unit
                length (e1 = (1, 0, a), e2 = (0, 1, b) for the plane z = a*x + b*y + c).
            mri_spacing: (3,) float64 voxel spacing (x, y, z) of the permuted MRI.
            mri_size: (3,) int64 size (X, Y, Z) of the permuted MRI.
            subject_id, num_str, profile: 0-d string arrays identifying the run.
            version: (1,) int32 file-format version (2).

    Returns:
        Dict with output_npz (path of the .npz), n_clean_matches (number of outlier-filtered
        matches given to the TPS fit), best_scale (histology scale chosen by the search),
        profile and elapsed_sec (wall time of this call).

    Raises:
        ValueError: If profile is not a key of PROFILES. Other errors from the steps (for
            example too few matches) propagate unchanged.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; expected one of {list(PROFILES)}")
    settings = PROFILES[profile]

    log_path = os.path.join(log_dir, f"{subject_id}_{num_str}_process.log")
    _reset_logging(log_path)
    logging.info("Process started")
    logging.info(f"Profile: {profile} {settings}")
    section_start = time.perf_counter()

    # ---- Load ----
    vols = load_section(subject_id, num_str, histo_type, spacing, orientation, flip, args.data_root)
    Ih_hr, Mh_hr, Im, Mm = vols["Ih_hr"], vols["Mh_hr"], vols["Im"], vols["Mm"]
    Ih_rgb_hr = vols.get("Ih_rgb_hr")
    logging.info(f"Loaded section {subject_id}/{num_str}")

    # ---- Resample histology to MRI spacing (HR -> LR) ----
    Ih, Mh, A_hr_to_lr = resample_histo_to_mri(Ih_hr=Ih_hr, Mh_hr=Mh_hr, Im=Im)

    # ---- Prepare histology (LR -> proc: pad to a multiple of the patch size) ----
    # scale_ratio (1, 1): no resizing here; the shrinkage scale is searched below.
    scale_ratio = (1, 1)
    proc_Ih, proc_Mh, A_lr_to_proc = prepare_histology(
        Ih=Ih, Mh=Mh, scale_ratio=scale_ratio, patch_size=args.patch_size,
        show_proc_img=False,
    )
    A_hr_to_proc = A_lr_to_proc @ A_hr_to_lr
    _ = np.linalg.inv(A_hr_to_proc)  # inverse (proc -> HR) is computed but not used
    logging.info(f"Prepared histology - proc_Ih size:{proc_Ih.GetSize()}")

    # ---- Prepare MRI (dilated mask, robust intensity normalisation) ----
    # n_slices is only the slice index of the optional preparation display.
    n_slices = Im.GetSize()[2] // args.n_slices_offset_divisor
    _, slab_Im, slab_Mm = prepare_mri_slab(
        Im=Im, Mm=Mm, n_slices=n_slices, show_proc_img=False, pclip=tuple(args.pclip),
    )
    logging.info(f"Prepared MRI slab - slab_Im size:{slab_Im.GetSize()}")

    # ---- Find ROI: best histology scale (shrinkage compensation) ----
    scales = np.arange(1.00, args.scale_max + 1e-6, args.scale_step)
    best_scale_info, _ = search_scale(
        proc_Ih=proc_Ih, histo_mask=proc_Mh, slab_Im=slab_Im, slab_Mm=slab_Mm,
        scales=scales, common_kwargs=common_kwargs,
    )
    best_scale = best_scale_info["scale"]
    logging.info(f"Best scale: {best_scale} score:{best_scale_info['score']}")

    # ---- Find ROI: best patch size / strides and z band ----
    # Candidate configurations (patch size and strides in pixels/voxels).
    patch_matrix = [
        {"patch": 96, "hist_stride": 64, "mri_stride": 8},
        {"patch": 64, "hist_stride": 32, "mri_stride": 4},
    ]
    best_patch, best_hist_stride, best_mri_stride, z_center, z_min, z_max = search_patch_config(
        proc_Ih=proc_Ih, proc_Mh=proc_Mh, slab_Im=slab_Im, slab_Mm=slab_Mm,
        best_scale=best_scale, patch_matrix=patch_matrix,
        z_min=None, z_max=None, common_kwargs=common_kwargs,
    )
    logging.info(f"Best patch:{best_patch} hist_stride:{best_hist_stride} mri_stride:{best_mri_stride} "
                 f"z_center:{z_center}")

    # ---- Fine matching (proc -> scaled histology), restricted to z_center +/- 60 slices ----
    scaled_proc_Ih, scaled_proc_Mh = rescale_histology(proc_Ih, proc_Mh, best_scale)

    # The z band returned by search_patch_config is replaced by a fixed +/-60-slice window.
    size = slab_Im.GetSize()
    Z = int(size[2]) if slab_Im.GetDimension() == 3 else 1
    z_min = max(0, int(z_center - 60))
    z_max = min(Z - 1, int(z_center + 60))

    all_matches = match_patches(
        proc_Ih=scaled_proc_Ih, histo_mask=scaled_proc_Mh,
        slab_Im=slab_Im, slab_Mm=slab_Mm,
        z_min=z_min, z_max=z_max,
        hist_patch=best_patch, hist_stride=best_hist_stride,
        mri_patch=best_patch, mri_stride=best_mri_stride,
        **common_kwargs,
    )

    # ---- Outlier filtering ----
    clean_matches, coarse_matches, pts_clean, sc_clean = filter_outliers(
        all_matches=all_matches["matches"], z_min=z_min, z_max=z_max,
        plane_dist_thresh=settings["outlier_plane_dist"], knn_params=settings["outlier_knn"],
    )
    logging.info(f"Fine matching - clean_matches:{len(clean_matches)}")

    # ---- ROI plane ----
    # extract_roi_plane is required, not only QC: plane_img (MRI sampled on a plane through the
    # clean matches) sets the common canvas, and its origin x0/y0 (MRI voxels) sets G's (u, v) frame.
    plane_img, x0, x1, y0, y1 = extract_roi_plane(
        slab_Im, scaled_proc_Ih, pts_clean, sc_clean, all_matches, best_patch,
        coarse_matches, clean_matches, z_min, z_max, device=device,
    )

    # ---- Common canvas ----
    Hc, Wc, oy_h, ox_h, oy_m, ox_m, hist_common, mri_common, mask_common = create_common_canvas(
        scaled_proc_Ih, scaled_proc_Mh, plane_img
    )
    logging.info(f"Common canvas - hist_common:{hist_common.shape}")

    # ---- Global in-plane transform G ----
    # MRI x/y range covered by the whole common canvas (1 canvas pixel = 1 MRI voxel), so that
    # G's (u, v) equal canvas (column, row).
    step = 1.0
    x_range_final = (x0 - ox_m * step, x0 + (Wc - 1 - ox_m) * step)
    y_range_final = (y0 - oy_m * step, y0 + (Hc - 1 - oy_m) * step)

    # Histology match centres moved to the common-canvas frame; MRI centres stay in voxels.
    clean_matches_common = matches_to_common_canvas(clean_matches=clean_matches, oy_h=oy_h, ox_h=ox_h)
    # Bounds for fit_global_plane's MRI filter, taken from the matches themselves.
    yy = np.array([m["mri_center_zyx"][1] for m in clean_matches_common], np.float32)
    xx = np.array([m["mri_center_zyx"][2] for m in clean_matches_common], np.float32)
    Hm_fit = int(np.ceil(yy.max())) + 1
    Wm_fit = int(np.ceil(xx.max())) + 1

    G = fit_global_plane(
        clean_matches_common, x_range_final, y_range_final,
        mri_hw=(Hm_fit, Wm_fit), hist_hw=(Hc, Wc), robust=True,
    )
    logging.info("Fitted initial in-plane G")

    # Histology image and mask on the common canvas as (1, Hc, Wc) sitk images.
    proc_Ih_common_arr = hist_common.astype(np.float32)[None, ...]
    proc_Ih_common_sitk = sitk.GetImageFromArray(proc_Ih_common_arr)
    proc_Mh_common_arr = mask_common.astype(np.float32)[None, ...]
    proc_Mh_common_sitk = sitk.GetImageFromArray(proc_Mh_common_arr)

    # ---- Initial surface: robust TPS fit (RANSAC, then MAD) ----
    # Adaptive RANSAC inlier threshold: fit a TPS with MAD rejection, take the residuals of all
    # matches (MRI voxels), and use median + 3 * 1.4826 * MAD, clipped to [3, 20] voxels.
    _tps_diag, _, _, _, _resid_diag, _ = fit_tps_robust(
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

    # G was fitted from clean_matches_common, and fit_tps_robust recovers uv by inverting G on
    # the histology coordinates it is given, so they must be in the same common-canvas frame;
    # passing clean_matches (local histology frame) would offset every uv by (ox_h, oy_h).
    tps_ransac, uv_in, xyz_in, inlier_mask, _, score = fit_tps_robust(
        clean_matches_common, G, method="ransac",
        n_iters=300, sample_size=ransac_sample_size, inlier_thresh=adaptive_thresh, min_inliers=30,
        device=device, normalize_uv=False, lam_hyp=1e-2, lam_final=1e-2,
    )
    logging.info(f"Init DDF - adaptive inlier_thresh={adaptive_thresh:.2f}, sample_size={ransac_sample_size}")
    matches_inliers = [m for m, keep in zip(clean_matches_common, inlier_mask.cpu().tolist()) if keep]

    # Second MAD pass on the RANSAC inliers; uv_in2/xyz_in2 are the surviving (u, v) and MRI xyz.
    tps_init_final, uv_in2, xyz_in2, inlier_mask2, _, score2 = fit_tps_robust(
        matches_inliers, G, method="mad",
        mad_thresh=2.5, mad_iterations=2, lam_hyp=1e-2, lam_final=1e-2,
        device=device, normalize_uv=False,
    )
    matches_inliers_final = [m for m, keep in zip(matches_inliers, inlier_mask2.cpu().tolist()) if keep]

    # Histology (x, y) of the same matches, in common-canvas pixels.
    hist_xy_in2 = np.asarray(
        [[float(m["hist_center_yx"][1]), float(m["hist_center_yx"][0])] for m in matches_inliers_final],
        dtype=np.float32,
    )

    # The in-plane part of the residual against G (a plane plus a 2D similarity) detects bad
    # matches well: the flexible TPS absorbs them (they pass RANSAC/MAD), G does not. Drop
    # matches whose MRI position lies more than INPLANE_MAX voxels in-plane from G(u, v).
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

    # ---- Residual TPS on top of G ----
    # 3D: MRI xyz offset from G(u, v); 2D: histology xy offset from G's similarity. Both add
    # zero-residual anchors on a grid far from the matches, so the surface falls back to G there.
    tps3d_res, uv_all_3d, dxyz_all, uv_anchor_3d = fit_residual_tps3d(
        uv_ctrl=uv_in2, xyz_ctrl=xyz_in2, G=G, device=device, lam=1e-1,
        normalize_uv=False, anchor_step=24.0, far_thr=35.0, max_anchors=1200, seed=42,
    )
    tps2d_res, uv_all_2d, dxy_all, uv_anchor_2d = fit_residual_tps2d(
        uv_ctrl=uv_in2, hist_xy_ctrl=hist_xy_in2, G=G, device=device, lam=2e-1,
        normalize_uv=False, anchor_step=24.0, far_thr=35.0, max_anchors=1200, seed=42,
    )

    # Per-section cap on the initial surface's displacement from G (tps_max_disp_vox in the
    # refinement): 1.25 x the 99th percentile of the kept matches' |xyz - G(u, v)|, clipped to
    # [8, 60] voxels, so it covers their curvature without following a single extreme match.
    _uvn2 = uv_in2.detach().cpu().numpy() if torch.is_tensor(uv_in2) else np.asarray(uv_in2)
    _xyzn2 = xyz_in2.detach().cpu().numpy() if torch.is_tensor(xyz_in2) else np.asarray(xyz_in2)
    _mag = np.linalg.norm(_xyzn2 - (_X0[None] + _uvn2[:, 0:1]*_e1[None] + _uvn2[:, 1:2]*_e2[None]), axis=1)
    clamp_vox = float(np.clip(1.25 * np.percentile(_mag, 99), 8.0, 60.0))
    logging.info(f"Init DDF - per-section clamp_vox={clamp_vox:.2f} (kept |dxyz| max={_mag.max():.1f} vox)")

    # ---- Resample histology onto the surface (common canvas) ----
    # xy_map_sym: (Hc, Wc, 2) histology (x, y) looked up for each canvas pixel; valid_hist marks
    # lookups inside the histology image.
    hist_on_init_tps, valid_hist, uv_map_sym, xy_map_sym, conf_hist, dmin_hist = resample_histology_on_surface(
        hist_img=proc_Ih_common_sitk, G=G, tps2d_residual=tps2d_res, ctrl_uv=uv_in2,
        canvas_shape=(Hc, Wc), device=device, is_mask=False, sigma_uv=20.0, min_conf=0.25,
        hard_mult=2.0, max_residual_px=20.0, return_extra=True,
    )
    hist_mask_on_init_tps = resample_histology_on_surface(
        hist_img=proc_Mh_common_sitk, G=G, tps2d_residual=tps2d_res, ctrl_uv=uv_in2,
        canvas_shape=(Hc, Wc), device=device, is_mask=True, sigma_uv=20.0, min_conf=0.25,
        hard_mult=2.0, max_residual_px=20.0, return_extra=False,
    )
    hist_mask_on_init_tps = (hist_mask_on_init_tps > 0.5)
    logging.info("Initial TPS on top of G fitted")

    # ---- Curved-sheet refinement ----
    # Initial surface = G + 3D residual TPS (absolute MRI voxel xyz). The refinement deforms the
    # sheet so that the MRI sampled on it matches hist_on_init_tps.
    init_tps_abs = ResidualToGSurface3D(G=G, residual_tps=tps3d_res, device=device).to(device)
    (xyz_norm_final, mri_on_final_sheet, valid_mri,
     _xyz_norm_base, _conf, _phi, _n0, _alpha, _rf_psi, _rf_beta) = refine_curved_sheet(
        slab_Im=slab_Im, slab_Mm=slab_Mm, init_G=G, init_tps=init_tps_abs,
        matches_for_warmstart=matches_inliers_final, device=device,
        hist_on_init_tps=hist_on_init_tps, hist_mask_on_init_tps=None,
        ctrl_uv_support=uv_in2, tps_sigma_uv=20.0, tps_min_conf=0.25, tps_max_disp_vox=clamp_vox,
        lam_area=args.lam_area, lam_shape=args.lam_shape, w_rigid_mode=args.w_rigid_mode,
        similarity_weights=settings["refine_similarity_weights"],
        gif_file=None,
    )
    logging.info("Refined curved sheet")

    # ---- QC figures in MRI (sheet) space ----
    # Valid canvas pixels: sheet inside the MRI, histology lookup inside the image and the mask.
    valid_display = valid_mri & valid_hist & hist_mask_on_init_tps
    plot_mri_on_sheet(
        hist_on_init_tps=hist_on_init_tps, valid_display=valid_display,
        mri_on_final_sheet=mri_on_final_sheet,
        save_file=os.path.join(log_dir, f"{subject_id}_{num_str}_histo_in_mri_space_all.png"),
    )
    # plot_mri_on_sheet's save_file_single saves only the MRI-on-sheet panel, so the
    # "Target Histology on Sheet" panel is saved separately.
    _save_target_histo_on_sheet(
        subject_id=subject_id, num_str=num_str, log_dir=log_dir,
        hist_on_init_tps=hist_on_init_tps, valid_display=valid_display,
    )

    # ---- Map back to HR histology (affine chain HR <-> common canvas) and QC ----
    A_hr_to_common, A_common_to_hr = build_hr_common_affines(
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

    # ---- Save the transform (keys and frames: see the docstring) ----
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
        profile=np.array(profile),
        version=np.array([2], dtype=np.int32),
    )
    logging.info(f"Saved transform grid -> {save_path}")

    # Free large objects and cached GPU memory before the next section.
    del all_matches, clean_matches, coarse_matches, matches_inliers, matches_inliers_final
    del tps_ransac, tps_init_final, tps3d_res, tps2d_res, init_tps_abs
    torch.cuda.empty_cache()

    elapsed_sec = time.perf_counter() - section_start
    logging.info(f"Process finished in {elapsed_sec:.2f}s")

    return {
        "output_npz": save_path,
        "n_clean_matches": len(clean_matches_common),
        "best_scale": best_scale,
        "profile": profile,
        "elapsed_sec": elapsed_sec,
    }


def _save_target_histo_on_sheet(subject_id, num_str, log_dir, hist_on_init_tps, valid_display):
    """Save the histology-on-sheet image, zeroed outside valid_display, as <prefix>_histo_in_mri_space.png.

    Same content and masking as the "Target Histology on Sheet" panel of plot_mri_on_sheet.
    """
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
    """Resample the MRI into HR histology pixels and save the overlay figures.

    Calls inverse.mri_on_histology_space (which writes <prefix>_mri_on_histo_all.png and
    <prefix>_mri_on_histo.png), logs the HR coverage, and, when RGB histology is available,
    writes <prefix>_mri_on_histo_rgb.png. A ValueError from the resampling (empty histology
    mask or empty valid_display) is logged and the step is skipped. histo_type and flip are
    not used.
    """
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
        # Empty histology mask or empty valid_display: nothing to project.
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
    plot_mri_on_histology_rgb(arr_Ih_hr_2d, arr_Ih_rgb_hr, mri_on_hr_updated,
                            tile=tile_size, save_file=save_file_rgb)

