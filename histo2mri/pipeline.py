"""Register one 2D histology section to a 3D MRI volume, running all pipeline steps in order.

register_section is the per-section driver called by the command-line tool (histo2mri.cli).
It ends with a curved sheet in MRI voxel space (xyz_norm, normalised (x, y, z) per axis) sampled
on the common canvas, plus the index affine from the common canvas back to the full-resolution
(HR) histology, and saves both to <subject>_<section>_transform_grid.npz. PROFILES holds named
parameter sets for stains/regions that need different settings.
"""
import contextlib
import logging
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from .canvas import create_common_canvas, extract_roi_plane
from .inverse import build_hr_common_affines, mri_on_histology_space
from .io import load_section, resolve_section
from .matching import match_patches, rescale_histology, search_patch_config, search_scale
from .outliers import filter_outliers
from .plane import fit_global_plane, matches_to_common_canvas
from .preprocessing import prepare_histology, prepare_mri_slab, resample_histo_to_mri
from .qc import (plot_histology_in_mri_space, plot_mri_on_histology_rgb, plot_mri_on_sheet, plot_roi_support,
                 show_matched_pairs, visualize_roi_from_matches)
from .refine import prepare_coords_from_G_canvas, refine_curved_sheet, sheet_folding_pct, tps_sheet_xyz_norm
from .tps import (InplaneScaledResidual, ResidualToGSurface3D, fit_residual_tps2d, fit_residual_tps3d,
                  fit_tps_robust, resample_histology_on_surface)


# Named settings for stains/regions that need different parameters. "default" is used for
# everything unless a section asks for another profile (--profile, or a "profile" column in
# the sections CSV). Keys:
#   outlier_plane_dist: distance threshold (MRI voxels) of the coarse RANSAC plane pre-filter
#     in filter_outliers.
#   outlier_knn: parameters of the stricter kNN outlier filter that follows it.
#   refine_similarity_weights: (w_ssim, w_mind) weights of the SSIM and MIND image loss in
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
        refine_similarity_weights=(1.0, 0.0),   # (w_ssim, w_mind)
    ),
    "he-cblock": dict(
        outlier_plane_dist=20.0,
        outlier_knn=dict(k=3, plane_mad_k=2.0, tukey_k=1.5, cc_radius_scale=2.0),
        refine_similarity_weights=(0.0, 1.0),
    ),
}

# Patch-matcher settings shared by search_scale, search_patch_config and match_patches.
# Fine matching (match_patches) adds an NGF term on top, see FINE_MATCH_W_NGF.
MATCHER_SETTINGS = dict(
    cell_size=8, bins=18,                 # HOG cell size (px) and orientation bins
    min_histo_mask_frac=0.2,              # keep histology patches with >= 20% tissue
    min_mri_mask_frac=0.50,               # keep MRI patches with >= 50% brain mask
    min_overlap_frac=0.35,                # minimum mask overlap of a histology/MRI pair
    topk_hog=64, w_hog=0.6, w_mind=0.4,   # HOG shortlist size and fusion weights
    mind_radius=2, mind_dilation=1,
    hist_batch=128, mri_chunk=16384, keep_mri_bank_on_gpu=True,
)

# NGF fusion weight of the fine matcher only, so its score is
# (0.6*HOG + 0.4*MIND + 0.2*NGF) / 1.2. Not part of MATCHER_SETTINGS, because the cached
# HOG+MIND matcher used by the scale and patch-config searches has no NGF term. Without NGF,
# mean TRE on 37 myelin sections got worse (1.91 -> 1.99 mm) and the surface folded more.
FINE_MATCH_W_NGF = 0.2

# Half-width of the fine-matching z window around z_center, in mm; converted to slices with
# the slab's slice spacing. It is a tilt allowance: about 26 deg of section tilt for a typical
# section, 16 deg for the largest ones. The coarse z band of search_patch_config is not used,
# because it only reflects the spread of a few coarse matches (often its 8-slice minimum),
# while the section's tilt makes the final sheet reach well beyond it. 24 mm equals the former
# fixed 60 slices at 0.4 mm MRI and gives 64 slices at 0.375 mm.
FINE_MATCH_Z_HALF_MM = 24.0

# Fold fix. Where neighbouring matches disagree, the in-plane part of the 3D residual TPS can
# fold the initial sheet, and the refinement keeps such folds (they show as swirls in the MRI
# on the histology). If the initial sheet folds, the in-plane part is scaled by these factors
# in turn until it no longer does; that sheet is refined as well and the result whose final
# sheet folds less is kept. Sections whose initial sheet does not fold are not affected.
FOLD_FIX_INPLANE_SCALES = (0.5, 0.25)

logger = logging.getLogger(__name__)


def _to_np(x):
    """Return x as a numpy array (torch tensors are detached and moved to the CPU)."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


@contextlib.contextmanager
def _section_log(log_path):
    """Write the package's INFO (and higher) log records to log_path while the block runs.

    The file is overwritten. The handler is attached to the "histo2mri" logger only, so a
    caller's own logging configuration (e.g. in a notebook) is left untouched.
    """
    pkg_logger = logging.getLogger("histo2mri")
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    old_level = pkg_logger.level
    if pkg_logger.getEffectiveLevel() > logging.INFO:
        pkg_logger.setLevel(logging.INFO)
    pkg_logger.addHandler(handler)
    try:
        yield
    finally:
        pkg_logger.removeHandler(handler)
        handler.close()
        pkg_logger.setLevel(old_level)


def register_section(bids_root, subject, sample, out_dir, *, stain=None,
                     profile="default", device=None, ransac_sample_size=5,
                     pad_multiple=64, pclip=(1, 99), scale_max=1.30, scale_step=0.05,
                     lam_area=2.0, lam_shape=0.0, w_rigid_mode="mask",
                     mri_on_histo_tile_size=348, matcher_settings=None, save_roi_qc=False,
                     verbose=False):
    """Register one histology section to its MRI volume and save the transform.

    Steps: load the section (io.load_section); resample the histology to MRI spacing and pad
    it to a multiple of pad_multiple; normalise the MRI intensities; search the histology
    scale and the patch/stride configuration, then match histology patches to MRI patches
    within +/-60 slices of the estimated z centre; remove outlier matches; sample the MRI on
    a plane through the matches (ROI plane) and place it and the histology on a common
    canvas; fit the global plane G; fit a TPS surface robustly (RANSAC with an adaptive
    threshold from a MAD fit, then a MAD refit), drop matches far in-plane from G, and fit
    3D and 2D residual TPS on top of G; resample the histology onto the canvas through G and
    the 2D residual; refine the curved sheet in the MRI so that the MRI sampled on it
    matches that histology (if the initial sheet folds, also refine one with the in-plane
    residual scaled down and keep the less folded result; see FOLD_FIX_INPLANE_SCALES);
    pull the MRI back to the HR histology for QC; and save the transform. Errors are not
    caught here; they propagate to the caller.

    Args:
        bids_root: Root of the BIDS dataset (layout: see io.load_section).
        subject: Subject label, with or without the "sub-" prefix.
        sample: Sample label, with or without the "sample-" prefix.
        out_dir: Output folder for this section; created if missing. All files below are
            written here.
        stain: Stain label; optional when the subject and sample identify one section.
            The cutting plane, in-plane rotation, flip and pixel size are read from the
            section's sidecars by io.load_section.
        profile: Key of PROFILES selecting the outlier-filter and refinement settings.
        device: torch device (or string) for matching, TPS fitting and refinement; defaults
            to "cuda:0" when CUDA is available, otherwise "cpu".
        ransac_sample_size: Minimal sample size of the RANSAC TPS fit (the command line
            uses 3 for NextBrain sections and 5 otherwise).
        pad_multiple: The prepared histology is padded to a multiple of this (pixels). The
            matching patch sizes themselves are searched (96 and 64 px).
        pclip: (low, high) percentiles for the MRI intensity normalisation.
        scale_max: Upper end of the histology scale search (grid 1.00..scale_max).
        scale_step: Step of the histology scale grid.
        lam_area: Weight of the area (properness) rigidity term in the refinement; 0 turns
            local rigidity off.
        lam_shape: Weight of the shape (orthonormality) rigidity term.
        w_rigid_mode: Spatial weight map of the rigidity terms: "mask", "conf", "edge" or
            "none" (see losses._build_rigidity_weight).
        mri_on_histo_tile_size: Checkerboard tile size (HR pixels) of the MRI-on-histology
            figures.
        matcher_settings: Patch-matcher settings; defaults to MATCHER_SETTINGS. The device
            is added automatically.
        save_roi_qc: Also save the three ROI QC figures (matched patch pairs, 3D view of the
            matches, hull support on histology and ROI plane).
        verbose: Log extra detail of the MRI-on-histology step.

    Files written to out_dir (prefix <sub-X>_<sample-Y>_<stain-Z>):
        _process.log: Log of this section (overwritten on each call).
        _histo_in_mri_space_all.png: Histology resampled onto the canvas, valid region,
            MRI on the final sheet and their checkerboard.
        _histo_in_mri_space.png: The histology-on-sheet panel alone.
        _mri_on_histo_all.png, _mri_on_histo.png: MRI resampled into HR histology pixels
            (composite with checkerboards, and the MRI alone). Skipped when there is no
            usable support.
        _mri_on_histo_rgb.png: Overlay on the RGB histology; only when an RGB image exists
            and the previous step was not skipped.
        _roi_pairs.png, _roi_3d.png, _roi_support.png: ROI QC figures; only with
            save_roi_qc=True.
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
            subject_id, sample, stain, section_id, profile: 0-d string arrays
                identifying the run.
            mri_origin, mri_direction: (3,) and (9,) float64 origin and direction cosines
                of the permuted MRI. With mri_spacing they map a sheet voxel position to
                world millimetres, which is what a 3D viewer needs.
            mri_permute: (3,) int64 ORIENTATION_PERMUTE order that was applied;
                mri_view_flip: (3,) bool the in-plane axes load_mri then reversed to reach
                the viewer frame (io.VIEW_AXES); mri_orientation, mri_canonical ("LPS") and
                mri_source name the cutting plane, the orientation the volume was brought
                to, and the file it came from. Together they undo load_mri and reach the
                stored T1w grid.
            histo_on_surface, mri_on_surface: (Hc, Wc) float32 histology resampled onto the
                sheet and the MRI sampled on it, on the common canvas. The full-resolution
                versions are not stored; regenerate them from the affines.
            A_hr_to_tif: (4, 4) float64 index affine mapping an HR histology pixel of the
                loaded (rotated and flipped) image back to a pixel of the cleanslide TIFF.
                Compose with A_common_to_hr to go from the canvas to that file.
            crop_origin_xy, source_shape_hw: (2,) int64 position of the cleanslide image
                within the raw _BF.tif and that slide's shape, or -1 when not cropped.
            histo_source: 0-d string array, the cleanslide TIFF that was read.
            version: (1,) int32 file-format version (5; version 4 lacked mri_view_flip and
                its volume was left in the canonical frame, version 3 lacked the frame and
                surface fields, version 2 named the section with subject_id and num_str).

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
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    matcher = dict(MATCHER_SETTINGS if matcher_settings is None else matcher_settings, device=device)

    os.makedirs(out_dir, exist_ok=True)
    # Resolved before the log is opened, because the log file is named after the section.
    section_id = resolve_section(bids_root, subject, sample, stain)["base"]
    log_path = os.path.join(out_dir, f"{section_id}_process.log")
    with _section_log(log_path):
        return _register(
            bids_root, subject, sample, stain, section_id, out_dir,
            profile=profile, device=device, ransac_sample_size=ransac_sample_size,
            pad_multiple=pad_multiple, pclip=pclip, scale_max=scale_max, scale_step=scale_step,
            lam_area=lam_area, lam_shape=lam_shape, w_rigid_mode=w_rigid_mode,
            mri_on_histo_tile_size=mri_on_histo_tile_size, save_roi_qc=save_roi_qc,
            verbose=verbose, settings=settings, matcher=matcher,
        )


def _register(bids_root, subject, sample, stain, section_id, out_dir, *,
              profile, device, ransac_sample_size, pad_multiple, pclip, scale_max, scale_step,
              lam_area, lam_shape, w_rigid_mode, mri_on_histo_tile_size, save_roi_qc, verbose,
              settings, matcher):
    """Body of register_section, run while the section log file is attached."""
    logger.info("Process started")
    logger.info(f"Profile: {profile} {settings}")
    section_start = time.perf_counter()

    # ---- Load ----
    vols = load_section(bids_root, subject, sample, stain)
    Ih_hr, Mh_hr, Im, Mm = vols["Ih_hr"], vols["Mh_hr"], vols["Im"], vols["Mm"]
    Ih_rgb_hr = vols.get("Ih_rgb_hr")
    paths = vols["paths"]
    logger.info(f"Loaded section {section_id}: {vols['orientation']}, "
                f"rotation {vols['rotation']:g} deg, flip {vols['flip']}, "
                f"spacing {tuple(round(v, 5) for v in vols['spacing'])} mm")

    # ---- Resample histology to MRI spacing (HR -> LR) ----
    Ih, Mh, A_hr_to_lr = resample_histo_to_mri(Ih_hr=Ih_hr, Mh_hr=Mh_hr, Im=Im)

    # ---- Prepare histology (LR -> proc: pad to a multiple of pad_multiple) ----
    # scale_ratio (1, 1): no resizing here; the shrinkage scale is searched below.
    scale_ratio = (1, 1)
    proc_Ih, proc_Mh, A_lr_to_proc = prepare_histology(
        Ih=Ih, Mh=Mh, scale_ratio=scale_ratio, pad_multiple=pad_multiple,
        show_proc_img=False,
    )
    A_hr_to_proc = A_lr_to_proc @ A_hr_to_lr
    logger.info(f"Prepared histology - proc_Ih size:{proc_Ih.GetSize()}")

    # ---- Prepare MRI (dilated mask, robust intensity normalisation) ----
    # n_slices is only the slice index of the optional preparation display.
    n_slices = Im.GetSize()[2] // 2
    _, slab_Im, slab_Mm = prepare_mri_slab(
        Im=Im, Mm=Mm, n_slices=n_slices, show_proc_img=False, pclip=tuple(pclip),
    )
    logger.info(f"Prepared MRI slab - slab_Im size:{slab_Im.GetSize()}")

    # ---- Find ROI: best histology scale (shrinkage compensation) ----
    scales = np.arange(1.00, scale_max + 1e-6, scale_step)
    best_scale_info, _ = search_scale(
        proc_Ih=proc_Ih, histo_mask=proc_Mh, slab_Im=slab_Im, slab_Mm=slab_Mm,
        scales=scales, common_kwargs=matcher,
    )
    best_scale = best_scale_info["scale"]
    logger.info(f"Best scale: {best_scale} score:{best_scale_info['score']}")

    # ---- Find ROI: best patch size / strides and z band ----
    # Candidate configurations (patch size and strides in pixels/voxels).
    patch_matrix = [
        {"patch": 96, "hist_stride": 64, "mri_stride": 8},
        {"patch": 64, "hist_stride": 32, "mri_stride": 4},
    ]
    best_patch, best_hist_stride, best_mri_stride, z_center, z_min, z_max = search_patch_config(
        proc_Ih=proc_Ih, proc_Mh=proc_Mh, slab_Im=slab_Im, slab_Mm=slab_Mm,
        best_scale=best_scale, patch_matrix=patch_matrix,
        z_min=None, z_max=None, common_kwargs=matcher,
    )
    logger.info(f"Best patch:{best_patch} hist_stride:{best_hist_stride} mri_stride:{best_mri_stride} "
                 f"z_center:{z_center}")

    # ---- Fine matching (proc -> scaled histology), restricted to z_center +/- FINE_MATCH_Z_HALF_MM ----
    scaled_proc_Ih, scaled_proc_Mh = rescale_histology(proc_Ih, proc_Mh, best_scale)

    # The z band returned by search_patch_config is replaced by a fixed window in mm.
    size = slab_Im.GetSize()
    Z = int(size[2]) if slab_Im.GetDimension() == 3 else 1
    z_half = int(round(FINE_MATCH_Z_HALF_MM / slab_Im.GetSpacing()[2])) if Z > 1 else 0
    z_min = max(0, int(z_center - z_half))
    z_max = min(Z - 1, int(z_center + z_half))
    logger.info(f"Fine matching z window: +/-{z_half} slices (+/-{FINE_MATCH_Z_HALF_MM:g} mm)")

    all_matches = match_patches(
        proc_Ih=scaled_proc_Ih, histo_mask=scaled_proc_Mh,
        slab_Im=slab_Im, slab_Mm=slab_Mm,
        z_min=z_min, z_max=z_max,
        hist_patch=best_patch, hist_stride=best_hist_stride,
        mri_patch=best_patch, mri_stride=best_mri_stride,
        w_ngf=FINE_MATCH_W_NGF,
        **matcher,
    )

    # ---- Outlier filtering ----
    clean_matches, coarse_matches, pts_clean, sc_clean = filter_outliers(
        all_matches=all_matches["matches"], z_min=z_min, z_max=z_max,
        plane_dist_thresh=settings["outlier_plane_dist"], knn_params=settings["outlier_knn"],
    )
    logger.info(f"Fine matching - clean_matches:{len(clean_matches)}")

    # ---- ROI plane ----
    # extract_roi_plane is required, not only QC: plane_img (MRI sampled on a plane through the
    # clean matches) sets the common canvas, and its origin x0/y0 (MRI voxels) sets G's (u, v) frame.
    plane_img, x0, x1, y0, y1 = extract_roi_plane(clean_matches, pts_clean, slab_Im, device)
    if save_roi_qc:
        _save_roi_qc(section_id, out_dir, all_matches, clean_matches, best_patch,
                     slab_Im, scaled_proc_Ih, plane_img, x0, y0, z_min, z_max)

    # ---- Common canvas ----
    Hc, Wc, oy_h, ox_h, oy_m, ox_m, hist_common, mri_common, mask_common = create_common_canvas(
        scaled_proc_Ih, scaled_proc_Mh, plane_img
    )
    logger.info(f"Common canvas - hist_common:{hist_common.shape}")

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
    logger.info("Fitted initial in-plane G")

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

    # G was fitted from clean_matches_common, and fit_tps_robust recovers uv by inverting G on
    # the histology coordinates it is given, so they must be in the same common-canvas frame;
    # passing clean_matches (local histology frame) would offset every uv by (ox_h, oy_h).
    tps_ransac, uv_in, xyz_in, inlier_mask, _, score = fit_tps_robust(
        clean_matches_common, G, method="ransac",
        n_iters=300, sample_size=ransac_sample_size, inlier_thresh=adaptive_thresh, min_inliers=30,
        device=device, normalize_uv=False, lam_hyp=1e-2, lam_final=1e-2,
    )
    logger.info(f"Init DDF - adaptive inlier_thresh={adaptive_thresh:.2f}, sample_size={ransac_sample_size}")
    # fit_tps_robust skips invalid matches; matches_to_common_canvas already removed them, so
    # the inlier mask lines up with the list. Fail loudly if that ever stops being true.
    if len(inlier_mask) != len(clean_matches_common):
        raise RuntimeError("TPS inlier mask does not line up with the match list")
    matches_inliers = [m for m, keep in zip(clean_matches_common, inlier_mask.cpu().tolist()) if keep]

    # Second MAD pass on the RANSAC inliers; uv_in2/xyz_in2 are the surviving (u, v) and MRI xyz.
    tps_init_final, uv_in2, xyz_in2, inlier_mask2, _, score2 = fit_tps_robust(
        matches_inliers, G, method="mad",
        mad_thresh=2.5, mad_iterations=2, lam_hyp=1e-2, lam_final=1e-2,
        device=device, normalize_uv=False,
    )
    if len(inlier_mask2) != len(matches_inliers):
        raise RuntimeError("TPS inlier mask does not line up with the match list")
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
    logger.info(f"Init DDF - in-plane filter: kept {int(_keep.sum())}/{len(_keep)} (max in-plane {_inp.max():.1f} vox)")

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
    logger.info(f"Init DDF - per-section clamp_vox={clamp_vox:.2f} (kept |dxyz| max={_mag.max():.1f} vox)")

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
    logger.info("Initial TPS on top of G fitted")

    # ---- Fold check of the initial sheet (see FOLD_FIX_INPLANE_SCALES) ----
    # The sheet the refinement starts from (tps_sheet_xyz_norm, before its B-spline warm
    # start), evaluated on the canvas nodes that lie inside the MRI and the histology mask.
    dims_xyz = slab_Im.GetSize()
    uv_canvas, _, _ = prepare_coords_from_G_canvas(G, Hc, Wc, device,
                                                   common_shift_yx=G.get("canvas_offset_yx", None))

    def _initial_folding(residual):
        sheet = ResidualToGSurface3D(G=G, residual_tps=residual, device=device).to(device)
        xyz = tps_sheet_xyz_norm(sheet, G, uv_canvas, Hc, Wc, *dims_xyz, device, ctrl_uv=uv_in2,
                                 sigma_uv=20.0, min_conf=0.25, max_disp_vox=clamp_vox)
        inside = _to_np(((xyz >= 0) & (xyz <= 1)).all(dim=-1))
        return sheet_folding_pct(xyz, inside & valid_hist & hist_mask_on_init_tps, dims_xyz, G)

    fold_initial = _initial_folding(tps3d_res)
    tps3d_res_shrunk = None
    if fold_initial > 0.0:
        for inplane_scale in FOLD_FIX_INPLANE_SCALES:
            tps3d_res_shrunk = InplaneScaledResidual(tps3d_res, G["n"], inplane_scale)
            fold_shrunk = _initial_folding(tps3d_res_shrunk)
            if fold_shrunk <= 0.0:
                break
        logger.info(f"Fold fix - initial sheet folds on {fold_initial:.3f}% of the canvas; "
                    f"in-plane residual x{inplane_scale:g}: {fold_shrunk:.3f}%")

    # ---- Curved-sheet refinement ----
    # Initial surface = G + 3D residual TPS (absolute MRI voxel xyz). The refinement deforms the
    # sheet so that the MRI sampled on it matches hist_on_init_tps.
    def _refine(residual):
        init_tps_abs = ResidualToGSurface3D(G=G, residual_tps=residual, device=device).to(device)
        refined = refine_curved_sheet(
            slab_Im=slab_Im, slab_Mm=slab_Mm, init_G=G, init_tps=init_tps_abs,
            matches_for_warmstart=matches_inliers_final, device=device,
            hist_on_init_tps=hist_on_init_tps, hist_mask_on_init_tps=None,
            ctrl_uv_support=uv_in2, tps_sigma_uv=20.0, tps_min_conf=0.25, tps_max_disp_vox=clamp_vox,
            lam_area=lam_area, lam_shape=lam_shape, w_rigid_mode=w_rigid_mode,
            similarity_weights=settings["refine_similarity_weights"],
            gif_file=None,
        )
        valid = refined[2] & valid_hist & hist_mask_on_init_tps
        return init_tps_abs, refined, sheet_folding_pct(refined[0], valid, dims_xyz, G)

    init_tps_abs, refined, fold_final = _refine(tps3d_res)
    if tps3d_res_shrunk is not None:
        result_shrunk = _refine(tps3d_res_shrunk)
        use_shrunk = result_shrunk[2] < fold_final
        logger.info(f"Fold fix - final sheet folding: original {fold_final:.3f}%, in-plane residual "
                    f"x{inplane_scale:g} {result_shrunk[2]:.3f}% -> "
                    f"{'using the scaled residual' if use_shrunk else 'keeping the original'}")
        if use_shrunk:
            init_tps_abs, refined, fold_final = result_shrunk
        del result_shrunk
    (xyz_norm_final, mri_on_final_sheet, valid_mri,
     _xyz_norm_base, _conf, _phi, _n0, _alpha, _rf_psi, _rf_beta) = refined
    logger.info("Refined curved sheet")

    # ---- QC figures in MRI (sheet) space ----
    # Valid canvas pixels: sheet inside the MRI, histology lookup inside the image and the mask.
    valid_display = valid_mri & valid_hist & hist_mask_on_init_tps
    plot_mri_on_sheet(
        hist_on_init_tps=hist_on_init_tps, valid_display=valid_display,
        mri_on_final_sheet=mri_on_final_sheet,
        save_file=os.path.join(out_dir, f"{section_id}_histo_in_mri_space_all.png"),
    )
    # plot_mri_on_sheet's save_file_single saves only the MRI-on-sheet panel, so the
    # "Target Histology on Sheet" panel is saved separately.
    _save_target_histo_on_sheet(
        section_id=section_id, out_dir=out_dir,
        hist_on_init_tps=hist_on_init_tps, valid_display=valid_display,
    )

    # Where the section sits in the volume, rather than flattened on the canvas.
    plot_histology_in_mri_space(
        xyz_norm_final=_to_np(xyz_norm_final), valid_surf=_to_np(valid_display),
        histo_on_surface=_to_np(hist_on_init_tps),
        mri_zyx=sitk.GetArrayFromImage(slab_Im), mri_size=slab_Im.GetSize(),
        save_file=os.path.join(out_dir, f"{section_id}_histo_in_mri_volume.png"),
    )

    # ---- Map back to HR histology (affine chain HR <-> common canvas) and QC ----
    A_hr_to_common, A_common_to_hr = build_hr_common_affines(
        proc_Ih, scaled_proc_Ih, A_hr_to_proc, oy_h, ox_h
    )
    logger.info("Composed A_common_to_hr")

    _save_mri_on_histo_space(
        section_id=section_id, out_dir=out_dir,
        tile_size=mri_on_histo_tile_size, verbose=verbose, Ih_hr=Ih_hr, Mh_hr=Mh_hr,
        proc_Ih=proc_Ih, scaled_proc_Ih=scaled_proc_Ih, A_hr_to_proc=A_hr_to_proc,
        oy_h=oy_h, ox_h=ox_h,
        xyz_norm_final=xyz_norm_final, xy_map_sym=xy_map_sym,
        valid_display=valid_display, mri_on_final_sheet=mri_on_final_sheet,
        slab_Im=slab_Im, device=device, Ih_rgb_hr=Ih_rgb_hr,
    )

    # ---- Save the transform (keys and frames: see the docstring) ----
    save_path = os.path.join(out_dir, f"{section_id}_transform_grid.npz")
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
        # Voxel -> world for the permuted volume, so the sheet can be placed in millimetres
        # rather than voxel numbers, and the permutation undone to reach the stored T1w.
        mri_origin=np.asarray(slab_Im.GetOrigin(), dtype=np.float64),
        mri_direction=np.asarray(slab_Im.GetDirection(), dtype=np.float64),
        mri_permute=np.asarray(vols["mri_permute"], dtype=np.int64),
        mri_view_flip=np.asarray(vols["mri_view_flip"], dtype=bool),
        mri_orientation=np.array(vols["orientation"]),
        mri_canonical=np.array(vols["mri_canonical"]),
        mri_source=np.array(paths["mri"]),
        # The two surfaces themselves, on the common canvas: histology resampled onto the
        # sheet, and the MRI sampled on it. Canvas-sized, so cheap to keep; the full-
        # resolution versions are regenerated from the affines above when wanted.
        histo_on_surface=_to_np(hist_on_init_tps).astype(np.float32),
        mri_on_surface=_to_np(mri_on_final_sheet).astype(np.float32),
        # Back to the histology files: A_hr_to_tif undoes the flip and the rotation, and the
        # crop values undo cleanslide's crop to reach the raw slide. -1 means "not cropped".
        A_hr_to_tif=np.asarray(vols["A_hr_to_tif"], dtype=np.float64),
        crop_origin_xy=np.asarray(vols["crop"]["crop_origin_xy"] or (-1, -1), dtype=np.int64),
        source_shape_hw=np.asarray(vols["crop"]["source_shape_hw"] or (-1, -1), dtype=np.int64),
        histo_source=np.array(paths["histo_gray"]),
        subject_id=np.array(paths["subject"]),
        sample=np.array(paths["sample"]),
        stain=np.array(paths["stain"]),
        section_id=np.array(section_id),
        profile=np.array(profile),
        version=np.array([5], dtype=np.int32),
    )
    logger.info(f"Saved transform grid -> {save_path}")

    # Free large objects and cached GPU memory before the next section.
    del all_matches, clean_matches, coarse_matches, matches_inliers
    del tps_ransac, tps_init_final, tps3d_res, tps3d_res_shrunk, tps2d_res, init_tps_abs, refined
    torch.cuda.empty_cache()

    elapsed_sec = time.perf_counter() - section_start
    logger.info(f"Process finished in {elapsed_sec:.2f}s")

    return {
        "output_npz": save_path,
        "n_clean_matches": len(clean_matches_common),
        "best_scale": best_scale,
        "profile": profile,
        "elapsed_sec": elapsed_sec,
    }


def _save_target_histo_on_sheet(section_id, out_dir, hist_on_init_tps, valid_display):
    """Save the histology-on-sheet image, zeroed outside valid_display, as <prefix>_histo_in_mri_space.png.

    Same content and masking as the "Target Histology on Sheet" panel of plot_mri_on_sheet.
    """
    h_disp = _to_np(hist_on_init_tps).astype(np.float32)
    if h_disp.ndim == 3 and h_disp.shape[0] == 1:
        h_disp = h_disp[0]
    h_disp = h_disp.copy()
    h_disp[~np.asarray(valid_display, dtype=bool)] = 0.0

    save_path = os.path.join(out_dir, f"{section_id}_histo_in_mri_space.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(h_disp, cmap="gray", origin="lower")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_mri_on_histo_space(section_id, out_dir, tile_size, verbose, Ih_hr, Mh_hr,
                              proc_Ih, scaled_proc_Ih, A_hr_to_proc, oy_h, ox_h,
                              xyz_norm_final, xy_map_sym, valid_display, mri_on_final_sheet,
                              slab_Im, device, Ih_rgb_hr=None):
    """Resample the MRI into HR histology pixels and save the overlay figures.

    Calls inverse.mri_on_histology_space (which writes <prefix>_mri_on_histo_all.png and
    <prefix>_mri_on_histo.png), logs the HR coverage, and, when RGB histology is available,
    writes <prefix>_mri_on_histo_rgb.png. A ValueError from the resampling (empty histology
    mask or empty valid_display) is logged and the step is skipped.
    """
    save_file = os.path.join(out_dir, f"{section_id}_mri_on_histo_all.png")
    save_file_single = os.path.join(out_dir, f"{section_id}_mri_on_histo.png")

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
            verbose=verbose,
        )
    except ValueError as e:
        # Empty histology mask or empty valid_display: nothing to project.
        logger.info(f"Dense HR QC skipped: {e}")
        return

    logger.info(f"Dense HR QC coverage: {final_valid_hr.mean():.6f}")

    if Ih_rgb_hr is None:
        logger.info("Dense HR QC: no RGB histology found, skipping the RGB overlay")
        return

    arr_Ih_hr = sitk.GetArrayFromImage(Ih_hr)
    arr_Ih_hr_2d = arr_Ih_hr[0] if arr_Ih_hr.ndim == 3 and arr_Ih_hr.shape[0] == 1 else arr_Ih_hr
    arr_Ih_rgb_hr = sitk.GetArrayFromImage(Ih_rgb_hr)[0]

    save_file_rgb = os.path.join(out_dir, f"{section_id}_mri_on_histo_rgb.png")
    plot_mri_on_histology_rgb(arr_Ih_hr_2d, arr_Ih_rgb_hr, mri_on_hr_updated,
                            tile=tile_size, save_file=save_file_rgb)



def _save_roi_qc(section_id, out_dir, all_matches, clean_matches, best_patch,
                 slab_Im, scaled_proc_Ih, plane_img, x0, y0, z_min, z_max):
    """Save the three ROI QC figures (<prefix>_roi_pairs.png, _roi_3d.png, _roi_support.png)."""
    prefix = os.path.join(out_dir, f"{section_id}")
    show_matched_pairs(
        clean_matches, all_matches["hist_img"], all_matches["mri_img"],
        patch=best_patch, ncols=2, save_file=f"{prefix}_roi_pairs.png",
        hist_mask=all_matches["hist_mask"], mri_mask=all_matches["mri_mask"],
    )
    visualize_roi_from_matches(
        matches=clean_matches, mri_vol=slab_Im, z_min=z_min, z_max=z_max, mode="mip",
        save_file=f"{prefix}_roi_3d.png",
    )
    plot_roi_support(clean_matches, scaled_proc_Ih, plane_img, x0, y0,
                     save_file=f"{prefix}_roi_support.png")
