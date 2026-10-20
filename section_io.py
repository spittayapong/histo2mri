"""Section/subject loading helpers for the publishable batch pipeline (pipeline_run.py).

Deliberately independent from annotations_run.py (which is internal-only tooling) so this
module can be published standalone. Reads the same sections-csv format
(subject_id, slice_id/block_id/stain, orientation, spacing, optional flip).
"""
import os

import numpy as np
import SimpleITK as sitk

ORIENTATION_PERMUTE = {
    "axial": [0, 1, 2],
    "coronal": [0, 2, 1],
    "sagittal": [1, 2, 0],
}


def parse_spacing(spacing_str):
    parts = [float(x) for x in spacing_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated spacing values, got: {spacing_str!r}")
    return tuple(parts)


def parse_orientation(orientation_str):
    key = orientation_str.strip().lower()
    if key not in ORIENTATION_PERMUTE:
        raise ValueError(f"Unknown orientation {orientation_str!r}; expected one of {list(ORIENTATION_PERMUTE)}")
    return key


def parse_flip(flip_str):
    """'x'/'y'/'z' letters (any combo, e.g. 'xy') -> (flip_x, flip_y, flip_z) bools for sitk.Flip.
    Blank/missing means no flip."""
    letters = set((flip_str or "").strip().lower())
    unknown = letters - {"x", "y", "z"}
    if unknown:
        raise ValueError(f"Unknown flip axis letters {sorted(unknown)} in {flip_str!r}; expected x/y/z")
    return ("x" in letters, "y" in letters, "z" in letters)


def build_num_str(row, ds_type):
    if ds_type == "NB":
        return f"{row['block_id']}_{row['stain']}_{row['slice_id']}"
    return row["slice_id"]


def load_volume_auto(base_path_no_ext, npz_key, spacing, pixel_type=None):
    """Load a volume as either .mnc (geometry trusted from the file header) or
    .npz (raw array; geometry set manually from `spacing`), whichever exists."""
    mnc_path = base_path_no_ext + ".mnc"
    npz_path = base_path_no_ext + ".npz"

    if os.path.exists(mnc_path):
        return sitk.ReadImage(mnc_path, pixel_type) if pixel_type else sitk.ReadImage(mnc_path)

    if os.path.exists(npz_path):
        arr = np.load(npz_path)[npz_key]
        is_vec = (arr.ndim == 4 and arr.shape[-1] == 3)   # RGB: (1,H,W,3) -> 3-component vector image
        img = sitk.GetImageFromArray(arr.astype(np.float32), isVector=is_vec)
        img.SetSpacing(spacing)
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        return img

    raise FileNotFoundError(f"Neither {mnc_path} nor {npz_path} exists")


def load_section(subject_id, num_str, histo_type, spacing, orientation, flip, data_root):
    """Load Ih_hr, Mh_hr, Im, Mm for one (subject, section) pair, with orientation
    permute applied to the MRI and flip applied to the histology, matching the
    conventions used across this pipeline's per-subject data layout."""
    histo_dir = os.path.join(data_root, subject_id, f"histo_{histo_type}", "gray")
    histo_mask_dir = os.path.join(data_root, subject_id, f"histo_{histo_type}", "mask")
    mri_dir = os.path.join(data_root, subject_id, "mri")

    histo_base = os.path.join(histo_dir, f"{subject_id}_{num_str}_gray")
    Ih_hr = load_volume_auto(histo_base, npz_key="gray", spacing=spacing, pixel_type=sitk.sitkFloat32)

    histo_mask_base = os.path.join(histo_mask_dir, f"{subject_id}_{num_str}_bimask")
    Mh_hr = load_volume_auto(histo_mask_base, npz_key="bimask", spacing=spacing, pixel_type=sitk.sitkFloat32)

    # RGB histology (optional; QC overlays only) -- same mnc-or-npz mechanism as gray/mask.
    histo_rgb_dir = os.path.join(data_root, subject_id, f"histo_{histo_type}", "rgb")
    rgb_base = os.path.join(histo_rgb_dir, f"{subject_id}_{num_str}_rgb")
    try:
        Ih_rgb_hr = load_volume_auto(rgb_base, npz_key="rgb", spacing=spacing)
    except FileNotFoundError:
        Ih_rgb_hr = None

    if any(flip):
        Ih_hr = sitk.Flip(Ih_hr, list(flip), flipAboutOrigin=False)
        Mh_hr = sitk.Flip(Mh_hr, list(flip), flipAboutOrigin=False)
        if Ih_rgb_hr is not None:
            Ih_rgb_hr = sitk.Flip(Ih_rgb_hr, list(flip), flipAboutOrigin=False)

    permute_axes = ORIENTATION_PERMUTE[orientation]

    mri_base = os.path.join(mri_dir, f"{subject_id}_T1")
    Im = load_volume_auto(mri_base, npz_key="T1", spacing=spacing, pixel_type=sitk.sitkFloat32)
    Im = sitk.PermuteAxes(Im, permute_axes)

    mri_mask_base = os.path.join(mri_dir, f"{subject_id}_bimask")
    Mm = load_volume_auto(mri_mask_base, npz_key="bimask", spacing=spacing)
    Mm = sitk.PermuteAxes(Mm, permute_axes)

    return {"Ih_hr": Ih_hr, "Mh_hr": Mh_hr, "Im": Im, "Mm": Mm, "Ih_rgb_hr": Ih_rgb_hr}
