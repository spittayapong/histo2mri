"""Resample histology to MRI spacing and prepare histology and MRI-slab images for matching."""
import numpy as np
import SimpleITK as sitk
from skimage import morphology
from skimage.morphology import ball
from .qc import show_proc_image


def voxel_index_affine_hr_to_lr(spacing_hr, spacing_lr):
    """
    HR voxel index -> LR voxel index (homogeneous 4x4).
    Assumes same origin/direction and only spacing changes.
    """
    spacing_hr = np.asarray(spacing_hr, dtype=np.float64)
    spacing_lr = np.asarray(spacing_lr, dtype=np.float64)

    scale = spacing_hr / spacing_lr  # HR index -> LR index

    A = np.eye(4, dtype=np.float64)
    A[0, 0] = scale[0]
    A[1, 1] = scale[1]
    A[2, 2] = scale[2]
    return A


def resample_histo_to_mri(Ih_hr, Mh_hr, Im):
    hist_size = Ih_hr.GetSize()
    hist_spacing = Ih_hr.GetSpacing()
    mri_spacing = Im.GetSpacing()

    print("\n-------- Original Image Size --------")
    print("MRI spacing:", mri_spacing)
    print("Histology size:", hist_size)
    print("Histology spacing:", hist_spacing)

    # Compute new size so that physical size stays identical
    hist_size_new = [
        int(np.ceil(hist_size[i] * (hist_spacing[i] / mri_spacing[i])))
        for i in range(3)
    ]

    # For moving mri back to high resolution histology space
    A_hr_to_lr = voxel_index_affine_hr_to_lr(hist_spacing, mri_spacing)

    # Resample histology to MRI spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(Ih_hr)
    resampler.SetOutputSpacing(mri_spacing)
    resampler.SetSize(hist_size_new)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(Ih_hr.GetOrigin())
    resampler.SetOutputDirection(Ih_hr.GetDirection())
    Ih = resampler.Execute(Ih_hr)

    # Resample histology mask to MRI spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(Mh_hr)
    resampler.SetOutputSpacing(mri_spacing)
    resampler.SetSize(hist_size_new)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(Mh_hr.GetOrigin())
    resampler.SetOutputDirection(Mh_hr.GetDirection())
    Mh = resampler.Execute(Mh_hr)

    return Ih, Mh, A_hr_to_lr


# ------------------ InStep : Histology Image preparation ------------------ #
# This step performs after finding approximately MRI slice
# Prepare histology for registration:
# - crop to mask
# - scale shrikaged tissue
# - normalize robust intensity
# - pad to square
def robust_intensity_prep(I, mask=None, pclip=(1, 99), eps=1e-8):
    """
    Robustly normalize a 2D MRI projection to [0,1].

    Supports:
      - NumPy arrays (returns NumPy array)
      - SimpleITK.Image (returns SimpleITK.Image with same origin/spacing/direction)

    Foreground:
      - If mask is None: I > 0 (computed in NumPy space)
      - If mask is provided:
          * NumPy array mask, or
          * SimpleITK.Image mask (will be converted to NumPy and treated as boolean)

    Returns:
        I01        : float32 image in [0,1] (NumPy or SimpleITK.Image)
        stats      : dict with 'lo', 'hi', 'mean', 'std', 'pclip'
    """
    # --- Detect SimpleITK input (without hard dependency if you import elsewhere)
    is_sitk = hasattr(I, "GetSize") and hasattr(I, "GetSpacing") and hasattr(I, "GetOrigin")

    if is_sitk:
        import SimpleITK as sitk
        I_sitk = I
        I_np = sitk.GetArrayFromImage(I_sitk).astype(np.float32)  # (z,y,x) for 3D, (y,x) for 2D
        meta = {
            "origin": I_sitk.GetOrigin(),
            "spacing": I_sitk.GetSpacing(),
            "direction": I_sitk.GetDirection(),
        }
    else:
        I_sitk = None
        I_np = np.asarray(I).astype(np.float32)
        meta = None

    # --- Prepare / convert mask
    if mask is None:
        mask_np = I_np > 0
    else:
        if hasattr(mask, "GetSize") and hasattr(mask, "GetSpacing") and hasattr(mask, "GetOrigin"):
            import SimpleITK as sitk
            mask_np = sitk.GetArrayFromImage(mask).astype(bool)
        else:
            mask_np = np.asarray(mask).astype(bool)

    # --- Clean non-finite
    I_np = I_np.copy()
    I_np[~np.isfinite(I_np)] = 0

    # --- Robust percentiles within mask (fallback if empty)
    has_fg = np.any(mask_np)
    vals = I_np[mask_np] if has_fg else I_np.ravel()

    if vals.size == 0:
        lo, hi = float(I_np.min()), float(I_np.max())
    else:
        lo, hi = np.percentile(vals, pclip)
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())

    # --- Clip + rescale to [0,1]
    I01_np = np.zeros_like(I_np, dtype=np.float32)
    denom = (hi - lo + eps)

    if has_fg:
        I01_np[mask_np] = np.clip((I_np[mask_np] - lo) / denom, 0, 1)
    else:
        I01_np = np.clip((I_np - lo) / denom, 0, 1).astype(np.float32)

    stats = {
        "lo": float(lo),
        "hi": float(hi),
        "mean": float(vals.mean()) if vals.size else float(I_np.mean()),
        "std": float(vals.std(ddof=0)) if vals.size else float(I_np.std(ddof=0)),
        "pclip": pclip,
    }

    # --- Convert back to SimpleITK.Image if needed (preserve metadata)
    if is_sitk:
        import SimpleITK as sitk
        out = sitk.GetImageFromArray(I01_np)  # matches array dimensionality
        out.SetOrigin(meta["origin"])
        out.SetSpacing(meta["spacing"])
        out.SetDirection(meta["direction"])
        return out

    return I01_np


def scale_image_xy_grid(img, sx, sy, interp=sitk.sitkLinear):
    """
    Resamples onto a new grid by changing size/spacing (identity transform).
    Z size/spacing unchanged.
    Returns: (img_scaled, scaled_size_xyz)
    """
    size = np.array(img.GetSize(), dtype=int)          # (x,y,z)
    spacing = np.array(img.GetSpacing(), dtype=float)  # (sx,sy,sz)

    # size after scaling (NOTE: truncation happens here)
    scaled_size = np.maximum(1, (size * np.array([sx, sy, 1.0])).astype(int))

    new_spacing = spacing / np.array([sx, sy, 1.0])

    img_scaled = sitk.Resample(
        img,
        scaled_size.tolist(),
        sitk.Transform(),          # identity
        interp,
        img.GetOrigin(),
        new_spacing.tolist(),
        img.GetDirection(),
        0,
        img.GetPixelID()
    )
    return img_scaled, scaled_size


def pad_xy_to_patch_multiple(img, patch_size, constant=0):
    """
    Pads ONLY X and Y to be divisible by patch_size. Z unchanged.
    Returns: (padded_img, pad_lower_xyz, pad_upper_xyz)
    """
    size = np.array(img.GetSize(), dtype=int)  # (x,y,z)

    target = size.copy()
    target[0] = int(np.ceil(size[0] / patch_size) * patch_size)
    target[1] = int(np.ceil(size[1] / patch_size) * patch_size)
    target[2] = size[2]

    pad_total = target - size
    pad_lower = (pad_total // 2).astype(int)
    pad_upper = (pad_total - pad_lower).astype(int)

    pad_lower[2] = 0
    pad_upper[2] = 0

    padded = sitk.ConstantPad(
        img,
        padLowerBound=pad_lower.tolist(),
        padUpperBound=pad_upper.tolist(),
        constant=constant
    )
    return padded, pad_lower, pad_upper


def affine_lr_to_proc_voxel(Ih_lr, scaled_size_xyz, pad_lower_xyz):
    """
    Returns A_lr_to_proc (4x4) in VOXEL (index) space such that:
      [i_proc] = A_lr_to_proc * [i_lr]

    Your pipeline:
      i_scaled ≈ S * i_lr
      i_proc   = i_scaled + pad_lower

    We use effective scales computed from integer sizes to match what was produced.
    """
    lr_size = np.array(Ih_lr.GetSize(), dtype=float)        # pre-scale
    scaled_size = np.array(scaled_size_xyz, dtype=float)    # post-scale, pre-pad
    pad_lower = np.array(pad_lower_xyz, dtype=float)

    s_eff = scaled_size / lr_size
    s_eff[2] = 1.0
    pad_lower[2] = 0.0

    A = np.eye(4, dtype=float)
    A[0, 0] = s_eff[0]
    A[1, 1] = s_eff[1]
    A[2, 2] = 1.0
    A[0, 3] = pad_lower[0]
    A[1, 3] = pad_lower[1]
    A[2, 3] = 0.0
    return A


def prepare_histology(Ih, Mh, scale_ratio, patch_size, show_proc_img=False, save_file=None):
    print("\n-------- Prepare Histology for Registration --------")

    sy, sx = scale_ratio

    # 1) scale LR (grid change) in XY only
    Ih_scaled, scaled_size = scale_image_xy_grid(Ih, sx, sy)
    Mh_scaled, _ = scale_image_xy_grid(Mh, sx, sy, interp=sitk.sitkNearestNeighbor)

    # 2) pad XY to patch multiple
    pad_Ih, pad_lower, pad_upper = pad_xy_to_patch_multiple(Ih_scaled, patch_size, constant=0)
    proc_Mh, _, _ = pad_xy_to_patch_multiple(Mh_scaled, patch_size, constant=0)

    # 3) LR->proc affine in voxel space (matches steps 1+2)
    A_lr_to_proc = affine_lr_to_proc_voxel(Ih, scaled_size, pad_lower)

    proc_Ih = pad_Ih

    if show_proc_img:
        show_proc_image(proc_Ih, proc_Mh, title="Histo", idx=0, save_file=save_file)

    return proc_Ih, proc_Mh, A_lr_to_proc


# ------------------ MRI Image preparation ------------------ #
# Prepare MRI for registration:
# - extract MRI slab (+/- n_slices around max_index)
# - normalize robust intensity

def dilate_mask(mri_mask, ball_size=3):
    mri_mask_array = sitk.GetArrayFromImage(mri_mask)

    selem3d = ball(ball_size)  # radius in voxels
    mri_mask_array_dilated = morphology.binary_dilation(mri_mask_array, footprint=selem3d)
    mri_mask_bool = mri_mask_array_dilated.astype(np.uint8)

    # Create a SimpleITK image from the numpy array
    dilated_mask_image = sitk.GetImageFromArray(mri_mask_bool)

    # Copy the header information (spacing, origin, direction) from the original image
    dilated_mask_image.SetSpacing(mri_mask.GetSpacing())
    dilated_mask_image.SetOrigin(mri_mask.GetOrigin())
    dilated_mask_image.SetDirection(mri_mask.GetDirection())

    return dilated_mask_image


def prepare_mri_slab(Im, Mm, n_slices, show_proc_img=False, save_file=None, pclip=(1, 99)):
    print("\n-------- Prepare MRI Slab for Registration --------")

    slab_Mm = dilate_mask(Mm, ball_size=1)
    slab_raw_Im = Im
    slab_Im = robust_intensity_prep(Im, slab_Mm, pclip=pclip)

    if show_proc_img:
        show_proc_image(slab_Im, slab_Mm, idx=n_slices, title="MRI", save_file=save_file)

    return slab_raw_Im, slab_Im, slab_Mm

