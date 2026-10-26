"""Resample histology to MRI spacing and prepare the histology and MRI slab for matching.

First stage after loading. Histology goes HR (full resolution) -> LR (resampled
to MRI spacing) -> proc (scaled in-plane and padded to a multiple of the patch
size). The voxel-index affines between these frames are 4x4 homogeneous
matrices acting on sitk index order (x, y, z, 1):
    i_lr   = A_hr_to_lr   @ i_hr    (diagonal: spacing_hr / spacing_lr)
    i_proc = A_lr_to_proc @ i_lr    (diagonal in-plane scale + lower padding)
The pipeline composes them as A_hr_to_proc = A_lr_to_proc @ A_hr_to_lr, which
the inverse stage uses to map results back to HR histology.
"""
import numpy as np
import SimpleITK as sitk
from skimage import morphology
from skimage.morphology import ball
from .qc import show_proc_image
import logging

logger = logging.getLogger(__name__)


def voxel_index_affine_hr_to_lr(spacing_hr, spacing_lr):
    """Build the HR -> LR voxel-index affine for a pure spacing change.

    Assumes both grids share origin and direction, so index 0 maps to index 0
    and each axis scales by spacing_hr / spacing_lr.

    Args:
        spacing_hr: (x, y, z) spacing of the HR grid.
        spacing_lr: (x, y, z) spacing of the LR grid.

    Returns:
        4x4 float64 diagonal affine mapping homogeneous (x, y, z, 1) HR indices
        to LR indices.
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
    """Resample HR histology and its mask to the MRI voxel spacing.

    The output keeps the histology origin and direction, takes the MRI
    spacing, and has size ceil(size_hr * spacing_hr / spacing_mri) so the
    physical extent is preserved. The image uses linear interpolation (no
    anti-aliasing smoothing is applied) and the mask nearest-neighbour.

    Args:
        Ih_hr: HR histology, 3D sitk.Image (x, y, z), usually z size 1.
        Mh_hr: HR histology tissue mask, same grid as Ih_hr.
        Im: MRI volume, sitk.Image; only its spacing is used.

    Returns:
        Tuple (Ih, Mh, A_hr_to_lr): LR histology and mask (sitk.Image), and the
        4x4 HR -> LR voxel-index affine.
    """
    hist_size = Ih_hr.GetSize()
    hist_spacing = Ih_hr.GetSpacing()
    mri_spacing = Im.GetSpacing()

    logger.debug('-------- Original Image Size --------')
    logger.debug('MRI spacing: %s', mri_spacing)
    logger.debug('Histology size: %s', hist_size)
    logger.debug('Histology spacing: %s', hist_spacing)

    # Compute new size so that physical size stays identical
    hist_size_new = [
        int(np.ceil(hist_size[i] * (hist_spacing[i] / mri_spacing[i])))
        for i in range(3)
    ]

    # Kept so results can be mapped back to HR histology later
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


# ------------------ Histology preparation ------------------ #
def robust_intensity_prep(I, mask=None, pclip=(1, 99), eps=1e-8):
    """Rescale an image to [0, 1] between robust percentiles of its foreground.

    Non-finite values are set to 0. The lo/hi percentiles (pclip) are taken
    over the foreground (mask, or I > 0 if mask is None; all pixels if the
    foreground is empty); if they coincide, the foreground min/max are used.
    Foreground pixels are rescaled and clipped to [0, 1]; background pixels are
    set to 0. Works on 2D or 3D input; the pipeline uses it on the MRI volume.

    Args:
        I: image, numpy array or sitk.Image.
        mask: optional foreground mask, numpy array or sitk.Image, same shape
            as I.
        pclip: (low, high) percentiles.
        eps: added to the denominator.

    Returns:
        float32 image in [0, 1]: a sitk.Image with the input origin, spacing and
        direction if I is a sitk.Image, else a numpy array. Summary statistics
        (lo, hi, mean, std) are computed internally but not returned.
    """
    # --- Detect SimpleITK input by duck typing
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
    """Rescale an image in x and y by changing its grid size and spacing.

    Resamples with the identity transform onto a grid with the same origin and
    direction, spacing / (sx, sy, 1) and size int(size * (sx, sy, 1)), so
    pixel index i maps to roughly i * s and z is unchanged. Pixels outside the
    input are 0.

    Args:
        img: 3D sitk.Image.
        sx: scale factor along x.
        sy: scale factor along y.
        interp: sitk interpolator.

    Returns:
        Tuple (img_scaled, scaled_size): resampled sitk.Image and its (x, y, z)
        size as an int array (truncated, at least 1).
    """
    size = np.array(img.GetSize(), dtype=int)          # (x,y,z)
    spacing = np.array(img.GetSpacing(), dtype=float)  # (sx,sy,sz)

    # Size after scaling; int() truncates.
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
    """Pad x and y with a constant so both sizes are multiples of patch_size.

    Padding is split evenly between the lower and upper side (the extra pixel,
    if any, goes to the upper side); z is not padded.

    Args:
        img: 3D sitk.Image.
        patch_size: patch edge length in pixels.
        constant: pad value.

    Returns:
        Tuple (padded, pad_lower, pad_upper): padded sitk.Image and the
        (x, y, z) int arrays of pixels added below and above.
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
    """Build the LR -> proc voxel-index affine.

    Models i_scaled ~= S * i_lr (scale_image_xy_grid) followed by
    i_proc = i_scaled + pad_lower (pad_xy_to_patch_multiple), so
    i_proc = A_lr_to_proc @ i_lr. S uses the effective scales
    scaled_size / lr_size from the integer image sizes; z is unchanged.

    Args:
        Ih_lr: LR histology before scaling, sitk.Image.
        scaled_size_xyz: (x, y, z) size after scaling, before padding.
        pad_lower_xyz: (x, y, z) lower padding in pixels.

    Returns:
        4x4 float affine mapping homogeneous (x, y, z, 1) LR indices to proc
        indices.
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
    """Scale the LR histology in-plane and pad it to a multiple of the patch size.

    The image is scaled with linear and the mask with nearest-neighbour
    interpolation (scale_image_xy_grid), then both are zero-padded in x and y
    (pad_xy_to_patch_multiple). Intensities are not normalised here.

    Args:
        Ih: LR histology (at MRI spacing), sitk.Image.
        Mh: LR histology mask, sitk.Image.
        scale_ratio: (sy, sx) in-plane scale factors; the pipeline uses (1, 1).
        patch_size: patch edge length in pixels.
        show_proc_img: if True, show the prepared image and mask.
        save_file: optional output path for that figure.

    Returns:
        Tuple (proc_Ih, proc_Mh, A_lr_to_proc): prepared histology and mask
        (sitk.Image) and the 4x4 LR -> proc voxel-index affine.
    """
    logger.debug('-------- Prepare Histology for Registration --------')

    sy, sx = scale_ratio

    # 1) scale LR (grid change) in XY only
    Ih_scaled, scaled_size = scale_image_xy_grid(Ih, sx, sy)
    Mh_scaled, _ = scale_image_xy_grid(Mh, sx, sy, interp=sitk.sitkNearestNeighbor)

    # 2) pad XY to patch multiple
    pad_Ih, pad_lower, pad_upper = pad_xy_to_patch_multiple(Ih_scaled, patch_size, constant=0)
    proc_Mh, _, _ = pad_xy_to_patch_multiple(Mh_scaled, patch_size, constant=0)

    # 3) LR -> proc voxel-index affine (steps 1 and 2)
    A_lr_to_proc = affine_lr_to_proc_voxel(Ih, scaled_size, pad_lower)

    proc_Ih = pad_Ih

    if show_proc_img:
        show_proc_image(proc_Ih, proc_Mh, title="Histo", idx=0, save_file=save_file)

    return proc_Ih, proc_Mh, A_lr_to_proc


# ------------------ MRI preparation ------------------ #

def dilate_mask(mri_mask, ball_size=3):
    """Dilate a 3D mask with a ball structuring element.

    Args:
        mri_mask: 3D sitk.Image mask.
        ball_size: ball radius in voxels.

    Returns:
        uint8 sitk.Image with the dilated mask and the input spacing, origin
        and direction.
    """
    mri_mask_array = sitk.GetArrayFromImage(mri_mask)

    selem3d = ball(ball_size)  # radius in voxels
    mri_mask_array_dilated = morphology.binary_dilation(mri_mask_array, footprint=selem3d)
    mri_mask_bool = mri_mask_array_dilated.astype(np.uint8)

    dilated_mask_image = sitk.GetImageFromArray(mri_mask_bool)

    # Copy the header information (spacing, origin, direction) from the original image
    dilated_mask_image.SetSpacing(mri_mask.GetSpacing())
    dilated_mask_image.SetOrigin(mri_mask.GetOrigin())
    dilated_mask_image.SetDirection(mri_mask.GetDirection())

    return dilated_mask_image


def prepare_mri_slab(Im, Mm, n_slices, show_proc_img=False, save_file=None, pclip=(1, 99)):
    """Prepare the MRI volume for matching (the "slab").

    The whole volume is kept (no slices are cropped). The mask is dilated by a
    ball of radius 1 voxel, and the intensities are rescaled to [0, 1] between
    the pclip percentiles inside the dilated mask, with zeros outside it
    (robust_intensity_prep).

    Args:
        Im: MRI volume, sitk.Image.
        Mm: MRI mask, sitk.Image.
        n_slices: slice index shown in the optional display; not used
            otherwise.
        show_proc_img: if True, show the prepared slab and mask.
        save_file: optional output path for that figure.
        pclip: (low, high) intensity percentiles.

    Returns:
        Tuple (slab_raw_Im, slab_Im, slab_Mm): the unmodified MRI, the
        normalised MRI and the dilated mask, all sitk.Image.
    """
    logger.debug('-------- Prepare MRI Slab for Registration --------')

    slab_Mm = dilate_mask(Mm, ball_size=1)
    slab_raw_Im = Im
    slab_Im = robust_intensity_prep(Im, slab_Mm, pclip=pclip)

    if show_proc_img:
        show_proc_image(slab_Im, slab_Mm, idx=n_slices, title="MRI", save_file=save_file)

    return slab_raw_Im, slab_Im, slab_Mm

