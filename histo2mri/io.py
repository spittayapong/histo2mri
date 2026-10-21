"""Load one histology section and its MRI volume, and parse the fields of the sections CSV.

First step of the pipeline (pipeline.register_section). Images are returned as SimpleITK
images; arrays read from .npz files are in numpy (z, y, x) order and geometry tuples are in
sitk (x, y, z) order. The MRI axes are permuted so that the histology cutting plane lies in
the x-y plane of the returned MRI (z is the slicing axis searched by the matcher); this
assumes the stored MRI's x, y and z axes are normal to the sagittal, coronal and axial planes.
"""
import os
import numpy as np
import SimpleITK as sitk


# sitk.PermuteAxes order for each section orientation: output axis i is input axis order[i].
# axial keeps the MRI as it is; coronal swaps y and z; sagittal maps (x, y, z) <- (y, z, x).
ORIENTATION_PERMUTE = {
    "axial": [0, 1, 2],
    "coronal": [0, 2, 1],
    "sagittal": [1, 2, 0],
}


def parse_spacing(spacing_str):
    """Parse a comma-separated spacing string such as "0.1,0.1,0.1".

    Args:
        spacing_str: Three comma-separated numbers, voxel spacing in sitk (x, y, z) order.
    Returns:
        Tuple of three floats.
    Raises:
        ValueError: If there are not exactly three values.
    """
    parts = [float(x) for x in spacing_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated spacing values, got: {spacing_str!r}")
    return tuple(parts)


def parse_orientation(orientation_str):
    """Normalise an orientation name and check that it is a key of ORIENTATION_PERMUTE.

    Args:
        orientation_str: "axial", "coronal" or "sagittal" (case and surrounding spaces ignored).
    Returns:
        The lower-case orientation name.
    Raises:
        ValueError: If the name is unknown.
    """
    key = orientation_str.strip().lower()
    if key not in ORIENTATION_PERMUTE:
        raise ValueError(f"Unknown orientation {orientation_str!r}; expected one of {list(ORIENTATION_PERMUTE)}")
    return key


def parse_flip(flip_str):
    """Parse the flip column into per-axis flags for sitk.Flip.

    Args:
        flip_str: Any combination of the letters x, y, z (e.g. "xy"; case ignored). Empty
            or None means no flip.
    Returns:
        (flip_x, flip_y, flip_z) booleans in sitk axis order.
    Raises:
        ValueError: If flip_str contains other letters.
    """
    letters = set((flip_str or "").strip().lower())
    unknown = letters - {"x", "y", "z"}
    if unknown:
        raise ValueError(f"Unknown flip axis letters {sorted(unknown)} in {flip_str!r}; expected x/y/z")
    return ("x" in letters, "y" in letters, "z" in letters)


def build_num_str(row, ds_type):
    """Build the section identifier used in file and folder names from a sections-CSV row.

    Args:
        row: Dict of one CSV row (column name -> string).
        ds_type: "NB" or "internal" (any value other than "NB" is treated as internal).
    Returns:
        For "NB": "<block_id>_<stain>_<slice_id>"; otherwise the row's slice_id.
    Raises:
        KeyError: If a required column is missing.
    """
    if ds_type == "NB":
        return f"{row['block_id']}_{row['stain']}_{row['slice_id']}"
    return row["slice_id"]


def load_volume_auto(base_path_no_ext, npz_key, spacing, pixel_type=None):
    """Load an image from <base>.mnc or, if that does not exist, from <base>.npz.

    A .mnc file is read with its header geometry (spacing, origin, direction). A .npz file
    gives a raw array: it is converted to float32 and given the spacing `spacing`, origin 0
    and identity direction. A 4D array whose last axis has length 3 (RGB, (1, H, W, 3)) is
    loaded as a 3-component vector image.

    Args:
        base_path_no_ext: Path without the extension.
        npz_key: Key of the array in the .npz file.
        spacing: (x, y, z) spacing set on images loaded from .npz.
        pixel_type: Optional sitk pixel type for reading .mnc; ignored for .npz.
    Returns:
        sitk.Image.
    Raises:
        FileNotFoundError: If neither file exists.
    """
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
    """Load the HR histology, its mask, the optional RGB histology, the MRI and the MRI mask.

    Expected layout (each file as .mnc, or as .npz with the array under the given key; .mnc
    is used when both exist):

        data_root/<subject_id>/histo_<histo_type>/gray/<subject_id>_<num_str>_gray     key "gray"
        data_root/<subject_id>/histo_<histo_type>/mask/<subject_id>_<num_str>_bimask   key "bimask"
        data_root/<subject_id>/histo_<histo_type>/rgb/<subject_id>_<num_str>_rgb       key "rgb" (optional)
        data_root/<subject_id>/mri/<subject_id>_T1                                    key "T1"
        data_root/<subject_id>/mri/<subject_id>_bimask                                key "bimask"

    .npz histology arrays are expected as (1, H, W) (RGB: (1, H, W, 3)), MRI arrays as (Z, Y, X). The
    histology gray image and mask, and the MRI, are read as float32; .mnc MRI masks and
    RGB images keep their stored pixel type. The flip is applied to the three histology
    images (about the image centre, so the physical extent is unchanged); the orientation
    permutation (ORIENTATION_PERMUTE) is applied to the MRI and its mask.

    Args:
        subject_id: Subject folder name; also the file-name prefix.
        num_str: Section identifier (see build_num_str).
        histo_type: Histology type, selects histo_<histo_type>.
        spacing: (x, y, z) spacing given to every image read from .npz (histology and MRI).
        orientation: Key of ORIENTATION_PERMUTE.
        flip: (flip_x, flip_y, flip_z) booleans (see parse_flip).
        data_root: Root data folder.
    Returns:
        Dict with sitk images "Ih_hr" (HR histology), "Mh_hr" (HR histology mask), "Im"
        (MRI), "Mm" (MRI mask) and "Ih_rgb_hr" (RGB histology, or None if no file exists).
    """
    histo_dir = os.path.join(data_root, subject_id, f"histo_{histo_type}", "gray")
    histo_mask_dir = os.path.join(data_root, subject_id, f"histo_{histo_type}", "mask")
    mri_dir = os.path.join(data_root, subject_id, "mri")

    histo_base = os.path.join(histo_dir, f"{subject_id}_{num_str}_gray")
    Ih_hr = load_volume_auto(histo_base, npz_key="gray", spacing=spacing, pixel_type=sitk.sitkFloat32)

    histo_mask_base = os.path.join(histo_mask_dir, f"{subject_id}_{num_str}_bimask")
    Mh_hr = load_volume_auto(histo_mask_base, npz_key="bimask", spacing=spacing, pixel_type=sitk.sitkFloat32)

    # RGB histology is optional and only used for QC overlays.
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

