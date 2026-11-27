"""Load one histology section and its MRI volume, from a BIDS dataset or from explicit files.

First step of the pipeline: load_section (pipeline.register_section) finds the files and
sidecar values in a BIDS dataset and hands them to load_files (pipeline.register_files),
which takes them directly. Images are returned as SimpleITK
images; arrays read from the histology TIFFs are promoted to numpy (z, y, x) order with a
single slice, and geometry tuples are in sitk (x, y, z) order. The MRI axes are permuted so
that the histology cutting plane lies in the x-y plane of the returned MRI (z is the slicing
axis searched by the matcher); the MRI is reoriented to LPS first, so the permutation means
the same thing whatever orientation the stored file happens to use, and the two in-plane axes
are then turned to the frame a viewer displays (VIEW_AXES), so that SectionRotation and
SectionFlip describe what the person filling in the sidecar sees on screen.

Expected layout, with <sub> = sub-<subject> and <base> = <sub>_sample-<sample>_stain-<stain>:

    <bids_root>/<sub>/micr/<base>_BF.json                             section metadata
    <bids_root>/<sub>/anat/<sub>_T1w.nii.gz                           MRI
    <bids_root>/derivatives/brainmask/<sub>/anat/<sub>_desc-brain_mask.nii.gz
    <bids_root>/derivatives/cleanslide/<sub>/micr/<base>_desc-clean_BF.tif    gray histology
    <bids_root>/derivatives/cleanslide/<sub>/micr/<base>_desc-clean_BF.json   its pixel size
    <bids_root>/derivatives/cleanslide/<sub>/micr/<base>_desc-tissue_mask.tif
    <bids_root>/derivatives/cleanslide/<sub>/micr/<base>_desc-cleanrgb_BF.tif (optional)

The histology spacing is read from the cleanslide sidecar (PixelSize, PixelSizeUnits) rather
than passed in, because that sidecar describes the cropped image actually loaded.

Three fields of the raw micr sidecar place the section in the MRI frame. They are applied in
this order, which is the order the angles of this dataset were measured with:

    SectionRotation     degrees, positive counter-clockwise (the cv2.getRotationMatrix2D
                        convention), the in-plane angle between slide and MRI. null means 0.
    SectionFlip         any of the letters x/y/z, or "" for none. "y" mirrors top-to-bottom
                        (np.flipud), "x" left-to-right (np.fliplr); "xy" is a 180 degree
                        rotation and so does *not* mirror.
    SectionOrientation  "axial", "coronal" or "sagittal", the cutting plane, which selects
                        the MRI axis permutation.

Rotation and flip must be right before matching: the patch matcher (HOG and MIND on fixed
orientations) is not rotation invariant, and plane.fit_global_plane forces its in-plane
linear part back to a proper rotation, so a mirror cannot be recovered downstream.
"""
import glob
import json
import math
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

# BIDS derivative pipelines this loader reads from.
BIDS_HISTO_SOURCE = "cleanslide"
BIDS_MRI_MASK_SOURCE = "brainmask"

# The MRI is brought to this anatomical orientation before ORIENTATION_PERMUTE is applied.
# The stored files are not consistent (LAI and RAS both occur, and the MINC data this
# pipeline was tuned on was LPS), and they differ by axis *signs*, so without this the same
# SectionFlip would mean different things for different subjects.
MRI_CANONICAL_ORIENTATION = "LPS"

# Where the in-plane axes point after the permutation: (+x, +y) = (screen-right, screen-down)
# of the radiological display FSLeyes and MRIcron show by default. The canonical frame alone
# leaves +y pointing Superior for a coronal plane, while image rows run downward, so the array
# is upside down with respect to every viewer and SectionFlip would have to carry a "y" that
# nobody filling in the sidecar can see. load_mri reverses the in-plane axes that disagree.
VIEW_AXES = {
    "axial": ("L", "P"),
    "coronal": ("L", "I"),
    "sagittal": ("A", "I"),
}

OPPOSITE_DIRECTION = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}

# PixelSizeUnits spellings accepted in a cleanslide sidecar, as a factor to millimetres.
PIXEL_SIZE_UNITS_TO_MM = {
    "um": 1e-3,
    "µm": 1e-3,
    "micron": 1e-3,
    "microns": 1e-3,
    "mm": 1.0,
    "millimeter": 1.0,
    "millimetre": 1.0,
}

_IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def parse_spacing(spacing_str):
    """Parse a comma-separated spacing string such as "0.1,0.1,0.1".

    Only needed to override the spacing a cleanslide sidecar gives; load_section reads it
    from the sidecar when no override is passed.

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
        ValueError: If the name is unknown or missing.
    """
    if orientation_str is None:
        raise ValueError(f"No section orientation given; expected one of {list(ORIENTATION_PERMUTE)}")
    key = str(orientation_str).strip().lower()
    if key not in ORIENTATION_PERMUTE:
        raise ValueError(f"Unknown orientation {orientation_str!r}; expected one of {list(ORIENTATION_PERMUTE)}")
    return key


def parse_flip(flip_str):
    """Parse the flip field into per-axis flags for sitk.Flip.

    Args:
        flip_str: Any combination of the letters x, y, z (e.g. "xy"; case ignored). Empty
            means no flip; None is rejected, because "not recorded" and "no flip" are
            different and only the second one is safe to assume.
    Returns:
        (flip_x, flip_y, flip_z) booleans in sitk axis order.
    Raises:
        ValueError: If flip_str is None or contains other letters.
    """
    if flip_str is None:
        raise ValueError('No section flip given; expected letters from "xyz", or "" for no flip')
    letters = set(str(flip_str).strip().lower())
    unknown = letters - {"x", "y", "z"}
    if unknown:
        raise ValueError(f"Unknown flip axis letters {sorted(unknown)} in {flip_str!r}; expected x/y/z")
    return ("x" in letters, "y" in letters, "z" in letters)


def parse_rotation(rotation):
    """Parse the in-plane rotation in degrees.

    Unlike the flip, a missing value has a safe meaning: every section of this dataset
    states its angle outright, and null is recorded as "no in-plane correction".

    Args:
        rotation: Number of degrees, positive counter-clockwise, or None for no rotation.
    Returns:
        Float degrees.
    Raises:
        ValueError: If rotation is not a number.
    """
    if rotation is None:
        return 0.0
    if isinstance(rotation, bool) or not isinstance(rotation, (int, float)):
        raise ValueError(f"Section rotation {rotation!r} is not a number of degrees")
    return float(rotation)


def _entity(value, prefix):
    """Return value carrying the BIDS entity prefix, adding it if it is not already there."""
    text = str(value).strip()
    return text if text.startswith(prefix) else prefix + text


def resolve_section(bids_root, subject, sample, stain=None):
    """Find the files of one section in a BIDS dataset.

    Args:
        bids_root: Dataset root (the folder holding dataset_description.json).
        subject: Subject label, with or without the "sub-" prefix.
        sample: Sample label, with or without the "sample-" prefix.
        stain: Stain label, with or without the "stain-" prefix. May be omitted when the
            subject and sample identify exactly one section.
    Returns:
        Dict of paths with the keys "histo_gray", "histo_mask", "histo_rgb",
        "histo_sidecar", "section_sidecar", "mri", "mri_mask", plus "subject", "sample",
        "stain" and "base" (the shared file-name stem).
    Raises:
        FileNotFoundError: If no section matches.
        ValueError: If several stains match and none was asked for.
    """
    sub = _entity(subject, "sub-")
    samp = _entity(sample, "sample-")
    histo_dir = os.path.join(bids_root, "derivatives", BIDS_HISTO_SOURCE, sub, "micr")

    suffix = "_desc-clean_BF.tif"
    candidates = sorted(glob.glob(os.path.join(histo_dir, f"{sub}_{samp}_stain-*{suffix}")))
    if stain is not None:
        wanted = _entity(stain, "stain-")
        candidates = [p for p in candidates if f"_{wanted}_" in os.path.basename(p)]

    if not candidates:
        asked = f" with {_entity(stain, 'stain-')}" if stain is not None else ""
        raise FileNotFoundError(f"No {BIDS_HISTO_SOURCE} section for {sub} {samp}{asked} in {histo_dir}")
    if len(candidates) > 1:
        stains = [os.path.basename(p).split("_stain-")[1][: -len(suffix)] for p in candidates]
        raise ValueError(f"{sub} {samp} has several stains {stains}; pass stain= to choose one")

    base = os.path.basename(candidates[0])[: -len(suffix)]
    found_stain = base.split("_stain-")[1]
    return {
        "histo_gray": candidates[0],
        "histo_mask": os.path.join(histo_dir, f"{base}_desc-tissue_mask.tif"),
        "histo_rgb": os.path.join(histo_dir, f"{base}_desc-cleanrgb_BF.tif"),
        "histo_sidecar": os.path.join(histo_dir, f"{base}_desc-clean_BF.json"),
        "section_sidecar": os.path.join(bids_root, sub, "micr", f"{base}_BF.json"),
        "mri": os.path.join(bids_root, sub, "anat", f"{sub}_T1w.nii.gz"),
        "mri_mask": os.path.join(bids_root, "derivatives", BIDS_MRI_MASK_SOURCE, sub, "anat",
                                 f"{sub}_desc-brain_mask.nii.gz"),
        "subject": sub,
        "sample": samp,
        "stain": found_stain,
        "base": base,
    }


def write_dataset_description(out_dir, version):
    """The dataset_description.json BIDS requires of every derivative.

    Args:
        out_dir: Root of the derivative being written; created if missing.
        version: Version of this package, recorded under GeneratedBy.
    Returns:
        Path of the file written.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dataset_description.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "Name": "histo2mri: 2D histology to 3D MRI registration",
            "BIDSVersion": "1.10.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "histo2mri", "Version": version}],
        }, handle, indent=2)
    return path


def find_sections(bids_root, subject=None, sample=None, stain=None):
    """List the sections of a dataset, narrowed by whichever entities are given.

    The dataset itself is the list of what there is to process; each entity left as None
    matches everything.

    Args:
        bids_root: Dataset root.
        subject: Subject label, with or without the "sub-" prefix, or None for all.
        sample: Sample label, with or without the "sample-" prefix, or None for all.
        stain: Stain label, with or without the "stain-" prefix, or None for all.
    Returns:
        List of dicts with "subject", "sample", "stain" and "base", sorted by file name.
    """
    sub = _entity(subject, "sub-") if subject else "sub-*"
    samp = _entity(sample, "sample-") if sample else "sample-*"
    st = _entity(stain, "stain-") if stain else "stain-*"

    suffix = "_desc-clean_BF.tif"
    pattern = os.path.join(bids_root, "derivatives", BIDS_HISTO_SOURCE, sub, "micr",
                           f"{sub}_{samp}_{st}{suffix}")

    sections = []
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)[: -len(suffix)]
        subject_id, sample_id, stain_id = base.split("_")[:3]
        sections.append({
            "subject": subject_id[len("sub-"):],
            "sample": sample_id[len("sample-"):],
            "stain": stain_id[len("stain-"):],
            "base": base,
        })
    return sections


def _read_json(path, what):
    """Read a JSON sidecar, naming what it was needed for if it is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_pixel_size_mm(sidecar_path):
    """Read the histology pixel size from a cleanslide sidecar, in millimetres.

    A section is one slice thick, so the file records no z size; the in-plane x value is
    used for z, as the MINC histology this pipeline was built on did.

    Args:
        sidecar_path: Path of the <base>_desc-clean_BF.json sidecar.
    Returns:
        (x, y, z) spacing in mm.
    Raises:
        FileNotFoundError: If the sidecar does not exist.
        ValueError: If PixelSize or PixelSizeUnits is missing or not understood.
    """
    sidecar = _read_json(sidecar_path, "Histology sidecar")

    pixel_size = sidecar.get("PixelSize")
    if not isinstance(pixel_size, (list, tuple)) or len(pixel_size) < 2:
        raise ValueError(f"{sidecar_path}: PixelSize {pixel_size!r} is not a list of at least two numbers")

    units = str(sidecar.get("PixelSizeUnits", "")).strip().lower()
    if units not in PIXEL_SIZE_UNITS_TO_MM:
        raise ValueError(f"{sidecar_path}: PixelSizeUnits {sidecar.get('PixelSizeUnits')!r} is not one of "
                         f"{sorted(PIXEL_SIZE_UNITS_TO_MM)}")

    to_mm = PIXEL_SIZE_UNITS_TO_MM[units]
    sx, sy = float(pixel_size[0]) * to_mm, float(pixel_size[1]) * to_mm
    return (sx, sy, sx)


def read_crop_metadata(sidecar_path):
    """Where the cleanslide image sits inside the raw slide it was cropped from.

    cleanslide crops to the tissue, so a clean-TIFF pixel is offset from the raw _BF.tif
    pixel of the same tissue. Its sidecar records the offset and the raw shape, which is
    what maps a result back onto the original slide or onto annotations drawn on it.

    Args:
        sidecar_path: Path of the <base>_desc-clean_BF.json sidecar.
    Returns:
        Dict with "crop_origin_xy" (x, y) and "source_shape_hw" (height, width); both None
        when the sidecar records no crop.
    Raises:
        FileNotFoundError: If the sidecar does not exist.
    """
    measurements = _read_json(sidecar_path, "Histology sidecar").get("Measurements", {})
    origin = measurements.get("crop_origin_xy")
    shape = measurements.get("source_shape_hw")
    return {
        "crop_origin_xy": tuple(int(v) for v in origin) if origin else None,
        "source_shape_hw": tuple(int(v) for v in shape) if shape else None,
    }


def read_section_metadata(section_sidecar_path):
    """Read and parse the fields that place a section in the MRI frame.

    Args:
        section_sidecar_path: Path of the raw <base>_BF.json sidecar.
    Returns:
        Dict with "orientation" (str), "rotation" (float degrees) and "flip"
        ((x, y, z) booleans).
    Raises:
        FileNotFoundError: If the sidecar does not exist.
        ValueError: If a field is missing or malformed.
    """
    sidecar = _read_json(section_sidecar_path, "Section sidecar")
    try:
        return {
            "orientation": parse_orientation(sidecar.get("SectionOrientation")),
            "rotation": parse_rotation(sidecar.get("SectionRotation")),
            "flip": parse_flip(sidecar.get("SectionFlip")),
        }
    except ValueError as e:
        raise ValueError(f"{section_sidecar_path}: {e}") from e


def load_slide(path, spacing, as_mask=False):
    """Load a histology TIFF as a single-slice 3D sitk image with the given spacing.

    The TIFFs carry no physical geometry, so the image is given `spacing`, origin 0 and
    identity direction. A 2D file becomes (1, H, W); an RGB file becomes a 3-component
    vector image. A mask is binarised to float32 {0, 1}, a gray image is cast to float32,
    and an RGB image keeps its stored pixel type.

    Args:
        path: Path of the TIFF.
        spacing: (x, y, z) spacing in mm.
        as_mask: Treat non-zero as foreground and return a binary float32 image.
    Returns:
        sitk.Image.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is neither 2D nor 3D.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Histology image not found: {path}")

    img = sitk.ReadImage(path)
    if img.GetDimension() not in (2, 3):
        raise ValueError(f"{path}: expected a 2D or 3D image, got {img.GetDimension()}D")

    is_vector = img.GetNumberOfComponentsPerPixel() > 1
    arr = sitk.GetArrayFromImage(img)
    if img.GetDimension() == 2:
        arr = arr[np.newaxis, ...]

    if as_mask:
        arr = (arr > 0).astype(np.float32)
    elif not is_vector:
        arr = arr.astype(np.float32)

    out = sitk.GetImageFromArray(arr, isVector=is_vector)
    out.SetSpacing(tuple(float(s) for s in spacing))
    out.SetOrigin((0.0, 0.0, 0.0))
    out.SetDirection(_IDENTITY_DIRECTION)
    return out


def index_affine(from_img, to_img, transform=None):
    """4x4 affine mapping a continuous index of from_img to one of to_img.

    Both images describe the same tissue, so the mapping goes through physical space and is
    read out of the images' own geometry rather than derived by hand. `transform` maps a
    physical point of from_img to one of to_img, as sitk.Resample uses it (output to input);
    None means the two share physical space, which is what sitk.Flip leaves behind.

    Args:
        from_img: 3D sitk.Image the index is given in.
        to_img: 3D sitk.Image the index is wanted in.
        transform: Optional sitk transform from from_img physical space to to_img physical space.
    Returns:
        (4, 4) float64 affine; apply to a homogeneous (x, y, z, 1) index.
    """
    corners = []
    for index in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        point = from_img.TransformContinuousIndexToPhysicalPoint([float(v) for v in index])
        if transform is not None:
            point = transform.TransformPoint(point)
        corners.append(np.array(to_img.TransformPhysicalPointToContinuousIndex(point)))

    affine = np.eye(4, dtype=np.float64)
    affine[:3, 3] = corners[0]
    affine[:3, :3] = np.column_stack([c - corners[0] for c in corners[1:]])
    return affine


def _rotation_transform(img, degrees):
    """The Euler transform rotate_in_plane applies, or None for no rotation."""
    if math.isclose(degrees, 0.0, abs_tol=1e-12):
        return None
    size = np.array(img.GetSize(), dtype=float)
    centre = np.array(img.TransformContinuousIndexToPhysicalPoint(((size - 1) / 2.0).tolist()))
    transform = sitk.Euler3DTransform()
    transform.SetCenter(centre.tolist())
    transform.SetRotation(0.0, 0.0, math.radians(degrees))
    return transform


def rotate_in_plane(img, degrees, interp=sitk.sitkLinear):
    """Rotate a single-slice image about its centre in the x-y plane.

    Positive is counter-clockwise, matching cv2.getRotationMatrix2D, which is the convention
    the section angles were measured with. The output grid is enlarged to the rotated
    bounding box, so no tissue is cut off, and keeps the input spacing and direction.

    The centre of rotation only shifts the result by a translation, which patch matching
    recovers on its own; the angle is what has to be right.

    Args:
        img: 3D sitk.Image, usually with z size 1.
        degrees: Rotation in degrees, positive counter-clockwise.
        interp: sitk interpolator; use sitkNearestNeighbor for a mask.
    Returns:
        sitk.Image, unchanged (the same object) when degrees is 0.
    """
    if math.isclose(degrees, 0.0, abs_tol=1e-12):
        return img

    size = np.array(img.GetSize(), dtype=float)
    spacing = np.array(img.GetSpacing(), dtype=float)
    theta = math.radians(degrees)

    width, height = size[0] * spacing[0], size[1] * spacing[1]
    cos_t, sin_t = abs(math.cos(theta)), abs(math.sin(theta))
    n_x = int(math.ceil((width * cos_t + height * sin_t) / spacing[0]))
    n_y = int(math.ceil((width * sin_t + height * cos_t) / spacing[1]))

    centre = np.array(img.TransformContinuousIndexToPhysicalPoint(((size - 1) / 2.0).tolist()))
    # Output index 0 sits half the enlarged extent before the centre, stepped along the
    # image's own axes: sitk.Flip returns a negative direction, and ignoring it here would
    # place the grid on the opposite side of the tissue and resample nothing. z is unchanged,
    # so its half-extent keeps the slices where the input's are.
    half_extent = np.array([(n_x - 1) / 2.0 * spacing[0],
                            (n_y - 1) / 2.0 * spacing[1],
                            (size[2] - 1) / 2.0 * spacing[2]])
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    origin = (centre - direction @ half_extent).tolist()

    transform = sitk.Euler3DTransform()
    transform.SetCenter(centre.tolist())
    transform.SetRotation(0.0, 0.0, theta)

    return sitk.Resample(img, [n_x, n_y, int(size[2])], transform, interp, origin,
                         img.GetSpacing(), img.GetDirection(), 0.0, img.GetPixelID())


def axis_directions(img):
    """The anatomical direction each index axis points toward, e.g. ("L", "I", "P").

    Args:
        img: sitk.Image.
    Returns:
        Tuple of three letters from LRAPSI, one per index axis (x, y, z in sitk order).
    """
    # Column `axis` of the direction matrix is that index axis's direction in world space,
    # and SimpleITK world space is LPS: +x Left, +y Posterior, +z Superior.
    direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    codes = []
    for axis in range(3):
        world = direction[:, axis]
        dominant = int(np.argmax(np.abs(world)))
        codes.append("LPS"[dominant] if world[dominant] > 0 else "RAI"[dominant])
    return tuple(codes)


def view_flips(img, orientation):
    """Which in-plane axes of a permuted volume must be reversed to reach VIEW_AXES.

    Args:
        img: sitk.Image, already permuted for `orientation`.
        orientation: Key of VIEW_AXES.
    Returns:
        [flip_x, flip_y, False], as sitk.Flip takes it.
    Raises:
        ValueError: If an in-plane axis runs along neither the wanted direction nor its
            opposite, which means the volume is oblique or the orientation is wrong.
    """
    codes = axis_directions(img)
    flips = []
    for axis, wanted in enumerate(VIEW_AXES[orientation]):
        if codes[axis] == wanted:
            flips.append(False)
        elif codes[axis] == OPPOSITE_DIRECTION[wanted]:
            flips.append(True)
        else:
            raise ValueError(f"{orientation}: in-plane axis {axis} points {codes[axis]}, "
                             f"expected {wanted} or {OPPOSITE_DIRECTION[wanted]}")
    return flips + [False]


def load_mri(path, orientation, pixel_type=None):
    """Load an MRI volume into the frame a viewer displays for the given cutting plane.

    The volume is reoriented to LPS, permuted so the cutting plane is x-y, and its in-plane
    axes are reversed where needed to reach VIEW_AXES: +x screen-right, +y screen-down.

    Args:
        path: Path of the MRI (.nii.gz or any format whose header carries the geometry).
        orientation: Key of ORIENTATION_PERMUTE.
        pixel_type: Optional sitk pixel type to read as.
    Returns:
        sitk.Image with the cutting plane in x-y and the search axis as z.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MRI image not found: {path}")
    img = sitk.ReadImage(path, pixel_type) if pixel_type else sitk.ReadImage(path)
    img = sitk.DICOMOrient(img, MRI_CANONICAL_ORIENTATION)
    img = sitk.PermuteAxes(img, ORIENTATION_PERMUTE[orientation])
    # flipAboutOrigin=False keeps the physical extent, so index -> millimetre is unchanged.
    return sitk.Flip(img, view_flips(img, orientation), flipAboutOrigin=False)


def pixel_size_to_spacing(pixel_size_mm):
    """Turn a histology pixel size into the (x, y, z) spacing load_slide takes.

    A section is one slice thick, so z takes the x size, as read_pixel_size_mm does.

    Args:
        pixel_size_mm: One number (square pixels), (x, y), or a full (x, y, z), in mm.
    Returns:
        (x, y, z) tuple of floats in mm.
    Raises:
        ValueError: If there are not one to three values, or one is not positive.
    """
    values = [float(pixel_size_mm)] if np.isscalar(pixel_size_mm) else [float(v) for v in pixel_size_mm]
    if not 1 <= len(values) <= 3:
        raise ValueError(f"Expected one to three pixel size values, got {pixel_size_mm!r}")
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"Pixel size must be positive, got {pixel_size_mm!r}")
    if len(values) == 1:
        values = values * 2
    if len(values) == 2:
        values.append(values[0])
    return tuple(values)


def load_files(histo, histo_mask, mri, mri_mask, *, orientation, pixel_size_mm,
               rotation=0.0, flip="", histo_rgb=None, crop=None):
    """Load one section from explicit files: histology, its mask, the MRI and its mask.

    The file-level counterpart of load_section, for data that is not in a BIDS dataset:
    what the sidecars would say is passed in. The histology is rotated and then flipped, in
    that order; the MRI and its mask are reoriented to LPS, permuted for the cutting plane
    and turned to the viewer frame (see load_mri).

    Args:
        histo: Gray histology image (a 2D TIFF or any format SimpleITK reads); its own
            geometry is ignored and pixel_size_mm is used instead.
        histo_mask: Tissue mask of the histology, same grid; non-zero is tissue.
        mri: MRI volume with its geometry in the header (e.g. .nii.gz).
        mri_mask: Brain mask on the MRI grid.
        orientation: Cutting plane, "axial", "coronal" or "sagittal".
        pixel_size_mm: Histology pixel size in mm; see pixel_size_to_spacing.
        rotation: In-plane rotation in degrees, positive counter-clockwise; None means 0.
        flip: Letters from x/y/z ("" for none), or an (x, y, z) tuple of booleans.
        histo_rgb: Optional RGB histology on the same grid, used only for QC figures.
        crop: Optional dict with "crop_origin_xy" and "source_shape_hw" (see
            read_crop_metadata); both are None when omitted.
    Returns:
        The dict load_section returns; "paths" holds "histo_gray", "histo_mask",
        "histo_rgb", "mri" and "mri_mask".
    Raises:
        FileNotFoundError: If a file, including a histo_rgb that was given, is missing.
        ValueError: If orientation, rotation, flip or pixel size is malformed.
    """
    orientation = parse_orientation(orientation)
    rotation = parse_rotation(rotation)
    flip = parse_flip(flip) if flip is None or isinstance(flip, str) else tuple(bool(f) for f in flip)
    if len(flip) != 3:
        raise ValueError(f"flip must give three axes (x, y, z), got {flip!r}")
    spacing = pixel_size_to_spacing(pixel_size_mm)
    paths = {"histo_gray": histo, "histo_mask": histo_mask, "histo_rgb": histo_rgb,
             "mri": mri, "mri_mask": mri_mask}

    Ih_hr = load_slide(histo, spacing)
    Mh_hr = load_slide(histo_mask, spacing, as_mask=True)
    # RGB histology is optional and only used for QC overlays.
    Ih_rgb_hr = load_slide(histo_rgb, spacing) if histo_rgb is not None else None

    # The image as the TIFF stores it, kept to record the way back to that file.
    Ih_tif = Ih_hr

    # Rotate, then flip: the order the section angles of this dataset were measured with.
    # The transform is rebuilt from the same input and angle rotate_in_plane uses, so the
    # recorded affine describes the rotation that was actually applied;
    # test_hr_to_tif_round_trips_a_marker runs both paths and fails if they ever diverge.
    rotation_transform = _rotation_transform(Ih_tif, rotation)
    Ih_hr = rotate_in_plane(Ih_hr, rotation)
    Mh_hr = rotate_in_plane(Mh_hr, rotation, interp=sitk.sitkNearestNeighbor)
    if Ih_rgb_hr is not None:
        Ih_rgb_hr = rotate_in_plane(Ih_rgb_hr, rotation)
    A_rotated_to_tif = index_affine(Ih_hr, Ih_tif, rotation_transform)

    A_flipped_to_rotated = np.eye(4, dtype=np.float64)
    if any(flip):
        Ih_rotated = Ih_hr
        # Flip about the image centre, so the physical extent is unchanged.
        Ih_hr = sitk.Flip(Ih_hr, list(flip), flipAboutOrigin=False)
        Mh_hr = sitk.Flip(Mh_hr, list(flip), flipAboutOrigin=False)
        if Ih_rgb_hr is not None:
            Ih_rgb_hr = sitk.Flip(Ih_rgb_hr, list(flip), flipAboutOrigin=False)
        # Flip leaves the tissue at the same physical points, so no transform is needed.
        A_flipped_to_rotated = index_affine(Ih_hr, Ih_rotated)

    A_hr_to_tif = A_rotated_to_tif @ A_flipped_to_rotated

    Im = load_mri(mri, orientation, pixel_type=sitk.sitkFloat32)
    Mm = load_mri(mri_mask, orientation)

    # Which axes load_mri reversed to reach the viewer frame: the difference between the
    # canonical permuted directions and the ones the loaded volume ended up with.
    canonical_directions = tuple("LPS"[axis] for axis in ORIENTATION_PERMUTE[orientation])
    mri_view_flip = tuple(canonical != loaded for canonical, loaded
                          in zip(canonical_directions, axis_directions(Im)))

    return {
        "Ih_hr": Ih_hr,
        "Mh_hr": Mh_hr,
        "Im": Im,
        "Mm": Mm,
        "Ih_rgb_hr": Ih_rgb_hr,
        "spacing": spacing,
        "orientation": orientation,
        "rotation": rotation,
        "flip": flip,
        "paths": paths,
        # The way back to the files this section came from: A_hr_to_tif undoes the flip and
        # the rotation, the crop metadata undoes cleanslide's crop, and permute with
        # MRI_CANONICAL_ORIENTATION and mri_view_flip undoes what load_mri did to the volume.
        "A_hr_to_tif": A_hr_to_tif,
        "crop": dict(crop) if crop else {"crop_origin_xy": None, "source_shape_hw": None},
        "mri_permute": tuple(ORIENTATION_PERMUTE[orientation]),
        "mri_canonical": MRI_CANONICAL_ORIENTATION,
        "mri_view_flip": mri_view_flip,
        "mri_view_axes": axis_directions(Im),
    }


def load_section(bids_root, subject, sample, stain=None, *,
                 orientation=None, rotation=None, flip=None, spacing=None):
    """Load one section's histology, its mask, the optional RGB histology, the MRI and its mask.

    Finds the section's files in the BIDS dataset (resolve_section), reads the placement
    fields and pixel size from its sidecars, and loads them with load_files. The spacing and
    the three placement fields come from the sidecars unless they are passed in, which lets
    a caller try another value without editing the dataset.

    Args:
        bids_root: Dataset root.
        subject: Subject label, with or without the "sub-" prefix.
        sample: Sample label, with or without the "sample-" prefix.
        stain: Stain label; optional when subject and sample identify one section.
        orientation: Override for SectionOrientation.
        rotation: Override for SectionRotation, in degrees counter-clockwise.
        flip: Override for SectionFlip, as the letters x/y/z.
        spacing: Override for the histology (x, y, z) spacing in mm.
    Returns:
        Dict with sitk images "Ih_hr" (HR histology), "Mh_hr" (HR histology mask), "Im"
        (MRI), "Mm" (MRI mask) and "Ih_rgb_hr" (RGB histology, or None if no file exists),
        plus the resolved "spacing", "orientation", "rotation", "flip", "paths" (as
        resolve_section returns them) and the frame fields described in load_files.
    Raises:
        FileNotFoundError: If a required file is missing.
        ValueError: If a sidecar field or an override is malformed.
    """
    paths = resolve_section(bids_root, subject, sample, stain)

    metadata = read_section_metadata(paths["section_sidecar"])
    orientation = metadata["orientation"] if orientation is None else orientation
    rotation = metadata["rotation"] if rotation is None else rotation
    flip = metadata["flip"] if flip is None else flip
    spacing = read_pixel_size_mm(paths["histo_sidecar"]) if spacing is None else tuple(spacing)

    vols = load_files(
        paths["histo_gray"], paths["histo_mask"], paths["mri"], paths["mri_mask"],
        orientation=orientation, pixel_size_mm=spacing, rotation=rotation, flip=flip,
        histo_rgb=paths["histo_rgb"] if os.path.exists(paths["histo_rgb"]) else None,
        crop=read_crop_metadata(paths["histo_sidecar"]),
    )
    vols["paths"] = paths
    return vols
