"""Optional first step: clean raw histology with cleanslide before registering it.

histo2mri registers a cleaned section: an inverted grayscale with the background at 0, its
tissue mask and, for QC, the colour section. cleanslide (a separate package,
https://github.com/spittayapong/cleanslide) makes those from a raw brightfield slide. With
the command-line option --cleanslide the two run in one go:

    clean_file   one raw slide given as a file (histo2mri register --cleanslide)
    clean_bids   the raw sections of a BIDS dataset (histo2mri bids --cleanslide)

Both always write cleanslide's QC panels, because a bad clean (tissue cut off, an artifact
left in) does not fail; it only shows up later as a bad registration. cleanslide is an
optional dependency: pip install "histo2mri[cleanslide]".
"""
import dataclasses
import glob
import json
import os

# Names of the files clean_file writes, after the prefix; the same descriptions cleanslide
# uses in a BIDS derivative.
CLEAN_SUFFIXES = {
    "histo_gray": "_desc-clean_BF.tif",
    "histo_mask": "_desc-tissue_mask.tif",
    "histo_rgb": "_desc-cleanrgb_BF.tif",
    "sidecar": "_desc-clean_BF.json",
}


def _cleanslide():
    """Import cleanslide, or say how to install it."""
    try:
        import cleanslide
        from cleanslide import io as cs_io, qc as cs_qc
        from cleanslide.params import CleanParams
        from cleanslide.pipeline import clean_section
    except ImportError as e:
        raise ImportError(
            "--cleanslide needs the cleanslide package; install it with "
            "pip install \"histo2mri[cleanslide]\"") from e
    return cleanslide, cs_io, cs_qc, CleanParams, clean_section


def parse_background_threshold(value):
    """An 8-bit grey level (int 0-255), "auto", or None for "not given".

    Returns:
        The int, "auto", or None.
    Raises:
        ValueError: If value is neither "auto" nor an integer in 0-255.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        level = int(value)
    except (TypeError, ValueError):
        raise ValueError(f'background threshold must be 0-255 or "auto", got {value!r}') from None
    if not 0 <= level <= 255:
        raise ValueError(f"background threshold {level} is outside the 0-255 range")
    return level


def _params(CleanParams, background_threshold, section=None):
    """cleanslide parameters: the option first, then the section's sidecar, then the default.

    The same precedence as cleanslide's own command line.
    """
    params = CleanParams()
    if background_threshold is not None:
        return dataclasses.replace(
            params, background_threshold=None if background_threshold == "auto" else background_threshold)
    if section is not None and section.declares_background_threshold:
        return dataclasses.replace(params, background_threshold=section.background_threshold)
    return params


def _write_tif(cv2, path, image, pixel_size_um):
    """8-bit TIFF, deflate, with resolution tags, as cleanslide writes its outputs."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    dpi = int(round(25400.0 / pixel_size_um))
    if not cv2.imwrite(path, image, [cv2.IMWRITE_TIFF_COMPRESSION, 8,
                                     cv2.IMWRITE_TIFF_XDPI, dpi, cv2.IMWRITE_TIFF_YDPI, dpi]):
        raise OSError(f"could not write {path}")


def _crop(info):
    """The crop fields of a cleanslide result, in the form io.read_crop_metadata returns."""
    origin, shape = info.get("crop_origin_xy"), info.get("source_shape_hw")
    return {
        "crop_origin_xy": tuple(int(v) for v in origin) if origin else None,
        "source_shape_hw": tuple(int(v) for v in shape) if shape else None,
    }


def clean_file(raw, pixel_size_um, out_dir, prefix, *, background_threshold=None):
    """Clean one raw brightfield slide and write the files histo2mri registers.

    Writes <out_dir>/<prefix> followed by each of CLEAN_SUFFIXES, plus the QC panels
    <prefix>_cleanslide_outputs.png and <prefix>_cleanslide_qc.png.

    Args:
        raw: Raw RGB slide (.tif, .tiff or .png).
        pixel_size_um: Pixel size in micrometres; one value, or (x, y) with x == y, because
            cleanslide does not support anisotropic pixels.
        out_dir: Folder to write into; created if missing.
        prefix: File-name prefix.
        background_threshold: Grey level above which a pixel is slide, "auto" to estimate
            it, or None for cleanslide's default.
    Returns:
        Dict with the paths "histo_gray", "histo_mask", "histo_rgb", "sidecar", the QC
        panel paths "qc_outputs" and "qc_stages", and "crop" (where the cleaned image sits
        in the raw slide; see io.read_crop_metadata).
    Raises:
        ImportError: If cleanslide is not installed.
        FileNotFoundError: If raw does not exist.
        ValueError: If the pixel size is anisotropic or the threshold is malformed.
    """
    cleanslide, _, cs_qc, CleanParams, clean_section = _cleanslide()
    import cv2   # a dependency of cleanslide

    sizes = [float(pixel_size_um)] if not isinstance(pixel_size_um, (list, tuple)) else [float(v) for v in pixel_size_um]
    if len(sizes) == 2 and abs(sizes[0] - sizes[1]) > 1e-9:
        raise ValueError(f"cleanslide needs square pixels, got {sizes}")
    size_um = sizes[0]

    if not os.path.exists(raw):
        raise FileNotFoundError(f"Raw histology not found: {raw}")
    image = cv2.imread(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read {raw}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    params = _params(CleanParams, parse_background_threshold(background_threshold))
    result = clean_section(rgb, size_um, params, keep_stages=True)

    os.makedirs(out_dir, exist_ok=True)
    paths = {key: os.path.join(out_dir, prefix + suffix) for key, suffix in CLEAN_SUFFIXES.items()}
    _write_tif(cv2, paths["histo_gray"], result.clean, size_um)
    _write_tif(cv2, paths["histo_mask"], result.mask, size_um)
    _write_tif(cv2, paths["histo_rgb"], result.rgb, size_um)
    with open(paths["sidecar"], "w", encoding="utf-8") as handle:
        json.dump({
            "PixelSize": [size_um, size_um],
            "PixelSizeUnits": "um",
            "Sources": [os.path.abspath(raw)],
            "GeneratedBy": [{"Name": "cleanslide", "Version": cleanslide.__version__}],
            "OutputFormatVersion": cleanslide.OUTPUT_FORMAT_VERSION,
            "Parameters": vars(params),
            "Measurements": result.info,
        }, handle, indent=2, default=str)

    paths["qc_outputs"] = cs_qc.output_panel(result, prefix, os.path.join(out_dir, f"{prefix}_cleanslide_outputs.png"))
    paths["qc_stages"] = cs_qc.stage_panel(result, prefix, os.path.join(out_dir, f"{prefix}_cleanslide_qc.png"))
    paths["crop"] = _crop(result.info)
    return paths


def _strip(label, prefix):
    text = str(label).strip()
    return text[len(prefix):] if text.startswith(prefix) else text


def clean_bids(bids_dir, out_root, *, participant_labels=None, sample_labels=None,
               stain_labels=None, background_threshold=None, progress=print):
    """Clean the raw sections of a BIDS dataset into a cleanslide derivative at out_root.

    Reads each <bids_dir>/sub-X/micr/*_BF.tif with its sidecar (pixel size, optional
    BackgroundThreshold) and writes what cleanslide's own command line writes:
    <out_root>/sub-X/micr/<section>_desc-*.tif/.json, <out_root>/dataset_description.json
    and the QC panels in <out_root>/qc/. The input dataset is only read. A section's earlier
    outputs are removed before it is cleaned, so a failed clean leaves nothing stale behind.

    Args:
        bids_dir: Raw BIDS dataset.
        out_root: Folder of the cleanslide derivative.
        participant_labels, sample_labels, stain_labels: Narrow the sections, with or
            without the entity prefix; None matches everything.
        background_threshold: Overrides every sidecar: a grey level or "auto"; None uses
            each sidecar's BackgroundThreshold, or cleanslide's default.
        progress: Called with one status line per section.
    Returns:
        List of dicts with "subject", "sample", "stain" (labels without prefix), "name"
        (the section file stem) and "status" ("ok" or "error"), plus "error" on failure.
    Raises:
        ImportError: If cleanslide is not installed.
        SystemExit: If no raw section matches.
    """
    cleanslide, cs_io, cs_qc, CleanParams, clean_section = _cleanslide()
    threshold = parse_background_threshold(background_threshold)

    subjects = [_strip(s, "sub-") for s in participant_labels] if participant_labels else None
    samples = {_strip(s, "sample-") for s in sample_labels} if sample_labels else None
    stains = {_strip(s, "stain-") for s in stain_labels} if stain_labels else None
    sections = [s for s in cs_io.find_sections(bids_dir, subjects)
                if (samples is None or s.sample in samples) and (stains is None or s.stain in stains)]
    if not sections:
        raise SystemExit(f"No raw section in {bids_dir} matches the label options")

    cs_io.write_dataset_description(out_root, cleanslide.__version__)
    qc_dir = os.path.join(out_root, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    results = []
    for i, section in enumerate(sections, start=1):
        entry = {"subject": section.subject, "sample": section.sample, "stain": section.stain,
                 "name": section.name}
        stale = glob.glob(os.path.join(cs_io.derivative_dir(out_root, section), f"{section.name}_desc-*"))
        for path in stale:
            os.remove(path)
        try:
            params = _params(CleanParams, threshold, section)
            result = clean_section(cs_io.read_image(section), section.pixel_size_um, params,
                                   keep_stages=True)
            cs_io.write_result(result, section, out_root, params, cleanslide.__version__,
                               cleanslide.OUTPUT_FORMAT_VERSION)
            cs_qc.output_panel(result, section.name, os.path.join(qc_dir, f"{section.name}_outputs.png"))
            cs_qc.stage_panel(result, section.name, os.path.join(qc_dir, f"{section.name}_qc.png"))
            entry["status"] = "ok"
            progress(f"[clean {i}/{len(sections)}] {section.name} ... DONE "
                     f"(background threshold {'auto' if params.background_threshold is None else params.background_threshold})")
        except Exception as e:
            for path in glob.glob(os.path.join(cs_io.derivative_dir(out_root, section), f"{section.name}_desc-*")):
                os.remove(path)
            entry.update(status="error", error=f"cleanslide: {type(e).__name__}: {e}")
            progress(f"[clean {i}/{len(sections)}] {section.name} ... FAILED ({type(e).__name__}: {e})")
        results.append(entry)
    return results
