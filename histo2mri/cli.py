"""Command-line entry point, with one subcommand per way of giving the input.

    histo2mri register --histo H --histo-mask M --mri T1 --mri-mask B -o OUT/PREFIX \\
                       --orientation coronal --histo-pixel-size 5 [--rotation DEG] [--flip xy]
    histo2mri bids <bids_dir> <output_dir> participant [--participant-label LABEL ...]

register runs one section given as explicit files (pipeline.register_files), as tools such as
FSL BET do; what a BIDS sidecar would hold (cutting plane, rotation, flip, pixel size) is
given as options. Its outputs are written as <OUT>/<PREFIX>_*. It exits with status 1 if the
section fails.

bids is a BIDS App. The dataset is the list of what there is to process: it takes the sections
the dataset holds, narrowed by the label options, and for each one calls
pipeline.register_section and writes that section's outputs to
<output_dir>/<sub-X>/<sub-X_sample-Y_stain-Z>/. The output folder is a BIDS derivative, so a
dataset_description.json is written at its root, and a run summary TSV records one row per
section.

Console output: one status line per section. Warnings from the pipeline are shown; with
--verbose, each pipeline step as it starts ("[k/N] name"), all log messages (including
per-iteration progress) and progress bars are shown.
Each section's own log file always contains the INFO-level messages.
"""
import matplotlib
matplotlib.use("Agg")
import argparse
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
import pandas as pd
import torch
from . import __version__
from .io import ORIENTATION_PERMUTE, find_sections, resolve_section, write_dataset_description
from .pipeline import PROFILES, register_files, register_section


def _pipeline_options():
    """Parent parser with the pipeline settings both subcommands share."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("pipeline settings")
    g.add_argument("--profile", default="default", choices=sorted(PROFILES),
                   help="Named parameter set (see histo2mri.pipeline.PROFILES); applies to the "
                        "whole run. Use he-cblock only for H&E cerebellum-block sections.")
    g.add_argument("--ransac-sample-size", type=int, default=5,
                   help="Matches per RANSAC TPS hypothesis. Sections that yield few matches "
                        "sparse section need 3; this will be derived from the match count later.")
    g.add_argument("--pad-multiple", type=int, default=64,
                   help="The histology is padded to a multiple of this (px).")
    g.add_argument("--pclip", nargs=2, type=float, default=(1, 99), metavar=("LOW", "HIGH"),
                   help="Percentiles for the MRI intensity normalisation.")
    g.add_argument("--scale-max", type=float, default=1.30, help="Upper end of the histology scale search.")
    g.add_argument("--scale-step", type=float, default=0.05, help="Step of the histology scale search.")
    g.add_argument("--device", default=None, help="cuda:0 / cpu; auto-detect if omitted")
    g.add_argument("--mri-on-histo-tile-size", type=int, default=348,
                   help="Checkerboard tile size (px) for the mri_on_histo overlay. ")
    g.add_argument("--lam-area", type=float, default=2.0,
                   help="Local rigidity: weight on log(area/area0)^2 (properness). 0 disables rigidity.")
    g.add_argument("--lam-shape", type=float, default=0.0,
                   help="Local rigidity: weight on first-fundamental-form drift (orthonormality).")
    g.add_argument("--w-rigid-mode", default="mask", choices=["mask", "conf", "edge", "none"],
                   help="Spatial rigidity weight map used by the refinement.")
    g.add_argument("--save-roi-qc", action="store_true",
                   help="Also save the ROI QC figures per section.")
    g.add_argument("--verbose", action="store_true",
                   help="Show all log messages and progress bars on the console.")
    return p


def build_arg_parser():
    """Build the argument parser."""
    common = _pipeline_options()
    p = argparse.ArgumentParser(
        prog="histo2mri", description="Register 2D histology sections to their MRI volume")
    p.add_argument("--version", action="version", version=f"histo2mri {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="{register,bids}")

    r = sub.add_parser("register", parents=[common],
                       help="register one section given as explicit files",
                       description="Register one histology section given as explicit files.")
    g = r.add_argument_group("inputs and output")
    g.add_argument("--histo", required=True, help="gray histology image (e.g. 2D TIFF)")
    g.add_argument("--histo-mask", required=True, help="tissue mask on the histology grid")
    g.add_argument("--mri", required=True, help="MRI volume with its geometry in the header (e.g. .nii.gz)")
    g.add_argument("--mri-mask", required=True, help="brain mask on the MRI grid")
    g.add_argument("--histo-rgb", default=None,
                   help="optional RGB histology on the histology grid, used only for QC figures")
    g.add_argument("-o", "--output", required=True, metavar="OUT/PREFIX",
                   help="output prefix: files are written as OUT/PREFIX_*. A path ending in '/' "
                        "uses the histology file name as the prefix.")
    g = r.add_argument_group("section placement (what a BIDS sidecar would hold)")
    g.add_argument("--orientation", required=True, choices=sorted(ORIENTATION_PERMUTE),
                   help="cutting plane of the section")
    g.add_argument("--histo-pixel-size", required=True, type=float, nargs="+", metavar="UM",
                   help="histology pixel size in micrometres: one value, or X Y")
    g.add_argument("--rotation", type=float, default=0.0,
                   help="in-plane rotation of the histology in degrees, positive "
                        "counter-clockwise, applied before the flip (default 0)")
    g.add_argument("--flip", default="",
                   help="axes to mirror after the rotation: letters from x/y ('y' = top-bottom, "
                        "'x' = left-right); default none")

    b = sub.add_parser("bids", parents=[common],
                       help="register the sections of a BIDS dataset (BIDS App)",
                       description="Register the histology sections of a BIDS dataset.")
    b.add_argument("bids_dir", help="BIDS dataset to read")
    b.add_argument("output_dir", help="where the derivative is written")
    b.add_argument("analysis_level", choices=["participant"],
                   help="level of the analysis; only 'participant' is supported")
    b.add_argument("--participant-label", nargs="+", metavar="LABEL",
                   help="only these subjects, with or without the 'sub-' prefix; "
                        "all of them when omitted")
    b.add_argument("--sample-label", nargs="+", metavar="LABEL",
                   help="only these samples, with or without the 'sample-' prefix")
    b.add_argument("--stain-label", nargs="+", metavar="LABEL",
                   help="only these stains, with or without the 'stain-' prefix")
    b.add_argument("--limit", type=int, default=None, help="Only process the first N sections (dry-run)")
    b.add_argument("--summary", default=None,
                   help="Run summary path; defaults to <output_dir>/run_log_<timestamp>.tsv")
    return p


def _setup_console_logging(verbose):
    """Show the package's warnings (or, with verbose, all its messages) on the console.

    A handler left by an earlier main() call in the same process is replaced, not stacked,
    so messages are not repeated and none goes to a stream that has since been closed.
    """
    pkg_logger = logging.getLogger("histo2mri")
    for old in [h for h in pkg_logger.handlers if getattr(h, "_histo2mri_console", False)]:
        pkg_logger.removeHandler(old)
    handler = logging.StreamHandler()
    handler._histo2mri_console = True
    handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    handler.setFormatter(logging.Formatter("    %(levelname)s %(name)s: %(message)s"))
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def _pipeline_kwargs(args):
    """The register_section / register_files keyword arguments shared by both subcommands."""
    return dict(
        profile=args.profile,
        device=torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")),
        ransac_sample_size=args.ransac_sample_size,
        pad_multiple=args.pad_multiple, pclip=tuple(args.pclip),
        scale_max=args.scale_max, scale_step=args.scale_step,
        lam_area=args.lam_area, lam_shape=args.lam_shape, w_rigid_mode=args.w_rigid_mode,
        mri_on_histo_tile_size=args.mri_on_histo_tile_size,
        save_roi_qc=args.save_roi_qc, verbose=args.verbose,
    )


def _done_line(label, profile, info):
    return (f"{label} ... DONE (profile={profile}, n_matches={info['n_clean_matches']}, "
            f"best_scale={info['best_scale']:.3f}, elapsed={info['elapsed_sec']:.1f}s)")


def _split_output(output):
    """OUT/PREFIX -> (OUT, PREFIX); PREFIX is None when output ends in a separator."""
    out_dir, prefix = os.path.split(output)
    return out_dir or ".", prefix or None


def run_register(args):
    """Run one section given as files; return the exit status (0 ok, 1 failed)."""
    if len(args.histo_pixel_size) > 2:
        raise SystemExit("--histo-pixel-size takes one value, or two (X Y)")
    out_dir, prefix = _split_output(args.output)
    label = prefix or os.path.basename(args.histo)
    print(f"{label} ... RUNNING", flush=True)
    start = time.perf_counter()
    try:
        info = register_files(
            args.histo, args.histo_mask, args.mri, args.mri_mask, out_dir,
            orientation=args.orientation, pixel_size_um=args.histo_pixel_size,
            rotation=args.rotation, flip=args.flip, histo_rgb=args.histo_rgb,
            section_id=prefix, **_pipeline_kwargs(args),
        )
    except Exception as e:
        print(f"{label} ... FAILED ({type(e).__name__}: {e}, "
              f"elapsed={time.perf_counter() - start:.1f}s)", flush=True)
        traceback.print_exc()
        return 1
    print(_done_line(label, args.profile, info), flush=True)
    print(f"Transform -> {info['output_npz']}")
    return 0


def _select_sections(args):
    """List the sections to run: what the dataset holds, narrowed by the label options.

    Each option takes several labels, so the selection is the union over them; an option
    left out matches everything.

    Returns:
        List of dicts with the keys subject, sample and stain, each section listed once.
    Raises:
        SystemExit: If no section of the dataset matches.
    """
    selected, seen = [], set()
    for subject in args.participant_label or [None]:
        for sample in args.sample_label or [None]:
            for stain in args.stain_label or [None]:
                for section in find_sections(args.bids_dir, subject, sample, stain):
                    if section["base"] not in seen:
                        seen.add(section["base"])
                        selected.append(section)

    if not selected:
        asked = {name: value for name, value in (
            ("participant", args.participant_label), ("sample", args.sample_label),
            ("stain", args.stain_label)) if value}
        raise SystemExit(f"No section in {args.bids_dir} matches {asked or 'the dataset'}")
    return sorted(selected, key=lambda s: s["base"])


def run_bids(args):
    """Run the pipeline on every selected section and write the derivative.

    For each section (up to --limit): resolve its BIDS files to get the section id, delete
    any previous output of that section under output_dir, and call
    pipeline.register_section, which reads the cutting plane, rotation, flip and pixel size
    from the section's sidecars. A failure in one section is reported (message and traceback
    on the console) and the loop continues.

    The run summary (--summary, default <output_dir>/run_log_<YYYYmmdd_HHMMSS>.tsv) is
    rewritten after every section, so it survives an interrupted run. It has one row per
    section with the columns subject, sample, stain, section_id, status ("ok" or "error"),
    profile, and either the values returned by register_section (output_npz,
    n_clean_matches, best_scale, elapsed_sec) or error and elapsed_sec.

    Returns:
        0; failed sections are recorded in the summary rather than in the exit status.
    """
    kwargs = _pipeline_kwargs(args)
    sections = _select_sections(args)
    if args.limit is not None:
        sections = sections[: args.limit]

    write_dataset_description(args.output_dir, __version__)
    summary_path = args.summary or os.path.join(
        args.output_dir, f"run_log_{datetime.now():%Y%m%d_%H%M%S}.tsv")

    total = len(sections)
    results = []
    for i, section in enumerate(sections, start=1):
        entry = {k: section[k] for k in ("subject", "sample", "stain")}
        entry["profile"] = args.profile
        row_start = time.perf_counter()
        try:
            # Resolved here as well as in register_section, so that a missing or ambiguous
            # section is reported against this row rather than aborting the batch.
            paths = resolve_section(args.bids_dir, section["subject"], section["sample"],
                                    section["stain"])
            section_id = paths["base"]
            entry["section_id"] = section_id
            print(f"[{i}/{total}] {section_id} ... RUNNING", flush=True)

            out_dir = os.path.join(args.output_dir, paths["subject"], section_id)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)

            info = register_section(
                args.bids_dir, section["subject"], section["sample"], out_dir,
                stain=section["stain"], **kwargs,
            )
            entry.update(status="ok", **info)
            print(f"[{i}/{total}] " + _done_line(section_id, args.profile, info), flush=True)
        except Exception as e:
            entry.update(status="error", error=f"{type(e).__name__}: {e}",
                         elapsed_sec=time.perf_counter() - row_start)
            label = entry.get("section_id") or f"{section['subject']}/{section['sample']}"
            print(f"[{i}/{total}] {label} ... FAILED "
                  f"({type(e).__name__}: {e}, elapsed={entry['elapsed_sec']:.1f}s)", flush=True)
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(entry)
        # Tab-separated, the format BIDS uses for every table it defines.
        pd.DataFrame(results).to_csv(summary_path, sep="\t", index=False, na_rep="n/a")

    print(f"\nDone. Run summary -> {summary_path}")
    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"{n_ok}/{len(results)} sections succeeded")
    return 0


def main(argv=None):
    """Parse the command line and run the chosen subcommand."""
    args = build_arg_parser().parse_args(argv)
    _setup_console_logging(args.verbose)
    status = run_register(args) if args.command == "register" else run_bids(args)
    sys.exit(status)


if __name__ == "__main__":
    main()
