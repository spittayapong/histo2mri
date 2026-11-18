"""Command-line entry point: a BIDS App that registers histology sections to their MRI.

    histo2mri <bids_dir> <output_dir> participant [--participant-label LABEL ...]

The dataset is the list of what there is to process: main() takes the sections it holds,
narrowed by the label options, and for each one calls pipeline.register_section and writes
that section's outputs to <output_dir>/<sub-X>/<sub-X_sample-Y_stain-Z>/. The output folder
is a BIDS derivative, so a dataset_description.json is written at its root, and a run summary
TSV records one row per section. Matplotlib is switched to the non-interactive Agg backend
on import.

Console output: one status line per section. Warnings from the pipeline are shown; with
--verbose, all log messages (including per-iteration progress) and progress bars are shown.
Each section's own log file always contains the INFO-level messages.
"""
import matplotlib
matplotlib.use("Agg")
import argparse
import logging
import os
import shutil
import time
import traceback
from datetime import datetime
import pandas as pd
import torch
from . import __version__
from .io import find_sections, resolve_section, write_dataset_description
from .pipeline import PROFILES, register_section


def build_arg_parser():
    """Build the argument parser of the BIDS App."""
    p = argparse.ArgumentParser(
        description="Register 2D histology sections of a BIDS dataset to their MRI volume")
    p.add_argument("bids_dir", help="BIDS dataset to read")
    p.add_argument("output_dir", help="where the derivative is written")
    p.add_argument("analysis_level", choices=["participant"],
                    help="level of the analysis; only 'participant' is supported")
    p.add_argument("--participant-label", nargs="+", metavar="LABEL",
                    help="only these subjects, with or without the 'sub-' prefix; "
                         "all of them when omitted")
    p.add_argument("--sample-label", nargs="+", metavar="LABEL",
                    help="only these samples, with or without the 'sample-' prefix")
    p.add_argument("--stain-label", nargs="+", metavar="LABEL",
                    help="only these stains, with or without the 'stain-' prefix")
    p.add_argument("--profile", default="default", choices=sorted(PROFILES),
                    help="Named parameter set (see histo2mri.pipeline.PROFILES); applies to the "
                         "whole run. Use he-cblock only for H&E cerebellum-block sections.")
    p.add_argument("--ransac-sample-size", type=int, default=5,
                    help="Matches per RANSAC TPS hypothesis. Sections that yield few matches "
                         "(NextBrain blocks) need 3; this will be derived from the match count later.")
    p.add_argument("--pad-multiple", type=int, default=64,
                    help="The histology is padded to a multiple of this (px).")
    p.add_argument("--pclip", nargs=2, type=float, default=(1, 99), metavar=("LOW", "HIGH"),
                    help="Percentiles for the MRI intensity normalisation.")
    p.add_argument("--scale-max", type=float, default=1.30, help="Upper end of the histology scale search.")
    p.add_argument("--scale-step", type=float, default=0.05, help="Step of the histology scale search.")
    p.add_argument("--device", default=None, help="cuda:0 / cpu; auto-detect if omitted")
    p.add_argument("--mri-on-histo-tile-size", type=int, default=348,
                    help="Checkerboard tile size (px) for the mri_on_histo overlay. "
                         "Use e.g. 2048 for 4 um histology.")
    p.add_argument("--lam-area", type=float, default=2.0,
                    help="Local rigidity: weight on log(area/area0)^2 (properness). 0 disables rigidity.")
    p.add_argument("--lam-shape", type=float, default=0.0,
                    help="Local rigidity: weight on first-fundamental-form drift (orthonormality).")
    p.add_argument("--w-rigid-mode", default="mask", choices=["mask", "conf", "edge", "none"],
                    help="Spatial rigidity weight map used by the refinement.")
    p.add_argument("--save-roi-qc", action="store_true",
                    help="Also save the three ROI QC figures per section.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N sections (dry-run)")
    p.add_argument("--summary", default=None,
                    help="Run summary path; defaults to <output_dir>/run_log_<timestamp>.tsv")
    p.add_argument("--verbose", action="store_true",
                    help="Show all log messages and progress bars on the console.")
    p.add_argument("--version", action="version", version=f"histo2mri {__version__}")
    return p


def _setup_console_logging(verbose):
    """Show the package's warnings (or, with verbose, all its messages) on the console."""
    pkg_logger = logging.getLogger("histo2mri")
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    handler.setFormatter(logging.Formatter("    %(levelname)s %(name)s: %(message)s"))
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.DEBUG if verbose else logging.INFO)


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


def main():
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
    """
    args = build_arg_parser().parse_args()
    _setup_console_logging(args.verbose)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

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
                stain=section["stain"],
                profile=args.profile, device=device,
                ransac_sample_size=args.ransac_sample_size,
                pad_multiple=args.pad_multiple, pclip=tuple(args.pclip),
                scale_max=args.scale_max, scale_step=args.scale_step,
                lam_area=args.lam_area, lam_shape=args.lam_shape, w_rigid_mode=args.w_rigid_mode,
                mri_on_histo_tile_size=args.mri_on_histo_tile_size,
                save_roi_qc=args.save_roi_qc, verbose=args.verbose,
            )
            entry.update(status="ok", **info)
            print(f"[{i}/{total}] {section_id} ... DONE "
                  f"(profile={args.profile}, n_matches={info['n_clean_matches']}, "
                  f"best_scale={info['best_scale']:.3f}, elapsed={info['elapsed_sec']:.1f}s)", flush=True)
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


if __name__ == "__main__":
    main()
