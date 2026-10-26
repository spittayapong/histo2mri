"""Command-line entry point: run the registration pipeline over the sections listed in a CSV.

Each row of the sections CSV describes one histology section; main() calls
pipeline.register_section for it, writes that section's outputs to
<save-dir>/<subject_id>/<histo-type>/<num_str>/, and keeps a summary CSV with one row per
section. Matplotlib is switched to the non-interactive Agg backend on import.

Console output: one status line per section. Warnings from the pipeline are shown; with
--verbose, all log messages (including per-iteration progress) and progress bars are shown.
Each section's own log file always contains the INFO-level messages.
"""
import matplotlib
matplotlib.use("Agg")
import argparse
import csv
import logging
import os
import shutil
import time
import traceback
from datetime import datetime
import pandas as pd
import torch
from .io import build_num_str, parse_flip, parse_orientation, parse_spacing
from .pipeline import PROFILES, register_section


def build_arg_parser():
    """Build the argument parser of the batch command-line tool."""
    p = argparse.ArgumentParser(description="Batch-run the full registration pipeline over sections")
    p.add_argument("--sections-csv", required=True,
                    help="One section per row. --ds-type internal: subject_id,slice_id,orientation,spacing. "
                         "--ds-type NB: subject_id,block_id,slice_id,orientation,stain,spacing. "
                         "spacing is a quoted 'x,y,z' string. Optional columns: 'flip' (letters x/y/z) "
                         "and 'profile' (overrides --profile for that row).")
    p.add_argument("--profile", default="default", choices=sorted(PROFILES),
                    help="Named parameter set (see histo2mri.pipeline.PROFILES). Use he-cblock only for "
                         "H&E cerebellum-block sections.")
    p.add_argument("--ds-type", default="internal", choices=["internal", "NB"],
                    help="Section naming scheme of the CSV; NB (NextBrain) also uses a RANSAC sample "
                         "size of 3 instead of 5.")
    p.add_argument("--histo-type", required=True,
                    help="Suffix in <data-root>/<subject_id>/histo_<histo_type>/, e.g. myelin/iron/he.")
    p.add_argument("--data-root", required=True, help="Root folder of the data (layout: see README).")
    p.add_argument("--save-dir", default="./results", help="Output root folder.")
    p.add_argument("--patch-size", type=int, default=64,
                    help="The histology is padded to a multiple of this (px).")
    p.add_argument("--pclip", nargs=2, type=float, default=(1, 99), metavar=("LOW", "HIGH"),
                    help="Percentiles for the MRI intensity normalisation.")
    p.add_argument("--scale-max", type=float, default=1.30, help="Upper end of the histology scale search.")
    p.add_argument("--scale-step", type=float, default=0.05, help="Step of the histology scale search.")
    p.add_argument("--device", default=None, help="cuda:0 / cpu; auto-detect if omitted")
    p.add_argument("--mri-on-histo-tile-size", type=int, default=348,
                    help="Checkerboard tile size (px) for the mri_on_histo overlay. "
                         "Use e.g. 2048 for NB sections.")
    p.add_argument("--lam-area", type=float, default=2.0,
                    help="Local rigidity: weight on log(area/area0)^2 (properness). 0 disables rigidity.")
    p.add_argument("--lam-shape", type=float, default=0.0,
                    help="Local rigidity: weight on first-fundamental-form drift (orthonormality).")
    p.add_argument("--w-rigid-mode", default="mask", choices=["mask", "conf", "edge", "none"],
                    help="Spatial rigidity weight map used by the refinement.")
    p.add_argument("--save-roi-qc", action="store_true",
                    help="Also save the three ROI QC figures per section.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N rows (dry-run)")
    p.add_argument("--log-csv", default=None, help="Summary log path; defaults to <save-dir>/run_log_<timestamp>.csv")
    p.add_argument("--verbose", action="store_true",
                    help="Show all log messages and progress bars on the console.")
    return p


def _setup_console_logging(verbose):
    """Show the package's warnings (or, with verbose, all its messages) on the console."""
    pkg_logger = logging.getLogger("histo2mri")
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    handler.setFormatter(logging.Formatter("    %(levelname)s %(name)s: %(message)s"))
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def main():
    """Run the pipeline on every section of --sections-csv and write a summary CSV.

    For each row (up to --limit rows): build the section identifier num_str
    (io.build_num_str), parse spacing, orientation and the optional flip column, pick the
    profile (the row's "profile" column, else --profile), delete any previous output of the
    section in <save-dir>/<subject_id>/<histo-type>/<num_str>, and call
    pipeline.register_section. A failure in one section is reported (message and traceback on
    the console) and the loop continues.

    The summary CSV (--log-csv, default <save-dir>/run_log_<YYYYmmdd_HHMMSS>.csv) is rewritten
    after every section, so it survives an interrupted run. It has one row per section with
    the columns subject_id, histo_type, num_str, status ("ok" or "error"), profile, and either
    the values returned by register_section (output_npz, n_clean_matches, best_scale,
    elapsed_sec) or error (the exception message) and elapsed_sec.
    """
    args = build_arg_parser().parse_args()
    _setup_console_logging(args.verbose)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ransac_sample_size = 3 if args.ds_type == "NB" else 5

    with open(args.sections_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    os.makedirs(args.save_dir, exist_ok=True)
    log_csv = args.log_csv or os.path.join(args.save_dir, f"run_log_{datetime.now():%Y%m%d_%H%M%S}.csv")

    total = len(rows)
    results = []
    for i, row in enumerate(rows, start=1):
        subject_id = row["subject_id"]
        histo_type = args.histo_type
        entry = {"subject_id": subject_id, "histo_type": histo_type}
        try:
            num_str = build_num_str(row, args.ds_type)
        except KeyError as e:
            entry.update(status="error", error=f"Missing column {e} required for --ds-type {args.ds_type}")
            print(f"[{i}/{total}] {subject_id} ... FAILED ({entry['error']})", flush=True)
            results.append(entry)
            pd.DataFrame(results).to_csv(log_csv, index=False)
            continue

        entry["num_str"] = num_str
        print(f"[{i}/{total}] {subject_id} {histo_type} {num_str} ... RUNNING", flush=True)
        row_start = time.perf_counter()
        try:
            spacing = parse_spacing(row["spacing"])
            orientation = parse_orientation(row["orientation"])
            flip = parse_flip(row.get("flip", ""))
            profile = (row.get("profile") or "").strip() or args.profile
            entry["profile"] = profile

            out_dir = os.path.join(args.save_dir, subject_id, histo_type, num_str)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)

            info = register_section(
                subject_id, num_str, histo_type, spacing, orientation, flip,
                args.data_root, out_dir,
                profile=profile, device=device, ransac_sample_size=ransac_sample_size,
                patch_size=args.patch_size, pclip=tuple(args.pclip),
                scale_max=args.scale_max, scale_step=args.scale_step,
                lam_area=args.lam_area, lam_shape=args.lam_shape, w_rigid_mode=args.w_rigid_mode,
                mri_on_histo_tile_size=args.mri_on_histo_tile_size,
                save_roi_qc=args.save_roi_qc, verbose=args.verbose,
            )
            entry.update(status="ok", **info)
            print(f"[{i}/{total}] {subject_id} {histo_type} {num_str} ... DONE "
                  f"(profile={profile}, n_matches={info['n_clean_matches']}, best_scale={info['best_scale']:.3f}, "
                  f"elapsed={info['elapsed_sec']:.1f}s)", flush=True)
        except Exception as e:
            entry.update(status="error", error=f"{type(e).__name__}: {e}",
                          elapsed_sec=time.perf_counter() - row_start)
            print(f"[{i}/{total}] {subject_id} {histo_type} {num_str} ... FAILED "
                  f"({type(e).__name__}: {e}, elapsed={entry['elapsed_sec']:.1f}s)", flush=True)
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(entry)
        pd.DataFrame(results).to_csv(log_csv, index=False)

    print(f"\nDone. Summary log -> {log_csv}")
    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"{n_ok}/{len(results)} sections succeeded")


if __name__ == "__main__":
    main()
