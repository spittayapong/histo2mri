"""Command-line entry point: run the pipeline over the sections listed in a CSV."""
import matplotlib
matplotlib.use("Agg")
import argparse
import contextlib
import csv
import os
import shutil
import time
import traceback
from datetime import datetime
import matplotlib
import pandas as pd
import torch
from .io import build_num_str, parse_flip, parse_orientation, parse_spacing
from .pipeline import register_section


def build_arg_parser():
    p = argparse.ArgumentParser(description="Batch-run the full registration pipeline over sections")
    p.add_argument("--sections-csv", required=True,
                    help="Same format as annotations_run.py: internal: subject_id,slice_id,orientation,spacing. "
                         "NB: subject_id,block_id,slice_id,orientation,stain,spacing. "
                         "spacing is a quoted 'x,y,z' string. Optional 'flip' column: letters x/y/z.")
    p.add_argument("--ds-type", default="internal", choices=["internal", "NB"])
    p.add_argument("--histo-type", required=True,
                    help="Suffix in data/<subject_id>/histo_<histo_type>/, e.g. myelin/iron/he.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--save-dir", default="./evaluations/")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--pclip", nargs=2, type=float, default=(1, 99), metavar=("LOW", "HIGH"))
    p.add_argument("--n-slices-offset-divisor", type=int, default=2,
                    help="n_slices = Im depth // this (only affects QC display slice index)")
    p.add_argument("--scale-max", type=float, default=1.30, help="Upper bound (inclusive-ish) for best-scale search")
    p.add_argument("--scale-step", type=float, default=0.05)
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
    p.add_argument("--limit", type=int, default=None, help="Only process the first N rows (dry-run)")
    p.add_argument("--log-csv", default=None, help="Summary log path; defaults to <save-dir>/run_log_<timestamp>.csv")
    p.add_argument("--verbose", action="store_true",
                    help="Show print()/tqdm output from the underlying modules. Off by default.")
    return p


@contextlib.contextmanager
def suppress_output(enabled=True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def main():
    args = build_arg_parser().parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    common_kwargs = dict(
        device=device,
        cell_size=8, bins=18,
        min_histo_mask_frac=0.2,
        min_mri_mask_frac=0.50,
        min_overlap_frac=0.35,
        topk_hog=64, w_hog=0.6, w_mind=0.4,
        mind_radius=2, mind_dilation=1,
        hist_batch=128,
        mri_chunk=16384,
        keep_mri_bank_on_gpu=True,
    )

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
            continue

        entry["num_str"] = num_str
        print(f"[{i}/{total}] {subject_id} {histo_type} {num_str} ... RUNNING", flush=True)
        row_start = time.perf_counter()
        try:
            spacing = parse_spacing(row["spacing"])
            orientation = parse_orientation(row["orientation"])
            flip = parse_flip(row.get("flip", ""))

            log_dir = os.path.join(args.save_dir, subject_id, histo_type, num_str)
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)

            with suppress_output(enabled=not args.verbose):
                info = register_section(
                    subject_id, num_str, histo_type, spacing, orientation, flip,
                    args, device, common_kwargs, log_dir,
                )
            entry.update(status="ok", **info)
            print(f"[{i}/{total}] {subject_id} {histo_type} {num_str} ... DONE "
                  f"(n_matches={info['n_clean_matches']}, best_scale={info['best_scale']:.3f}, "
                  f"elapsed={info['elapsed_sec']:.1f}s)", flush=True)
        except Exception as e:
            entry.update(status="error", error=f"{type(e).__name__}: {e}",
                          elapsed_sec=time.perf_counter() - row_start)
            print(f"[{i}/{total}] {subject_id} {histo_type} {num_str} ... FAILED "
                  f"({type(e).__name__}: {e}, elapsed={entry['elapsed_sec']:.1f}s)", flush=True)
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
        results.append(entry)

    pd.DataFrame(results).to_csv(log_csv, index=False)
    print(f"\nDone. Summary log -> {log_csv}")
    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"{n_ok}/{len(results)} sections succeeded")



if __name__ == "__main__":
    main()
