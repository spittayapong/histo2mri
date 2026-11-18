# histo2mri

Registration of 2D histology sections to a 3D MRI volume: each section is mapped onto a curved
sheet in MRI voxel space, so a point on the slide and a point in the volume can be converted
into each other in both directions.

## Installation

```bash
pip install .
```

Python 3.11 or newer. A plain install gives the CPU build of PyTorch, which works but is slow;
for a CUDA build, install torch from the index that matches your driver before installing this
package. `pip install .[test]` adds pytest, `.[dev]` adds ipykernel.

## Input: a BIDS dataset

The pipeline reads one section at a time from a BIDS dataset, with the histology and the brain
mask taken from two derivative pipelines. With `<sub>` = `sub-<subject>` and `<base>` =
`<sub>_sample-<sample>_stain-<stain>`:

```
<bids_root>/
  <sub>/anat/<sub>_T1w.nii.gz                             MRI
  <sub>/micr/<base>_BF.json                               section metadata (see below)
  derivatives/brainmask/<sub>/anat/<sub>_desc-brain_mask.nii.gz
  derivatives/cleanslide/<sub>/micr/<base>_desc-clean_BF.tif    histology, greyscale
                                   <base>_desc-clean_BF.json   its pixel size
                                   <base>_desc-tissue_mask.tif tissue mask
                                   <base>_desc-cleanrgb_BF.tif optional, QC overlays only
```

The histology pixel size comes from `PixelSize` and `PixelSizeUnits` in the cleanslide sidecar,
not from the command line. The MRI and the brain mask carry their own geometry in their
headers. The histology derivative is produced by
[cleanslide](https://github.com/spittayapong/cleanslide), which removes the slide background
and the mounting artifacts, crops to the tissue and segments it.

cleanslide itself is not required: this package only reads those files and does not import it.
Histology that is already clean and has a tissue mask can be used directly, by writing the
greyscale section, its mask and a sidecar carrying `PixelSize` and `PixelSizeUnits` into the
paths above; the RGB image stays optional. `crop_origin_xy` and `source_shape_hw` under
`Measurements` in the sidecar are optional too: when they are absent the result can still be
mapped back to the TIFF given, just not to a larger slide it was cropped from.

### Section metadata

Three fields in the raw `micr` sidecar place the section in the MRI frame. They cannot be
recovered from the images and must be filled in:

| field | values | meaning |
|---|---|---|
| `SectionOrientation` | `axial`, `coronal`, `sagittal` | the cutting plane |
| `SectionRotation` | degrees, positive counter-clockwise | in-plane angle between slide and MRI |
| `SectionFlip` | any of `x`, `y`, `z`, or `""` | mirror needed to match the MRI |

```json
{
  "PixelSize": [4.0, 4.0],
  "PixelSizeUnits": "um",
  "SectionOrientation": "coronal",
  "SectionRotation": 0,
  "SectionFlip": "x"
}
```

They are applied in that order — **rotate, then flip**. `SectionRotation` follows the
`cv2.getRotationMatrix2D` convention, so a positive angle turns the image counter-clockwise.
In `SectionFlip`, `x` mirrors left-to-right (`np.fliplr`) and `y` top-to-bottom (`np.flipud`);
`z` is a no-op for a single section. Note that `"xy"` is a 180° rotation and so does **not**
mirror, while `"x"` or `"y"` alone does.

Two things are worth knowing when filling these in:

- The values are measured against the MRI **as a viewer shows it**: the loader reorients the
  volume to LPS, permutes it for the cutting plane, and then turns the two in-plane axes so
  that +x is screen-right and +y is screen-down in the radiological display FSLeyes and
  MRIcron use by default. For a coronal section that is `+x` Left, `+y` Inferior; sagittal is
  `+x` Anterior, `+y` Inferior; axial is `+x` Left, `+y` Posterior. Because the volume is
  canonicalised first, the same letters mean the same mounting whatever orientation the file
  happens to be stored in — but check once that your viewer is not mirroring left and right.
- Getting the flip wrong is not self-correcting. The patch matcher is not rotation invariant,
  and the global plane fit forces its in-plane part back to a proper rotation, so a mirrored
  section cannot be recovered later: it will register plausibly onto the wrong side.

## Usage

histo2mri is a [BIDS App](https://bids-apps.neuroimaging.io/): it takes the dataset, the
folder to write the derivative into, and the analysis level. The dataset is the list of what
there is to process, so with no label options every section in it is run:

```bash
histo2mri /path/to/bids /path/to/derivatives/histo2mri participant
```

`--participant-label`, `--sample-label` and `--stain-label` narrow that down. Each takes
several labels and matches everything when left out, so this runs one section:

```bash
histo2mri /path/to/bids out participant --participant-label NBP5816 --sample-label A2109
```

and this runs every H&E section of two subjects:

```bash
histo2mri /path/to/bids out participant --participant-label NBP5816 NBP8518 --stain-label HE
```

The output folder is a BIDS derivative: a `dataset_description.json` is written at its root.
Useful options:

| option | default | purpose |
|---|---|---|
| `--profile` | `default` | parameter set; see Profiles |
| `--device` | auto | `cuda:0` or `cpu` |
| `--ransac-sample-size` | 5 | matches per RANSAC TPS hypothesis; sections with few matches need 3 |
| `--pad-multiple` | 64 | the histology is padded to a multiple of this |
| `--scale-max`, `--scale-step` | 1.30, 0.05 | histology scale search |
| `--pclip` | 1 99 | percentiles for MRI intensity normalisation |
| `--mri-on-histo-tile-size` | 348 | checkerboard tile for the overlay; use ~2048 for 4 µm histology |
| `--lam-area`, `--lam-shape` | 2.0, 0.0 | local rigidity in the refinement: area (properness) and shape (orthonormality); 0 disables |
| `--w-rigid-mode` | `mask` | where rigidity applies: `mask`, `conf`, `edge` or `none` |
| `--save-roi-qc` | off | also write the three ROI QC figures |
| `--limit`, `--summary`, `--verbose` | | dry-run, summary path, full logging |

One `--profile` applies to the whole run, so sections needing `he-cblock` are run separately.
A run summary (`run_log_<timestamp>.tsv`, one row per section) is rewritten after every
section, so an interrupted run keeps its results and a failed section is visible without
opening each log.

### Profiles

`default` suits most sections. `he-cblock` is for H&E sections of cerebellum blocks, where the
internal granular layer is the darkest structure and the molecular layer the brightest, so
contrast does not follow MRI intensity order: it uses MIND rather than SSIM in the refinement,
and a stricter outlier filter for sparse matches.

## Outputs

Each section is written to `<output_dir>/<sub>/<base>/`:

| file | contents |
|---|---|
| `<base>_transform_grid.npz` | the transform; fields below |
| `<base>_histo_in_mri_space.png` | histology resampled on the sheet |
| `<base>_histo_in_mri_space_all.png` | that, the valid region, the MRI on the sheet, and their checkerboard |
| `<base>_histo_in_mri_volume.png` | the sheet in 3D in the volume, and the MRI slices it cuts |
| `<base>_mri_on_histo.png`, `_mri_on_histo_all.png` | the MRI resampled into full-resolution histology pixels |
| `<base>_mri_on_histo_rgb.png` | the same over the RGB histology, when one exists |
| `<base>_roi_pairs.png`, `_roi_3d.png`, `_roi_support.png` | with `--save-roi-qc` |
| `<base>_process.log` | this section's log |

### The transform

`<base>_transform_grid.npz` (`version` 5) holds everything needed to map in both directions,
but no full-resolution rasters: those are regenerated from these fields when wanted.

**The sheet**, on the common canvas of shape `(Hc, Wc)`:

| field | contents |
|---|---|
| `xyz_norm_final` | `(Hc, Wc, 3)` MRI position `(x, y, z)` divided by `(X-1, Y-1, Z-1)` |
| `xy_hist_map` | `(Hc, Wc, 2)` histology position in canvas-histology pixels |
| `valid_surf` | `(Hc, Wc)` 1 where the sheet is inside the volume, the image and its mask |
| `histo_on_surface`, `mri_on_surface` | `(Hc, Wc)` the histology resampled on the sheet, and the MRI sampled on it |
| `ox_h`, `oy_h`, `G_e1`, `G_e2` | the canvas offset and the in-plane axes of the global plane |

**To the histology files.** `A_common_to_hr` maps a canvas pixel to a pixel of the
full-resolution histology *as loaded*, that is after the rotation and flip. `A_hr_to_tif`
undoes those, giving a pixel of the cleanslide TIFF named by `histo_source`; adding
`crop_origin_xy` reaches the raw `_BF.tif`, whose shape is `source_shape_hw`.

**To the MRI file and to millimetres.** `mri_origin`, `mri_spacing`, `mri_direction` and
`mri_size` describe the permuted volume, so a sheet position becomes world millimetres —
which is what a 3D viewer needs. `mri_permute`, `mri_view_flip`, `mri_orientation` and
`mri_canonical` say how that volume was built from `mri_source`, so the in-plane reversal, the
permutation and the LPS reorientation can be undone to reach the stored grid.

`subject_id`, `sample`, `stain`, `section_id` and `profile` identify the run.

## Requirements

A CUDA GPU and enough RAM: about 35 GB per section for 0.375 mm MRI slabs of about
410 × 440 × 370 voxels (peak in the coarse search), and about 50 GB for 4 µm histology (peak
when writing the full-resolution outputs). The pipeline runs on the CPU without a GPU, more
slowly.

## How the histology reaches MRI resolution

The full-resolution histology is Gaussian-smoothed in plane before it is downsampled to the
MRI voxel size, with `sigma = (spacing_mri - spacing_histo) / 2` per axis
(`preprocessing.anti_alias_sigma_mm`) — the usual `(f - 1) / 2` rule for a downsampling factor
`f`, which is about 12 at 31.75 µm and 100 at 4 µm against 0.4 mm MRI. Input histology
therefore does not need to be pre-smoothed. The tissue mask is not smoothed; it stays binary
and is resampled nearest-neighbour.

## Limitations

- **Very small sections.** Patch matching runs at MRI resolution with 64–96 px patches, so a
  section that is small at that resolution (roughly under 200 × 200 px, i.e. about 8 × 8 cm at
  0.4 mm) yields only a handful of matches (3–12 on the NextBrain sections we tested). The
  initial surface then rests on very few points, and a single wrong match can decide the
  result. If no match is consistent with the fitted plane, the pipeline currently stops with an
  `IndexError` (in `np.percentile`) instead of a clear message. A denser patch grid gives more
  matches but did not improve accuracy in our tests. A proper fix is planned.
- **RANSAC on sparse sections.** The RANSAC stage of the TPS fit requires 30 inliers, which a
  section with a handful of matches can never reach; it then logs `no valid hypothesis found;
  falling back to all points` and the MAD refit does the robust fitting. This is expected, not
  a failure. Lowering the threshold was measured and did not improve accuracy.
- **Fine-matching z window.** Fine matching searches ±24 mm (`FINE_MATCH_Z_HALF_MM` in
  `histo2mri/pipeline.py`) around the slice found by the coarse search, which tolerates about
  26° of section tilt for a typical section. The value was set on 0.375–0.4 mm MRI; for strongly
  tilted or very large sections, or other MRI, it may need adjusting.

## Tests

```bash
pytest                 # everything, including the end-to-end smoke test
pytest -m "not slow"   # unit tests only
```

The smoke test builds a synthetic BIDS section whose histology is a slice of its own MRI, so a
correct registration exists by construction, and checks that the pipeline runs and writes every
documented transform field. It asserts nothing about accuracy.

## Citation

Cite the software with the metadata in [CITATION.cff](CITATION.cff), which GitHub also
offers through its "Cite this repository" button:

> Pittayapong, S., Bachratá, B., Erber, I., Endmayr, V., Menard, C., Bogner, W.,
> Höftberger, R., Hametner, S., & Grabner, G. *Single-Section Histology-MRI Brain
> Registration: An Automated 2D-to-3D Approach Without Auxiliary Data* (version 0.1.0)
> [Computer software]. https://github.com/spittayapong/histo2mri

Until a stable release, the tags here are pre-releases: quote the one you used, or the
commit hash from `git rev-parse --short HEAD`, alongside the version. The paper reference
will be added here.

## License

MIT; see [LICENSE](LICENSE).
