# histo2mri

Registration of 2D histology sections to a 3D MRI volume: each section is mapped onto a curved
sheet in MRI voxel space.

<!-- Installation, usage, demo and citation will be added before the release. -->

## Limitations

- **Very small sections.** Patch matching runs at MRI resolution with 64–96 px patches, so a
  section that is small at that resolution (roughly under 200 × 200 px, i.e. about 8 × 8 cm at
  0.4 mm) yields only a handful of matches (3–12 on the NextBrain sections we tested). The
  initial surface then rests on very few points, and a single wrong match can decide the
  result. If no match is consistent with the fitted plane, the pipeline currently stops with an
  `IndexError` (in `np.percentile`) instead of a clear message. A denser patch grid gives more
  matches but did not improve accuracy in our tests. A proper fix is planned.
- **Input histology must be pre-smoothed.** The pipeline downsamples the full-resolution
  histology to MRI voxel size with linear interpolation and no anti-aliasing (a factor of
  about 75-100 for 4-5 µm histology and 0.4 mm MRI). Smooth the histology to roughly MRI
  resolution beforehand, as was done for the data we tested; otherwise cell-level texture
  aliases into noise. Moving this smoothing into the pipeline is planned.
- **MRI input must be MINC.** The MRI and its mask are read from `.mnc` files, whose header
  gives their spacing; the `spacing` column of the sections CSV applies to the histology only.
- **Fine-matching z window.** Fine matching searches ±24 mm (`FINE_MATCH_Z_HALF_MM` in
  `histo2mri/pipeline.py`) around the slice found by the coarse search, which tolerates about
  26° of section tilt for a typical section. The value was set on 0.375–0.4 mm MRI; for strongly
  tilted or very large sections, or other MRI, it may need adjusting.
