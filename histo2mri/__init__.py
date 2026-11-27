"""histo2mri: 2D histology to 3D MRI registration.

Each section is mapped onto a curved sheet in MRI voxel space, so a point on the slide and a
point in the volume convert into each other in both directions.

    from histo2mri import register_section
    info = register_section(bids_root, "NBP5816", "A2109", out_dir)

Reads a BIDS dataset and writes a derivative holding, per section, the transform (.npz) and
the QC figures. register_files does the same for one section given as explicit files. The transform carries the frames needed to return to the original slide and
to the original MRI, so full-resolution outputs are regenerated rather than stored.
"""
from .pipeline import PROFILES, register_files, register_section

__all__ = [
    "PROFILES",
    "register_files",
    "register_section",
    "__version__",
]

__version__ = "0.1.0a2"
