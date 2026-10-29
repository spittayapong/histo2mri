"""Curved-sheet refinement of the histology section surface inside the MRI volume.

This is the image-driven stage of pipeline.register_section, run after the TPS fit on top
of the global plane G (tps) and before the mapping back to HR histology (inverse). The
section is represented as xyz_norm: an (H, W, 3) grid over the common canvas holding MRI
positions normalised to [0, 1] per axis in (x, y, z) order, so that voxel index =
xyz_norm * (dim - 1). MRI tensors are indexed (z, y, x). The refinement maximises the
similarity between the MRI sampled on the sheet and the histology on the common canvas.

Base surface. refine_curved_sheet evaluates the TPS surface on the canvas plane grid,
clamps its displacement from the G plane and blends it toward G where match support is
low (confidence conf in [0, 1], 1 near matches, 0 far from them), then adds a B-spline
residual warm-started from the matches. The result, xyz_norm_base, and its unit normal
field n0 stay fixed during the refinement.

Refinement fields. Four B-spline fields on the canvas, optimised in turn at each level of
a coarse-to-fine pyramid (pyramid_refinement):

  phi      2D stationary velocity field. Integrated by scaling and squaring into a
           displacement that resamples the base surface (and its normal) tangentially,
           i.e. it slides points within the sheet. Not gated. It changes which base
           point each canvas pixel maps to; it does not warp the histology image.
  alpha    Scalar offset along the base normal. Not gated.
  rf_psi   A second, independent velocity field, multiplied by (1 - conf) before
           integration and composed after phi.
  rf_beta  A second scalar normal offset, multiplied by (1 - conf).

rf_psi and rf_beta therefore only add freedom where the base surface has little match
support and has fallen back toward the flat G plane. Together, a tangential and a normal
component can represent any 3D displacement, but the tangential part goes through the same
diffeomorphic integration as phi instead of being a free vector field that could fold the
sheet.

Why the sheet cannot fold. The flow of a smooth stationary velocity field is a
diffeomorphism. Scaling and squaring approximates it by dividing the velocity by 2^n,
which gives a displacement small enough to be invertible, and composing that displacement
with itself n times. A composition of invertible maps is invertible, so the Jacobian
determinant stays positive and the sheet does not fold (provided the initial step is small
compared with the grid spacing).

Why rigidity terms are needed. Not folding is not the same as keeping shape. The B-spline
bending energy penalises second differences of the control points, and second derivatives
vanish on any affine field, so a field that locally expands or compresses area costs
nothing. The sheet can then "melt": exchange area between regions without folding (on a
real section refined with bending energy alone, the 1st-99th percentiles of the local area
ratio to the base surface were about 0.38 and 3.47, with no folded pixels).

In the decomposition of Staring, Klein and Pluim (2007, "A rigidity penalty term for
nonrigid registration"), a transform with Jacobian J is locally rigid when three
conditions hold: affinity (second derivatives are zero; the bending energy),
orthonormality (J^T J = I) and properness (det J = 1). The last two are added
here on the sheet itself rather than on phi's 2D Jacobian, through the first fundamental
form (E, F, G) of xyz_norm compared with that of xyz_norm_base (E0, F0, G0):

  properness      lam_area  * log(area / area0)^2, with area = sqrt(E*G - F^2). The log
                  form is symmetric: a 2x expansion and a 2x compression cost the same,
                  whereas (area/area0 - 1)^2 would under-penalise compression, the
                  direction in which the sheet melts.
  orthonormality  lam_shape * ((E-E0)^2 + 2(F-F0)^2 + (G-G0)^2) / area0^2.

Because both terms are computed from xyz_norm, one penalty constrains all four fields
(alpha has the same blindness as phi: a linear ramp in alpha tilts the sheet at zero
bending cost). Because they are measured against the base surface, they do not penalise
the curvature already established by the matches, only what the refinement adds. Both
are weighted per pixel by losses._build_rigidity_weight. By default lam_area is on and
lam_shape is off: melting is mostly an exchange of area, and leaving shear free lets the
real sectioning deformation through.

Units. xyz_norm * (dim - 1) gives voxel-index coordinates. For isotropic voxels this
differs from physical space by a uniform scale, which cancels in every ratio against the
base surface, so no voxel spacing is needed. Anisotropic data would need a spacing weight.
"""
import matplotlib
import warnings
import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import os
import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from .bspline import BSplineFFD2D, BSplineResidualSurface3D, ScalarNormalOffsetField, _copy_ctrl, _upsample_ctrl, bending_energy, sheet_inside_volume_penalty
from .losses import _build_rigidity_weight, combined_masked_ssim_mind_loss, combined_masked_ssim_ngf_loss, first_fundamental_form, surface_area_ratio
from .qc import _plot_surface
from .sampling import grid_sample_2d_det, _prepare_data_tensor, sample_mri_cubic_3d, sample_mri_on_sheet
import logging

logger = logging.getLogger(__name__)


# Refinement
def warm_start_residual_ctrl_from_matches(
    residual, init_tps, init_G, matches, H, W, Xdim, Ydim, Zdim,
    device="cuda:0", n_iter=300, lr=1e-2, lam_bend=1e-4,
    hist_key="hist_center_yx", mri_key="mri_center_zyx",
):
    """Fit a 3D B-spline residual so that the TPS surface passes through the matched MRI points.

    For each valid match, the histology position is mapped to plane coordinates (u, v)
    through the inverse of G's in-plane similarity, (u, v) = A2^-1 (xy - t_xy), and the
    TPS surface is evaluated there. The target is the MRI match position minus the TPS
    position, normalised per axis by (dim - 1). The residual is fitted at the canvas pixel
    where the match lies on the sheet, i.e. the canvas pixel of plane point (u, v) (the
    inverse of the pixel -> plane mapping of prepare_coords_from_G_canvas), normalised by
    (W - 1, H - 1); this is the grid on which refine_curved_sheet evaluates the residual.
    Its control points are fitted with Adam on a smooth-L1 loss plus
    lam_bend * bending_energy. Matches without
    both keys, or with an MRI position of (-1, -1, -1), are skipped.

    Args:
        residual: BSplineResidualSurface3D whose control points are optimised in place.
        init_tps: callable mapping a (N, 2) tensor of plane (u, v) to (N, 3) MRI voxel
            positions (x, y, z).
        init_G: global plane dict; uses "t_xy", "A2_invT", "x_range", "y_range",
            "plane_step" and, if present, "canvas_offset_yx".
        matches: list of match dicts.
        H: common-canvas height in pixels.
        W: common-canvas width in pixels.
        Xdim: MRI slab size along x, in voxels.
        Ydim: MRI slab size along y, in voxels.
        Zdim: MRI slab size along z, in voxels.
        device: torch device string.
        n_iter: number of Adam iterations.
        lr: Adam learning rate.
        lam_bend: weight of the bending energy on the control points.
        hist_key: match key of the histology position (y, x), in common-canvas pixels.
        mri_key: match key of the MRI position (z, y, x), in slab voxels.

    Returns:
        The same `residual` module, with fitted control points.
    """
    dev = torch.device(device)

    # build match tensors
    hist_xy = []
    xyz = []
    for m in matches:
        if hist_key not in m or mri_key not in m:
            continue
        z, y, x = m[mri_key]
        if (z, y, x) == (-1, -1, -1):
            continue
        hy, hx = m[hist_key]
        hist_xy.append((float(hx), float(hy)))   # x,y
        xyz.append((float(x), float(y), float(z)))  # x,y,z

    U = torch.tensor(np.asarray(hist_xy, np.float32), device=dev)   # (N,2) hist x,y
    X = torch.tensor(np.asarray(xyz, np.float32), device=dev)       # (N,3)

    # inverse G: hist(x,y) -> uv for TPS eval
    t = torch.as_tensor(init_G["t_xy"], dtype=torch.float32, device=dev)
    A2_invT = torch.as_tensor(init_G["A2_invT"], dtype=torch.float32, device=dev)
    uv_ctrl = (U - t) @ A2_invT

    with torch.no_grad():
        pred = init_tps(uv_ctrl)            # (N,3) xyz
        dxyz = X - pred                     # vector residual in voxel
        target = torch.stack([
            dxyz[:, 0] / max(Xdim - 1, 1),
            dxyz[:, 1] / max(Ydim - 1, 1),
            dxyz[:, 2] / max(Zdim - 1, 1),
        ], dim=1)                           # normalized residual vector

    # Fit the residual where each match lies on the sheet. refine_curved_sheet evaluates it on
    # the canvas grid, and canvas pixel (x, y) is plane point
    # (u, v) = ((x - ox) * step + u0, (y - oy) * step + v0) (prepare_coords_from_G_canvas), so
    # the match's canvas pixel is that mapping inverted at uv_ctrl. (Using the histology pixel U
    # instead misplaces the fit by G's in-plane similarity: a median of ~17 px on the
    # evaluation sections.)
    u0, v0 = float(init_G["x_range"][0]), float(init_G["y_range"][0])
    step = float(init_G.get("plane_step", 1.0))
    Wm = int(round((float(init_G["x_range"][1]) - u0) / step)) + 1
    Hm = int(round((float(init_G["y_range"][1]) - v0) / step)) + 1
    if "canvas_offset_yx" in init_G:
        oy, ox = map(int, init_G["canvas_offset_yx"])
    else:
        oy, ox = (H - Hm) // 2, (W - Wm) // 2
    px = (uv_ctrl[:, 0] - u0) / step + ox
    py = (uv_ctrl[:, 1] - v0) / step + oy
    uv_norm = torch.stack([px / max(W - 1, 1), py / max(H - 1, 1)], dim=1)

    opt = torch.optim.Adam(residual.parameters(), lr=lr)
    for _ in range(n_iter):
        pred_d = residual(uv_norm)  # (N,3), H=None path
        loss = F.smooth_l1_loss(pred_d, target) + lam_bend * bending_energy(residual.ctrl)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return residual


def _to_t_1_1_hw(x, device):
    """Convert a 2D SimpleITK image or array, (H, W) or (1, H, W), to a float32 tensor (1, 1, H, W)."""
    if isinstance(x, sitk.Image):
        a = sitk.GetArrayFromImage(x)
    else:
        a = np.asarray(x)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"Expected 2D or [1,H,W], got {a.shape}")
    return torch.as_tensor(a, dtype=torch.float32, device=device)[None, None]


def make_soft_dilated_mask(
    mask_t,                 # [1,1,H,W], bool/0-1
    dilate_px=10,           # dilation radius in pixels
    blur_ks=15,             # odd kernel size for soft edge
    blur_passes=1,
    min_val=0.0,
    max_val=1.0,
):
    """Dilate a binary mask and soften its edge.

    The mask is thresholded at 0.5, dilated by max pooling and then box-filtered
    `blur_passes` times. The box filter pads with zeros, so values also fall off within
    about blur_ks / 2 pixels of the image border.

    Args:
        mask_t: tensor (1, 1, H, W), boolean or 0/1.
        dilate_px: dilation radius in pixels.
        blur_ks: box-filter size in pixels; made odd if even; <= 1 disables blurring.
        blur_passes: number of box-filter passes (at least 1 when blurring).
        min_val: lower clamp of the result.
        max_val: upper clamp of the result.

    Returns:
        Float tensor (1, 1, H, W) in [min_val, max_val].
    """
    m = (mask_t > 0.5).float()

    # 1) dilate
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=dilate_px)

    # 2) soften boundary
    if blur_ks > 1:
        if blur_ks % 2 == 0:
            blur_ks += 1
        for _ in range(max(1, int(blur_passes))):
            m = F.avg_pool2d(m, kernel_size=blur_ks, stride=1, padding=blur_ks // 2)

    # 3) clamp
    m = m.clamp(min_val, max_val)
    return m


def prepare_coords_from_G_canvas(G, H, W, device, common_shift_yx=None):
    """Compute plane and normalised pixel coordinates for every common-canvas pixel.

    The plane grid is the one G was fitted on: u runs over G["x_range"] and v over
    G["y_range"] in steps of G["plane_step"]. Canvas pixel (y, x) maps to
    u = (x - ox) * step + u0 and v = (y - oy) * step + v0. The offset (oy, ox) is
    `common_shift_yx` if given, else G["canvas_offset_yx"] if present, else the offset
    that centres the plane grid in the canvas.

    Args:
        G: global plane dict; uses "x_range", "y_range", optional "plane_step" and
            optional "canvas_offset_yx".
        H: canvas height in pixels.
        W: canvas width in pixels.
        device: torch device.
        common_shift_yx: optional (oy, ox) offset of the plane grid in the canvas.

    Returns:
        Tuple (uv_plane_flat, uv_norm_pix, valid_uv):
            uv_plane_flat: float tensor (H*W, 2), plane (u, v) per pixel, row-major.
            uv_norm_pix: float tensor (H, W, 2), (x / (W-1), y / (H-1)) per pixel.
            valid_uv: bool tensor (H, W), True where (u, v) lies inside G's ranges.
    """
    u0, u1 = map(float, G["x_range"])
    v0, v1 = map(float, G["y_range"])
    step = float(G.get("plane_step", 1.0))

    us = np.arange(u0, u1 + 1e-6, step, dtype=np.float32)
    vs = np.arange(v0, v1 + 1e-6, step, dtype=np.float32)
    Wm, Hm = len(us), len(vs)

    if common_shift_yx is None:
        if "canvas_offset_yx" in G:
            oy, ox = map(int, G["canvas_offset_yx"])
        else:
            oy = (H - Hm) // 2
            ox = (W - Wm) // 2
    else:
        oy, ox = map(int, common_shift_yx)

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    uu = (xx - ox) * step + u0
    vv = (yy - oy) * step + v0

    uv_plane_flat = torch.from_numpy(np.stack([uu, vv], axis=-1).reshape(-1, 2)).to(device=device, dtype=torch.float32)

    uv_norm_pix = torch.empty((H, W, 2), device=device, dtype=torch.float32)
    uv_norm_pix[..., 0] = torch.arange(W, device=device, dtype=torch.float32)[None, :] / max(W - 1, 1)
    uv_norm_pix[..., 1] = torch.arange(H, device=device, dtype=torch.float32)[:, None] / max(H - 1, 1)

    valid_uv = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)
    valid_uv = torch.from_numpy(valid_uv).to(device=device)

    return uv_plane_flat, uv_norm_pix, valid_uv


def _sheet_frame_like_tps(
    xyz_norm, Xdim, Ydim, Zdim,
    matches_xyz=None, scores=None,
    elev=24, azim=36, alpha=0.9, n_contours=12, cmap="viridis"
):
    """Render the sheet as a 3D surface coloured by depth and return an RGB frame (for the GIF); matches_xyz and scores are not drawn."""
    xyz = xyz_norm.detach().cpu().numpy() if torch.is_tensor(xyz_norm) else np.asarray(xyz_norm)
    Xs = xyz[..., 0] * (Xdim - 1)
    Ys = xyz[..., 1] * (Ydim - 1)
    Zs = xyz[..., 2] * (Zdim - 1)

    valid = (Xs >= 0) & (Xs <= Xdim - 1) & (Ys >= 0) & (Ys <= Ydim - 1) & (Zs >= 0) & (Zs <= Zdim - 1)
    Xp = np.where(valid, Xs, np.nan)
    Yp = np.where(valid, Ys, np.nan)
    Zp = np.where(valid, Zs, np.nan)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    zmin, zmax = np.nanmin(Zp), np.nanmax(Zp)
    norm = plt.Normalize(zmin, zmax)
    facecolors = matplotlib.colormaps[cmap](norm(Zp))
    ax.plot_surface(Xp, Yp, Zp, facecolors=facecolors, linewidth=0, antialiased=True, shade=False, alpha=alpha)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label("Depth (Z)")

    try:
        ax.contour(Xp, Yp, Zp, levels=n_contours, colors="k", linewidths=0.4, alpha=0.5)
    except Exception:
        pass

    ax.set_xlim(0, Xdim - 1); ax.set_ylim(0, Ydim - 1); ax.set_zlim(0, Zdim - 1)
    ax.set_box_aspect((Xdim, Ydim, Zdim))
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()  # RGB
    plt.close(fig)
    return frame


def pyramid_refinement(
    mri_t,
    mri_mask_t,
    hist_t,
    hist_mask_t,
    xyz_norm_base,       # [H,W,3] fixed base surface (TPS + residual warm-started from matches)
    n0_base,              # [H,W,3] fixed unit normal field of xyz_norm_base
    conf,                 # [H,W] match-support confidence in [0,1] (from _support_conf_from_ctrl_uv)
    uv_norm_pix,
    grid_id_2d,           # identity sampling grid in [-1,1] units, (1,H,W,2)
    H, W,
    levels,
    device,
    save_file=None,
    frame_cb=None,        # callback(xyz_norm) -> save gif frame
    frame_every=20,
    patience=10,
    base_supersample=32,   # upsample xyz_norm_base/n0_base this many times before phi resamples them
    phi_n_steps=6,          # scaling-and-squaring integration steps for phi's velocity field
    rf_psi_n_steps=6,       # scaling-and-squaring integration steps for rf_psi's velocity field
    # ---- local rigidity -------------------------------------------------------
    lam_area=1.0,           # properness  : weight on log(area/area0)^2
    lam_shape=0.0,          # orthonormality: weight on ||[[E,F],[F,G]] - base||^2 (OFF by default)
    w_rigid_mode="mask",    # "mask" | "conf" | "edge" | "none"
    rigid_edge_k=2.0,       # only used by w_rigid_mode="edge"
    similarity_weights=(1.0, 0.0, 0.0),   # (w_ssim, w_ngf, w_mind) of the image loss
):
    """Refine the base sheet with four B-spline fields over a coarse-to-fine pyramid.

    The sheet is
        xyz_norm(x) = S(p(x)) + (alpha(x) + g(x) * rf_beta(x)) * N(p(x)),
        p = (id + d_psi) o (id + d_phi),
    where S and N are the base surface and its unit normal, g = 1 - conf, d_phi is phi
    integrated by scaling and squaring, and d_psi is (g * rf_psi) integrated the same
    way. See the module docstring for the role of each field and of the rigidity terms.

    At each level the four fields are created on a gh x gw control grid, initialised by
    bilinear upsampling of the previous level's control points (zero at the first
    level). Three blocks then run in turn, each optimising some fields with Adam while
    the others are frozen: "Phi" (phi), "Alpha" (alpha) and "ResidualFree" (rf_psi and
    rf_beta jointly). Each block minimises
        image loss + bending energy of its fields + lamSheet * stay-inside-volume penalty
        + lamOut * normal offset outside the soft mask (Alpha and ResidualFree only)
        + lamRigid * (lam_area * L_area + lam_shape * L_shape),
    and stops early after `patience` iterations without a decrease of the total loss.
    The image loss compares the MRI sampled on the sheet (bilinear, zero outside the
    MRI mask) with hist_t, masked by a soft-dilated histology mask (8 px dilation,
    11 px blur). The base surface and normal are upsampled by `base_supersample` before
    they are resampled, so the tangential lookup is smoother than bilinear interpolation
    on the canvas grid. After each level and at the end, percentiles of the area ratio
    against the base surface are printed for pixels with non-zero rigidity weight.

    Args:
        mri_t: MRI slab tensor (1, 1, Z, Y, X).
        mri_mask_t: MRI mask tensor (1, 1, Z, Y, X); the sampled MRI is zero outside it.
        hist_t: histology on the common canvas, tensor (1, 1, H, W).
        hist_mask_t: histology mask tensor (1, 1, H, W), thresholded at 0.5.
        xyz_norm_base: fixed base sheet, tensor (H, W, 3), normalised (x, y, z).
        n0_base: unit normal field of xyz_norm_base, tensor (H, W, 3).
        conf: match-support confidence, tensor (H, W) in [0, 1]; rf_psi and rf_beta are
            multiplied by (1 - conf).
        uv_norm_pix: tensor (H, W, 2), (x / (W-1), y / (H-1)) per canvas pixel; the
            points where the B-spline fields are evaluated.
        grid_id_2d: identity sampling grid for grid_sample, tensor (1, H, W, 2) in
            [-1, 1], (x, y) order.
        H: canvas height in pixels.
        W: canvas width in pixels.
        levels: list of dicts, one per pyramid level (coarse to fine), with keys
            gh, gw: control-grid size;
            n_phi, lr_phi, lam_phi: iterations, learning rate and bending weight for phi;
            n_alpha, lr_alpha, lam_alpha: the same for alpha;
            n_rf, lr_rf, lam_rf: the same for rf_psi and rf_beta (one shared bending weight);
            lamSheet: weight of sheet_inside_volume_penalty;
            lamOut: weight of the squared normal offset outside the soft mask;
            lamRigid (optional, default 1.0): multiplier on the rigidity loss.
        device: torch device.
        save_file: optional path prefix; if set, a plot of the final normalised z is
            saved to "<save_file>_final.png".
        frame_cb: optional callable(xyz_norm), called every `frame_every` iterations, on
            the last iteration of each block and once on the final sheet.
        frame_every: iteration interval for frame_cb.
        patience: early-stopping patience per block, in iterations.
        base_supersample: upsampling factor applied to the base surface and normal.
        phi_n_steps: number of scaling-and-squaring steps for phi.
        rf_psi_n_steps: number of scaling-and-squaring steps for rf_psi.
        lam_area: weight of the properness term log(area / area0)^2.
        lam_shape: weight of the orthonormality term (first-fundamental-form drift).
        w_rigid_mode: per-pixel rigidity weight, "mask", "conf", "edge" or "none"
            (see losses._build_rigidity_weight).
        rigid_edge_k: edge gain for w_rigid_mode="edge".
        similarity_weights: (w_ssim, w_ngf, w_mind). The image loss is always
            losses.combined_masked_ssim_mind_loss (similarity_metric is fixed to "mind"
            in the code), so w_ngf has no effect. The parameter profiles use (1, 0, 0)
            (SSIM only) and (0, 0, 1) (MIND only).

    Returns:
        Tuple (phi, alpha, rf_psi, rf_beta, xyz_norm_final):
            phi: BSplineFFD2D, velocity field at the finest level.
            alpha: ScalarNormalOffsetField at the finest level.
            rf_psi: BSplineFFD2D, ungated velocity field at the finest level.
            rf_beta: ScalarNormalOffsetField, ungated, at the finest level.
            xyz_norm_final: tensor (H, W, 3), refined sheet in normalised (x, y, z),
                computed without gradient.
    """
    similarity_metric = "mind"   # "ngf" or "mind": selects the image loss
    w_ssim, w_ngf, w_mind = similarity_weights
    hist_mask_fixed = (hist_mask_t > 0.5).float().detach()
    mask_soft = make_soft_dilated_mask(hist_mask_fixed, dilate_px=8, blur_ks=11, blur_passes=1).detach()

    gate = (1.0 - conf).clamp(0.0, 1.0).detach()          # [H,W]
    gate3 = gate.unsqueeze(-1)                             # [H,W,1] for broadcasting onto 3-vectors

    # ---- Local rigidity setup ------------------------------------------------
    # dims recovers voxel-index coords from xyz_norm. Isotropic spacing => the
    # physical scale cancels in every ratio below, so no voxel spacing is needed.
    Zdim_r, Ydim_r, Xdim_r = mri_t.shape[-3:]
    dims_t = torch.tensor(
        [float(Xdim_r - 1), float(Ydim_r - 1), float(Zdim_r - 1)],
        device=device, dtype=torch.float32,
    )

    with torch.no_grad():
        E0, F0, G0 = first_fundamental_form(xyz_norm_base.detach(), dims_t)
        area0_sq = (E0 * G0 - F0 * F0).clamp_min(1e-12)
        area0 = area0_sq.sqrt()
        E0, F0, G0, area0 = E0.detach(), F0.detach(), G0.detach(), area0.detach()

        w_rigid = _build_rigidity_weight(
            w_rigid_mode, mask_soft, conf, hist_t, edge_k=rigid_edge_k
        ).detach()
        w_sum = w_rigid.sum().clamp_min(1e-8)

    rigid_on = (float(lam_area) > 0.0) or (float(lam_shape) > 0.0)
    logger.debug(f'Rigidity: mode={w_rigid_mode} lam_area={lam_area} lam_shape={lam_shape} active_px={int((w_rigid > 0).sum())}/{H * W}')

    def _rigidity_terms(xyz_norm):
        """Return (L_area, L_shape) as w_rigid-weighted means; both are zero when the sheet keeps the base metric."""
        if not rigid_on:
            z = torch.zeros((), device=device, dtype=torch.float32)
            return z, z
        E, Fq, G = first_fundamental_form(xyz_norm, dims_t)
        area = (E * G - Fq * Fq).clamp_min(1e-12).sqrt()
        # properness: det J = 1, in symmetric log form
        L_area = (w_rigid * torch.log(area / area0).pow(2)).sum() / w_sum
        if float(lam_shape) > 0.0:
            # orthonormality: J^T J = I, i.e. the metric tensor must not drift
            shape = ((E - E0).pow(2) + 2.0 * (Fq - F0).pow(2) + (G - G0).pow(2)) / area0_sq
            L_shape = (w_rigid * shape).sum() / w_sum
        else:
            L_shape = torch.zeros((), device=device, dtype=torch.float32)
        return L_area, L_shape

    def _rigidity_loss(xyz_norm, cfg):
        """Return ((L_area, L_shape), weighted rigidity loss) for one pyramid level."""
        L_area, L_shape = _rigidity_terms(xyz_norm)
        lam = float(cfg.get("lamRigid", 1.0))
        return (L_area, L_shape), lam * (float(lam_area) * L_area + float(lam_shape) * L_shape)

    def _report_area(tag, xyz_norm):
        """Print area-ratio percentiles (sheet / base) over pixels with non-zero rigidity weight."""
        with torch.no_grad():
            r = surface_area_ratio(xyz_norm, xyz_norm_base, dims_t.detach().cpu().numpy())
            m = (w_rigid > 0).detach().cpu().numpy()
            if m.sum() == 0:
                return
            p = np.percentile(r[m], [1, 5, 50, 95, 99])
            logger.debug(f'    [{tag}] area ratio final/base p1,p5,p50,p95,p99 = {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f} {p[4]:.3f}')

    xyz_base_img = xyz_norm_base.permute(2, 0, 1).unsqueeze(0).detach()   # [1,3,H,W]
    n0_base_img = n0_base.permute(2, 0, 1).unsqueeze(0).detach()          # [1,3,H,W]

    if base_supersample and base_supersample > 1:
        Hs, Ws = H * int(base_supersample), W * int(base_supersample)
        xyz_base_img = F.interpolate(xyz_base_img, size=(Hs, Ws), mode="bicubic", align_corners=True)
        n0_base_img = F.interpolate(n0_base_img, size=(Hs, Ws), mode="bicubic", align_corners=True)
        n0_base_img = n0_base_img / (n0_base_img.norm(dim=1, keepdim=True) + 1e-8)

    def _resample_surface(grid2d):
        """Look up the (supersampled) base surface and unit normal at a (1, H, W, 2) sampling grid."""
        xyz = F.grid_sample(xyz_base_img, grid2d, mode="bilinear", padding_mode="border", align_corners=True)
        n0s = F.grid_sample(n0_base_img, grid2d, mode="bilinear", padding_mode="border", align_corners=True)
        xyz = xyz[0].permute(1, 2, 0)                       # (H,W,3)
        n0s = n0s[0].permute(1, 2, 0)
        n0s = n0s / (n0s.norm(dim=-1, keepdim=True) + 1e-8)
        return xyz, n0s

    def _scaling_and_squaring_2d(v, n_steps=6):
        # v: (H,W,2) stationary velocity field, same normalized [-1,1] units as grid_id_2d.
        # Integrates the flow ODE d(phi)/dt = v(phi(t)) from t=0 (identity) to t=1 by
        # repeated self-composition of a small initial step v / 2^n_steps (scaling and
        # squaring, as in diffeomorphic registration, e.g. DARTEL): a small smooth step
        # composed with itself stays invertible, unlike one large displacement.
        d = v / (2 ** n_steps)
        for _ in range(n_steps):
            d_img = d.permute(2, 0, 1).unsqueeze(0)                     # (1,2,H,W)
            query = grid_id_2d + d.unsqueeze(0)                         # (1,H,W,2)
            d_warped = grid_sample_2d_det(d_img, query, padding_mode="border")
            d = d + d_warped[0].permute(1, 2, 0)                        # (H,W,2)
        return d

    def _compose_disp_2d(base_grid2d, extra_disp):
        # base_grid2d: (1,H,W,2) sampling grid (e.g. identity + phi's integrated displacement).
        # extra_disp:  (H,W,2) an additional displacement (e.g. rf_psi's gated, integrated
        # velocity), evaluated at base_grid2d's positions and added on top. This composes
        # "extra" after "base" (extra_map o base_map), which is diffeomorphic if both are.
        extra_img = extra_disp.permute(2, 0, 1).unsqueeze(0)                # (1,2,H,W)
        extra_at_base = grid_sample_2d_det(extra_img, base_grid2d, padding_mode="border")
        return base_grid2d + extra_at_base.permute(0, 2, 3, 1)              # (1,H,W,2)

    def _run_block(name, params, lr, n_iter, build_xyz_fn, reg_fn, print_extra_fn):
        """Optimise `params` with Adam on image loss + reg_fn, with early stopping on the total loss."""
        opt = torch.optim.Adam([{"params": params, "lr": lr}])
        best_loss, no_improve = float("inf"), 0
        for it in range(n_iter):
            xyz_norm, out_val = build_xyz_fn()

            if frame_cb is not None and (it % max(1, int(frame_every)) == 0 or it == n_iter - 1):
                frame_cb(xyz_norm)

            mri2d = sample_mri_on_sheet(mri_t, mri_mask_t, xyz_norm)
            if similarity_metric == "mind":
                loss_img, comp = combined_masked_ssim_mind_loss(mri2d, hist_t, mask_soft, w_ssim=w_ssim, w_mind=w_mind)
            else:
                loss_img, comp = combined_masked_ssim_ngf_loss(mri2d, hist_t, mask_soft, w_ssim=w_ssim, w_ngf=w_ngf)
            L_reg_terms, loss_extra = reg_fn(xyz_norm, out_val)
            loss = loss_img + loss_extra

            opt.zero_grad()
            loss.backward()
            opt.step()

            if it % 50 == 0 or it == n_iter - 1:
                metric_name = "ngf" if "ngf" in comp else "mind"
                logger.debug(f"[{name}] it={it:04d}/{n_iter} loss={loss.item():.6f} img={loss_img.item():.6f} {print_extra_fn(L_reg_terms)} ssim={comp['ssim']:.4f} {metric_name}={comp[metric_name]:.4f}")

            if loss.item() < best_loss:
                best_loss, no_improve = loss.item(), 0
            else:
                no_improve += 1
            if no_improve >= patience:
                logger.debug(f'[{name}] Early stopping at it={it} (no improvement for {patience} iters)')
                break

    phi = None
    alpha = None
    rf_psi = None
    rf_beta = None

    logger.debug('===== Pyramid refinement: phi (velocity) + alpha (normal) + rf_psi (gated velocity) + rf_beta (gated normal) =====')
    for li, cfg in enumerate(levels):
        gh, gw = cfg["gh"], cfg["gw"]

        if phi is None:
            phi = BSplineFFD2D(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(phi.ctrl.detach(), gh, gw)
            phi = BSplineFFD2D(gh, gw, device=device).to(device)
            _copy_ctrl(phi.ctrl, ctrl_up)

        if alpha is None:
            alpha = ScalarNormalOffsetField(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(alpha.ctrl.detach(), gh, gw)
            alpha = ScalarNormalOffsetField(gh, gw, device=device).to(device)
            _copy_ctrl(alpha.ctrl, ctrl_up)

        if rf_psi is None:
            rf_psi = BSplineFFD2D(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(rf_psi.ctrl.detach(), gh, gw)
            rf_psi = BSplineFFD2D(gh, gw, device=device).to(device)
            _copy_ctrl(rf_psi.ctrl, ctrl_up)

        if rf_beta is None:
            rf_beta = ScalarNormalOffsetField(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(rf_beta.ctrl.detach(), gh, gw)
            rf_beta = ScalarNormalOffsetField(gh, gw, device=device).to(device)
            _copy_ctrl(rf_beta.ctrl, ctrl_up)

        # ---- Step Phi: freeze alpha, rf_psi and rf_beta; optimise the tangential resample ----
        with torch.no_grad():
            a_fix = alpha(uv_norm_pix, H=H, W=W)
            disp_psi_fix = _scaling_and_squaring_2d(gate3 * rf_psi(uv_norm_pix, H=H, W=W), n_steps=rf_psi_n_steps)
            rf_beta_fix = gate3 * rf_beta(uv_norm_pix, H=H, W=W)

        logger.debug(f'--- Level {li} ({gh}x{gw}) Step Phi ---')

        def _build_phi():
            v = phi(uv_norm_pix, H=H, W=W)
            disp = _scaling_and_squaring_2d(v, n_steps=phi_n_steps)
            grid2d = grid_id_2d + disp.unsqueeze(0)
            grid2d = _compose_disp_2d(grid2d, disp_psi_fix)
            xyz_resampled, n0_resampled = _resample_surface(grid2d)
            xyz_norm = xyz_resampled + a_fix * n0_resampled + rf_beta_fix * n0_resampled
            return xyz_norm, None

        def _reg_phi(xyz_norm, _):
            L_bend = bending_energy(phi.ctrl)
            L_sheet = sheet_inside_volume_penalty(xyz_norm)
            (L_area, L_shape), L_rigid = _rigidity_loss(xyz_norm, cfg)
            return (L_bend, L_sheet, L_area, L_shape),\
                   cfg["lam_phi"] * L_bend + cfg["lamSheet"] * L_sheet + L_rigid

        _run_block(
            f"L{li} Phi", phi.parameters(), cfg["lr_phi"], cfg["n_phi"],
            _build_phi, _reg_phi,
            lambda t: f"bend={t[0].item():.2e} sheet={t[1].item():.2e} area={t[2].item():.2e} shape={t[3].item():.2e}"
        )

        # ---- Step Alpha: freeze phi, rf_psi and rf_beta; optimise the normal offset ----
        with torch.no_grad():
            disp_fix = _scaling_and_squaring_2d(phi(uv_norm_pix, H=H, W=W), n_steps=phi_n_steps)
            grid2d_fix = grid_id_2d + disp_fix.unsqueeze(0)
            disp_psi_fix = _scaling_and_squaring_2d(gate3 * rf_psi(uv_norm_pix, H=H, W=W), n_steps=rf_psi_n_steps)
            grid2d_fix = _compose_disp_2d(grid2d_fix, disp_psi_fix)
            xyz_resampled_fix, n0_resampled_fix = _resample_surface(grid2d_fix)
            rf_beta_fix = gate3 * rf_beta(uv_norm_pix, H=H, W=W)

        logger.debug(f'--- Level {li} ({gh}x{gw}) Step Alpha ---')

        def _build_alpha():
            a = alpha(uv_norm_pix, H=H, W=W)
            xyz_norm = xyz_resampled_fix + a * n0_resampled_fix + rf_beta_fix * n0_resampled_fix
            return xyz_norm, a

        def _reg_alpha(xyz_norm, a):
            L_bend = bending_energy(alpha.ctrl)
            L_sheet = sheet_inside_volume_penalty(xyz_norm)
            L_out = (a[..., 0].pow(2) * (1.0 - mask_soft[0, 0])).mean()
            # alpha has the same affine blindness as phi: a linear ramp in alpha
            # tilts the sheet at zero bending cost, which stretches local area.
            (L_area, L_shape), L_rigid = _rigidity_loss(xyz_norm, cfg)
            return (L_bend, L_sheet, L_out, L_area, L_shape),\
                   cfg["lam_alpha"] * L_bend + cfg["lamSheet"] * L_sheet + cfg["lamOut"] * L_out + L_rigid

        _run_block(
            f"L{li} Alpha", alpha.parameters(), cfg["lr_alpha"], cfg["n_alpha"],
            _build_alpha, _reg_alpha,
            lambda t: f"bend={t[0].item():.2e} sheet={t[1].item():.2e} out={t[2].item():.2e} area={t[3].item():.2e} shape={t[4].item():.2e}"
        )

        # ---- Step ResidualFree: freeze phi and alpha; jointly optimise rf_psi (tangential,
        # composed after phi) and rf_beta (normal), both gated by 1 - conf ----
        with torch.no_grad():
            disp_fix = _scaling_and_squaring_2d(phi(uv_norm_pix, H=H, W=W), n_steps=phi_n_steps)
            grid2d_phi_fix = grid_id_2d + disp_fix.unsqueeze(0)
            a_fix = alpha(uv_norm_pix, H=H, W=W)

        logger.debug(f'--- Level {li} ({gh}x{gw}) Step ResidualFree (rf_psi tangential + rf_beta normal, gated by 1-conf) ---')

        def _build_rf():
            v_psi = rf_psi(uv_norm_pix, H=H, W=W)
            disp_psi = _scaling_and_squaring_2d(gate3 * v_psi, n_steps=rf_psi_n_steps)
            grid2d = _compose_disp_2d(grid2d_phi_fix, disp_psi)
            xyz_resampled, n0_resampled = _resample_surface(grid2d)
            rf_beta_gated = gate3 * rf_beta(uv_norm_pix, H=H, W=W)
            xyz_norm = xyz_resampled + a_fix * n0_resampled + rf_beta_gated * n0_resampled
            return xyz_norm, rf_beta_gated

        def _reg_rf(xyz_norm, rf_beta_gated):
            L_bend_psi = bending_energy(rf_psi.ctrl)
            L_bend_beta = bending_energy(rf_beta.ctrl)
            L_sheet = sheet_inside_volume_penalty(xyz_norm)
            L_out = (rf_beta_gated[..., 0].pow(2) * (1.0 - mask_soft[0, 0])).mean()
            (L_area, L_shape), L_rigid = _rigidity_loss(xyz_norm, cfg)
            return (L_bend_psi, L_bend_beta, L_sheet, L_out, L_area, L_shape),\
                   cfg["lam_rf"] * (L_bend_psi + L_bend_beta) + cfg["lamSheet"] * L_sheet\
                   + cfg["lamOut"] * L_out + L_rigid

        _run_block(
            f"L{li} ResidualFree", list(rf_psi.parameters()) + list(rf_beta.parameters()), cfg["lr_rf"], cfg["n_rf"],
            _build_rf, _reg_rf,
            lambda t: f"bend_psi={t[0].item():.2e} bend_beta={t[1].item():.2e} sheet={t[2].item():.2e} "
                      f"out={t[3].item():.2e} area={t[4].item():.2e} shape={t[5].item():.2e}"
        )

        with torch.no_grad():
            _dv = _scaling_and_squaring_2d(phi(uv_norm_pix, H=H, W=W), n_steps=phi_n_steps)
            _g2 = _compose_disp_2d(
                grid_id_2d + _dv.unsqueeze(0),
                _scaling_and_squaring_2d(gate3 * rf_psi(uv_norm_pix, H=H, W=W), n_steps=rf_psi_n_steps),
            )
            _xyz_l, _n0_l = _resample_surface(_g2)
            _report_area(
                f"L{li} end",
                _xyz_l + alpha(uv_norm_pix, H=H, W=W) * _n0_l
                       + gate3 * rf_beta(uv_norm_pix, H=H, W=W) * _n0_l,
            )

    with torch.no_grad():
        disp_final = _scaling_and_squaring_2d(phi(uv_norm_pix, H=H, W=W), n_steps=phi_n_steps)
        grid2d_phi_final = grid_id_2d + disp_final.unsqueeze(0)
        disp_psi_final = _scaling_and_squaring_2d(gate3 * rf_psi(uv_norm_pix, H=H, W=W), n_steps=rf_psi_n_steps)
        grid2d_final = _compose_disp_2d(grid2d_phi_final, disp_psi_final)
        xyz_resampled_final, n0_resampled_final = _resample_surface(grid2d_final)
        a_final = alpha(uv_norm_pix, H=H, W=W)
        rf_beta_final = gate3 * rf_beta(uv_norm_pix, H=H, W=W)
        xyz_norm_final = xyz_resampled_final + a_final * n0_resampled_final + rf_beta_final * n0_resampled_final
        _report_area("FINAL", xyz_norm_final)
        if frame_cb is not None:
            frame_cb(xyz_norm_final)
        if save_file is not None:
            _plot_surface(
                xyz_norm_final[..., 2],
                "Pyramid refinement – final surface (phi+alpha+gated rf_psi/rf_beta)",
                save_file=f"{save_file}_final.png",
            )

    return phi, alpha, rf_psi, rf_beta, xyz_norm_final


def _xyz_from_G_uv(uv_plane_flat, G, H, W, device):
    """Return the G-plane voxel positions X0 + u*e1 + v*e2 as a (H, W, 3) tensor."""
    uv = uv_plane_flat.detach().cpu().numpy().reshape(H, W, 2).astype(np.float32)
    uu, vv = uv[..., 0], uv[..., 1]
    X0 = np.asarray(G["X0"], np.float32)
    e1 = np.asarray(G["e1"], np.float32)
    e2 = np.asarray(G["e2"], np.float32)
    xyz_g = X0[None, None, :] + uu[..., None] * e1[None, None, :] + vv[..., None] * e2[None, None, :]
    return torch.from_numpy(xyz_g).to(device=device, dtype=torch.float32)


def _support_conf_from_ctrl_uv(uv_plane_flat, ctrl_uv, H, W, device, sigma_uv=20.0, min_conf=0.25):
    """Return the (H, W) match-support confidence from the distance to the nearest control (u, v)."""
    if ctrl_uv is None:
        return torch.ones((H, W), device=device, dtype=torch.float32)
    cuv = ctrl_uv.detach().cpu().numpy() if torch.is_tensor(ctrl_uv) else np.asarray(ctrl_uv, np.float32)
    if cuv.ndim != 2 or cuv.shape[0] == 0 or cuv.shape[1] != 2:
        return torch.ones((H, W), device=device, dtype=torch.float32)

    uv = uv_plane_flat.detach().cpu().numpy().astype(np.float32)
    d, _ = cKDTree(cuv.astype(np.float32)).query(uv, k=1)
    conf = np.exp(-(d * d) / (2.0 * float(sigma_uv) ** 2))
    conf = np.clip((conf - float(min_conf)) / (1.0 - float(min_conf) + 1e-8), 0.0, 1.0)
    conf = conf ** 2
    conf[d > 2.0 * float(sigma_uv)] = 0.0
    return torch.from_numpy(conf.reshape(H, W)).to(device=device, dtype=torch.float32)


def tps_sheet_xyz_norm(init_tps, G, uv_plane_flat, H, W, Xdim, Ydim, Zdim, device,
                       ctrl_uv=None, sigma_uv=20.0, min_conf=0.25, max_disp_vox=16.0):
    """Return the TPS sheet in normalised xyz, clamped and blended toward the G plane by confidence.

    This is the base sheet of refine_curved_sheet before its B-spline warm start (step 2 of
    its docstring). init_tps maps (N, 2) plane (u, v) to (N, 3) voxel xyz; its displacement
    from G is clamped to max_disp_vox voxels and scaled by the support confidence of
    ctrl_uv. Returns an (H, W, 3) tensor, xyz / (dim - 1) per axis.
    """
    with torch.no_grad():
        xyz_tps = init_tps(uv_plane_flat.detach()).view(H, W, 3)  # voxel xyz
        xyz_g = _xyz_from_G_uv(uv_plane_flat, G, H, W, device)

        disp = xyz_tps - xyz_g
        dn = torch.linalg.norm(disp, dim=-1, keepdim=True).clamp_min(1e-6)
        disp = disp * torch.clamp(float(max_disp_vox) / dn, max=1.0)

        conf = _support_conf_from_ctrl_uv(
            uv_plane_flat, ctrl_uv, H, W, device,
            sigma_uv=sigma_uv, min_conf=min_conf
        )
        xyz_safe = xyz_g + conf[..., None] * disp

        xyz_norm = torch.stack([
            xyz_safe[..., 0] / max(Xdim - 1, 1),
            xyz_safe[..., 1] / max(Ydim - 1, 1),
            xyz_safe[..., 2] / max(Zdim - 1, 1),
        ], dim=-1)
    return xyz_norm


def sheet_folding_pct(xyz_norm, valid, dims_xyz, G):
    """Percentage of folded canvas nodes of a sheet.

    The sheet's tangents (central differences of xyz_norm * (dim - 1), i.e. voxel units) are
    projected onto G's in-plane basis (e1, e2), giving a signed 2x2 Jacobian determinant per
    canvas node; the sheet is folded where it is <= 0. Only interior nodes are counted: valid
    nodes whose four neighbours are valid, excluding the canvas border.

    Args:
        xyz_norm: tensor or array (H, W, 3), sheet in normalised (x, y, z).
        valid: bool tensor or array (H, W), nodes to evaluate.
        dims_xyz: MRI size (X, Y, Z), e.g. slab_Im.GetSize().
        G: global plane dict; uses "e1" and "e2".

    Returns:
        Folded share of the interior valid nodes in percent (0.0 if there are none).
    """
    P = xyz_norm.detach().cpu().numpy() if torch.is_tensor(xyz_norm) else np.asarray(xyz_norm)
    P = P.astype(np.float64) * (np.asarray(dims_xyz, np.float64) - 1.0)
    e1 = np.asarray(G["e1"], np.float64)
    e2 = np.asarray(G["e2"], np.float64)
    e1, e2 = e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)
    Pv, Pu = np.gradient(P, axis=0), np.gradient(P, axis=1)
    det = (Pu @ e1) * (Pv @ e2) - (Pu @ e2) * (Pv @ e1)

    v = valid.detach().cpu().numpy() if torch.is_tensor(valid) else np.asarray(valid)
    v = v.astype(bool)
    inner = v & np.roll(v, 1, 0) & np.roll(v, -1, 0) & np.roll(v, 1, 1) & np.roll(v, -1, 1)
    inner[[0, -1], :] = False
    inner[:, [0, -1]] = False
    return 100.0 * float((det[inner] <= 0).mean()) if inner.any() else 0.0


def refine_curved_sheet(
    slab_Im, slab_Mm,
    init_G, init_tps, matches_for_warmstart,
    device,
    hist_on_init_tps=None,
    hist_mask_on_init_tps=None,
    ctrl_uv_support=None,
    tps_sigma_uv=20.0,
    tps_min_conf=0.25,
    tps_max_disp_vox=16.0,
    save_file=None,
    gif_file=None,         # e.g. ".../refine_sheet.gif"
    gif_every=20,          # capture every N iters
    gif_fps=10,
    # ---- local rigidity (see module docstring) --------------------------------
    lam_area=1.0,
    lam_shape=0.0,
    w_rigid_mode="mask",
    rigid_edge_k=2.0,
    lam_rigid_levels=(1.0, 1.0, 1.0),   # per-pyramid-level multiplier on the rigidity loss
    similarity_weights=(1.0, 0.0, 0.0),  # (w_ssim, w_ngf, w_mind), passed to pyramid_refinement
):
    """Refine the histology sheet inside the MRI slab by image similarity.

    Steps:
      1. Compute plane (u, v) and normalised pixel coordinates for every common-canvas
         pixel (prepare_coords_from_G_canvas, using G["canvas_offset_yx"] if present).
      2. Build the base sheet. init_tps is evaluated on the canvas plane grid; its
         displacement from the G plane is clamped to tps_max_disp_vox voxels and scaled
         by the support confidence conf, so the sheet falls back to G far from matches.
         conf is computed from the distance d to the nearest ctrl_uv_support point:
         exp(-d^2 / (2 sigma^2)), rescaled so that tps_min_conf maps to 0, clipped to
         [0, 1], squared, and set to 0 for d > 2 sigma. A 12 x 12 B-spline residual
         warm-started from the matches (warm_start_residual_ctrl_from_matches) is
         added. The unit normal n0 is computed by central differences of the base
         sheet in normalised coordinates.
      3. Run pyramid_refinement on three levels with 12 x 12, 24 x 24 and 32 x 32
         control grids and fixed per-level iteration counts, learning rates and
         regularisation weights; lam_rigid_levels scales the rigidity loss per level.
      4. Sample the MRI on the refined sheet with Catmull-Rom cubic interpolation.
    Optionally, the sheet is rendered every `gif_every` iterations into a GIF.

    Args:
        slab_Im: MRI slab, SimpleITK image (from preprocessing.prepare_mri_slab).
        slab_Mm: MRI slab mask, SimpleITK image on the same grid.
        init_G: global plane dict; uses X0, e1, e2, A2_invT, t_xy, x_range, y_range,
            plane_step and, if present, canvas_offset_yx.
        init_tps: callable mapping a (N, 2) tensor of plane (u, v) to (N, 3) MRI voxel
            positions (x, y, z); in the pipeline, tps.ResidualToGSurface3D (G plus the
            residual 3D TPS).
        matches_for_warmstart: list of match dicts with "hist_center_yx" (common-canvas
            pixels) and "mri_center_zyx" (slab voxels), used for the residual warm start.
        device: torch device string.
        hist_on_init_tps: histology on the common canvas, SimpleITK image or array of
            shape (H, W) or (1, H, W). Required; defines H and W.
        hist_mask_on_init_tps: optional histology mask on the same grid; None means
            all ones.
        ctrl_uv_support: plane (u, v) of the inlier matches, tensor or array (M, 2);
            None gives conf = 1 everywhere.
        tps_sigma_uv: sigma of the confidence fall-off, in plane units.
        tps_min_conf: confidence floor that is mapped to 0.
        tps_max_disp_vox: maximum TPS displacement from the G plane, in voxels.
        save_file: optional path prefix for the final-surface plot (see
            pyramid_refinement).
        gif_file: optional path of a GIF showing the sheet during the optimisation.
        gif_every: GIF frame interval, in iterations.
        gif_fps: GIF frame rate.
        lam_area: weight of the properness (area) term; see pyramid_refinement.
        lam_shape: weight of the orthonormality (shape) term; see pyramid_refinement.
        w_rigid_mode: per-pixel rigidity weight mode; see pyramid_refinement.
        rigid_edge_k: edge gain for w_rigid_mode="edge".
        lam_rigid_levels: multipliers on the rigidity loss for the three pyramid
            levels; if shorter than 3, the last value is repeated.
        similarity_weights: (w_ssim, w_ngf, w_mind) for the image loss; see
            pyramid_refinement.

    Returns:
        Tuple of 10 elements:
            xyz_norm_final: tensor (H, W, 3), refined sheet in normalised (x, y, z).
            mri_on_final_sheet: numpy array (H, W), MRI sampled on the refined sheet
                (Catmull-Rom, border padding, not masked).
            valid_out: numpy bool array (H, W), True where xyz_norm_final is finite and
                inside [0, 1]^3.
            xyz_norm_base: tensor (H, W, 3), base sheet before refinement.
            conf: tensor (H, W), match-support confidence in [0, 1].
            phi_refined: BSplineFFD2D, refined phi.
            n0: tensor (H, W, 3), unit normal of the base sheet (normalised coordinates).
            alpha_refined: ScalarNormalOffsetField, refined alpha.
            rf_psi_refined: BSplineFFD2D, refined rf_psi (before gating).
            rf_beta_refined: ScalarNormalOffsetField, refined rf_beta (before gating).
    """
    logger.debug('-------- Refine curved sheet --------')

    mri_t = _prepare_data_tensor(slab_Im, device)
    mri_mask_t = _prepare_data_tensor(slab_Mm, device)
    Zdim, Ydim, Xdim = mri_t.shape[-3:]

    if hist_on_init_tps is None:
        raise ValueError("hist_on_init_tps is required for this mode.")

    hist_t = _to_t_1_1_hw(hist_on_init_tps, device)
    if hist_mask_on_init_tps is None:
        hist_mask_t = torch.ones_like(hist_t)
    else:
        hist_mask_t = (_to_t_1_1_hw(hist_mask_on_init_tps, device) > 0.5).float()

    _, _, H, W = hist_t.shape

    matches_xyz = []
    matches_scores = []
    has_scores = True
    for m in matches_for_warmstart:
        if "mri_center_zyx" not in m:
            continue
        z, y, x = m["mri_center_zyx"]
        if (z, y, x) == (-1, -1, -1):
            continue
        matches_xyz.append((float(x), float(y), float(z)))
        if "score" in m:
            matches_scores.append(float(m["score"]))
        else:
            has_scores = False

    matches_xyz = np.asarray(matches_xyz, dtype=np.float32)
    matches_scores = np.asarray(matches_scores, dtype=np.float32) if (has_scores and len(matches_scores) == len(matches_xyz)) else None

    writer = None
    if gif_file is not None:
        gif_dir = os.path.dirname(gif_file)
        if gif_dir:
            os.makedirs(gif_dir, exist_ok=True)
        writer = imageio.get_writer(gif_file, mode="I", fps=max(1, int(gif_fps)))

    def _frame_cb(xyz_norm):
        if writer is None:
            return
        frame = _sheet_frame_like_tps(
            xyz_norm=xyz_norm,
            Xdim=Xdim, Ydim=Ydim, Zdim=Zdim,
            matches_xyz=matches_xyz,
            scores=matches_scores,
        )
        writer.append_data(frame)

    try:
        uv_plane_flat_init, uv_norm_pix, _ = prepare_coords_from_G_canvas(
            init_G, H, W, device, common_shift_yx=init_G.get("canvas_offset_yx", None)
        )

        grid_h, grid_w = 12, 12
        residual_warm = BSplineResidualSurface3D(grid_h, grid_w, device=device).to(device)
        residual_warm = warm_start_residual_ctrl_from_matches(
            residual=residual_warm,
            init_tps=init_tps,
            init_G=init_G,
            matches=matches_for_warmstart,
            H=H, W=W, Xdim=Xdim, Ydim=Ydim, Zdim=Zdim,
            device=device
        )

        with torch.no_grad():
            xyz_norm_tps = tps_sheet_xyz_norm(
                init_tps=init_tps,
                G=init_G,
                uv_plane_flat=uv_plane_flat_init,
                H=H, W=W,
                Xdim=Xdim, Ydim=Ydim, Zdim=Zdim,
                device=device,
                ctrl_uv=ctrl_uv_support,
                sigma_uv=tps_sigma_uv,
                min_conf=tps_min_conf,
                max_disp_vox=tps_max_disp_vox,
            )
            _frame_cb(xyz_norm_tps)

            delta_norm_init = residual_warm(uv_norm_pix, H=H, W=W)
            xyz_norm_base = (xyz_norm_tps + delta_norm_init).detach()

            S = xyz_norm_base.clamp(0, 1)
            Su = torch.zeros_like(S)
            Sv = torch.zeros_like(S)
            Su[:, 1:-1] = (S[:, 2:] - S[:, :-2]) * 0.5
            Su[:, 0] = S[:, 1] - S[:, 0]
            Su[:, -1] = S[:, -1] - S[:, -2]
            Sv[1:-1, :] = (S[2:, :] - S[:-2, :]) * 0.5
            Sv[0, :] = S[1, :] - S[0, :]
            Sv[-1, :] = S[-1, :] - S[-2, :]
            n0 = torch.cross(Su, Sv, dim=-1)
            n0 = (n0 / (n0.norm(dim=-1, keepdim=True) + 1e-8)).detach()

            # Same confidence field that blends the TPS with the G-plane fallback above;
            # (1 - conf) gates rf_psi and rf_beta, so they are free only where the base
            # surface lacks match support.
            conf = _support_conf_from_ctrl_uv(
                uv_plane_flat_init, ctrl_uv_support, H, W, device,
                sigma_uv=tps_sigma_uv, min_conf=tps_min_conf
            ).detach()

        grid_id_2d = torch.stack([
            2 * uv_norm_pix[..., 0] - 1.0,
            2 * uv_norm_pix[..., 1] - 1.0,
        ], dim=-1).unsqueeze(0)   # (1,H,W,2)

        _lr = list(lam_rigid_levels) + [lam_rigid_levels[-1]] * 3
        levels = [
            dict(gh=12, gw=12, n_phi=100, lr_phi=2e-3, lam_phi=8e-4,
                            n_alpha=100, lr_alpha=5e-4, lam_alpha=1e-2,
                            n_rf=60, lr_rf=2e-3, lam_rf=8e-3,
                            lamSheet=2e-3, lamOut=1e-2, lamRigid=_lr[0]),
            dict(gh=24, gw=24, n_phi=120, lr_phi=1e-3, lam_phi=6e-4,
                            n_alpha=120, lr_alpha=2.5e-4, lam_alpha=7.5e-3,
                            n_rf=30, lr_rf=1e-3, lam_rf=6e-3,
                            lamSheet=1.5e-3, lamOut=1e-2, lamRigid=_lr[1]),
            dict(gh=32, gw=32, n_phi=140, lr_phi=7e-4, lam_phi=4e-4,
                            n_alpha=140, lr_alpha=1.75e-4, lam_alpha=5e-3,
                            n_rf=40, lr_rf=7e-4, lam_rf=4e-3,
                            lamSheet=1e-3, lamOut=1e-2, lamRigid=_lr[2]),
        ]

        # Deterministic refinement, so repeated runs give identical results. F.grid_sample
        # accumulates the gradient w.r.t. its input with GPU atomic adds (run-to-run noise);
        # the two calls that need that gradient use grid_sample_2d_det, whose gather backward
        # is deterministic in torch's deterministic mode. The remaining grid_sample calls only
        # differentiate w.r.t. the grid, which is deterministic, so their warnings are
        # silenced. The caller's deterministic-mode setting is restored afterwards.
        _det_prev = torch.are_deterministic_algorithms_enabled()
        _det_prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"grid_sampler_.*_backward_cuda does not have a deterministic")
                warnings.filterwarnings("ignore", message=r".*CUBLAS_WORKSPACE_CONFIG.*")
                phi_refined, alpha_refined, rf_psi_refined, rf_beta_refined, xyz_norm_final = pyramid_refinement(
                    mri_t=mri_t,
                    mri_mask_t=mri_mask_t,
                    hist_t=hist_t,
                    hist_mask_t=hist_mask_t,
                    xyz_norm_base=xyz_norm_base,
                    n0_base=n0,
                    conf=conf,
                    uv_norm_pix=uv_norm_pix,
                    grid_id_2d=grid_id_2d,
                    H=H, W=W,
                    levels=levels,
                    device=device,
                    save_file=save_file,
                    frame_cb=_frame_cb if writer is not None else None,
                    frame_every=gif_every,
                    lam_area=lam_area,
                    lam_shape=lam_shape,
                    w_rigid_mode=w_rigid_mode,
                    rigid_edge_k=rigid_edge_k,
                    similarity_weights=similarity_weights,
                )
        finally:
            torch.use_deterministic_algorithms(_det_prev, warn_only=_det_prev_warn)

        with torch.no_grad():
            P_vox_xyz = torch.stack([
                xyz_norm_final[..., 0] * (Xdim - 1),
                xyz_norm_final[..., 1] * (Ydim - 1),
                xyz_norm_final[..., 2] * (Zdim - 1),
            ], dim=-1)

            mri_on_final_sheet = sample_mri_cubic_3d(
                mri_t, P_vox_xyz, kernel="catmull-rom", padding_mode="border"
            )[0, 0].detach().cpu().numpy()

        valid_mri = (
            torch.isfinite(xyz_norm_final).all(dim=-1) &
            (xyz_norm_final[..., 0] >= 0) & (xyz_norm_final[..., 0] <= 1) &
            (xyz_norm_final[..., 1] >= 0) & (xyz_norm_final[..., 1] <= 1) &
            (xyz_norm_final[..., 2] >= 0) & (xyz_norm_final[..., 2] <= 1)
        )

        valid_out = valid_mri.detach().cpu().numpy()

        return xyz_norm_final, mri_on_final_sheet, valid_out, xyz_norm_base, conf, phi_refined, n0, alpha_refined, rf_psi_refined, rf_beta_refined
    finally:
        if writer is not None:
            writer.close()
            logger.debug(f'Saved GIF to: {gif_file}')

