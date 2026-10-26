"""Thin-plate-spline (TPS) models built on top of the global plane G.

Provides a 3D TPS surface (u, v) -> MRI (x, y, z) and a 2D TPS map (u, v) -> (x, y), a
robust (RANSAC / MAD) TPS fit to the patch matches, residual TPS fits on top of G, and
resampling of the histology onto the initial surface before refinement. (u, v) are the
plane coordinates of G in MRI voxels, MRI positions are (x, y, z) voxels of the slab, and
histology positions are (x, y) pixels of the common canvas. Matches store MRI points as
(z, y, x) and histology points as (y, x).
"""
import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .plane import hist_xy_from_G_uv, xyz_from_G_uv
import logging

logger = logging.getLogger(__name__)


def _tps_kernel_r2logr_2d(r2, eps=1e-9):
    """2D-map TPS kernel r2 * log(r + eps), with r = sqrt(max(r2, eps)); 0 at r2 = 0.

    Unlike ``_tps_kernel_r2logr`` (3D surface) it adds eps inside the log and has no explicit
    zeroing at r2 = 0. The values differ slightly, so the two kernels are not interchangeable.
    """
    r = torch.sqrt(torch.clamp(r2, min=eps))
    return r2 * torch.log(r + eps)


class ThinPlateSplineMap2D(nn.Module):
    """Vector-valued 2D TPS map M(u, v) = [x(u, v), y(u, v)].

    Each output component is an affine term plus a weighted sum of TPS kernels centred on
    the control points. Uses ``_tps_kernel_r2logr_2d``, which is numerically different
    from the kernel of ``ThinPlateSplineSurface3D``. In the pipeline it models the
    histology-pixel residual of G (see ``fit_residual_tps2d``).

    Args:
        device: torch device of the fitted parameters.
        dtype: torch dtype of the fitted parameters.

    Call ``fit_from_uv_xy`` before use. ``forward(uv_query)`` takes an (M, 2) tensor of
    (u, v) in the units of the control points and returns an (M, 2) tensor of (x, y) in
    the units of the targets.
    """
    def __init__(self, device=None, dtype=torch.float32):
        super().__init__()
        self._device = device
        self._dtype = dtype

        self.register_buffer("ctrl_uv", None, persistent=False)
        self.register_buffer("mean_uv", None, persistent=False)
        self.register_buffer("scale_uv", None, persistent=False)

        self.register_buffer("w_x", None, persistent=False)
        self.register_buffer("a_x", None, persistent=False)
        self.register_buffer("w_y", None, persistent=False)
        self.register_buffer("a_y", None, persistent=False)

        self.lam = 0.0

    def fit_from_uv_xy(self, ctrl_uv, targets_xy, lam=0.0, normalize_uv=False):
        """Fit the map to control points by solving the regularised TPS system.

        Control points closer than 1e-5 in (u, v) are merged and their targets averaged.
        The kernel diagonal is damped by lam (at least 1e-6); if the solve fails, lam is
        raised by factors of 10 up to 1000x, and a least-squares solve is the last resort.
        The lam actually used is stored in ``self.lam``.

        Args:
            ctrl_uv: (N, 2) control-point parameters (u, v), N >= 3.
            targets_xy: (N, 2) target values (x, y) at the control points.
            lam: smoothing (ridge) weight on the kernel matrix.
            normalize_uv: standardise (u, v) by the control-point mean and std first.

        Returns:
            self, fitted.
        """
        ctrl_uv = torch.as_tensor(ctrl_uv, dtype=self._dtype, device=self._device)
        targets_xy = torch.as_tensor(targets_xy, dtype=self._dtype, device=self._device)

        if ctrl_uv.ndim != 2 or ctrl_uv.shape[1] != 2:
            raise ValueError("ctrl_uv must be (N,2)")
        if targets_xy.ndim != 2 or targets_xy.shape[1] != 2:
            raise ValueError("targets_xy must be (N,2)")
        if ctrl_uv.shape[0] != targets_xy.shape[0]:
            raise ValueError("ctrl_uv and targets_xy must have same N")

        device = ctrl_uv.device
        dtype = ctrl_uv.dtype
        N = ctrl_uv.shape[0]
        if N < 3:
            raise ValueError("Need at least 3 control points for TPS2D")

        qtol = 1e-5
        key = torch.round(ctrl_uv / qtol).to(torch.int64)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        if uniq.shape[0] < N:
            Nu = uniq.shape[0]
            uv_sum = torch.zeros((Nu, 2), device=device, dtype=dtype)
            xy_sum = torch.zeros((Nu, 2), device=device, dtype=dtype)
            cnt = torch.zeros((Nu, 1), device=device, dtype=dtype)

            uv_sum.index_add_(0, inv, ctrl_uv)
            xy_sum.index_add_(0, inv, targets_xy)
            cnt.index_add_(0, inv, torch.ones((N, 1), device=device, dtype=dtype))

            ctrl_uv = uv_sum / cnt
            targets_xy = xy_sum / cnt
            N = ctrl_uv.shape[0]

        if normalize_uv:
            mean_uv = ctrl_uv.mean(0, keepdim=True)
            scale_uv = ctrl_uv.std(0, keepdim=True)
            scale_uv = torch.where(scale_uv == 0, torch.ones_like(scale_uv), scale_uv)
            uv_n = (ctrl_uv - mean_uv) / scale_uv
        else:
            mean_uv = torch.zeros((1, 2), device=device, dtype=dtype)
            scale_uv = torch.ones((1, 2), device=device, dtype=dtype)
            uv_n = ctrl_uv

        diff = uv_n[:, None, :] - uv_n[None, :, :]
        r2 = (diff ** 2).sum(-1)
        K = _tps_kernel_r2logr_2d(r2)

        ones = torch.ones((N, 1), device=device, dtype=dtype)
        Pm = torch.cat([ones, uv_n], dim=1)

        B = torch.cat([targets_xy, torch.zeros((3, 2), device=device, dtype=dtype)], dim=0)

        I_N = torch.eye(N, device=device, dtype=dtype)
        Z_33 = torch.zeros((3, 3), device=device, dtype=dtype)

        lam_base = max(float(lam), 1e-6)
        sol = None
        lam_used = lam_base

        for fac in [1.0, 10.0, 100.0, 1000.0]:
            lam_try = lam_base * fac
            L_top = torch.cat([K + lam_try * I_N, Pm], dim=1)
            L_bot = torch.cat([Pm.T, Z_33], dim=1)
            L = torch.cat([L_top, L_bot], dim=0)
            try:
                sol = torch.linalg.solve(L, B)
                lam_used = lam_try
                break
            except RuntimeError:
                sol = None

        if sol is None:
            L_top = torch.cat([K + (lam_base * 1000.0) * I_N, Pm], dim=1)
            L_bot = torch.cat([Pm.T, Z_33], dim=1)
            L = torch.cat([L_top, L_bot], dim=0)
            sol = torch.linalg.lstsq(L, B).solution
            lam_used = lam_base * 1000.0

        w = sol[:N, :]
        a = sol[N:, :]

        self.ctrl_uv = ctrl_uv
        self.mean_uv = mean_uv
        self.scale_uv = scale_uv
        self.lam = float(lam_used)

        self.w_x, self.w_y = w[:, 0], w[:, 1]
        self.a_x, self.a_y = a[:, 0], a[:, 1]
        return self

    def _normalize_uv(self, uv):
        return (uv - self.mean_uv) / self.scale_uv

    def _ctrl_uv_norm(self):
        return (self.ctrl_uv - self.mean_uv) / self.scale_uv

    def _eval_scalar(self, uv_query, w, a):
        uv_query = torch.as_tensor(uv_query, dtype=self._dtype, device=self.ctrl_uv.device)
        uv_n = self._normalize_uv(uv_query)
        ctrl_n = self._ctrl_uv_norm()
        diff = uv_n[:, None, :] - ctrl_n[None, :, :]
        r2 = (diff ** 2).sum(-1)
        Uker = _tps_kernel_r2logr_2d(r2)
        u_n = uv_n[:, 0]
        v_n = uv_n[:, 1]
        return a[0] + a[1] * u_n + a[2] * v_n + Uker @ w

    def forward(self, uv_query):
        if uv_query.ndim != 2 or uv_query.shape[1] != 2:
            raise ValueError("uv_query must be (M,2)")
        x = self._eval_scalar(uv_query, self.w_x, self.a_x)
        y = self._eval_scalar(uv_query, self.w_y, self.a_y)
        return torch.stack([x, y], dim=1)


def uv_support_confidence(uv_flat, ctrl_uv, Hout, Wout, sigma_uv=20.0, min_conf=0.25, hard_mult=2.0):
    """Compute how well each query (u, v) is supported by nearby control points.

    With d the distance to the nearest control point, the confidence is
    g = exp(-d^2 / (2 sigma_uv^2)), rescaled so that g = min_conf maps to 0 and g = 1 to 1,
    clipped to [0, 1], squared, and set to 0 where d > hard_mult * sigma_uv. It is used to
    fade the TPS residual back to G away from the matches.

    Args:
        uv_flat: (Hout*Wout, 2) query (u, v), in the same units as ctrl_uv.
        ctrl_uv: (N, 2) control-point (u, v), numpy array or torch tensor.
        Hout: output grid height.
        Wout: output grid width.
        sigma_uv: Gaussian width in (u, v) units.
        min_conf: Gaussian value that is mapped to confidence 0.
        hard_mult: distance cut-off in multiples of sigma_uv.

    Returns:
        Tuple (conf, d): conf is an (Hout, Wout) float32 array in [0, 1]; d is the
        (Hout, Wout) nearest-control distance.
    """
    cuv = ctrl_uv.detach().cpu().numpy() if torch.is_tensor(ctrl_uv) else np.asarray(ctrl_uv, np.float32)
    tree = cKDTree(cuv.astype(np.float32))
    d, _ = tree.query(np.asarray(uv_flat, np.float32), k=1)
    d = d.reshape(Hout, Wout)
    conf = np.exp(-(d * d) / (2.0 * float(sigma_uv) ** 2))
    conf = np.clip((conf - float(min_conf)) / (1.0 - float(min_conf) + 1e-8), 0.0, 1.0)
    conf = conf ** 2
    conf[d > float(hard_mult) * float(sigma_uv)] = 0.0
    return conf.astype(np.float32), d


@torch.no_grad()
def fit_residual_tps3d(
    uv_ctrl,
    xyz_ctrl,
    G,
    device="cuda:0",
    lam=2e-2,
    normalize_uv=False,
    anchor_step=24.0,
    far_thr=35.0,
    max_anchors=1200,
    seed=42,
):
    """Fit a 3D TPS to the MRI residual of the matches with respect to the plane G.

    The targets are dxyz = xyz_ctrl - G(uv_ctrl). Zero-residual anchors from
    ``build_G_anchors_uv_xyz`` are added where the matches are sparse, so the residual
    decays to zero (the surface falls back to G) instead of extrapolating far from the
    data. The full surface is G(uv) + tps_res(uv) (see ``ResidualToGSurface3D``).

    Args:
        uv_ctrl: (N, 2) plane coordinates of the matches, numpy array or tensor.
        xyz_ctrl: (N, 3) matched MRI (x, y, z) in voxels.
        G: global plane dict from ``plane.fit_global_plane``.
        device: torch device for the TPS.
        lam: TPS smoothing weight.
        normalize_uv: standardise (u, v) before fitting.
        anchor_step: spacing of the anchor grid in (u, v) voxels.
        far_thr: minimum distance in (u, v) from any match for an anchor to be kept.
        max_anchors: maximum number of anchors (random subset if exceeded).
        seed: seed of the anchor subsampling.

    Returns:
        Tuple (tps_res, uv_all, dxyz_all, uv_anchor):
            tps_res: fitted ``ThinPlateSplineSurface3D`` mapping (u, v) -> dxyz in voxels.
            uv_all: (N + M, 2) float32 control (u, v), matches followed by anchors.
            dxyz_all: (N + M, 3) float32 residual targets (zeros for the anchors).
            uv_anchor: (M, 2) float32 anchor (u, v).
    """
    uv_np = uv_ctrl.detach().cpu().numpy() if torch.is_tensor(uv_ctrl) else np.asarray(uv_ctrl, np.float32)
    xyz_np = xyz_ctrl.detach().cpu().numpy() if torch.is_tensor(xyz_ctrl) else np.asarray(xyz_ctrl, np.float32)

    xyz_g_ctrl = xyz_from_G_uv(uv_np, G)
    dxyz_ctrl = xyz_np - xyz_g_ctrl

    uv_anchor, _ = build_G_anchors_uv_xyz(
        uv_ctrl=uv_np, G=G,
        anchor_step=anchor_step, far_thr=far_thr,
        max_anchors=max_anchors, seed=seed
    )
    dxyz_anchor = np.zeros((len(uv_anchor), 3), np.float32)

    uv_all = np.concatenate([uv_np, uv_anchor], axis=0).astype(np.float32)
    dxyz_all = np.concatenate([dxyz_ctrl, dxyz_anchor], axis=0).astype(np.float32)

    tps_res = ThinPlateSplineSurface3D(device=torch.device(device), dtype=torch.float32)
    tps_res.fit_from_uv_xyz(
        ctrl_uv=torch.from_numpy(uv_all).to(device),
        targets_xyz=torch.from_numpy(dxyz_all).to(device),
        lam=float(lam),
        normalize_uv=normalize_uv,
    ).to(device)
    return tps_res, uv_all, dxyz_all, uv_anchor


@torch.no_grad()
def fit_residual_tps2d(
    uv_ctrl,
    hist_xy_ctrl,
    G,
    device="cuda:0",
    lam=2e-2,
    normalize_uv=False,
    anchor_step=24.0,
    far_thr=35.0,
    max_anchors=1200,
    seed=42,
):
    """Fit a 2D TPS to the histology residual of the matches with respect to G.

    The targets are dxy = hist_xy_ctrl - G_hist(uv_ctrl), where G_hist is the in-plane
    similarity of G (``plane.hist_xy_from_G_uv``). Zero-residual anchors from
    ``build_G_anchors_uv_xyz`` keep the map close to G away from the matches.

    Args:
        uv_ctrl: (N, 2) plane coordinates of the matches, numpy array or tensor.
        hist_xy_ctrl: (N, 2) matched histology (x, y) in common-canvas pixels.
        G: global plane dict from ``plane.fit_global_plane``.
        device: torch device for the TPS.
        lam: TPS smoothing weight.
        normalize_uv: standardise (u, v) before fitting.
        anchor_step: spacing of the anchor grid in (u, v) voxels.
        far_thr: minimum distance in (u, v) from any match for an anchor to be kept.
        max_anchors: maximum number of anchors (random subset if exceeded).
        seed: seed of the anchor subsampling.

    Returns:
        Tuple (tps2d_res, uv_all, dxy_all, uv_anchor):
            tps2d_res: fitted ``ThinPlateSplineMap2D`` mapping (u, v) -> dxy in pixels.
            uv_all: (N + M, 2) float32 control (u, v), matches followed by anchors.
            dxy_all: (N + M, 2) float32 residual targets (zeros for the anchors).
            uv_anchor: (M, 2) float32 anchor (u, v).
    """
    uv_np = uv_ctrl.detach().cpu().numpy() if torch.is_tensor(uv_ctrl) else np.asarray(uv_ctrl, np.float32)
    hist_xy_np = hist_xy_ctrl.detach().cpu().numpy() if torch.is_tensor(hist_xy_ctrl) else np.asarray(hist_xy_ctrl, np.float32)

    hist_xy_g_ctrl = hist_xy_from_G_uv(uv_np, G)
    dxy_ctrl = hist_xy_np - hist_xy_g_ctrl

    uv_anchor, _ = build_G_anchors_uv_xyz(
        uv_ctrl=uv_np, G=G,
        anchor_step=anchor_step, far_thr=far_thr,
        max_anchors=max_anchors, seed=seed
    )
    dxy_anchor = np.zeros((len(uv_anchor), 2), np.float32)

    uv_all = np.concatenate([uv_np, uv_anchor], axis=0).astype(np.float32)
    dxy_all = np.concatenate([dxy_ctrl, dxy_anchor], axis=0).astype(np.float32)

    tps2d_res = ThinPlateSplineMap2D(device=torch.device(device), dtype=torch.float32)
    tps2d_res.fit_from_uv_xy(
        ctrl_uv=torch.from_numpy(uv_all).to(device),
        targets_xy=torch.from_numpy(dxy_all).to(device),
        lam=float(lam),
        normalize_uv=normalize_uv,
    ).to(device)
    return tps2d_res, uv_all, dxy_all, uv_anchor


@torch.no_grad()
def resample_histology_on_surface(
    hist_img,
    G,
    tps2d_residual,
    ctrl_uv,
    device="cuda:0",
    canvas_shape=None,
    common_shift_yx=None,
    fill_invalid=0.0,
    is_mask=False,
    sigma_uv=20.0,
    min_conf=0.25,
    hard_mult=2.0,
    max_residual_px=20.0,
    return_extra=False,
):
    """Resample the histology onto the plane grid of G using G plus the 2D TPS residual.

    For each output pixel with plane coordinates (u, v), the histology position is
    G_hist(u, v) + conf(u, v) * dxy(u, v), where dxy is the residual TPS clamped to a
    length of ``max_residual_px`` and conf is ``uv_support_confidence`` (the residual fades
    out away from the matches). The image is then sampled with ``grid_sample``.

    The output grid is either the plane grid of G (``canvas_shape=None``) or a canvas of
    ``canvas_shape`` in which the plane grid starts at row/column ``common_shift_yx``
    (centred if None). Pixels outside the plane range of G or outside the histology are
    marked invalid.

    Args:
        hist_img: 2D histology image (SimpleITK image or array, (H, W) or (1, H, W)) in
            the frame that G maps to, in the pipeline the common canvas.
        G: global plane dict from ``plane.fit_global_plane``.
        tps2d_residual: residual map from ``fit_residual_tps2d``.
        ctrl_uv: (N, 2) (u, v) of the matches, used for the support confidence.
        device: torch device.
        canvas_shape: optional (Hc, Wc) of the output canvas.
        common_shift_yx: optional (oy, ox) of the plane grid inside the canvas.
        fill_invalid: value for invalid pixels; None keeps the sampled values.
        is_mask: use nearest-neighbour sampling and threshold the result at 0.5.
        sigma_uv: see ``uv_support_confidence``.
        min_conf: see ``uv_support_confidence``.
        hard_mult: see ``uv_support_confidence``.
        max_residual_px: maximum length of the TPS residual, in histology pixels.
        return_extra: also return the intermediate maps.

    Returns:
        hist_out, an (Hout, Wout) float32 array. With ``return_extra=True``, the tuple
        (hist_out, valid, uv, xy, conf, dmin): valid is an (Hout, Wout) bool mask, uv the
        (Hout, Wout, 2) plane coordinates, xy the (Hout, Wout, 2) sampled histology
        (x, y), conf the support confidence and dmin the nearest-match distance in (u, v).
    """
    if isinstance(hist_img, sitk.Image):
        h = sitk.GetArrayFromImage(hist_img)
    else:
        h = np.asarray(hist_img)
    if h.ndim == 3 and h.shape[0] == 1:
        h = h[0]
    h = h.astype(np.float32)
    Hh, Wh = h.shape

    u0, u1 = map(float, G["x_range"])
    v0, v1 = map(float, G["y_range"])
    step = float(G.get("plane_step", 1.0))

    us = np.arange(u0, u1 + 1e-6, step, dtype=np.float32)
    vs = np.arange(v0, v1 + 1e-6, step, dtype=np.float32)
    Wm, Hm = len(us), len(vs)

    if canvas_shape is None:
        uu, vv = np.meshgrid(us, vs)
        in_uv = np.ones_like(uu, dtype=bool)
    else:
        Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])
        if common_shift_yx is None:
            oy = (Hc - Hm) // 2
            ox = (Wc - Wm) // 2
        else:
            oy, ox = int(common_shift_yx[0]), int(common_shift_yx[1])
        yy, xx = np.meshgrid(np.arange(Hc, dtype=np.float32), np.arange(Wc, dtype=np.float32), indexing="ij")
        uu = (xx - ox) * step + u0
        vv = (yy - oy) * step + v0
        in_uv = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)

    Hout, Wout = uu.shape
    uv_flat = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    xy_g = hist_xy_from_G_uv(uv_flat, G).reshape(Hout, Wout, 2)
    dxy = tps2d_residual(torch.from_numpy(uv_flat).to(device)).detach().cpu().numpy().reshape(Hout, Wout, 2)

    dn = np.linalg.norm(dxy, axis=-1, keepdims=True) + 1e-8
    dxy = dxy * np.minimum(1.0, float(max_residual_px) / dn)

    conf, dmin = uv_support_confidence(uv_flat, ctrl_uv, Hout, Wout, sigma_uv=sigma_uv, min_conf=min_conf, hard_mult=hard_mult)
    xy = xy_g + conf[..., None] * dxy

    xh = xy[..., 0]
    yh = xy[..., 1]
    valid = in_uv & (xh >= 0) & (xh <= (Wh - 1)) & (yh >= 0) & (yh <= (Hh - 1))

    hist_t = torch.from_numpy(h).to(device=device)[None, None]
    xn = 2.0 * (xh / max(Wh - 1, 1)) - 1.0
    yn = 2.0 * (yh / max(Hh - 1, 1)) - 1.0
    grid = np.stack([xn, yn], axis=-1).astype(np.float32)
    grid_t = torch.from_numpy(grid).to(device=device)[None]

    mode = "nearest" if is_mask else "bilinear"
    out = F.grid_sample(hist_t, grid_t, mode=mode, padding_mode="zeros", align_corners=True)
    hist_out = out[0, 0].detach().cpu().numpy()

    if is_mask:
        hist_out = (hist_out > 0.5).astype(np.float32)

    if fill_invalid is not None:
        hist_out = hist_out.copy()
        hist_out[~valid] = float(fill_invalid)

    if return_extra:
        return hist_out, valid, uv_flat.reshape(Hout, Wout, 2), xy, conf, dmin
    return hist_out


@torch.no_grad()
def fit_tps_robust(
    matches_filtered,
    G,
    device="cuda",
    # key controls:
    hist_key: str = "hist_center_yx",
    mri_key: str = "mri_center_zyx",
    # Robustness selection:
    method: str = "ransac",      # "ransac" | "mad" | "all"
    # RANSAC controls:
    n_iters: int = 200,
    sample_size: int = 30,
    inlier_thresh: float = 5.0,
    min_inliers: int = 20,
    score_type: str = "inliers",
    truncated_cap: float = 10.0,
    seed: int = 0,
    # MAD controls:
    mad_thresh: float = 3.5,
    mad_iterations: int = 1,
    # TPS controls:
    lam_hyp: float = 1e-3,
    lam_final: float = 5e-3,
    normalize_uv: bool = False,
):
    """Robustly fit a 3D TPS surface (u, v) -> MRI (x, y, z) to the patch matches.

    The histology point of each match is mapped to plane coordinates by inverting the
    in-plane similarity of G, uv = (hist_xy - t_xy) @ A2_invT, and the MRI point is the
    target. Outlier matches are then rejected with one of three methods:

    * ``"ransac"``: fit TPS hypotheses (smoothing ``lam_hyp``) on random subsets of
      ``sample_size`` matches; a match is an inlier if its 3D residual is
      <= ``inlier_thresh`` voxels. Hypotheses with fewer than ``min_inliers`` inliers are
      ignored. Hypotheses are scored by the inlier count or by the negative truncated
      residual sum (``score_type="truncated"``). Stops early when a new best hypothesis has
      more than 90% inliers. Falls back to all matches if no hypothesis is valid.
    * ``"mad"``: fit on all matches, then up to ``mad_iterations`` times keep matches with
      residual <= median + mad_thresh * 1.4826 * MAD and refit; stops when the set does
      not change or would have fewer than 3 matches.
    * ``"all"``: keep every match.

    The final surface is refitted on the inliers with smoothing ``lam_final``.

    Args:
        matches_filtered: list of match dicts. Histology points must be in the frame G
            was fitted in (the common canvas in the pipeline).
        G: global plane dict; needs ``t_xy`` and one of ``A2_invT``, ``A2`` or (``s``,
            ``R2``).
        device: torch device.
        hist_key: match key of the histology point, (y, x).
        mri_key: match key of the MRI point, (z, y, x); (-1, -1, -1) entries are skipped.
        method: ``"ransac"``, ``"mad"`` or ``"all"``.
        n_iters: number of RANSAC hypotheses.
        sample_size: matches per RANSAC hypothesis.
        inlier_thresh: RANSAC inlier threshold, in MRI voxels.
        min_inliers: minimum inlier count for a RANSAC hypothesis to be scored.
        score_type: ``"inliers"`` or ``"truncated"``.
        truncated_cap: residual cap, in voxels, for ``score_type="truncated"``.
        seed: seed of the RANSAC sampler.
        mad_thresh: number of robust standard deviations above the median for MAD.
        mad_iterations: maximum number of MAD reject-refit rounds (at least 1).
        lam_hyp: TPS smoothing for RANSAC hypotheses and MAD rounds.
        lam_final: TPS smoothing for the final refit.
        normalize_uv: standardise (u, v) before each TPS fit.

    Returns:
        Tuple (tps_final, uv_in, xyz_in, inlier_mask, residuals, best_score):
            tps_final: ``ThinPlateSplineSurface3D`` fitted on the inliers.
            uv_in: (K, 2) inlier (u, v) tensor.
            xyz_in: (K, 3) inlier MRI (x, y, z) tensor.
            inlier_mask: (N,) bool tensor over the valid matches, in input order (aligned
                with the input list only if every match is valid).
            residuals: (N,) 3D residuals of tps_final at all valid matches, in voxels.
            best_score: inlier count for "mad" and "all"; for "ransac" the best score
                (int or negative float), or None if no hypothesis was valid.

    Raises:
        ValueError: if fewer than 3 valid matches, fewer than ``sample_size`` for RANSAC,
            or an unknown method or score type is given.
        KeyError: if G lacks the in-plane transform keys.
    """
    logger.debug(f'-------- Estimate Initial Curved Sheet (method={method}) --------')
    device = torch.device(device)

    # Control points: histology (x, y) mapped to (u, v) through G, and MRI (x, y, z).
    pts_hist_xy, pts_xyz = [], []  # hist (x,y), MRI (x,y,z)
    for m in matches_filtered:
        if hist_key not in m or mri_key not in m:
            continue
        z, y, x = m[mri_key]
        if (z, y, x) == (-1, -1, -1):
            continue
        cy, cx = m[hist_key]
        pts_hist_xy.append((float(cx), float(cy)))
        pts_xyz.append((float(x), float(y), float(z)))

    if len(pts_hist_xy) < 3:
        raise ValueError(f"Not enough valid matches to fit TPS: N={len(pts_hist_xy)}")

    U = torch.as_tensor(np.asarray(pts_hist_xy, dtype=np.float32), device=device)   # (N,2)
    xyz_ctrl = torch.as_tensor(np.asarray(pts_xyz, dtype=np.float32), device=device) # (N,3)

    t = torch.as_tensor(G["t_xy"], dtype=torch.float32, device=device)
    if "A2_invT" in G:
        A2_invT = torch.as_tensor(G["A2_invT"], dtype=torch.float32, device=device)
        uv_ctrl = (U - t) @ A2_invT
    elif "A2" in G:
        A2 = torch.as_tensor(G["A2"], dtype=torch.float32, device=device)
        uv_ctrl = (U - t) @ torch.linalg.inv(A2).T
    elif ("s" in G) and ("R2" in G):
        s = torch.as_tensor(G["s"], dtype=torch.float32, device=device)
        R2 = torch.as_tensor(G["R2"], dtype=torch.float32, device=device)
        uv_ctrl = ((U - t) @ R2) / s
    else:
        raise KeyError("G must contain one of: A2_invT, A2, or (s and R2), plus t_xy.")

    N = uv_ctrl.shape[0]

    def _final_refit(inlier_mask, lam):
        uv_in = uv_ctrl[inlier_mask]
        xyz_in = xyz_ctrl[inlier_mask]
        tps = ThinPlateSplineSurface3D(device=device, dtype=torch.float32)
        tps.fit_from_uv_xyz(
            ctrl_uv=uv_in, targets_xyz=xyz_in, lam=lam, normalize_uv=normalize_uv
        ).to(device)
        residuals_final = _tps_residuals(tps, uv_ctrl, xyz_ctrl)
        return tps, uv_in, xyz_in, residuals_final

    if method == "all":
        inlier_mask = torch.ones(N, device=device, dtype=torch.bool)
        tps_final, uv_in, xyz_in, residuals_final = _final_refit(inlier_mask, lam_final)
        best_score = int(inlier_mask.sum())
        return tps_final, uv_in, xyz_in, inlier_mask, residuals_final, best_score

    if method == "mad":
        tps = ThinPlateSplineSurface3D(device=device, dtype=torch.float32)
        tps.fit_from_uv_xyz(
            ctrl_uv=uv_ctrl, targets_xyz=xyz_ctrl, lam=lam_hyp, normalize_uv=normalize_uv
        ).to(device)
        inlier_mask = torch.ones(N, device=device, dtype=torch.bool)
        residuals = _tps_residuals(tps, uv_ctrl, xyz_ctrl)

        for _ in range(max(1, mad_iterations)):
            median = residuals.median()
            mad = (residuals - median).abs().median()
            scale = 1.4826 * mad + 1e-8
            thresh = median + mad_thresh * scale
            new_mask = residuals <= thresh
            if int(new_mask.sum()) < 3 or torch.equal(new_mask, inlier_mask):
                break
            inlier_mask = new_mask
            tps = ThinPlateSplineSurface3D(device=device, dtype=torch.float32)
            tps.fit_from_uv_xyz(
                ctrl_uv=uv_ctrl[inlier_mask],
                targets_xyz=xyz_ctrl[inlier_mask],
                lam=lam_hyp,
                normalize_uv=normalize_uv,
            ).to(device)
            residuals = _tps_residuals(tps, uv_ctrl, xyz_ctrl)

        tps_final, uv_in, xyz_in, residuals_final = _final_refit(inlier_mask, lam_final)
        best_score = int(inlier_mask.sum())
        return tps_final, uv_in, xyz_in, inlier_mask, residuals_final, best_score

    if method != "ransac":
        raise ValueError("method must be one of {'ransac', 'mad', 'all'}")
    if N < max(3, sample_size):
        raise ValueError(f"Not enough points for RANSAC TPS: N={N}, sample_size={sample_size}")

    dtype = uv_ctrl.dtype
    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(seed)
    best_inlier_mask = None
    best_score = None
    all_idx = torch.arange(N, device=device)

    for _ in range(n_iters):
        perm = torch.randperm(N, generator=g_cpu, device="cpu")[:sample_size]
        idx = all_idx[perm.to(device)]
        try:
            tps_h = _fit_tps_on_subset(
                uv_ctrl, xyz_ctrl, idx,
                device=device, dtype=dtype,
                lam=lam_hyp, normalize_uv=normalize_uv
            )
        except RuntimeError:
            continue

        residuals = _tps_residuals(tps_h, uv_ctrl, xyz_ctrl)
        inlier_mask = residuals <= inlier_thresh
        num_inliers = int(inlier_mask.sum())
        if num_inliers < min_inliers:
            continue

        if score_type == "inliers":
            score = num_inliers
        elif score_type == "truncated":
            cost = torch.clamp(residuals, max=truncated_cap).sum().item()
            score = -cost
        else:
            raise ValueError("score_type must be 'inliers' or 'truncated'")

        if (best_score is None) or (score > best_score):
            best_score = score
            best_inlier_mask = inlier_mask.clone()
            if num_inliers > 0.9 * N:
                break

    if best_inlier_mask is None:
        logger.warning('RANSAC TPS: no valid hypothesis found; falling back to all points.')
        best_inlier_mask = torch.ones(N, device=device, dtype=torch.bool)

    tps_final, uv_in, xyz_in, residuals_final = _final_refit(best_inlier_mask, lam_final)
    return tps_final, uv_in, xyz_in, best_inlier_mask, residuals_final, best_score


class ResidualToGSurface3D(torch.nn.Module):
    """Absolute surface S(u, v) = G(u, v) + residual_tps(u, v) in MRI voxels.

    Combines the plane of G with a residual TPS from ``fit_residual_tps3d``. The
    evaluation goes through numpy on the CPU, so no gradients flow through it.

    Args:
        G: global plane dict from ``plane.fit_global_plane``.
        residual_tps: callable (M, 2) tensor -> (M, 3) residual, e.g. the
            ``ThinPlateSplineSurface3D`` from ``fit_residual_tps3d``.
        device: stored as ``self.device``; the output is placed on the device of the input.

    ``forward(uv_query)`` takes an (M, 2) tensor of (u, v) and returns an (M, 3) float32
    tensor of MRI (x, y, z) voxels.
    """
    def __init__(self, G, residual_tps, device="cuda:0"):
        super().__init__()
        self.G = G
        self.residual_tps = residual_tps
        self.device = torch.device(device)

    def forward(self, uv_query):
        uv = uv_query.detach().cpu().numpy().astype(np.float32)
        xyz_g = xyz_from_G_uv(uv, self.G)
        dxyz = self.residual_tps(uv_query).detach().cpu().numpy().astype(np.float32)
        xyz = xyz_g + dxyz
        return torch.from_numpy(xyz).to(device=uv_query.device, dtype=torch.float32)


def build_G_anchors_uv_xyz(
    uv_ctrl,                 # [N,2] uv used by TPS fit
    G,
    anchor_step=24.0,        # uv grid spacing
    far_thr=35.0,            # keep anchors only if far from real controls
    max_anchors=1200,
    seed=42,
):
    """Build anchor points on the plane of G in regions without matches.

    Anchors lie on a regular (u, v) grid over the plane range of G and are kept only if
    they are at least ``far_thr`` from every control point. Pinning a TPS to G at these
    anchors stops it from swirling or extrapolating where there are no matches.

    Args:
        uv_ctrl: (N, 2) control-point (u, v), numpy array or tensor.
        G: global plane dict (uses x_range, y_range, X0, e1, e2).
        anchor_step: grid spacing in (u, v) voxels.
        far_thr: minimum distance to the nearest control point, in (u, v) voxels.
        max_anchors: if more anchors remain, keep a random subset of this size.
        seed: seed of the random subset.

    Returns:
        Tuple (uv_anchor, xyz_anchor): (M, 2) float32 anchor (u, v) and (M, 3) float32
        MRI (x, y, z) of the anchors on the plane of G.
    """
    uv = uv_ctrl.detach().cpu().numpy() if torch.is_tensor(uv_ctrl) else np.asarray(uv_ctrl, np.float32)
    uv = uv.astype(np.float32)

    u0, u1 = map(float, G["x_range"])
    v0, v1 = map(float, G["y_range"])
    us = np.arange(u0, u1 + 1e-6, float(anchor_step), dtype=np.float32)
    vs = np.arange(v0, v1 + 1e-6, float(anchor_step), dtype=np.float32)
    UU, VV = np.meshgrid(us, vs)
    uv_grid = np.stack([UU.ravel(), VV.ravel()], axis=1)  # [M,2]

    if len(uv) > 0:
        tree = cKDTree(uv)
        d, _ = tree.query(uv_grid, k=1)
        keep = d >= float(far_thr)
        uv_anchor = uv_grid[keep]
    else:
        uv_anchor = uv_grid

    if len(uv_anchor) > int(max_anchors):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(uv_anchor), size=int(max_anchors), replace=False)
        uv_anchor = uv_anchor[idx]

    # G plane: xyz = X0 + u*e1 + v*e2
    X0 = np.asarray(G["X0"], np.float32)
    e1 = np.asarray(G["e1"], np.float32)
    e2 = np.asarray(G["e2"], np.float32)
    xyz_anchor = X0[None, :] + uv_anchor[:, 0:1] * e1[None, :] + uv_anchor[:, 1:2] * e2[None, :]

    return uv_anchor.astype(np.float32), xyz_anchor.astype(np.float32)


def _tps_kernel_r2logr(r2, eps=1e-9):
    """3D-surface TPS kernel r2 * log(r), with r = sqrt(max(r2, eps)); exactly 0 at r2 = 0.

    Takes squared distances so no square root of zero is needed. Unlike
    ``_tps_kernel_r2logr_2d`` there is no eps inside the log; the two are not interchangeable.
    """
    r = torch.clamp(r2, min=eps).sqrt()
    U = r2 * torch.log(r)
    # Exact 0 at r = 0.
    U = torch.where(r2 > 0, U, torch.zeros_like(U))
    return U


@torch.no_grad()
def _tps_residuals(tps, uv_all, xyz_all):
    """Return the (N,) Euclidean distances between tps(uv_all) and xyz_all."""
    pred = tps(uv_all)                     # (N,3)
    return (xyz_all - pred).norm(dim=1)    # (N,)


class ThinPlateSplineSurface3D(nn.Module):
    """Vector-valued TPS surface S(u, v) = [x(u, v), y(u, v), z(u, v)].

    Each output component is an affine term in (u, v) plus a weighted sum of TPS kernels
    (``_tps_kernel_r2logr``) centred on the control points. In the pipeline the targets
    are either absolute MRI positions (``fit_tps_robust``) or residuals to G
    (``fit_residual_tps3d``), both in MRI voxels; outputs have the units of the targets.

    Args:
        device: torch device of the fitted parameters.
        dtype: torch dtype of the fitted parameters.

    Methods (after ``fit_from_uv_xyz``):
        forward(uv_query): (M, 2) tensor -> (M, 3) tensor.
        predict(U, V): broadcastable arrays -> (X, Y, Z), each of the broadcast shape.
        partials(U, V): (S_u, S_v), each (3, *shape).
        normals(U, V): unit normals, (3, *shape).
    """

    def __init__(self, device=None, dtype=torch.float32):
        super().__init__()
        self._device = device
        self._dtype = dtype

        # Fit parameters
        self.register_buffer("ctrl_uv", None, persistent=False)   # (N,2)
        self.register_buffer("mean_uv", None, persistent=False)   # (1,2)
        self.register_buffer("scale_uv", None, persistent=False)  # (1,2)

        self.lam: float = 0.0

        self.register_buffer("w_x", None, persistent=False)
        self.register_buffer("a_x", None, persistent=False)
        self.register_buffer("w_y", None, persistent=False)
        self.register_buffer("a_y", None, persistent=False)
        self.register_buffer("w_z", None, persistent=False)
        self.register_buffer("a_z", None, persistent=False)

        # Set to the identity by fit_from_uv_xyz; never read by this class.
        self.register_buffer("pca_origin", None, persistent=False)  # (3,)
        self.register_buffer("pca_axes", None, persistent=False)    # (3,3)

    def fit_from_uv_xyz(self,
                        ctrl_uv,
                        targets_xyz,
                        lam: float = 0.0,
                        normalize_uv: bool = False):
        """Fit the surface to control points by solving the regularised TPS system.

        Control points closer than 1e-5 in (u, v) are merged and their targets averaged.
        The kernel diagonal is damped by lam (at least 1e-6); if the solve fails, lam is
        raised by factors of 10 up to 1000x, and a least-squares solve is the last resort.
        The lam actually used is stored in ``self.lam``.

        Args:
            ctrl_uv: (N, 2) control-point parameters (u, v), N >= 3.
            targets_xyz: (N, 3) target values at the control points.
            lam: smoothing (ridge) weight on the kernel matrix.
            normalize_uv: standardise (u, v) by the control-point mean and std first.

        Returns:
            self, fitted.

        Raises:
            ValueError: on wrong shapes or fewer than 3 (unique) control points.
        """
        ctrl_uv = torch.as_tensor(ctrl_uv, dtype=self._dtype, device=self._device)
        targets_xyz = torch.as_tensor(targets_xyz, dtype=self._dtype, device=self._device)

        if ctrl_uv.ndim != 2 or ctrl_uv.shape[1] != 2:
            raise ValueError("ctrl_uv must be (N,2)")
        if targets_xyz.ndim != 2 or targets_xyz.shape[1] != 3:
            raise ValueError("targets_xyz must be (N,3)")
        if ctrl_uv.shape[0] != targets_xyz.shape[0]:
            raise ValueError("ctrl_uv and targets_xyz must have same N")

        device = ctrl_uv.device
        dtype = ctrl_uv.dtype
        N = ctrl_uv.shape[0]
        if N < 3:
            raise ValueError("Need at least 3 control points for TPS3D.")

        # Merge duplicate/near-duplicate uv to avoid singular K.
        # Quantize uv, then average xyz for duplicated keys.
        qtol = 1e-5
        key = torch.round(ctrl_uv / qtol).to(torch.int64)  # (N,2)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        if uniq.shape[0] < N:
            Nu = uniq.shape[0]
            uv_sum = torch.zeros((Nu, 2), device=device, dtype=dtype)
            xyz_sum = torch.zeros((Nu, 3), device=device, dtype=dtype)
            cnt = torch.zeros((Nu, 1), device=device, dtype=dtype)

            uv_sum.index_add_(0, inv, ctrl_uv)
            xyz_sum.index_add_(0, inv, targets_xyz)
            cnt.index_add_(0, inv, torch.ones((N, 1), device=device, dtype=dtype))

            ctrl_uv = uv_sum / cnt
            targets_xyz = xyz_sum / cnt
            N = ctrl_uv.shape[0]
            if N < 3:
                raise ValueError("Too few unique control points after deduplication.")

        self.pca_origin = torch.zeros(3, device=device, dtype=dtype)
        self.pca_axes = torch.eye(3, device=device, dtype=dtype)

        # Normalize (u,v)
        if normalize_uv:
            mean_uv = ctrl_uv.mean(0, keepdim=True)   # (1,2)
            scale_uv = ctrl_uv.std(0, keepdim=True)   # (1,2)
            scale_uv = torch.where(scale_uv == 0, torch.ones_like(scale_uv), scale_uv)
            uv_n = (ctrl_uv - mean_uv) / scale_uv
        else:
            mean_uv = torch.zeros((1, 2), device=device, dtype=dtype)
            scale_uv = torch.ones((1, 2), device=device, dtype=dtype)
            uv_n = ctrl_uv

        # TPS system
        diff = uv_n[:, None, :] - uv_n[None, :, :]   # (N,N,2)
        r2 = (diff ** 2).sum(-1)                     # (N,N)
        K = _tps_kernel_r2logr(r2)                   # (N,N)

        ones = torch.ones((N, 1), device=device, dtype=dtype)
        Pm = torch.cat([ones, uv_n], dim=1)          # (N,3)

        B = torch.cat(
            [targets_xyz, torch.zeros((3, 3), device=device, dtype=dtype)],
            dim=0
        )                                            # (N+3,3)

        I_N = torch.eye(N, device=device, dtype=dtype)
        Z_33 = torch.zeros((3, 3), device=device, dtype=dtype)

        # Robust solve with damping escalation + lstsq fallback.
        lam_base = float(lam)
        if lam_base < 1e-6:
            lam_base = 1e-6

        sol = None
        lam_used = lam_base
        for fac in [1.0, 10.0, 100.0, 1000.0]:
            lam_try = lam_base * fac
            L_top = torch.cat([K + lam_try * I_N, Pm], dim=1)     # (N, N+3)
            L_bot = torch.cat([Pm.T, Z_33], dim=1)                # (3, N+3)
            L = torch.cat([L_top, L_bot], dim=0)                  # (N+3, N+3)
            try:
                sol = torch.linalg.solve(L, B)
                lam_used = lam_try
                break
            except RuntimeError:
                sol = None

        if sol is None:
            # Last resort: least-squares solution
            L_top = torch.cat([K + (lam_base * 1000.0) * I_N, Pm], dim=1)
            L_bot = torch.cat([Pm.T, Z_33], dim=1)
            L = torch.cat([L_top, L_bot], dim=0)
            sol = torch.linalg.lstsq(L, B).solution
            lam_used = lam_base * 1000.0

        w = sol[:N, :]   # (N,3)
        a = sol[N:, :]   # (3,3)

        # Store parameters
        self.ctrl_uv = ctrl_uv
        self.mean_uv = mean_uv
        self.scale_uv = scale_uv
        self.lam = float(lam_used)

        self.w_x, self.w_y, self.w_z = w[:, 0], w[:, 1], w[:, 2]
        self.a_x, self.a_y, self.a_z = a[:, 0], a[:, 1], a[:, 2]

        return self

    def _normalize_uv(self, U, V) -> torch.Tensor:
        """Normalize (U,V) into (u_n, v_n) flattened to (M,2)."""
        U = torch.as_tensor(U, dtype=self._dtype, device=self.ctrl_uv.device)
        V = torch.as_tensor(V, dtype=self._dtype, device=self.ctrl_uv.device)
        U_b, V_b = torch.broadcast_tensors(U, V)
        uv = torch.stack([U_b.reshape(-1), V_b.reshape(-1)], dim=1)  # (M,2)
        uv_n = (uv - self.mean_uv) / self.scale_uv
        return uv_n

    def _ctrl_uv_norm(self) -> torch.Tensor:
        return (self.ctrl_uv - self.mean_uv) / self.scale_uv

    def _tps_eval_scalar(self, U, V,
                          w: torch.Tensor,
                          a: torch.Tensor) -> torch.Tensor:
        """Evaluate one scalar TPS component at query (U,V)."""
        device = self.ctrl_uv.device
        dtype = self.ctrl_uv.dtype

        U_t = torch.as_tensor(U, dtype=dtype, device=device)
        V_t = torch.as_tensor(V, dtype=dtype, device=device)
        U_b, V_b = torch.broadcast_tensors(U_t, V_t)
        shape = U_b.shape

        uv_n = self._normalize_uv(U_b, V_b)             # (M,2)
        ctrl_n = self._ctrl_uv_norm()                   # (N,2)
        diff = uv_n[:, None, :] - ctrl_n[None, :, :]    # (M,N,2)
        r2   = (diff**2).sum(-1)                        # (M,N)
        Uker = _tps_kernel_r2logr(r2)                   # (M,N)

        u_n = uv_n[:, 0]
        v_n = uv_n[:, 1]
        val = (a[0]
               + a[1] * u_n
               + a[2] * v_n
               + Uker @ w)                              # (M,)

        return val.reshape(shape)

    def forward(self, uv_query: torch.Tensor) -> torch.Tensor:
        """Evaluate the surface at query points.

        Args:
            uv_query: (M, 2) tensor of (u, v) in the units of ``ctrl_uv``.

        Returns:
            (M, 3) tensor of (x, y, z) in the units of the targets.
        """
        if uv_query.ndim != 2 or uv_query.shape[1] != 2:
            raise ValueError("uv_query must be (M,2)")
        U = uv_query[:, 0]
        V = uv_query[:, 1]
        X = self._tps_eval_scalar(U, V, self.w_x, self.a_x)
        Y = self._tps_eval_scalar(U, V, self.w_y, self.a_y)
        Z = self._tps_eval_scalar(U, V, self.w_z, self.a_z)
        return torch.stack([X, Y, Z], dim=1)            # (M,3)

    def predict(self, U, V) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the surface on a grid or on broadcastable arrays.

        Args:
            U: u values, array or tensor.
            V: v values, broadcastable with U.

        Returns:
            Tuple (X, Y, Z) of tensors with the broadcast shape of U and V.
        """
        X = self._tps_eval_scalar(U, V, self.w_x, self.a_x)
        Y = self._tps_eval_scalar(U, V, self.w_y, self.a_y)
        Z = self._tps_eval_scalar(U, V, self.w_z, self.a_z)
        return X, Y, Z

    def partials(self, U, V):
        """Compute the first derivatives of the surface with respect to u and v.

        Uses the analytic kernel derivative dU/dr = 2 r log(r) + r and divides by the
        (u, v) normalisation scale, so the derivatives are with respect to raw (u, v).

        Args:
            U: u values, array or tensor.
            V: v values, broadcastable with U.

        Returns:
            Tuple (S_u, S_v), each a tensor of shape (3, *shape).
        """
        device = self.ctrl_uv.device
        dtype = self.ctrl_uv.dtype

        U_t = torch.as_tensor(U, dtype=dtype, device=device)
        V_t = torch.as_tensor(V, dtype=dtype, device=device)
        U_b, V_b = torch.broadcast_tensors(U_t, V_t)
        shape = U_b.shape

        uv_n = self._normalize_uv(U_b, V_b)             # (M,2)
        ctrl_n = self._ctrl_uv_norm()                   # (N,2)
        diff = uv_n[:, None, :] - ctrl_n[None, :, :]    # (M,N,2)
        r2   = (diff**2).sum(-1)                        # (M,N)
        r    = torch.sqrt(torch.clamp(r2, min=1e-8))    # (M,N)

        # dU/dr = 2r log r + r (for U(r) = r^2 log r)
        dU_dr = torch.zeros_like(r)
        mask = r > 0
        r_safe = r[mask]
        dU_dr[mask] = 2 * r_safe * torch.log(r_safe) + r_safe

        # du, dv per kernel center
        diffx_over_r = torch.zeros_like(r)
        diffy_over_r = torch.zeros_like(r)
        diffx_over_r[mask] = diff[:, :, 0][mask] / r_safe
        diffy_over_r[mask] = diff[:, :, 1][mask] / r_safe

        du = diffx_over_r * dU_dr    # (M,N)
        dv = diffy_over_r * dU_dr    # (M,N)

        def dcomp(w, a):
            df_du_n = a[1] + (du @ w)   # (M,)
            df_dv_n = a[2] + (dv @ w)   # (M,)
            return df_du_n, df_dv_n

        dx_du_n, dx_dv_n = dcomp(self.w_x, self.a_x)
        dy_du_n, dy_dv_n = dcomp(self.w_y, self.a_y)
        dz_du_n, dz_dv_n = dcomp(self.w_z, self.a_z)

        su = float(self.scale_uv[0, 0])
        sv = float(self.scale_uv[0, 1])

        Su_vec = torch.stack([dx_du_n / su,
                              dy_du_n / su,
                              dz_du_n / su], dim=1)      # (M,3)
        Sv_vec = torch.stack([dx_dv_n / sv,
                              dy_dv_n / sv,
                              dz_dv_n / sv], dim=1)      # (M,3)

        Su_grid = Su_vec.reshape(shape + (3,))           # (*shape,3)
        Sv_grid = Sv_vec.reshape(shape + (3,))

        dims = [Su_grid.dim() - 1] + list(range(0, Su_grid.dim() - 1))
        Su = Su_grid.permute(dims)                       # (3,*shape)
        Sv = Sv_grid.permute(dims)

        return Su, Sv

    def normals(self, U, V) -> torch.Tensor:
        """Compute unit surface normals S_u x S_v / |S_u x S_v|.

        Args:
            U: u values, array or tensor.
            V: v values, broadcastable with U.

        Returns:
            Tensor of shape (3, *shape); zero where the cross product vanishes.
        """
        Su, Sv = self.partials(U, V)           # (3,*shape)

        def move_first_to_last(t):
            return t.permute(list(range(1, t.dim())) + [0])  # (*shape,3)

        SuT = move_first_to_last(Su)
        SvT = move_first_to_last(Sv)

        N = torch.cross(SuT, SvT, dim=-1)     # (*shape,3)
        nrm = torch.linalg.norm(N, dim=-1, keepdim=True)
        nrm = torch.where(nrm == 0,
                          torch.ones_like(nrm),
                          nrm)
        N_unit = N / nrm

        dims = [N_unit.dim() - 1] + list(range(0, N_unit.dim() - 1))
        return N_unit.permute(dims)           # (3,*shape)


@torch.no_grad()
def _fit_tps_on_subset(uv, xyz, idx, device, dtype, lam, normalize_uv=False):
    """Fit a ThinPlateSplineSurface3D on the subset uv[idx] -> xyz[idx] and return it."""
    tps = ThinPlateSplineSurface3D(device=device, dtype=dtype)
    tps.fit_from_uv_xyz(
        ctrl_uv=uv[idx],
        targets_xyz=xyz[idx],
        lam=lam,
        normalize_uv=normalize_uv,
    )
    return tps

