"""Curved-sheet refinement of the histology surface inside the MRI volume."""
import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import os
import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from .bspline import BSplineFFD2D, BSplineResidualSurface3D, ScalarNormalOffsetField, _copy_ctrl, _upsample_ctrl, bending_energy, bending_energy_2d, sheet_inside_volume_penalty
from .losses import _build_rigidity_weight, combined_masked_ssim_mind_loss, combined_masked_ssim_ngf_loss, first_fundamental_form, surface_area_ratio
from .qc import _plot_surface
from .sampling import _prepare_data_tensor, sample_mri_cubic_3d, sample_mri_on_sheet


# Refinement
def warm_start_residual_ctrl_from_matches(
    residual, init_tps, init_G, matches, H, W, Xdim, Ydim, Zdim,
    device="cuda:0", n_iter=300, lr=1e-2, lam_bend=1e-4,
    hist_key="hist_center_yx", mri_key="mri_center_zyx",
):
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

    # residual model domain is normalized hist pixel coords [0,1]
    uv_norm = torch.stack([
        U[:, 0] / max(W - 1, 1),
        U[:, 1] / max(H - 1, 1),
    ], dim=1)

    opt = torch.optim.Adam(residual.parameters(), lr=lr)
    for _ in range(n_iter):
        pred_d = residual(uv_norm)  # (N,3), H=None path
        loss = F.smooth_l1_loss(pred_d, target) + lam_bend * bending_energy(residual.ctrl)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return residual


def _to_t_1_1_hw(x, device):
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
    # UV from same definition as resample_mri_on_G_tps
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
    facecolors = plt.cm.get_cmap(cmap)(norm(Zp))
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
    xyz_norm_base,       # [H,W,3] FIXED base surface (tps + frozen matches-warm-started residual)
    n0_base,              # [H,W,3] FIXED unit normal field of xyz_norm_base
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
    # ---- v12: local rigidity -------------------------------------------------
    lam_area=1.0,           # properness  : weight on log(area/area0)^2
    lam_shape=0.0,          # orthonormality: weight on ||[[E,F],[F,G]] - base||^2 (OFF by default)
    w_rigid_mode="mask",    # "mask" | "conf" | "edge" | "none"
    rigid_edge_k=2.0,       # only used by w_rigid_mode="edge"
    similarity_weights=(1.0, 0.0, 0.0),   # (w_ssim, w_ngf, w_mind) of the image loss
):
    """
    v11 refinement: four fields, alternated per pyramid level.

      phi     : 2D STATIONARY VELOCITY field, integrated via scaling-and-
                squaring into a diffeomorphic (non-folding by construction)
                displacement, then used to RESAMPLE xyz_norm_base
                tangentially. UNCONDITIONAL -- active everywhere, not gated
                by confidence.
      alpha   : scalar offset along the (resampled) surface normal.
                UNCONDITIONAL, same as phi.
      rf_psi  : a SECOND, independent 2D stationary velocity field (same
                mechanism as phi), gated by (1 - conf) and composed ON TOP
                of phi's already-integrated resample (its own separate
                scaling-and-squaring integration, then applied via a
                warp-of-a-warp lookup) -- an extra tangential correction
                that only matters where phi/alpha's "trust the base
                surface" assumption breaks down (no local match support,
                TPS fell back to the flat G-plane there). (NOTE: an
                earlier diagnostic pass suggested this composition
                reintroduced ~26% folding vs phi alone -- that turned out
                to be a measurement bug (torch.autograd.grad through the
                scaling-and-squaring recursion accumulates numerical error
                that grows with more steps); verified correct with
                finite-differences on the converged, no_grad output: 0%
                folded. Composition is fine.)
      rf_beta : a SECOND, independent scalar normal offset (same mechanism
                as alpha), also gated by (1 - conf) -- the extra "lift off
                the surface" correction in the same low-confidence regions.

      rf_psi + rf_beta together replace v10's single free 3-vector
      residual_free field: any 3D direction is exactly tangential + normal,
      so no expressiveness is lost -- but the tangential part now goes
      through the same diffeomorphic-integration treatment as phi, so it
      can no longer fold the surface the way an unconstrained free vector
      could.
    """
    similarity_metric = "mind"   # "ngf" or "mind" -- switch here to test MIND instead of NGF
    w_ssim, w_ngf, w_mind = similarity_weights
    hist_mask_fixed = (hist_mask_t > 0.5).float().detach()
    mask_soft = make_soft_dilated_mask(hist_mask_fixed, dilate_px=8, blur_ks=11, blur_passes=1).detach()

    gate = (1.0 - conf).clamp(0.0, 1.0).detach()          # [H,W]
    gate3 = gate.unsqueeze(-1)                             # [H,W,1] for broadcasting onto 3-vectors

    # ---- v12: local rigidity setup -------------------------------------------
    # dims recovers voxel-index coords from xyz_norm. Isotropic spacing => the
    # physical scale cancels in every ratio below, so no GetSpacing() is needed.
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
    print(f"[v12 rigidity] mode={w_rigid_mode} lam_area={lam_area} lam_shape={lam_shape} "
          f"active_px={int((w_rigid > 0).sum())}/{H*W}")

    def _rigidity_terms(xyz_norm):
        """(L_area, L_shape): w-weighted means. Both zero on a perfectly rigid refinement."""
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
        L_area, L_shape = _rigidity_terms(xyz_norm)
        lam = float(cfg.get("lamRigid", 1.0))
        return (L_area, L_shape), lam * (float(lam_area) * L_area + float(lam_shape) * L_shape)

    def _report_area(tag, xyz_norm):
        with torch.no_grad():
            r = surface_area_ratio(xyz_norm, xyz_norm_base, dims_t.detach().cpu().numpy())
            m = (w_rigid > 0).detach().cpu().numpy()
            if m.sum() == 0:
                return
            p = np.percentile(r[m], [1, 5, 50, 95, 99])
            print(f"    [{tag}] area ratio final/base p1,p5,p50,p95,p99 = "
                  f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f} {p[4]:.3f}")

    xyz_base_img = xyz_norm_base.permute(2, 0, 1).unsqueeze(0).detach()   # [1,3,H,W]
    n0_base_img = n0_base.permute(2, 0, 1).unsqueeze(0).detach()          # [1,3,H,W]

    if base_supersample and base_supersample > 1:
        Hs, Ws = H * int(base_supersample), W * int(base_supersample)
        xyz_base_img = F.interpolate(xyz_base_img, size=(Hs, Ws), mode="bicubic", align_corners=True)
        n0_base_img = F.interpolate(n0_base_img, size=(Hs, Ws), mode="bicubic", align_corners=True)
        n0_base_img = n0_base_img / (n0_base_img.norm(dim=1, keepdim=True) + 1e-8)

    def _resample_surface(grid2d):
        xyz = F.grid_sample(xyz_base_img, grid2d, mode="bilinear", padding_mode="border", align_corners=True)
        n0s = F.grid_sample(n0_base_img, grid2d, mode="bilinear", padding_mode="border", align_corners=True)
        xyz = xyz[0].permute(1, 2, 0)                       # (H,W,3)
        n0s = n0s[0].permute(1, 2, 0)
        n0s = n0s / (n0s.norm(dim=-1, keepdim=True) + 1e-8)
        return xyz, n0s

    def _scaling_and_squaring_2d(v, n_steps=6):
        # v: (H,W,2) stationary velocity field, same normalized [-1,1] units as grid_id_2d.
        # Integrates the flow ODE d(phi)/dt = v(phi(t)) from t=0 (identity) to t=1 by
        # repeated self-composition of a tiny initial step -- this is the standard
        # diffeomorphic-registration trick (Christensen fluid / VoxelMorph-diff / DARTEL):
        # a smooth field composed with itself can't introduce folding, unlike one large jump.
        d = v / (2 ** n_steps)
        for _ in range(n_steps):
            d_img = d.permute(2, 0, 1).unsqueeze(0)                     # (1,2,H,W)
            query = grid_id_2d + d.unsqueeze(0)                         # (1,H,W,2)
            d_warped = F.grid_sample(d_img, query, mode="bilinear", padding_mode="border", align_corners=True)
            d = d + d_warped[0].permute(1, 2, 0)                        # (H,W,2)
        return d

    def _compose_disp_2d(base_grid2d, extra_disp):
        # base_grid2d: (1,H,W,2) sampling grid (e.g. phi's identity + integrated displacement).
        # extra_disp:  (H,W,2) an ADDITIONAL displacement (e.g. rf_psi's gated, integrated
        # velocity), evaluated AT base_grid2d's positions and added on top -- i.e. this
        # composes "extra" AFTER "base" (extra_map ∘ base_map), which stays diffeomorphic
        # if both individually are, without needing to touch base's own computation at all.
        extra_img = extra_disp.permute(2, 0, 1).unsqueeze(0)                # (1,2,H,W)
        extra_at_base = F.grid_sample(extra_img, base_grid2d, mode="bilinear", padding_mode="border", align_corners=True)
        return base_grid2d + extra_at_base.permute(0, 2, 3, 1)              # (1,H,W,2)

    def _run_block(name, params, lr, n_iter, build_xyz_fn, reg_fn, print_extra_fn):
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
                print(f"[{name}] it={it:04d}/{n_iter} loss={loss.item():.6f} img={loss_img.item():.6f} "
                      f"{print_extra_fn(L_reg_terms)} ssim={comp['ssim']:.4f} {metric_name}={comp[metric_name]:.4f}")

            if loss.item() < best_loss:
                best_loss, no_improve = loss.item(), 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"[{name}] Early stopping at it={it} (no improvement for {patience} iters)")
                break

    phi = None
    alpha = None
    rf_psi = None
    rf_beta = None

    print("===== Pyramid refinement v11: phi (velocity) + alpha (normal) + rf_psi (gated velocity) + rf_beta (gated normal) =====")
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

        # ---- Step Phi: freeze alpha + rf_psi + rf_beta, optimize tangential resample ----
        with torch.no_grad():
            a_fix = alpha(uv_norm_pix, H=H, W=W)
            disp_psi_fix = _scaling_and_squaring_2d(gate3 * rf_psi(uv_norm_pix, H=H, W=W), n_steps=rf_psi_n_steps)
            rf_beta_fix = gate3 * rf_beta(uv_norm_pix, H=H, W=W)

        print(f"\n--- Level {li} ({gh}x{gw}) Step Phi ---")

        def _build_phi():
            v = phi(uv_norm_pix, H=H, W=W)
            disp = _scaling_and_squaring_2d(v, n_steps=phi_n_steps)
            grid2d = grid_id_2d + disp.unsqueeze(0)
            grid2d = _compose_disp_2d(grid2d, disp_psi_fix)
            xyz_resampled, n0_resampled = _resample_surface(grid2d)
            xyz_norm = xyz_resampled + a_fix * n0_resampled + rf_beta_fix * n0_resampled
            return xyz_norm, None

        def _reg_phi(xyz_norm, _):
            L_bend = bending_energy_2d(phi.ctrl)
            L_sheet = sheet_inside_volume_penalty(xyz_norm)
            (L_area, L_shape), L_rigid = _rigidity_loss(xyz_norm, cfg)
            return (L_bend, L_sheet, L_area, L_shape),\
                   cfg["lamA"] * L_bend + cfg["lamSheet"] * L_sheet + L_rigid

        _run_block(
            f"L{li} Phi", phi.parameters(), cfg["lrA"], cfg["nA"],
            _build_phi, _reg_phi,
            lambda t: f"bend={t[0].item():.2e} sheet={t[1].item():.2e} area={t[2].item():.2e} shape={t[3].item():.2e}"
        )

        # ---- Step Alpha: freeze phi + rf_psi + rf_beta, optimize normal offset ----
        with torch.no_grad():
            disp_fix = _scaling_and_squaring_2d(phi(uv_norm_pix, H=H, W=W), n_steps=phi_n_steps)
            grid2d_fix = grid_id_2d + disp_fix.unsqueeze(0)
            disp_psi_fix = _scaling_and_squaring_2d(gate3 * rf_psi(uv_norm_pix, H=H, W=W), n_steps=rf_psi_n_steps)
            grid2d_fix = _compose_disp_2d(grid2d_fix, disp_psi_fix)
            xyz_resampled_fix, n0_resampled_fix = _resample_surface(grid2d_fix)
            rf_beta_fix = gate3 * rf_beta(uv_norm_pix, H=H, W=W)

        print(f"--- Level {li} ({gh}x{gw}) Step Alpha ---")

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
                   cfg["lamB"] * L_bend + cfg["lamSheet"] * L_sheet + cfg["lamOut"] * L_out + L_rigid

        _run_block(
            f"L{li} Alpha", alpha.parameters(), cfg["lrB"], cfg["nB"],
            _build_alpha, _reg_alpha,
            lambda t: f"bend={t[0].item():.2e} sheet={t[1].item():.2e} out={t[2].item():.2e} area={t[3].item():.2e} shape={t[4].item():.2e}"
        )

        # ---- Step ResidualFree: freeze phi + alpha, jointly optimize rf_psi (tangential,
        # composed on top of phi) + rf_beta (normal), both gated by 1-conf ----
        with torch.no_grad():
            disp_fix = _scaling_and_squaring_2d(phi(uv_norm_pix, H=H, W=W), n_steps=phi_n_steps)
            grid2d_phi_fix = grid_id_2d + disp_fix.unsqueeze(0)
            a_fix = alpha(uv_norm_pix, H=H, W=W)

        print(f"--- Level {li} ({gh}x{gw}) Step ResidualFree (rf_psi tangential + rf_beta normal, gated by 1-conf) ---")

        def _build_rf():
            v_psi = rf_psi(uv_norm_pix, H=H, W=W)
            disp_psi = _scaling_and_squaring_2d(gate3 * v_psi, n_steps=rf_psi_n_steps)
            grid2d = _compose_disp_2d(grid2d_phi_fix, disp_psi)
            xyz_resampled, n0_resampled = _resample_surface(grid2d)
            rf_beta_gated = gate3 * rf_beta(uv_norm_pix, H=H, W=W)
            xyz_norm = xyz_resampled + a_fix * n0_resampled + rf_beta_gated * n0_resampled
            return xyz_norm, rf_beta_gated

        def _reg_rf(xyz_norm, rf_beta_gated):
            L_bend_psi = bending_energy_2d(rf_psi.ctrl)
            L_bend_beta = bending_energy(rf_beta.ctrl)
            L_sheet = sheet_inside_volume_penalty(xyz_norm)
            L_out = (rf_beta_gated[..., 0].pow(2) * (1.0 - mask_soft[0, 0])).mean()
            (L_area, L_shape), L_rigid = _rigidity_loss(xyz_norm, cfg)
            return (L_bend_psi, L_bend_beta, L_sheet, L_out, L_area, L_shape),\
                   cfg["lamF"] * (L_bend_psi + L_bend_beta) + cfg["lamSheet"] * L_sheet\
                   + cfg["lamOut"] * L_out + L_rigid

        _run_block(
            f"L{li} ResidualFree", list(rf_psi.parameters()) + list(rf_beta.parameters()), cfg["lrF"], cfg["nF"],
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
                "Pyramid refinement v11 – final surface (phi+alpha+gated rf_psi/rf_beta)",
                save_file=f"{save_file}_final.png",
            )

    return phi, alpha, rf_psi, rf_beta, xyz_norm_final


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
    # ---- v12: local rigidity (see module header) -----------------------------
    lam_area=1.0,
    lam_shape=0.0,
    w_rigid_mode="mask",
    rigid_edge_k=2.0,
    lam_rigid_levels=(1.0, 1.0, 1.0),   # per-pyramid-level multiplier on the rigidity loss
    similarity_weights=(1.0, 0.0, 0.0),  # (w_ssim, w_ngf, w_mind), passed to pyramid_refinement
):
    print("\n-------- Refine Curved Sheet (v12: + local rigidity) --------")

    def _xyz_from_G_uv(uv_plane_flat, G, H, W, device):
        uv = uv_plane_flat.detach().cpu().numpy().reshape(H, W, 2).astype(np.float32)
        uu, vv = uv[..., 0], uv[..., 1]
        X0 = np.asarray(G["X0"], np.float32)
        e1 = np.asarray(G["e1"], np.float32)
        e2 = np.asarray(G["e2"], np.float32)
        xyz_g = X0[None, None, :] + uu[..., None] * e1[None, None, :] + vv[..., None] * e2[None, None, :]
        return torch.from_numpy(xyz_g).to(device=device, dtype=torch.float32)

    def _support_conf_from_ctrl_uv(uv_plane_flat, ctrl_uv, H, W, device, sigma_uv=20.0, min_conf=0.25):
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

    def _build_xyz_norm_tps_safe(init_tps, G, uv_plane_flat, H, W, Xdim, Ydim, Zdim, device,
                                 ctrl_uv=None, sigma_uv=20.0, min_conf=0.25, max_disp_vox=16.0):
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
            xyz_norm_tps = _build_xyz_norm_tps_safe(
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

            # Same confidence field used to blend TPS vs G-plane fallback above --
            # reused here to gate residual_free's freedom (low conf = untrustworthy
            # base surface = residual_free gets real freedom there).
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
            dict(gh=12, gw=12, nA=100, lrA=2e-3, lamA=8e-4,
                            nB=100, lrB=5e-4, lamB=1e-2,
                            nF=60, lrF=2e-3, lamF=8e-3,
                            lamSheet=2e-3, lamOut=1e-2, lamRigid=_lr[0]),
            dict(gh=24, gw=24, nA=120, lrA=1e-3, lamA=6e-4,
                            nB=120, lrB=2.5e-4, lamB=7.5e-3,
                            nF=30, lrF=1e-3, lamF=6e-3,
                            lamSheet=1.5e-3, lamOut=1e-2, lamRigid=_lr[1]),
            dict(gh=32, gw=32, nA=140, lrA=7e-4, lamA=4e-4,
                            nB=140, lrB=1.75e-4, lamB=5e-3,
                            nF=40, lrF=7e-4, lamF=4e-3,
                            lamSheet=1e-3, lamOut=1e-2, lamRigid=_lr[2]),
        ]

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

        with torch.no_grad():
            P_vox_xyz = torch.stack([
                xyz_norm_final[..., 0] * (Xdim - 1),
                xyz_norm_final[..., 1] * (Ydim - 1),
                xyz_norm_final[..., 2] * (Zdim - 1),
            ], dim=-1)

            mri_on_final_sheet = sample_mri_cubic_3d(
                mri_t, P_vox_xyz, kernel="catmull-rom", padding_mode="border"
            )[0, 0].detach().cpu().numpy()

        if torch.is_tensor(xyz_norm_final):
            valid_mri = (
                torch.isfinite(xyz_norm_final).all(dim=-1) &
                (xyz_norm_final[..., 0] >= 0) & (xyz_norm_final[..., 0] <= 1) &
                (xyz_norm_final[..., 1] >= 0) & (xyz_norm_final[..., 1] <= 1) &
                (xyz_norm_final[..., 2] >= 0) & (xyz_norm_final[..., 2] <= 1)
            )
        else:
            xyz_norm_final = np.asarray(xyz_norm_final)
            valid_mri = (
                np.isfinite(xyz_norm_final).all(axis=-1) &
                (xyz_norm_final[..., 0] >= 0) & (xyz_norm_final[..., 0] <= 1) &
                (xyz_norm_final[..., 1] >= 0) & (xyz_norm_final[..., 1] <= 1) &
                (xyz_norm_final[..., 2] >= 0) & (xyz_norm_final[..., 2] <= 1)
            )

        valid_out = valid_mri.detach().cpu().numpy()

        return xyz_norm_final, mri_on_final_sheet, valid_out, xyz_norm_base, conf, phi_refined, n0, alpha_refined, rf_psi_refined, rf_beta_refined
    finally:
        if writer is not None:
            writer.close()
            print(f"Saved GIF to: {gif_file}")

