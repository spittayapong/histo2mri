import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os

from pose_estimation_v6 import make_G_tensors, uv_from_hist_grid_inverse_G
from utilities import sample_mri_cubic_3d

# ============================================================
# 4-stage pipeline:
#   Stage 0 – Data + TPS init (you already did)
#   Stage 1 – G fixed, refine TPS + B-spline (coarse)
#   Stage 2 – tiny refinement of G (translation only)
#   Stage 3 – G fixed (refined), final TPS + B-spline refinement (Pyramid Refinement)
# ============================================================

# ============================================================
# Helper : B-spline surface + losses
# ============================================================

def bspline_basis_1d(t, grid_size):
    """
    t: (...,) normalized parameter in [0,1]
    grid_size: number of control points (>=4)
    """
    # Map t∈[0,1] to u∈[0, grid_size-3]  (open-uniform B-spline)
    u = t * (grid_size - 3)

    idx0 = torch.floor(u).clamp(0, grid_size - 4).long()
    du   = u - idx0

    w0 = ((1 - du) ** 3) / 6
    w1 = (3*du**3 - 6*du**2 + 4) / 6
    w2 = (-3*du**3 + 3*du**2 + 3*du + 1) / 6
    w3 = (du ** 3) / 6

    w = torch.stack([w0, w1, w2, w3], dim=-1)  # (...,4)
    return idx0, w

class BSplineResidualSurface3D(nn.Module):
    """
    ΔS(u_norm, v_norm): 3D residual in normalized MRI coords [0,1]^3

      ΔS(u,v) = sum_i,j B_i(u)*B_j(v)*ctrl[i,j,:]

    ctrl: (grid_h, grid_w, 3)
    """
    def __init__(self, grid_h, grid_w, device="cuda"):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.ctrl = nn.Parameter(
            torch.zeros(grid_h, grid_w, 3, device=device, dtype=torch.float32)
        )

    def forward(self, uv_norm, H=None, W=None):
        if H is None:
            u_norm = uv_norm[:, 0]
            v_norm = uv_norm[:, 1]
            dS = self._eval_surface(u_norm, v_norm)
            return dS
        else:
            u_norm = uv_norm[..., 0]
            v_norm = uv_norm[..., 1]
            dS = self._eval_surface(u_norm, v_norm)
            return dS.view(H, W, 3)

    def _eval_surface(self, u_norm, v_norm):
        iu0, wu = bspline_basis_1d(u_norm, self.grid_w)
        iv0, wv = bspline_basis_1d(v_norm, self.grid_h)

        orig_shape = u_norm.shape
        iu0 = iu0.reshape(-1)
        iv0 = iv0.reshape(-1)
        wu  = wu.reshape(-1, 4)
        wv  = wv.reshape(-1, 4)

        N_pts = iu0.shape[0]
        pts = torch.zeros((N_pts, 3), device=self.ctrl.device)

        for du in range(4):
            for dv in range(4):
                idx_u = iu0 + du
                idx_v = iv0 + dv
                w = wu[:, du] * wv[:, dv]           # (N,)
                ctrl_ij = self.ctrl[idx_v, idx_u, :]  # (N,3)
                pts += ctrl_ij * w.unsqueeze(-1)

        return pts.view(*orig_shape, 3)

def bending_energy(ctrl):
    """2nd-derivative smoothness penalty on residual ctrl grid."""
    dx2 = ctrl[2:, 1:-1] - 2*ctrl[1:-1, 1:-1] + ctrl[:-2, 1:-1]
    dy2 = ctrl[1:-1, 2:] - 2*ctrl[1:-1, 1:-1] + ctrl[1:-1, :-2]
    return (dx2.pow(2).mean() + dy2.pow(2).mean())

def sheet_inside_volume_penalty(xyz_norm_total):
    """
    Encourage the curved sheet to stay inside [0,1]^3 in normalized coords.
    """
    below = F.relu(-xyz_norm_total)
    above = F.relu(xyz_norm_total - 1)
    return (below**2 + above**2).mean()

def project_tangential(delta, n0):
    """
    Remove the component of `delta` along the local surface normal `n0`.

    delta, n0: [...,3] tensors in normalized [0,1]^3 space, n0 assumed unit length.
    Keeps only the in-plane (tangential) part of delta — the normal direction
    is reserved exclusively for the scalar alpha*n0 offset field, so residual
    and alpha stop fighting over the same degree of freedom.
    """
    comp = (delta * n0).sum(dim=-1, keepdim=True)
    return delta - comp * n0

# -----------------------------
# MIND-SSC (2D) implementation
# -----------------------------
def _shift2d(x, dy, dx):
    """Shift x by (dy,dx) with replicate padding. x: (N,C,H,W)"""
    N, C, H, W = x.shape
    pad_t = max(dy, 0)
    pad_b = max(-dy, 0)
    pad_l = max(dx, 0)
    pad_r = max(-dx, 0)
    xpad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    y0 = pad_t - dy
    x0 = pad_l - dx
    return xpad[:, :, y0:y0+H, x0:x0+W]

def mind_descriptor_2d(img, radius=1, dilation=1, sigma=0.8, eps=1e-8):
    """
    MIND-SSC-like descriptor for 2D images.
    img: (N,1,H,W) recommended.
    radius: patch half-size for local averaging of squared differences
    dilation: neighbor offset dilation
    sigma: smoothing for the SSD map (controls descriptor robustness)
    returns: (N, D, H, W) where D = number of neighbor offsets (here 8)
    """
    # 8-neighborhood offsets (dy,dx)
    offs = [
        (-1,  0), ( 1,  0), ( 0, -1), ( 0,  1),
        (-1, -1), (-1,  1), ( 1, -1), ( 1,  1),
    ]
    # apply dilation
    offs = [(dy*dilation, dx*dilation) for dy,dx in offs]

    # local SSD for each offset
    ssd_list = []
    for dy, dx in offs:
        shifted = _shift2d(img, dy, dx)
        diff2 = (img - shifted).pow(2)
        # local average of diff2 over patch (radius)
        k = 2*radius + 1
        diff2 = F.avg_pool2d(diff2, kernel_size=k, stride=1, padding=radius)
        ssd_list.append(diff2)

    ssd = torch.cat(ssd_list, dim=1)  # (N,8,H,W)

    # estimate local variance (as in MIND): use minimum SSD across offsets
    v = torch.min(ssd, dim=1, keepdim=True).values
    v = v + eps

    # normalize + exponentiate
    desc = torch.exp(-(ssd / v))

    # optional smoothing for stability
    if sigma is not None and sigma > 0:
        # separable gaussian via conv
        win = int(max(3, 2*round(3*sigma)+1))
        g = torch.arange(win, device=img.device, dtype=img.dtype) - win//2
        g = torch.exp(-0.5*(g/sigma)**2)
        g = (g / (g.sum() + eps)).view(1, 1, win, 1)
        desc = F.conv2d(desc, g.repeat(desc.shape[1], 1, 1, 1), padding=(win//2, 0), groups=desc.shape[1])
        desc = F.conv2d(desc, g.transpose(2,3).repeat(desc.shape[1], 1, 1, 1), padding=(0, win//2), groups=desc.shape[1])

    # normalize descriptor per-pixel (optional but helpful)
    desc = desc / (desc.mean(dim=1, keepdim=True) + eps)
    return desc  # (N,8,H,W)

def mind_loss_torch(img1, img2, radius=1, dilation=1, sigma=0.8, p=1, eps=1e-8):
    """
    Lp distance between MIND descriptors.
    p=1 (L1) is typical and robust.
    """
    d1 = mind_descriptor_2d(img1, radius=radius, dilation=dilation, sigma=sigma, eps=eps)
    d2 = mind_descriptor_2d(img2, radius=radius, dilation=dilation, sigma=sigma, eps=eps)
    if p == 1:
        return (d1 - d2).abs().mean()
    else:
        return ((d1 - d2).abs() + eps).pow(p).mean()

# ---------- SSIM + NGF losses ----------
def _gaussian_kernel(win=11, sigma=1.5, device='cuda', dtype=torch.float32):
    g = torch.arange(win, device=device, dtype=dtype) - win//2
    g = torch.exp(-0.5*(g/sigma)**2)
    g = (g / g.sum()).view(1, 1, win)
    k = g.transpose(1, 2) @ g
    return k

def ssim_loss_torch(img1, img2, win=11, sigma=1.5, eps=1e-8):
    N, C, H, W = img1.shape
    device, dtype = img1.device, img1.dtype
    k = _gaussian_kernel(win, sigma, device, dtype).repeat(C, 1, 1, 1)

    mu1 = F.conv2d(img1, k, padding=win//2, groups=C)
    mu2 = F.conv2d(img2, k, padding=win//2, groups=C)
    mu1_sq = mu1*mu1; mu2_sq = mu2*mu2; mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, k, padding=win//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, k, padding=win//2, groups=C) - mu2_sq
    sigma12   = F.conv2d(img1*img2, k, padding=win//2, groups=C) - mu1_mu2

    C1 = 0.01**2; C2 = 0.03**2
    num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps
    ssim_map = num / den
    return 1.0 - ssim_map.mean()

def _sobel_kernels(device, dtype):
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=dtype).view(1,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=dtype).view(1,1,3,3)
    return sx, sy

def ngf_loss_torch(img1, img2, eps=1e-8):
    device, dtype = img1.device, img1.dtype
    sx, sy = _sobel_kernels(device, dtype)

    g1x = F.conv2d(img1, sx, padding=1); g1y = F.conv2d(img1, sy, padding=1)
    g2x = F.conv2d(img2, sx, padding=1); g2y = F.conv2d(img2, sy, padding=1)

    dot = g1x*g2x + g1y*g2y
    n1  = g1x.pow(2) + g1y.pow(2)
    n2  = g2x.pow(2) + g2y.pow(2)
    ngf2 = (dot*dot) / (n1*n2 + eps)
    return 1.0 - ngf2.mean()

def combined_ssim_ngf_loss(warped, hist, w_ssim=0.5, w_ngf=0.5):
    w = (warped - warped.min()) / (warped.max() - warped.min() + 1e-8)
    h = (hist   - hist.min())   / (hist.max()   - hist.min()   + 1e-8)
    L_ssim = ssim_loss_torch(w, h, win=11, sigma=1.5)
    L_ngf  = ngf_loss_torch(w, h)
    return w_ssim*L_ssim + w_ngf*L_ngf, {"ssim": L_ssim.item(), "ngf": L_ngf.item()}

def combined_masked_ssim_ngf_loss(warped, hist, mask, w_ssim=0.5, w_ngf=0.5, eps=1e-8):
    m = (mask > 0.1).float()
    w = _norm01_t(warped)
    h = _norm01_t(hist)

    # SSIM proxy with masking
    L_ssim = ssim_loss_torch(w * m, h * m)

    # masked NGF
    sx, sy = _sobel_kernels(w.device, w.dtype)
    wgx = F.conv2d(w, sx, padding=1); wgy = F.conv2d(w, sy, padding=1)
    hgx = F.conv2d(h, sx, padding=1); hgy = F.conv2d(h, sy, padding=1)
    dot = wgx * hgx + wgy * hgy
    n1 = torch.sqrt(wgx * wgx + wgy * wgy + eps)
    n2 = torch.sqrt(hgx * hgx + hgy * hgy + eps)
    ngf2 = (dot * dot) / (n1 * n2 + eps)
    L_ngf = ((1.0 - ngf2) * m).sum() / (m.sum() + eps)

    loss = w_ssim * L_ssim + w_ngf * L_ngf
    return loss, {"ssim": float(L_ssim.detach().cpu()), "ngf": float(L_ngf.detach().cpu())}

def combined_ngf_mind_loss(
    warped, hist,
    w_ngf=0.5, w_mind=0.5,
    mind_radius=1, mind_dilation=1, mind_sigma=0.8,
):
    # normalize to [0,1] (same style as your current)
    w = (warped - warped.min()) / (warped.max() - warped.min() + 1e-8)
    h = (hist   - hist.min())   / (hist.max()   - hist.min()   + 1e-8)

    L_ngf  = ngf_loss_torch(w, h)
    L_mind = mind_loss_torch(w, h, radius=mind_radius, dilation=mind_dilation, sigma=mind_sigma, p=1)

    loss = w_ngf * L_ngf + w_mind * L_mind
    return loss, {"ngf": float(L_ngf.detach().cpu()), "mind": float(L_mind.detach().cpu())}

class BSplineFFD2D(nn.Module):
    """
    2D B-spline free-form deformation (FFD) on image plane:
      disp(u,v) = sum_i,j B_i(u) B_j(v) ctrl[j,i,:]
    ctrl: (grid_h, grid_w, 2) in *normalized grid_sample units* ([-1,1] space)
    """
    def __init__(self, grid_h, grid_w, device="cuda"):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.ctrl = nn.Parameter(torch.zeros(grid_h, grid_w, 2, device=device, dtype=torch.float32))

    def forward(self, uv_norm_pix, H, W):
        # uv_norm_pix: (H,W,2) in [0,1] (x_norm, y_norm)
        u_norm = uv_norm_pix[..., 0]
        v_norm = uv_norm_pix[..., 1]
        disp = self._eval(u_norm, v_norm)   # (H,W,2) in normalized [-1,1] units
        return disp

    def _eval(self, u_norm, v_norm):
        iu0, wu = bspline_basis_1d(u_norm, self.grid_w)
        iv0, wv = bspline_basis_1d(v_norm, self.grid_h)

        orig_shape = u_norm.shape
        iu0 = iu0.reshape(-1)
        iv0 = iv0.reshape(-1)
        wu  = wu.reshape(-1, 4)
        wv  = wv.reshape(-1, 4)

        N = iu0.shape[0]
        out = torch.zeros((N, 2), device=self.ctrl.device, dtype=self.ctrl.dtype)

        for du in range(4):
            for dv in range(4):
                idx_u = iu0 + du
                idx_v = iv0 + dv
                w = wu[:, du] * wv[:, dv]                      # (N,)
                c = self.ctrl[idx_v, idx_u, :]                 # (N,2)
                out += c * w.unsqueeze(-1)
        return out.view(*orig_shape, 2)

def bending_energy_2d(ctrl):
    """2nd-derivative smoothness penalty on 2D ctrl grid."""
    dx2 = ctrl[2:, 1:-1] - 2*ctrl[1:-1, 1:-1] + ctrl[:-2, 1:-1]
    dy2 = ctrl[1:-1, 2:] - 2*ctrl[1:-1, 1:-1] + ctrl[1:-1, :-2]
    return (dx2.pow(2).mean() + dy2.pow(2).mean())

class ScalarNormalOffsetField(nn.Module):
    """
    Scalar offset alpha(u,v) as B-spline surface.
    ctrl: (grid_h, grid_w, 1)
    Output alpha in "normalized MRI coords" (i.e., same units as xyz_norm_total).
    """
    def __init__(self, grid_h, grid_w, device="cuda"):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.ctrl = nn.Parameter(torch.zeros(grid_h, grid_w, 1, device=device, dtype=torch.float32))

    def forward(self, uv_norm_pix, H, W):
        u_norm = uv_norm_pix[..., 0]
        v_norm = uv_norm_pix[..., 1]
        a = self._eval(u_norm, v_norm)   # (H,W,1)
        return a

    def _eval(self, u_norm, v_norm):
        iu0, wu = bspline_basis_1d(u_norm, self.grid_w)
        iv0, wv = bspline_basis_1d(v_norm, self.grid_h)

        orig_shape = u_norm.shape
        iu0 = iu0.reshape(-1)
        iv0 = iv0.reshape(-1)
        wu  = wu.reshape(-1, 4)
        wv  = wv.reshape(-1, 4)

        N = iu0.shape[0]
        out = torch.zeros((N, 1), device=self.ctrl.device, dtype=self.ctrl.dtype)

        for du in range(4):
            for dv in range(4):
                idx_u = iu0 + du
                idx_v = iv0 + dv
                w = wu[:, du] * wv[:, dv]
                c = self.ctrl[idx_v, idx_u, :]      # (N,1)
                out += c * w.unsqueeze(-1)
        return out.view(*orig_shape, 1)

def _prepare_data_tensor(sitk_img, device):
    arr = sitk.GetArrayFromImage(sitk_img)
    # Add a channel dimension if the image is scalar, then a batch dimension.
    # 2D input        (H, W)     -> (1, 1, H, W)
    # 3D input:
    #   - (1, H, W)   -> treat as 2D slice -> (1, 1, H, W)
    #   - (Z, Y, X)   -> 3D volume         -> (1, 1, Z, Y, X)
    # 4D (already has channels)            -> just prepend batch.
    tensor = torch.as_tensor(arr, dtype=torch.float32, device=device)

    # If a 3D array has a singleton depth (1, H, W), squeeze it to 2D.
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)                # (H, W)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)              # (1, H, W)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)              # (1, Z, Y, X)
    elif tensor.ndim != 4:
        raise ValueError(f"Unexpected image dims {tensor.shape}; expected 2D or 3D scalar image.")

    return tensor.unsqueeze(0)                    # prepend batch -> (1, C, ...)

def prepare_coords(G, img_size, device):
    # ============================================================
    # Precompute coords shared by all stages
    # ============================================================
    H, W = img_size

    # plane coords from initial G (matches)
    Gt = make_G_tensors(G, device=device)
    uv_plane_flat_init = uv_from_hist_grid_inverse_G(H, W, Gt, device=device)  # (HW,2)

    # normalized pixel coords for B-spline (fixed)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    uv_norm_pix = torch.empty(H, W, 2, device=device)
    uv_norm_pix[..., 0] = xs / (W - 1)
    uv_norm_pix[..., 1] = ys / (H - 1)

    return Gt, uv_plane_flat_init, uv_norm_pix

def _plot_surface(z_tensor, title, save_file=None):
    z = z_tensor.detach().cpu().numpy()          # (H, W)
    ys, xs = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # light red fill, no colormap; thin red wire for shape
    surf = ax.plot_surface(
        xs, ys, z,
        color=(0.95, 0.45, 0.45, 0.35),  # RGBA (soft red, 35% opaque)
        edgecolor=(0.82, 0.20, 0.20, 0.6),
        linewidth=0.3,
        antialiased=True
    )

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    plt.tight_layout()

    # Save figure
    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        fig.savefig(save_file, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_file}")

    plt.show()

def exec_stage1(mri_t, 
                hist_t, 
                tps3d,
                residual,
                uv_plane_flat_init, 
                uv_norm_pix, 
                Xdim, Ydim, Zdim, 
                H, W, 
                device,
                save_file=None
                ):
    # ============================================================
    # Stage 1 – G fixed, refine TPS + B-spline (coarse)
    # ============================================================
    w_ssim = 0.5
    w_ngf  = 0.5
    # w_mind = 0.5
    lam_bend  = 1e-4
    lam_sheet = 1e-4   # small “stay inside volume” penalty

    lr_tps_stage1 = 1e-4
    lr_bs_stage1  = 1e-3
    n_iter_stage1 = 1600

    optimizer_stage1 = torch.optim.Adam([
        {"params": tps3d.parameters(),    "lr": lr_tps_stage1},
        {"params": residual.parameters(), "lr": lr_bs_stage1},
    ])

    # # Plot surface before stage1 refinement
    # with torch.no_grad():
    #     Xyz = tps3d(uv_plane_flat_init)           # (HW,3)
    #     xyz_norm_tps = torch.stack([
    #         Xyz[:, 0] / (Xdim - 1),
    #         Xyz[:, 1] / (Ydim - 1),
    #         Xyz[:, 2] / (Zdim - 1),
    #     ], dim=1).view(H, W, 3)
    #     delta_norm = residual(uv_norm_pix, H=H, W=W)  # (H,W,3)
    #     xyz_norm_total = xyz_norm_tps + delta_norm
    #     _plot_surface(xyz_norm_total[..., 2], "Init TPS+residual", save_file=f"{save_file}_init.png")


    print("===== Stage 1: G fixed, optimizing TPS + B-spline =====")
    for it in range(n_iter_stage1):

        Xyz = tps3d(uv_plane_flat_init)     # (HW,3)
        X = Xyz[:, 0]
        Y = Xyz[:, 1]
        Z = Xyz[:, 2]

        xyz_norm_tps = torch.stack([
            X / (Xdim - 1),
            Y / (Ydim - 1),
            Z / (Zdim - 1),
        ], dim=1).view(H, W, 3)

        delta_norm = residual(uv_norm_pix, H=H, W=W)
        xyz_norm_total = xyz_norm_tps + delta_norm

        # grid_sample coords
        xg = 2*xyz_norm_total[..., 0] - 1.0
        yg = 2*xyz_norm_total[..., 1] - 1.0
        zg = 2*xyz_norm_total[..., 2] - 1.0
        grid = torch.stack([xg, yg, zg], dim=-1).unsqueeze(0).unsqueeze(1)

        warped3d = F.grid_sample(
            mri_t, grid,
            mode='bilinear', padding_mode='border',
            align_corners=True,
        )
        warped2d = warped3d[:, :, 0, :, :]  # (1,1,H,W)

        loss_img, comp = combined_ssim_ngf_loss(warped2d, hist_t, w_ssim, w_ngf)
        # loss_img, comp = combined_ngf_mind_loss(warped2d, hist_t, w_ngf=0.5, w_mind=0.5)
        L_bend  = bending_energy(residual.ctrl)
        L_sheet = sheet_inside_volume_penalty(xyz_norm_total)

        loss = loss_img + lam_bend * L_bend + lam_sheet * L_sheet
        # loss = loss_img + lam_bend * L_bend
        # loss = loss_img

        optimizer_stage1.zero_grad()
        loss.backward()
        optimizer_stage1.step()

        if it % 50 == 0 or it == n_iter_stage1 - 1:
            print(
                f"[Stage1 TPS+FFD] it={it:04d} "
                f"loss={loss.item():.6f} "
                f"img={loss_img.item():.6f} "
                f"bend={L_bend.item():.2e} sheet={L_sheet.item():.2e} "
                f"ssim={comp['ssim']:.4f} ngf={comp['ngf']:.4f}"
                # f"mind={comp['mind']:.4f} ngf={comp['ngf']:.4f}"
            )
    
    # # Plot surface after stage1 refinement
    # with torch.no_grad():
    #     Xyz = tps3d(uv_plane_flat_init)
    #     xyz_norm_tps = torch.stack([
    #         Xyz[:, 0] / (Xdim - 1),
    #         Xyz[:, 1] / (Ydim - 1),
    #         Xyz[:, 2] / (Zdim - 1),
    #     ], dim=1).view(H, W, 3)
    #     delta_norm = residual(uv_norm_pix, H=H, W=W)
    #     xyz_norm_total = xyz_norm_tps + delta_norm
    #     _plot_surface(xyz_norm_total[..., 2], "Stage1 – G fixed, refine TPS + B-spline", save_file=f"{save_file}_stage1.png")

    return tps3d, residual


def exec_stage1_new(
    mri_t,
    mri_mask_t,
    hist_t,
    hist_mask_t,
    tps3d,
    residual,
    uv_plane_flat_init,
    uv_norm_pix,
    Xdim, Ydim, Zdim,
    H, W,
    device,
    shape_params=None,
    shape_update_every=10,
    shape_soften_ks=5,
    save_file=None
):
    # ============================================================
    # Stage 1 – G fixed, refine TPS + B-spline (coarse)
    # ============================================================
    w_ssim = 0.5
    w_ngf = 0.5
    lam_bend = 1e-3
    lam_sheet = 5e-4

    lr_tps_stage1 = 1e-4
    lr_bs_stage1 = 3e-4
    n_iter_stage1 = 200

    if shape_params is None:
        shape_params = dict(
            hist_thr=0.25, valid_thr=0.05,
            dmax_px=18, seed_dilate=1, close_iter=1, min_size=250
        )

    optimizer_stage1 = torch.optim.Adam([
        {"params": tps3d.parameters(), "lr": lr_tps_stage1},
        {"params": residual.parameters(), "lr": lr_bs_stage1},
    ])

    # Plot surface before stage1 refinement
    with torch.no_grad():
        Xyz = tps3d(uv_plane_flat_init)
        xyz_norm_tps = torch.stack([
            Xyz[:, 0] / (Xdim - 1),
            Xyz[:, 1] / (Ydim - 1),
            Xyz[:, 2] / (Zdim - 1),
        ], dim=1).view(H, W, 3)
        delta_norm = residual(uv_norm_pix, H=H, W=W)
        xyz_norm_total = xyz_norm_tps + delta_norm
        _plot_surface(xyz_norm_total[..., 2], "Init TPS+residual", save_file=f"{save_file}_init.png")

    shape_gate = None

    print("===== Stage 1: G fixed, optimizing TPS + B-spline (ROI masked) =====")
    for it in range(n_iter_stage1):
        Xyz = tps3d(uv_plane_flat_init)
        X = Xyz[:, 0]
        Y = Xyz[:, 1]
        Z = Xyz[:, 2]

        xyz_norm_tps = torch.stack([
            X / (Xdim - 1),
            Y / (Ydim - 1),
            Z / (Zdim - 1),
        ], dim=1).view(H, W, 3)

        delta_norm = residual(uv_norm_pix, H=H, W=W)
        xyz_norm_total = xyz_norm_tps + delta_norm

        xg = 2 * xyz_norm_total[..., 0] - 1.0
        yg = 2 * xyz_norm_total[..., 1] - 1.0
        zg = 2 * xyz_norm_total[..., 2] - 1.0
        grid = torch.stack([xg, yg, zg], dim=-1).unsqueeze(0).unsqueeze(1)

        warped3d = F.grid_sample(
            mri_t, grid,
            mode="bilinear", padding_mode="border",
            align_corners=True,
        )
        warped2d = warped3d[:, :, 0, :, :]  # (1,1,H,W)

        valid3d = F.grid_sample(
            mri_mask_t, grid,
            mode="bilinear", padding_mode="zeros",
            align_corners=True,
        )
        valid2d = valid3d[:, :, 0, :, :].clamp(0, 1)

        if (shape_gate is None) or (it % shape_update_every == 0):
            with torch.no_grad():
                shape_gate = shape_constrained_final_mask(
                    hist_mask_t, valid2d.detach(), **shape_params
                ).to(device=warped2d.device, dtype=warped2d.dtype)

                if shape_soften_ks > 1:
                    shape_gate = F.avg_pool2d(
                        shape_gate,
                        kernel_size=shape_soften_ks,
                        stride=1,
                        padding=shape_soften_ks // 2
                    ).clamp(0, 1)

                shape_gate = (shape_gate > 0.5).float()

        loss_img, comp = combined_masked_ssim_ngf_loss(
            warped2d, hist_t, shape_gate, w_ssim=w_ssim, w_ngf=w_ngf
        )

        DEBUG_STAGE1_VIS = True
        DEBUG_STAGE1_EVERY = 200

        if DEBUG_STAGE1_VIS and (it % DEBUG_STAGE1_EVERY == 0 or it == n_iter_stage1 - 1):
            with torch.no_grad():
                mri_np = warped2d[0, 0].detach().cpu().numpy()
                hist_np = hist_t[0, 0].detach().cpu().numpy()
                mask_np = shape_gate[0, 0].detach().cpu().numpy()  # if you use shape mask in Stage-1

                def _norm01(a, eps=1e-8):
                    a = a.astype(np.float32)
                    return (a - a.min()) / (a.max() - a.min() + eps)

                def _checker(a, b, tile=24):
                    a = _norm01(a); b = _norm01(b)
                    yy, xx = np.indices(a.shape)
                    m = ((yy // tile + xx // tile) % 2).astype(bool)
                    return np.where(m, a, b)

                mri_masked = mri_np * mask_np

                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                ax[0].imshow(hist_np, cmap="gray"); ax[0].set_title(f"Histology (Stage-1 it={it})")
                ax[1].imshow(mri_masked, cmap="gray"); ax[1].set_title("MRI masked (Stage-1)")
                ax[2].imshow(mask_np, cmap="gray"); ax[2].set_title("shape_mask (Stage-1)")
                ax[3].imshow(_checker(hist_np, mri_masked, tile=24), cmap="gray"); ax[3].set_title("Checkerboard")
                for a in ax:
                    a.axis("on")
                plt.tight_layout()
                plt.show()

        L_bend = bending_energy(residual.ctrl)
        L_sheet = sheet_inside_volume_penalty(xyz_norm_total)
        loss = loss_img + lam_bend * L_bend + lam_sheet * L_sheet

        optimizer_stage1.zero_grad()
        loss.backward()
        optimizer_stage1.step()

        if it % 50 == 0 or it == n_iter_stage1 - 1:
            print(
                f"[Stage1 TPS+FFD] it={it:04d} "
                f"loss={loss.item():.6f} "
                f"img={loss_img.item():.6f} "
                f"bend={L_bend.item():.2e} sheet={L_sheet.item():.2e} "
                f"ssim={comp['ssim']:.4f} ngf={comp['ngf']:.4f}"
            )

    # Plot surface after stage1 refinement
    with torch.no_grad():
        Xyz = tps3d(uv_plane_flat_init)
        xyz_norm_tps = torch.stack([
            Xyz[:, 0] / (Xdim - 1),
            Xyz[:, 1] / (Ydim - 1),
            Xyz[:, 2] / (Zdim - 1),
        ], dim=1).view(H, W, 3)
        delta_norm = residual(uv_norm_pix, H=H, W=W)
        xyz_norm_total = xyz_norm_tps + delta_norm
        _plot_surface(
            xyz_norm_total[..., 2],
            "Stage1 – G fixed, refine TPS + B-spline",
            save_file=f"{save_file}_stage1.png"
        )

    return tps3d, residual


def exec_stage2(
    mri_t,
    hist_t,
    tps3d,
    residual,
    Gt,
    uv_norm_pix,
    Xdim, Ydim, Zdim,
    H, W,
    device,
    save_file=None
):
    # ============================================================
    # Stage 2 – SAFE refine G: translation-only (s,R2 fixed)
    # ============================================================
    w_ssim = 0.5
    w_ngf  = 0.5

    s_const = Gt["s"].detach()
    R2_const = Gt["R2"].detach()
    t0 = Gt["t"].detach()
    Ares_const = Gt.get(
        "A_residual",
        torch.eye(2, device=device, dtype=R2_const.dtype)
    ).detach()

    # only translation learnable
    t_param = nn.Parameter(t0.clone())

    lr_G = 1e-3
    n_iter_stage2 = 150
    lam_t = 1e-2  # strong anchor to avoid floating

    optimizer_stage2 = torch.optim.Adam(
        [{"params": [t_param], "lr": lr_G}]
    )

    def uv_from_hist_grid_inverse_G_translation_only(H, W, s, R2, t_vec, Ares, device):
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"
        )
        U_math = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # (HW,2)
        A2 = s * (R2 @ Ares)
        A2_invT = torch.linalg.inv(A2).T
        P = (U_math - t_vec) @ A2_invT
        return P

    print("===== Stage 2: SAFE refine G (translation-only) =====")
    for it in range(n_iter_stage2):
        uv_plane_flat_ref = uv_from_hist_grid_inverse_G_translation_only(
            H, W, s_const, R2_const, t_param, Ares_const, device=device
        )

        # TPS sheet in 3D
        Xyz = tps3d(uv_plane_flat_ref)  # (HW,3)
        xyz_norm_tps = torch.stack([
            Xyz[:, 0] / (Xdim - 1),
            Xyz[:, 1] / (Ydim - 1),
            Xyz[:, 2] / (Zdim - 1),
        ], dim=1).view(H, W, 3)

        # frozen residual
        with torch.no_grad():
            delta_norm = residual(uv_norm_pix, H=H, W=W)
        xyz_norm_total = xyz_norm_tps + delta_norm

        # sample MRI on current sheet
        xg = 2 * xyz_norm_total[..., 0] - 1.0
        yg = 2 * xyz_norm_total[..., 1] - 1.0
        zg = 2 * xyz_norm_total[..., 2] - 1.0
        grid = torch.stack([xg, yg, zg], dim=-1).unsqueeze(0).unsqueeze(1)

        warped3d = F.grid_sample(
            mri_t, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        warped2d = warped3d[:, :, 0, :, :]

        loss_img, comp = combined_ssim_ngf_loss(warped2d, hist_t, w_ssim, w_ngf)
        L_reg_t = (t_param - t0).pow(2).sum()

        loss = loss_img + lam_t * L_reg_t

        optimizer_stage2.zero_grad()
        loss.backward()
        optimizer_stage2.step()

        if it % 25 == 0 or it == n_iter_stage2 - 1:
            print(
                f"[Stage2 SAFE] it={it:04d}/{n_iter_stage2} "
                f"loss={loss.item():.6f} img={loss_img.item():.6f} "
                f"reg_t={L_reg_t.item():.2e} "
                f"ssim={comp['ssim']:.4f} ngf={comp['ngf']:.4f}"
            )

    t_refined = t_param.detach()

    print("=== Stage2 SAFE G delta ===")
    print("Δt:", (t_refined - t0).detach().cpu().numpy())

    # fixed uv for next stage
    with torch.no_grad():
        uv_plane_flat_stage3 = uv_from_hist_grid_inverse_G_translation_only(
            H, W, s_const, R2_const, t_refined, Ares_const, device=device
        ).detach()

        # optional debug surface
        Xyz = tps3d(uv_plane_flat_stage3)
        xyz_norm_tps = torch.stack([
            Xyz[:, 0] / (Xdim - 1),
            Xyz[:, 1] / (Ydim - 1),
            Xyz[:, 2] / (Zdim - 1),
        ], dim=1).view(H, W, 3)
        delta_norm = residual(uv_norm_pix, H=H, W=W)
        xyz_norm_total = xyz_norm_tps + delta_norm
        _plot_surface(
            xyz_norm_total[..., 2],
            "Stage2 SAFE - refining translation only",
            save_file=f"{save_file}_stage2_safe.png" if save_file is not None else None
        )

    # keep return signature compatible with caller
    return uv_plane_flat_stage3, s_const, R2_const, t_refined


def sample_mri_on_sheet(mri_t, mri_mask_t, xyz_norm_sheet):
    xg = 2*xyz_norm_sheet[..., 0] - 1.0
    yg = 2*xyz_norm_sheet[..., 1] - 1.0
    zg = 2*xyz_norm_sheet[..., 2] - 1.0

    # Build grid: (1, 1, H, W, 3)
    grid = torch.stack([xg, yg, zg], dim=-1).unsqueeze(0).unsqueeze(1)  # (1,1,H,W,3)

    # Build grid: (1, 1, H, W, 3)
    warped3d = F.grid_sample(
        mri_t, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
        )
    warped3d_mask = F.grid_sample(
        mri_mask_t, grid,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
        )
    
    # Extract the 2D slice: (1, 1, H, W)
    warped2d = warped3d[:, :, 0, :, :]
    warped2d_mask = warped3d_mask[:, :, 0, :, :]

    # Ensure mask is binary (if it is not already)
    warped2d_mask = (warped2d_mask > 0).float()

    # Apply mask
    warped2d_masked = warped2d * warped2d_mask

    # return warped3d[:, :, 0, :, :]  # (1,1,H,W)
    return warped2d_masked

# Multi-Resolution Pyramid
# ---------- helper: upsample B-spline ctrl grids ----------
def _upsample_ctrl(ctrl, out_h, out_w, mode="bilinear"):
    """
    ctrl: (gh, gw, C)
    returns: (out_h, out_w, C)
    """
    # gh, gw, C = ctrl.shape
    x = ctrl.permute(2, 0, 1).unsqueeze(0)  # (1,C,gh,gw)
    x = F.interpolate(x, size=(out_h, out_w), mode=mode, align_corners=True)
    return x.squeeze(0).permute(1, 2, 0).contiguous()

def init_scalar_ctrl_from_field(a0_hw1, gh, gw):
    """Downsample a per-pixel scalar field [H,W,1] to a (gh,gw,1) B-spline ctrl init."""
    a = a0_hw1.permute(2, 0, 1).unsqueeze(0)              # [1,1,H,W]
    a = F.interpolate(a, size=(gh, gw), mode="bilinear", align_corners=True)
    return a.squeeze(0).permute(1, 2, 0).contiguous()      # [gh,gw,1]

def _copy_ctrl(dst_param, src_tensor):
    with torch.no_grad():
        dst_param.copy_(src_tensor)

def pyramid_refinement(mri_t,
                       mri_mask_t,
                       hist_t,
                       xyz_norm_base, 
                       uv_norm_pix, 
                       n0,
                       grid_id_2d,
                       H, W,
                       levels, 
                       device,
                       save_file=None
                       ):
    w_ssim, w_ngf, w_mind = 0.5, 0.5, 0.5

    phi = None
    alpha = None

    print("===== Mixed A->B per level coarse->fine =====")
    for li, cfg in enumerate(levels):
        gh, gw = cfg["gh"], cfg["gw"]

        # --- init or upsample phi ---
        if phi is None:
            phi = BSplineFFD2D(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(phi.ctrl.data, gh, gw)
            phi = BSplineFFD2D(gh, gw, device=device).to(device)
            _copy_ctrl(phi.ctrl, ctrl_up)

        # --- init or upsample alpha ---
        if alpha is None:
            alpha = ScalarNormalOffsetField(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(alpha.ctrl.data, gh, gw)
            alpha = ScalarNormalOffsetField(gh, gw, device=device).to(device)
            _copy_ctrl(alpha.ctrl, ctrl_up)

        # ========================================================
        # (A) Freeze surface, optimize phi against MRI(slice from current alpha)
        # ========================================================
        opt_phi = torch.optim.Adam([{"params": phi.parameters(), "lr": cfg["lrA"]}])

        # compute fixed MRI slice for this level from current alpha (surface fixed during A)
        with torch.no_grad():
            a_det = alpha(uv_norm_pix, H=H, W=W)
            xyz_norm_det = xyz_norm_base + a_det * n0
            mri2d_det = sample_mri_on_sheet(mri_t, mri_mask_t, xyz_norm_det).detach()

        print(f"\n--- Level {li} ({gh}x{gw}) Step A: optimize phi, surface frozen ---")
        for it in range(cfg["nA"]):
            disp = phi(uv_norm_pix, H=H, W=W)               # (H,W,2)
            grid2d = grid_id_2d + disp.unsqueeze(0)         # (1,H,W,2)

            hist_warped = F.grid_sample(
                hist_t, grid2d, mode="bilinear", padding_mode="border", align_corners=True
            )

            loss_img, comp = combined_ssim_ngf_loss(mri2d_det, hist_warped, w_ssim=w_ssim, w_ngf=w_ngf)
            # loss_img, comp = combined_ngf_mind_loss(mri2d_det, hist_warped, w_ngf=0.5, w_mind=0.5)
            L_bend = bending_energy_2d(phi.ctrl)
            loss = loss_img + cfg["lamA"] * L_bend

            opt_phi.zero_grad()
            loss.backward()
            opt_phi.step()

            if it % 50 == 0 or it == cfg["nA"] - 1:
                print(f"[L{li} A] it={it:04d}/{cfg['nA']} loss={loss.item():.6f} img={loss_img.item():.6f} "
                    f"bend={L_bend.item():.2e} ssim={comp['ssim']:.4f} ngf={comp['ngf']:.4f}")
                    # f"bend={L_bend.item():.2e} mind={comp['mind']:.4f} ngf={comp['ngf']:.4f}")

        # freeze phi result (fixed warped histology for B)
        with torch.no_grad():
            disp_det = phi(uv_norm_pix, H=H, W=W)
            grid2d_det = grid_id_2d + disp_det.unsqueeze(0)
            hist_phi_fixed_lvl = F.grid_sample(
                hist_t, grid2d_det, mode="bilinear", padding_mode="border", align_corners=True
            ).detach()

        # ========================================================
        # (B) Freeze phi, optimize alpha (surface) against fixed warped histology
        # ========================================================
        opt_theta = torch.optim.Adam([{"params": alpha.parameters(), "lr": cfg["lrB"]}])

        print(f"--- Level {li} ({gh}x{gw}) Step B: optimize surface (alpha), phi frozen ---")
        for it in range(cfg["nB"]):
            a = alpha(uv_norm_pix, H=H, W=W)                # (H,W,1)
            xyz_norm_theta = xyz_norm_base + a * n0

            mri2d_theta = sample_mri_on_sheet(mri_t, mri_mask_t, xyz_norm_theta)

            loss_img, comp = combined_ssim_ngf_loss(mri2d_theta, hist_phi_fixed_lvl, w_ssim=w_ssim, w_ngf=w_ngf)
            # loss_img, comp = combined_ngf_mind_loss(mri2d_theta, hist_phi_fixed_lvl, w_ngf=0.5, w_mind=0.5)
            L_bend = bending_energy(alpha.ctrl)
            L_sheet = sheet_inside_volume_penalty(xyz_norm_theta)

            loss = loss_img + cfg["lamB"] * L_bend + cfg["lamSheet"] * L_sheet

            opt_theta.zero_grad()
            loss.backward()
            opt_theta.step()

            if it % 50 == 0 or it == cfg["nB"] - 1:
                print(f"[L{li} B] it={it:04d}/{cfg['nB']} loss={loss.item():.6f} img={loss_img.item():.6f} "
                        f"bend={L_bend.item():.2e} sheet={L_sheet.item():.2e} "
                        f"ssim={comp['ssim']:.4f} ngf={comp['ngf']:.4f}")
                        # f"mind={comp['mind']:.4f} ngf={comp['ngf']:.4f}")
    
    # Plot surface after pyramid refinement
    with torch.no_grad():
        a_final = alpha(uv_norm_pix, H=H, W=W)          # (H,W,1)
        xyz_norm_final = xyz_norm_base + a_final * n0    # (H,W,3) in [0,1]
        _plot_surface(xyz_norm_final[..., 2], "Pyramid refinement – final surface", save_file=f"{save_file}_final.png")

    return phi, alpha

def resample_mri_on_final_curved_sheet(slab_raw_Im, 
                                       proc_Ih,
                                       phi,
                                       alpha,
                                       grid_id_2d,                                       
                                       uv_norm_pix, 
                                       xyz_norm_base,
                                       n0,
                                       Xdim, Ydim, Zdim, 
                                       H, W,
                                       device
                                       ):
    mri_raw_t = _prepare_data_tensor(slab_raw_Im, device)
    hist_t = _prepare_data_tensor(proc_Ih, device)

    with torch.no_grad():
        # final warped histology
        disp_final = phi(uv_norm_pix, H=H, W=W)
        grid2d_final = grid_id_2d + disp_final.unsqueeze(0)
        hist_warped_final = F.grid_sample(hist_t, grid2d_final, mode="bilinear",
                                        padding_mode="border", align_corners=True).detach()
        
        hist_warped_final = hist_warped_final[0,0].detach().cpu().numpy()

        # mri on final curved sheet
        a = alpha(uv_norm_pix, H=H, W=W)
        xyz_norm_final = (xyz_norm_base + a * n0).detach()
        # mri2d_final = sample_mri_on_sheet(xyz_norm_final).detach()
        
        # convert normalized [0,1] -> voxel coords
        P_vox_xyz = torch.stack([
            xyz_norm_final[..., 0] * (Xdim - 1),
            xyz_norm_final[..., 1] * (Ydim - 1),
            xyz_norm_final[..., 2] * (Zdim - 1),
        ], dim=-1)  # (H,W,3) voxel coords (x,y,z)

        mri_on_final_sheet = sample_mri_cubic_3d(
            mri_raw_t,                 # (1,1,Z,Y,X)
            P_vox_xyz,             # (H,W,3)
            kernel="catmull-rom",  # or "bspline"
            padding_mode="border",  # your sampler supports: zeros|reflect|wrap
        )  # (1,1,H,W)

        mri_on_final_sheet = mri_on_final_sheet.squeeze().cpu().numpy()

        return hist_warped_final, mri_on_final_sheet


# Main Refinement Function
def refine_curved_sheet(slab_Im, slab_raw_Im, slab_Mm, proc_Ih, init_G, init_tps, device, save_file=None):
    print("\n-------- Refine Curved Sheet --------")

    # Initial image tensors
    mri_t = _prepare_data_tensor(slab_Im, device)    
    mri_mask_t = _prepare_data_tensor(slab_Mm, device)
    hist_t = _prepare_data_tensor(proc_Ih, device)
    _, _, H, W = hist_t.shape
    Zdim, Ydim, Xdim = mri_t.shape[-3:]


    # Precompute coords shared by all stages
    Gt, uv_plane_flat_init, uv_norm_pix = prepare_coords(G=init_G, img_size=(H,W), device=device)
  
    grid_h = 12
    grid_w = 12
    residual = BSplineResidualSurface3D(grid_h, grid_w, device=device).to(device)


    # Stage 1 – G fixed, refine TPS + B-spline (coarse)  
    tps3d, residual = exec_stage1(mri_t=mri_t,
                                  hist_t=hist_t,
                                  tps3d=init_tps,
                                  residual=residual,
                                  uv_plane_flat_init=uv_plane_flat_init,
                                  uv_norm_pix=uv_norm_pix,
                                  Xdim=Xdim, Ydim=Ydim, Zdim=Zdim,
                                  H=H, W=W,
                                  device=device,
                                  save_file=save_file
                                  )
    
    # Stage 2 – refine full G (s, R2, t), TPS+FFD frozen
    uv_plane_flat_stage3, _, _, _ = exec_stage2(mri_t=mri_t,
                                                                     hist_t=hist_t,
                                                                     tps3d=tps3d,
                                                                     residual=residual,
                                                                     Gt=Gt,
                                                                     uv_norm_pix=uv_norm_pix, 
                                                                     Xdim=Xdim, Ydim=Ydim, Zdim=Zdim,
                                                                     H=H, W=W,
                                                                     device=device,
                                                                     save_file=save_file
                                                                     )
    
    # Build a fixed base sheet from your current parameters
    with torch.no_grad():
        # choose uv_plane_flat_init or uv_plane_flat_stage3 depending on where you are
        uv_plane_flat_fixed = uv_plane_flat_stage3.detach()

        Xyz = tps3d(uv_plane_flat_fixed)  # (HW,3)
        X = Xyz[:, 0]; Y = Xyz[:, 1]; Z = Xyz[:, 2]

        xyz_norm_tps = torch.stack([
            X / (Xdim - 1),
            Y / (Ydim - 1),
            Z / (Zdim - 1),
        ], dim=1).view(H, W, 3)

        delta_norm_fixed = residual(uv_norm_pix, H=H, W=W)  # use your current residual; freeze later
        xyz_norm_base = (xyz_norm_tps + delta_norm_fixed).detach()  # (H,W,3) in [0,1] coords

    # Compute fixed normals from base sheet (xyz_norm_base)
    with torch.no_grad():
        S = xyz_norm_base  # (H,W,3)

        # finite differences (central-ish), in normalized coords
        Su = torch.zeros_like(S)
        Sv = torch.zeros_like(S)

        Su[:, 1:-1] = (S[:, 2:] - S[:, :-2]) * 0.5
        Su[:, 0]    = (S[:, 1] - S[:, 0])
        Su[:, -1]   = (S[:, -1] - S[:, -2])

        Sv[1:-1, :] = (S[2:, :] - S[:-2, :]) * 0.5
        Sv[0, :]    = (S[1, :] - S[0, :])
        Sv[-1, :]   = (S[-1, :] - S[-2, :])

        n = torch.cross(Su, Sv, dim=-1)
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
        n0 = n.detach()  # (H,W,3), fixed
    
    # Multi-Resolution Pyramid Refinement
    # ---------- build identity 2D grid in [-1,1] once ----------
    base_x = 2*uv_norm_pix[..., 0] - 1.0
    base_y = 2*uv_norm_pix[..., 1] - 1.0
    grid_id_2d = torch.stack([base_x, base_y], dim=-1).unsqueeze(0)  # (1,H,W,2)

    # Coarse->fine schedules  
    levels = [
        dict(gh=12, gw=12, nA=500, lrA=2e-3, lamA=8e-4,
                        nB=350, lrB=1e-3, lamB=2e-3, lamSheet=2e-3),
        dict(gh=24, gw=24, nA=600, lrA=1e-3, lamA=6e-4,
                        nB=450, lrB=7e-4, lamB=1.5e-3, lamSheet=1.5e-3),
        dict(gh=32, gw=32, nA=700, lrA=7e-4, lamA=4e-4,
                        nB=550, lrB=5e-4, lamB=1.0e-3, lamSheet=1.0e-3),
    ]

    phi, alpha = pyramid_refinement(mri_t=mri_t,
                                    mri_mask_t=mri_mask_t,
                                    hist_t=hist_t,
                                    xyz_norm_base=xyz_norm_base,
                                    uv_norm_pix=uv_norm_pix,
                                    n0=n0,
                                    grid_id_2d=grid_id_2d,
                                    H=H, W=W,
                                    levels=levels,
                                    device=device,
                                    save_file=save_file
                                    )
    
    hist_warped_final, mri_on_final_sheet = resample_mri_on_final_curved_sheet(slab_raw_Im=slab_raw_Im,
                                                                               proc_Ih=proc_Ih,
                                                                               phi=phi,
                                                                               alpha=alpha,
                                                                               grid_id_2d=grid_id_2d,
                                                                               uv_norm_pix=uv_norm_pix,
                                                                               xyz_norm_base=xyz_norm_base,
                                                                               n0=n0,
                                                                               Xdim=Xdim, Ydim=Ydim, Zdim=Zdim,
                                                                               H=H, W=W,
                                                                               device=device
                                                                               )

    torch.cuda.empty_cache()
    
    return hist_warped_final, mri_on_final_sheet, phi, alpha, uv_norm_pix, grid_id_2d, xyz_norm_base, n0








# ---------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------
from matplotlib.path import Path
from scipy.spatial import ConvexHull

def build_roi_soft_from_inliers_hull(inlier_matches, H, W, device, tissue_mask=None, expand_px=12, blur_ks=11):
    # points in (x,y)
    pts = np.array([[m["hist_center_yx"][1], m["hist_center_yx"][0]] for m in inlier_matches], dtype=np.float32)
    if len(pts) < 3:
        raise ValueError("Need >=3 inlier points for hull ROI.")

    hull = ConvexHull(pts)
    poly = pts[hull.vertices]  # polygon vertices

    yy, xx = np.mgrid[0:H, 0:W]
    q = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = Path(poly).contains_points(q).reshape(H, W).astype(np.float32)

    roi = torch.from_numpy(inside)[None, None].to(device)

    if expand_px > 0:
        k = 2 * expand_px + 1
        roi = F.max_pool2d(roi, kernel_size=k, stride=1, padding=expand_px)

    if tissue_mask is not None:
        # tissue_mask: [1,1,H,W] float/bool
        roi = roi * (tissue_mask > 0.5).float()

    if blur_ks > 1:
        if blur_ks % 2 == 0:
            blur_ks += 1
        roi = F.avg_pool2d(roi, kernel_size=blur_ks, stride=1, padding=blur_ks // 2)

    return roi.clamp(0, 1)

def sample_mri_on_sheet_with_mask(mri_t, mri_mask_t, xyz_norm_sheet, pad_mode="zeros"):
    """
    xyz_norm_sheet: [H,W,3] in [0,1]
    Returns:
      mri2d: [1,1,H,W]
      valid2d: [1,1,H,W] (sampled MRI mask on sheet)
    """
    xg = 2 * xyz_norm_sheet[..., 0] - 1.0
    yg = 2 * xyz_norm_sheet[..., 1] - 1.0
    zg = 2 * xyz_norm_sheet[..., 2] - 1.0
    grid = torch.stack([xg, yg, zg], dim=-1).unsqueeze(0).unsqueeze(1)  # [1,1,H,W,3]

    mri3d = F.grid_sample(mri_t, grid, mode="bilinear", padding_mode=pad_mode, align_corners=True)
    msk3d = F.grid_sample(mri_mask_t, grid, mode="bilinear", padding_mode=pad_mode, align_corners=True)

    mri2d = mri3d[:, :, 0, :, :]
    # valid2d = (msk3d[:, :, 0, :, :] > 0.5).float()
    valid2d = msk3d[:, :, 0, :, :].clamp(0, 1)   # keep soft mask
    return mri2d, valid2d


# ---------------------------------------------------------
# Masked losses (MIND + orientation)
# ---------------------------------------------------------
def _norm01_t(x):
    return (x - x.amin(dim=(-2, -1), keepdim=True)) / (x.amax(dim=(-2, -1), keepdim=True) - x.amin(dim=(-2, -1), keepdim=True) + 1e-8)

def masked_mind_loss(img1, img2, mask, radius=1, dilation=1, sigma=0.8):
    d1 = mind_descriptor_2d(img1, radius=radius, dilation=dilation, sigma=sigma)  # [N,D,H,W]
    d2 = mind_descriptor_2d(img2, radius=radius, dilation=dilation, sigma=sigma)
    diff = (d1 - d2).abs().mean(dim=1, keepdim=True)  # [N,1,H,W]
    return (diff * mask).sum() / (mask.sum() + 1e-8)

def masked_orient_loss(img1, img2, mask, eps=1e-8):
    # HOG-like orientation consistency (differentiable)
    sx, sy = _sobel_kernels(img1.device, img1.dtype)
    g1x = F.conv2d(img1, sx, padding=1); g1y = F.conv2d(img1, sy, padding=1)
    g2x = F.conv2d(img2, sx, padding=1); g2y = F.conv2d(img2, sy, padding=1)

    dot = g1x * g2x + g1y * g2y
    n1 = torch.sqrt(g1x * g1x + g1y * g1y + eps)
    n2 = torch.sqrt(g2x * g2x + g2y * g2y + eps)

    cos = (dot / (n1 * n2 + eps)).abs()  # orientation ignoring sign
    loss_map = 1.0 - cos
    return (loss_map * mask).sum() / (mask.sum() + 1e-8)

def combined_masked_mind_orient(warped, hist, mask, w_mind=0.7, w_orient=0.3,
                                mind_radius=1, mind_dilation=1, mind_sigma=0.8):
    w = _norm01_t(warped)
    h = _norm01_t(hist)
    Lm = masked_mind_loss(w, h, mask, radius=mind_radius, dilation=mind_dilation, sigma=mind_sigma)
    Lo = masked_orient_loss(w, h, mask)
    return w_mind * Lm + w_orient * Lo, {"mind": float(Lm.detach().cpu()), "orient": float(Lo.detach().cpu())}

# ---------------------------------------------------------
# ROI-guided refinement
# ---------------------------------------------------------

# def prealign_hist_to_sheet_by_mask(hist_t, hist_mask_t, target_valid_t, iters=200, lr=5e-3):
#     dev = hist_t.device
#     tgt = (target_valid_t > 0.5).float().detach()

#     log_s = torch.zeros(1, device=dev, requires_grad=True)
#     ang   = torch.zeros(1, device=dev, requires_grad=True)
#     tx    = torch.zeros(1, device=dev, requires_grad=True)
#     ty    = torch.zeros(1, device=dev, requires_grad=True)

#     opt = torch.optim.Adam([log_s, ang, tx, ty], lr=lr)

#     with torch.enable_grad():
#         for _ in range(iters):
#             s = torch.exp(log_s).clamp(0.85, 1.20)
#             c = torch.cos(ang)
#             sn = torch.sin(ang)
#             tnx = 0.35 * torch.tanh(tx)
#             tny = 0.35 * torch.tanh(ty)

#             row1 = torch.stack([s * c, -s * sn, tnx], dim=-1)  # [1,3]
#             row2 = torch.stack([s * sn,  s * c, tny], dim=-1)  # [1,3]
#             A = torch.stack([row1, row2], dim=1)               # [1,2,3]

#             grid = F.affine_grid(A, hist_t.size(), align_corners=True)
#             mw = F.grid_sample(hist_mask_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

#             inter = (mw * tgt).sum()
#             dice = (2.0 * inter) / (mw.sum() + tgt.sum() + 1e-8)
#             reg = 1e-3 * (log_s.pow(2) + ang.pow(2) + tx.pow(2) + ty.pow(2)).sum()
#             loss = (1.0 - dice) + reg

#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#     with torch.no_grad():
#         s = torch.exp(log_s)
#         c = torch.cos(ang)
#         sn = torch.sin(ang)
#         tnx = 0.35 * torch.tanh(tx)
#         tny = 0.35 * torch.tanh(ty)

#         row1 = torch.stack([s * c, -s * sn, tnx], dim=-1)
#         row2 = torch.stack([s * sn,  s * c, tny], dim=-1)
#         A = torch.stack([row1, row2], dim=1)

#         grid = F.affine_grid(A, hist_t.size(), align_corners=True)
#         hist_w = F.grid_sample(hist_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
#         mask_w = F.grid_sample(hist_mask_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
#         mask_w = (mask_w > 0.5).float()

#     return hist_w, mask_w, A

import scipy.ndimage as ndi

def shape_constrained_final_mask(
    hist_mask_w, valid_soft_w,
    hist_thr=0.25, valid_thr=0.05,
    dmax_px=18, seed_dilate=2,
    close_iter=1, min_size=250,
):
    H = (hist_mask_w[0, 0].detach().cpu().numpy() > hist_thr)
    V = (valid_soft_w[0, 0].detach().cpu().numpy() > valid_thr)

    D = ndi.distance_transform_edt(~H)
    cand = V & (D <= dmax_px)

    seed = ndi.binary_dilation(H, iterations=seed_dilate) & V
    cand = cand | seed

    lab, n = ndi.label(cand)
    if n > 0:
        scores = []
        for i in range(1, n + 1):
            comp = (lab == i)
            scores.append((comp & seed).sum())
        best = int(np.argmax(scores)) + 1
        M = (lab == best)
    else:
        M = seed

    M = ndi.binary_closing(M, iterations=close_iter)
    M = ndi.binary_fill_holes(M)

    lab2, n2 = ndi.label(M)
    if n2 > 0:
        M2 = np.zeros_like(M, dtype=bool)
        for i in range(1, n2 + 1):
            comp = (lab2 == i)
            if comp.sum() >= min_size:
                M2 |= comp
        M = M2

    # keep only valid region
    M = M & V

    return torch.from_numpy(M.astype(np.float32))[None, None].to(hist_mask_w.device)

def _debug_sheet_view(tag, hist_t, mri2d, shape_mask, tile=24):
    hist_np = hist_t[0, 0].detach().cpu().numpy()
    mri_np  = mri2d[0, 0].detach().cpu().numpy()
    mask_np = shape_mask[0, 0].detach().cpu().numpy()
    mri_masked_np = mri_np * mask_np

    def _norm01(a, eps=1e-8):
        a = a.astype(np.float32)
        return (a - a.min()) / (a.max() - a.min() + eps)

    def _checker(a, b, tile=24):
        a = _norm01(a); b = _norm01(b)
        hh, ww = a.shape
        yy, xx = np.indices((hh, ww))
        m = ((yy // tile + xx // tile) % 2).astype(bool)
        return np.where(m, a, b)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(hist_np, cmap="gray"); ax[0].set_title(f"Histology (ref) [{tag}]")
    ax[1].imshow(mri_masked_np, cmap="gray"); ax[1].set_title(f"MRI [{tag}] (shape mask)")
    ax[2].imshow(mask_np, cmap="gray"); ax[2].set_title(f"shape mask [{tag}]")
    ax[3].imshow(_checker(hist_np, mri_masked_np, tile=tile), cmap="gray"); ax[3].set_title(f"Checkerboard [{tag}]")
    for a in ax:
        a.axis("on")
    plt.tight_layout()
    plt.show()

def refine_curved_sheet_roi(
    slab_Im, slab_raw_Im, slab_Mm, proc_Ih, proc_Mh,
    init_G, init_tps,
    inlier_matches,        # <- from your filtered matches
    device="cuda:0",
    run_stage12=True,      # run your existing Stage1/2 first for stabilization
    save_file=None
):
    mri_t = _prepare_data_tensor(slab_Im, device)
    mri_mask_t = _prepare_data_tensor(slab_Mm, device)
    hist_t = _prepare_data_tensor(proc_Ih, device)
    hist_mask_t = (_prepare_data_tensor(proc_Mh, device) > 0.5).float()
    _, _, H, W = hist_t.shape
    Zdim, Ydim, Xdim = mri_t.shape[-3:]

    Gt, uv_plane_flat_init, uv_norm_pix = prepare_coords(G=init_G, img_size=(H, W), device=device)

    # init residual + optional Stage1/2
    residual = BSplineResidualSurface3D(12, 12, device=device).to(device)
    tps3d = init_tps

    with torch.no_grad():
        # plain init-TPS on init-G grid (no residual refinement yet)
        Xyz0 = tps3d(uv_plane_flat_init)  # [HW,3]
        xyz_norm0 = torch.stack([
            Xyz0[:, 0] / (Xdim - 1),
            Xyz0[:, 1] / (Ydim - 1),
            Xyz0[:, 2] / (Zdim - 1),
        ], dim=1).view(H, W, 3).detach()

        mri2d0, valid0 = sample_mri_on_sheet_with_mask(
            mri_t, mri_mask_t, xyz_norm0, pad_mode="zeros"
        )

    # define parameters for roi on mri sheet
    shape_params = dict(hist_thr=0.25, valid_thr=0.05, dmax_px=18, seed_dilate=1, close_iter=1, min_size=250)
    shape_update_every = 20
    shape_soften_ks  = 5

    # initiate roi on mri sheet
    hist_mask_np = hist_mask_t[0, 0].detach().cpu().numpy()
    shape_mask0 = shape_constrained_final_mask(
        hist_mask_t, valid0, **shape_params
    )
    shape_mask0_np = shape_mask0[0, 0].detach().cpu().numpy()
    valid0_np = valid0[0, 0].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(hist_mask_np, cmap="gray"); ax[0].set_title("Histology mask (pre Stage-1/2)")
    ax[1].imshow(valid0_np, cmap="gray"); ax[1].set_title("Valid mask (pre Stage-1/2)")
    ax[2].imshow(shape_mask0_np, cmap="gray"); ax[2].set_title("shape_constrained_final_mask")
    for a in ax:
        a.axis("on")
    plt.tight_layout()
    plt.show()

    # ---------------- DEBUG: baseline before Stage-1/2 ----------------
    DEBUG_PRE_STAGE12 = True
    if DEBUG_PRE_STAGE12:
        with torch.no_grad():
            x = xyz_norm0[..., 0]; y = xyz_norm0[..., 1]; z = xyz_norm0[..., 2]

            hist_np = hist_t[0, 0].detach().cpu().numpy()
            mri_np = mri2d0[0, 0].detach().cpu().numpy()

            def _norm01(a, eps=1e-8):
                a = a.astype(np.float32)
                return (a - a.min()) / (a.max() - a.min() + eps)

            def _checker(a, b, tile=24):
                a = _norm01(a); b = _norm01(b)
                hh, ww = a.shape
                yy, xx = np.indices((hh, ww))
                m = ((yy // tile + xx // tile) % 2).astype(bool)
                return np.where(m, a, b)

            # MRI masked by shape mask
            mri_masked_np = mri_np * shape_mask0_np

            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax[0].imshow(hist_np, cmap="gray"); ax[0].set_title("Histology (ref)")
            ax[1].imshow(mri_masked_np, cmap="gray"); ax[1].set_title("MRI @ pre Stage-1/2 (shape_mask0)")
            ax[2].imshow(shape_mask0_np, cmap="gray"); ax[2].set_title("shape_mask0")
            ax[3].imshow(_checker(hist_np, mri_masked_np, tile=24), cmap="gray"); ax[3].set_title("Checkerboard")
            for a in ax:
                a.axis("on")
            plt.tight_layout()
            plt.show()
    # ---------------- end DEBUG block ----------------


    if run_stage12:
        # tps3d, residual = exec_stage1(
        #     mri_t=mri_t, hist_t=hist_t, tps3d=tps3d, residual=residual,
        #     uv_plane_flat_init=uv_plane_flat_init, uv_norm_pix=uv_norm_pix,
        #     Xdim=Xdim, Ydim=Ydim, Zdim=Zdim, H=H, W=W, device=device, save_file=save_file
        # )

        tps3d, residual = exec_stage1_new(
            mri_t=mri_t,
            mri_mask_t=mri_mask_t,
            hist_t=hist_t,
            hist_mask_t=hist_mask_t,
            tps3d=tps3d,
            residual=residual,
            uv_plane_flat_init=uv_plane_flat_init,
            uv_norm_pix=uv_norm_pix,
            Xdim=Xdim, Ydim=Ydim, Zdim=Zdim,
            H=H, W=W,
            device=device,
            shape_params=shape_params,
            shape_update_every=shape_update_every,
            shape_soften_ks=shape_soften_ks,
            save_file=save_file
        )

        # ---------------- DEBUG: after Stage-1 ----------------
        DEBUG_POST_STAGE1 = True
        if DEBUG_POST_STAGE1:
            with torch.no_grad():
                Xyz1 = tps3d(uv_plane_flat_init)
                xyz_norm_tps1 = torch.stack([
                    Xyz1[:, 0] / (Xdim - 1),
                    Xyz1[:, 1] / (Ydim - 1),
                    Xyz1[:, 2] / (Zdim - 1),
                ], dim=1).view(H, W, 3)
                delta_norm1 = residual(uv_norm_pix, H=H, W=W)
                xyz_norm_stage1 = xyz_norm_tps1 + delta_norm1

                mri2d_stage1, valid_stage1 = sample_mri_on_sheet_with_mask(
                    mri_t, mri_mask_t, xyz_norm_stage1, pad_mode="zeros"
                )
                shape_mask_stage1 = shape_constrained_final_mask(
                    hist_mask_t, valid_stage1, **shape_params
                )
                _debug_sheet_view("post Stage-1", hist_t, mri2d_stage1, shape_mask_stage1, tile=24)
        # ---------------- end DEBUG block ----------------
        
        # uv_plane_flat_fixed, _, _, _ = exec_stage2(
        #     mri_t=mri_t, hist_t=hist_t, tps3d=tps3d, residual=residual, Gt=Gt, uv_norm_pix=uv_norm_pix,
        #     Xdim=Xdim, Ydim=Ydim, Zdim=Zdim, H=H, W=W, device=device, save_file=save_file
        # )

        # # ---------------- DEBUG: after Stage-2 ----------------
        # DEBUG_POST_STAGE2 = True
        # if DEBUG_POST_STAGE2:
        #     with torch.no_grad():
        #         Xyz2 = tps3d(uv_plane_flat_fixed)
        #         xyz_norm_tps2 = torch.stack([
        #             Xyz2[:, 0] / (Xdim - 1),
        #             Xyz2[:, 1] / (Ydim - 1),
        #             Xyz2[:, 2] / (Zdim - 1),
        #         ], dim=1).view(H, W, 3)
        #         delta_norm2 = residual(uv_norm_pix, H=H, W=W)
        #         xyz_norm_stage2 = xyz_norm_tps2 + delta_norm2

        #         mri2d_stage2, valid_stage2 = sample_mri_on_sheet_with_mask(
        #             mri_t, mri_mask_t, xyz_norm_stage2, pad_mode="zeros"
        #         )
        #         shape_mask_stage2 = shape_constrained_final_mask(
        #             hist_mask_t, valid_stage2, **shape_params
        #         )
        #         _debug_sheet_view("post Stage-2", hist_t, mri2d_stage2, shape_mask_stage2, tile=24)
        # # ---------------- end DEBUG block ----------------

        # uv_plane_flat_fixed = uv_plane_flat_fixed.detach()
    
    uv_plane_flat_fixed = uv_plane_flat_init.detach()   

    # build base sheet
    with torch.no_grad():
        Xyz = tps3d(uv_plane_flat_fixed)  # [HW,3]
        xyz_norm_tps = torch.stack([
            Xyz[:, 0] / (Xdim - 1),
            Xyz[:, 1] / (Ydim - 1),
            Xyz[:, 2] / (Zdim - 1),
        ], dim=1).view(H, W, 3)
        delta_norm = residual(uv_norm_pix, H=H, W=W)
        xyz_norm_base = (xyz_norm_tps + delta_norm).detach()

        # normals
        S = xyz_norm_base
        Su = torch.zeros_like(S); Sv = torch.zeros_like(S)
        Su[:, 1:-1] = 0.5 * (S[:, 2:] - S[:, :-2]); Su[:, 0] = S[:, 1] - S[:, 0]; Su[:, -1] = S[:, -1] - S[:, -2]
        Sv[1:-1, :] = 0.5 * (S[2:, :] - S[:-2, :]); Sv[0, :] = S[1, :] - S[0, :]; Sv[-1, :] = S[-1, :] - S[-2, :]
        n0 = torch.cross(Su, Sv, dim=-1)
        n0 = n0 / (n0.norm(dim=-1, keepdim=True) + 1e-8)
        n0 = n0.detach()

    # identity 2D grid
    base_x = 2 * uv_norm_pix[..., 0] - 1.0
    base_y = 2 * uv_norm_pix[..., 1] - 1.0
    grid_id_2d = torch.stack([base_x, base_y], dim=-1).unsqueeze(0)  # [1,H,W,2]
    
    roi_soft = hist_mask_t

    DEBUG_EVERY = 100
    PLOT_SHEET = True
    PLOT_STRIDE = 6

    def _plot_sheet_move(xyz_norm_cur, xyz_norm_ref, tag, Xdim, Ydim, Zdim, stride=6):
        cur = xyz_norm_cur.detach().cpu()
        ref = xyz_norm_ref.detach().cpu()

        d = torch.linalg.norm(cur - ref, dim=-1).numpy()  # normalized displacement
        print(f"[{tag}] disp mean={d.mean():.5f}, p95={np.percentile(d,95):.5f}, max={d.max():.5f}")

        X = (cur[..., 0].numpy() * (Xdim - 1))
        Y = (cur[..., 1].numpy() * (Ydim - 1))
        Z = (cur[..., 2].numpy() * (Zdim - 1))

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(d, cmap="magma")
        ax1.set_title(f"{tag}: |delta xyz|")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_wireframe(
            X[::stride, ::stride],
            Y[::stride, ::stride],
            Z[::stride, ::stride],
            rstride=1, cstride=1,
            color=(0.9, 0.25, 0.25),
            linewidth=0.55, alpha=0.95
        )
        ax2.set_title(f"{tag}: curved sheet")
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

    # # multilevel A/B (same idea as your pyramid_refinement)
    # levels = [
    #     dict(gh=12, gw=12, nA=50, lrA=1e-3, lamA=8e-3, lamA_out=5e-1,
    #          nB=150, lrB=1e-4, lamB=2e-3, lamSheet=2e-3, lamB_out=5e-3),
    #     dict(gh=24, gw=24, nA=100, lrA=1e-3, lamA=6e-3, lamA_out=7e-1,
    #          nB=300, lrB=7e-4, lamB=1.5e-3, lamSheet=1.5e-3, lamB_out=7e-3),
    #     # dict(
    #     #     gh=32, gw=32,
    #     #     nA=150, lrA=3e-4, lamA=1.2e-3, lamA_out=1e-4,
    #     #     nB=300, lrB=2e-4, lamB=2.0e-3, lamSheet=1e-3, lamB_out=9e-3
    #     #     )
    # ]
    # Coarse->fine schedules  
    # levels = [
    #     dict(gh=12, gw=12, nA=500, lrA=2e-3, lamA=8e-4,
    #                     nB=350, lrB=1e-3, lamB=2e-3, lamSheet=2e-3),
    #     dict(gh=24, gw=24, nA=600, lrA=1e-3, lamA=6e-4,
    #                     nB=450, lrB=7e-4, lamB=1.5e-3, lamSheet=1.5e-3),
    #     dict(gh=32, gw=32, nA=700, lrA=7e-4, lamA=4e-4,
    #                     nB=550, lrB=5e-4, lamB=1.0e-3, lamSheet=1.0e-3),
    # ]
    levels = [
        dict(gh=12, gw=12, nA=50, lrA=2e-3, lamA=8e-4,
                        nB=350, lrB=1e-3, lamB=2e-3, lamSheet=2e-3),
        dict(gh=24, gw=24, nA=50, lrA=1e-3, lamA=6e-4,
                        nB=450, lrB=7e-4, lamB=1.5e-3, lamSheet=1.5e-3),
        dict(gh=32, gw=32, nA=50, lrA=7e-4, lamA=4e-4,
                        nB=550, lrB=5e-4, lamB=1.0e-3, lamSheet=1.0e-3),
    ]

    phi, alpha = None, None
    print("===== ROI-guided multi-resolution refinement (MIND + orient) =====")



    for li, cfg in enumerate(levels):
        gh, gw = cfg["gh"], cfg["gw"]

        if phi is None:
            phi = BSplineFFD2D(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(phi.ctrl.data, gh, gw)
            phi = BSplineFFD2D(gh, gw, device=device).to(device)
            _copy_ctrl(phi.ctrl, ctrl_up)

        if alpha is None:
            alpha = ScalarNormalOffsetField(gh, gw, device=device).to(device)
        else:
            ctrl_up = _upsample_ctrl(alpha.ctrl.data, gh, gw)
            alpha = ScalarNormalOffsetField(gh, gw, device=device).to(device)
            _copy_ctrl(alpha.ctrl, ctrl_up)

        # -------- Warp histology to sampled MRI on curved sheet --------
        # ---------------- A: optimize phi, alpha frozen ----------------
        opt_phi = torch.optim.Adam([{"params": phi.parameters(), "lr": cfg["lrA"]}])

        with torch.no_grad():
            a_det = alpha(uv_norm_pix, H=H, W=W)
            xyz_det = xyz_norm_base + a_det * n0
            mri2d_det, valid_det = sample_mri_on_sheet_with_mask(
                mri_t, mri_mask_t, xyz_det, pad_mode="zeros"
            )
            valid_det = valid_det.detach()
        
        shape_gate_A = None
        roi2d = roi_soft[0, 0]  # histology mask

        for it in range(cfg["nA"]):           

            disp_raw = phi(uv_norm_pix, H=H, W=W)
            # disp = disp_raw * roi_gate2d.unsqueeze(-1)
            disp = disp_raw
            grid2d = grid_id_2d + disp.unsqueeze(0)

            hist_w = F.grid_sample(hist_t, grid2d, mode="bilinear", padding_mode="border", align_corners=True)

            hist_mask_w = F.grid_sample(
                hist_mask_t, grid2d, mode="nearest",
                padding_mode="zeros", align_corners=True
            )

            # find roi in sampled MRI
            if (shape_gate_A is None) or (it % shape_update_every == 0):
                with torch.no_grad():
                    shape_gate_A = shape_constrained_final_mask(
                        hist_mask_w.detach(), valid_det.detach(), **shape_params
                    ).to(device=hist_w.device, dtype=hist_w.dtype)
                    if shape_soften_ks > 1:
                        shape_gate_A = F.avg_pool2d(
                            shape_gate_A, kernel_size=shape_soften_ks, stride=1, padding=shape_soften_ks // 2
                        ).clamp(0, 1)
                    shape_gate_A = (shape_gate_A > 0.5).float()

            # extract only roi in sampled MRI, and use it as reference image for warping histology
            mri_used  = mri2d_det * shape_gate_A  
            # mri_used  = mri2d_det
            # mask = (roi_w * shape_gate_A).clamp(0, 1)
            # mask = hist_mask_w

            # loss_img, comp = combined_masked_mind_orient(
            #     mri2d_det, hist_w, mask,
            #     w_mind=0.7, w_orient=0.3,
            #     mind_radius=1, mind_dilation=1, mind_sigma=0.8
            # )
            loss_img, comp = combined_ssim_ngf_loss(
                mri_used, hist_w, w_ssim=0.5, w_ngf=0.5
            )
            # loss_img, comp = combined_masked_ssim_ngf_loss(mri_used, hist_w, shape_gate_A, w_ssim=0.5, w_ngf=0.5)

            L_bend = bending_energy_2d(phi.ctrl)
            L_out = ((disp_raw[...,0]**2 + disp_raw[...,1]**2) * (1.0 - roi2d)).mean()

            # loss = loss_img + cfg["lamA"] * L_bend + cfg["lamA_out"] * L_out
            loss = loss_img + cfg["lamA"] * L_bend

            DEBUG_A_VIS = True
            DEBUG_A_EVERY = 50

            if DEBUG_A_VIS and (it % DEBUG_A_EVERY == 0 or it == cfg["nA"] - 1):
                with torch.no_grad():
                    mri_np = mri_used[0, 0].detach().cpu().numpy()
                    h_w_np = hist_w[0, 0].detach().cpu().numpy()

                    def _norm01(a, eps=1e-8):
                        a = a.astype(np.float32)
                        return (a - a.min()) / (a.max() - a.min() + eps)

                    def _checker(a, b, tile=24):
                        a = _norm01(a); b = _norm01(b)
                        yy, xx = np.indices(a.shape)
                        mm = ((yy // tile + xx // tile) % 2).astype(bool)
                        return np.where(mm, a, b)

                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(mri_np, cmap="gray")
                    ax[0].set_title(f"MRI on surface (A) L{li} it={it}")

                    ax[1].imshow(h_w_np, cmap="gray")
                    ax[1].set_title("Warped histology")

                    ax[2].imshow(_checker(mri_np, h_w_np, tile=24), cmap="gray")
                    ax[2].set_title("Checkerboard")

                    for a in ax:
                        a.axis("on")
                    plt.tight_layout()
                    plt.show()

            opt_phi.zero_grad()
            loss.backward()
            opt_phi.step()

            if it % 50 == 0 or it == cfg["nA"] - 1:
                # print(f"[L{li} A] it={it:04d}/{cfg['nA']} loss={loss.item():.6f} "
                #       f"img={loss_img.item():.6f} bend={L_bend.item():.2e} out={L_out.item():.2e} "
                #       f"mind={comp['mind']:.4f} orient={comp['orient']:.4f}")
                print(f"[L{li} A] it={it:04d}/{cfg['nA']} loss={loss.item():.6f} "
                      f"img={loss_img.item():.6f} bend={L_bend.item():.2e} out={L_out.item():.2e} "
                      f"ssim={comp['ssim']:.4f} ngf={comp['ngf']:.4f}")
 
        # freeze warped hist ROI for B
        with torch.no_grad():
            # disp_det = phi(uv_norm_pix, H=H, W=W) * roi_gate2d.unsqueeze(-1)
            disp_det = phi(uv_norm_pix, H=H, W=W)
            grid2d_det = grid_id_2d + disp_det.unsqueeze(0)
            hist_phi = F.grid_sample(hist_t, grid2d_det, mode="bilinear", padding_mode="zeros", align_corners=True).detach()
            hist_mask_phi = F.grid_sample(hist_mask_t, grid2d_det, mode="nearest",padding_mode="zeros", align_corners=True).detach()  

        # -------- B: optimize alpha, phi frozen --------
        opt_alpha = torch.optim.Adam([{"params": alpha.parameters(), "lr": cfg["lrB"]}])
        shape_gate_B = None

        for it in range(cfg["nB"]):
            a_raw = alpha(uv_norm_pix, H=H, W=W)
            xyz_theta = xyz_norm_base + a_raw * n0
            mri2d_theta, valid_theta = sample_mri_on_sheet_with_mask(mri_t, mri_mask_t, xyz_theta, pad_mode="zeros")

            if (shape_gate_B is None) or (it % shape_update_every == 0):
                with torch.no_grad():
                    shape_gate_B = shape_constrained_final_mask(
                        hist_mask_phi, valid_theta.detach(), **shape_params
                    ).to(device=mri2d_theta.device, dtype=mri2d_theta.dtype)
                    if shape_soften_ks > 1:
                        shape_gate_B = F.avg_pool2d(
                            shape_gate_B, kernel_size=shape_soften_ks, stride=1, padding=shape_soften_ks // 2
                        ).clamp(0, 1)            

            # mask = hist_mask_phi
            # loss_img, comp = combined_masked_mind_orient(
            #     mri2d_theta, hist_phi, mask,
            #     w_mind=0.7, w_orient=0.3,
            #     mind_radius=1, mind_dilation=1, mind_sigma=0.8
            # )
            loss_img, comp = combined_ssim_ngf_loss(
                mri2d_theta, hist_phi, w_ssim=0.5, w_ngf=0.5
            )
            # loss_img, comp = combined_masked_ssim_ngf_loss(
            #     mri2d_theta, hist_phi, mask, w_ssim=0.5, w_ngf=0.5
            # )

            L_bend = bending_energy(alpha.ctrl)
            L_sheet = sheet_inside_volume_penalty(xyz_theta)
            L_out = (a_raw[...,0].pow(2) * (1.0 - hist_mask_phi[0,0])).mean()

            # magnitude penalty in ROI (soft constraint)
            lam_z = 5e-4   # tune: 1e-3, 5e-3, 1e-2
            L_z = (a_raw[..., 0].pow(2) * hist_mask_phi[0, 0]).mean()

            # gradient penalty to avoid local spikes in z offset
            lam_z_smooth = 1e-3
            az = a_raw[..., 0]
            dzx = az[:, 1:] - az[:, :-1]
            dzy = az[1:, :] - az[:-1, :]
            L_z_smooth = dzx.pow(2).mean() + dzy.pow(2).mean()

            # loss = loss_img + cfg["lamB"] * L_bend + cfg["lamSheet"] * L_sheet + cfg["lamB_out"] * L_out
            loss = loss_img + cfg["lamB"] * L_bend + cfg["lamSheet"] * L_sheet
            # loss = (
            #     loss_img
            #     + cfg["lamB"] * L_bend
            #     + cfg["lamSheet"] * L_sheet
            #     + cfg["lamB_out"] * L_out
            #     + lam_z * L_z
            #     + lam_z_smooth * L_z_smooth
            # )

            opt_alpha.zero_grad()
            loss.backward()
            opt_alpha.step()

            if it % 50 == 0 or it == cfg["nB"] - 1:
                # print(f"[L{li} B] it={it:04d}/{cfg['nB']} loss={loss.item():.6f} "
                #       f"img={loss_img.item():.6f} bend={L_bend.item():.2e} sheet={L_sheet.item():.2e} "
                #       f"out={L_out.item():.2e} mind={comp['mind']:.4f} orient={comp['orient']:.4f}")
                print(f"[L{li} B] it={it:04d}/{cfg['nB']} loss={loss.item():.6f} "
                      f"img={loss_img.item():.6f} bend={L_bend.item():.2e} sheet={L_sheet.item():.2e} "
                      f"out={L_out.item():.2e} ssim={comp['ssim']:.4f} ngf={comp['ngf']:.4f}")
            
            if PLOT_SHEET and (it % DEBUG_EVERY == 0 or it == cfg["nB"] - 1):
                with torch.no_grad():
                    a_vis = alpha(uv_norm_pix, H=H, W=W)
                    xyz_vis = xyz_norm_base + a_vis * n0
                    _plot_sheet_move(xyz_vis, xyz_norm_base, f"L{li} B it={it}", Xdim, Ydim, Zdim, stride=PLOT_STRIDE)


    with torch.no_grad():
        # same gating as training
        disp_f = phi(uv_norm_pix, H=H, W=W)
        grid2d_f = grid_id_2d + disp_f.unsqueeze(0)

        hist_f = F.grid_sample(
            hist_t, grid2d_f, mode="bilinear",
            padding_mode="zeros", align_corners=True
        )
        hist_mask_f = F.grid_sample(
            hist_mask_t, grid2d_f, mode="nearest",
            padding_mode="zeros", align_corners=True
        )

        a_f = alpha(uv_norm_pix, H=H, W=W)
        xyz_f = xyz_norm_base + a_f * n0  # [H,W,3] in [0,1]

        # sample MRI with same sampler used for final output
        mri_src_t = mri_t
        Zr, Yr, Xr = mri_src_t.shape[-3:]
        P_vox_xyz = torch.stack([
            xyz_f[..., 0] * (Xr - 1),
            xyz_f[..., 1] * (Yr - 1),
            xyz_f[..., 2] * (Zr - 1),
        ], dim=-1)
        
        mri_f = sample_mri_cubic_3d(
            mri_src_t, P_vox_xyz,
            kernel="catmull-rom",
            padding_mode="zeros",
        )



        # build valid_soft on final sheet
        # xg_f = 2 * xyz_f[..., 0] - 1.0
        # yg_f = 2 * xyz_f[..., 1] - 1.0
        # zg_f = 2 * xyz_f[..., 2] - 1.0
        # grid3d_f = torch.stack([xg_f, yg_f, zg_f], dim=-1).unsqueeze(0).unsqueeze(1)

        # valid_soft_f = F.grid_sample(
        #     mri_mask_t, grid3d_f, mode="bilinear",
        #     padding_mode="zeros", align_corners=True
        # )[:, :, 0, :, :].clamp(0, 1)

        # final_mask = shape_constrained_final_mask(
        #     hist_mask_w, valid_soft_f, **shape_params
        # ).to(device=hist_f.device, dtype=hist_f.dtype)

        mri_f = torch.where(hist_mask_f > 0.5, mri_f, torch.zeros_like(mri_f))
        # hist_f = torch.where(final_mask > 0.5, hist_f, torch.zeros_like(hist_f))
        hist_warped_final = hist_f[0, 0].cpu().numpy()
        mri_on_final_sheet = mri_f[0,0].cpu().numpy()
        valid_mask = hist_mask_f[0, 0].cpu().numpy()

        # ---------------- DEBUG: after pyramid ----------------
        DEBUG_AFTER_PYRAMID = True
        if DEBUG_AFTER_PYRAMID:
            with torch.no_grad():
                def _norm01(a, eps=1e-8):
                    a = a.astype(np.float32)
                    return (a - a.min()) / (a.max() - a.min() + eps)

                def _checker(a, b, tile=24):
                    a = _norm01(a); b = _norm01(b)
                    yy, xx = np.indices(a.shape)
                    m = ((yy // tile + xx // tile) % 2).astype(bool)
                    return np.where(m, a, b)

                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                ax[0].imshow(hist_warped_final, cmap="gray"); ax[0].set_title("Histology on final sheet")
                ax[1].imshow(mri_on_final_sheet, cmap="gray"); ax[1].set_title("MRI on final sheet")
                ax[2].imshow(valid_mask, cmap="gray"); ax[2].set_title("Valid mask")
                ax[3].imshow(_checker(hist_warped_final, mri_on_final_sheet, tile=24), cmap="gray"); ax[3].set_title("Checkerboard")
                for a in ax:
                    a.axis("on")
                plt.tight_layout()
                plt.show()
        # ---------------- end DEBUG block ----------------



    return {
        "hist_warped_final": hist_warped_final,
        "mri_on_final_sheet": mri_on_final_sheet,
        "phi": phi,
        "alpha": alpha,
        "roi_soft": roi_soft.detach().cpu().numpy(),    # [1,1,H,W]
        "uv_norm_pix": uv_norm_pix,
        "grid_id_2d": grid_id_2d,
        "xyz_norm_base": xyz_norm_base,
        "n0": n0,
        "valid_mask": valid_mask,
    }
