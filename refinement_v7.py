import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


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


# -----------------------------
# MIND-SSC (2D) implementation
# -----------------------------

# ---------- SSIM + NGF losses ----------

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

    return warped2d_masked

# Multi-Resolution Pyramid
# ---------- helper: upsample B-spline ctrl grids ----------
def _upsample_ctrl(ctrl, out_h, out_w, mode="bilinear"):
    """
    ctrl: (gh, gw, C)
    returns: (out_h, out_w, C)
    """
    x = ctrl.permute(2, 0, 1).unsqueeze(0)  # (1,C,gh,gw)
    x = F.interpolate(x, size=(out_h, out_w), mode=mode, align_corners=True)
    return x.squeeze(0).permute(1, 2, 0).contiguous()


def _copy_ctrl(dst_param, src_tensor):
    with torch.no_grad():
        dst_param.copy_(src_tensor)


# Main Refinement Function

# ---------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------


# ---------------------------------------------------------
# Masked losses (MIND + orientation)
# ---------------------------------------------------------

# ---------------------------------------------------------
# ROI-guided refinement
# ---------------------------------------------------------
