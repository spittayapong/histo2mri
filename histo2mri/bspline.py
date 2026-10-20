"""B-spline surfaces, free-form deformations and smoothness penalties used by the refinement."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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

