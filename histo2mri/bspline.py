"""Cubic B-spline control-grid fields and penalties used by the curved-sheet refinement.

All fields are evaluated at normalised canvas coordinates (u, v) in [0, 1], with u along
the canvas columns (x) and v along the rows (y). Control grids are stored as
(grid_h, grid_w, C) tensors indexed [row (v), column (u)]. Three parameterisations are
provided: a 3D residual of the surface in normalised MRI coordinates (xyz_norm), a 2D
free-form vector field in grid_sample units, and a scalar offset along the surface normal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bspline_basis_1d(t, grid_size):
    """Return the first control index and the 4 cubic B-spline weights for each t.

    Uses a uniform (unclamped) cubic B-spline: t in [0, 1] is mapped to
    u = t * (grid_size - 3), so the grid_size control points span grid_size - 3 segments
    and the end control points are not interpolated. The segment index is clamped to the
    valid range, so t outside [0, 1] extrapolates the end segment polynomially.

    Args:
        t: tensor of shape (...), normalised parameter in [0, 1].
        grid_size: number of control points along this axis (>= 4).

    Returns:
        Tuple (idx0, w): idx0 is a long tensor of shape (...) with the first of the 4
        control indices idx0..idx0+3; w is a tensor of shape (..., 4) with their weights.
    """
    # t in [0, 1] -> u in [0, grid_size - 3] (uniform cubic B-spline)
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
    """3D residual surface dS(u, v) = sum_ij B_i(u) B_j(v) ctrl[j, i, :].

    Cubic B-spline field over normalised canvas coordinates (u, v) in [0, 1] whose value
    is a 3D displacement in normalised MRI coordinates (each of x, y, z divided by its
    size - 1, the same units as xyz_norm). The refinement fits it to the match residuals
    of the initial TPS surface and adds it to that surface. Control points start at zero.

    Args:
        grid_h: number of control points along v (rows), >= 4.
        grid_w: number of control points along u (columns), >= 4.
        device: torch device of the control grid.

    Attributes:
        ctrl: trainable (grid_h, grid_w, 3) control-point displacements.

    forward(uv_norm, H=None, W=None):
        uv_norm: (N, 2) normalised (u, v) if H is None, otherwise (..., 2) with H*W
        points (typically (H, W, 2)). Returns (N, 3), or (H, W, 3) when H and W are given.
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
    """Return the mean squared second difference of a control grid along rows and columns.

    Discrete smoothness penalty without the mixed (cross) derivative term. Used for the
    3D residual surface and the scalar normal-offset fields.

    Args:
        ctrl: (grid_h, grid_w, C) control grid, grid_h and grid_w >= 3.

    Returns:
        Scalar tensor: mean of squared second differences along axis 0 plus that along
        axis 1, over interior control points.
    """
    dx2 = ctrl[2:, 1:-1] - 2*ctrl[1:-1, 1:-1] + ctrl[:-2, 1:-1]
    dy2 = ctrl[1:-1, 2:] - 2*ctrl[1:-1, 1:-1] + ctrl[1:-1, :-2]
    return (dx2.pow(2).mean() + dy2.pow(2).mean())


def sheet_inside_volume_penalty(xyz_norm_total):
    """Penalise surface points that leave the MRI volume.

    Squared hinge on each normalised coordinate outside [0, 1], averaged over all elements.

    Args:
        xyz_norm_total: tensor (..., 3) of surface positions in normalised MRI coordinates.

    Returns:
        Scalar tensor; zero when every coordinate lies in [0, 1].
    """
    below = F.relu(-xyz_norm_total)
    above = F.relu(xyz_norm_total - 1)
    return (below**2 + above**2).mean()


class BSplineFFD2D(nn.Module):
    """2D B-spline free-form vector field f(u, v) = sum_ij B_i(u) B_j(v) ctrl[j, i, :].

    Cubic B-spline field over normalised canvas coordinates (u, v) in [0, 1] whose value
    is a 2D vector (x, y) in grid_sample units, where the full canvas width or height
    spans [-1, 1]. The refinement uses it as a stationary velocity field for in-plane
    resampling of the surface and integrates it by scaling and squaring. Control points
    start at zero (identity).

    Args:
        grid_h: number of control points along v (rows), >= 4.
        grid_w: number of control points along u (columns), >= 4.
        device: torch device of the control grid.

    Attributes:
        ctrl: trainable (grid_h, grid_w, 2) control-point vectors.

    forward(uv_norm_pix, H, W):
        uv_norm_pix: (..., 2) normalised (u, v) = (x_norm, y_norm), typically (H, W, 2).
        H, W are accepted but not used. Returns (..., 2) vectors in grid_sample units.
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


class ScalarNormalOffsetField(nn.Module):
    """Scalar B-spline field alpha(u, v) = sum_ij B_i(u) B_j(v) ctrl[j, i, 0].

    Cubic B-spline field over normalised canvas coordinates (u, v) in [0, 1]. The
    refinement multiplies it by the unit surface normal to move the surface out of plane,
    so alpha is in normalised MRI coordinates (the units of xyz_norm). Control points start
    at zero.

    Args:
        grid_h: number of control points along v (rows), >= 4.
        grid_w: number of control points along u (columns), >= 4.
        device: torch device of the control grid.

    Attributes:
        ctrl: trainable (grid_h, grid_w, 1) control-point offsets.

    forward(uv_norm_pix, H, W):
        uv_norm_pix: (..., 2) normalised (u, v), typically (H, W, 2). H, W are accepted
        but not used. Returns (..., 1) offsets.
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


def _upsample_ctrl(ctrl, out_h, out_w, mode="bilinear"):
    """Resize a (gh, gw, C) control grid to (out_h, out_w, C) with F.interpolate (align_corners).

    Warm start for the next pyramid level; approximate, not exact B-spline knot refinement.
    """
    x = ctrl.permute(2, 0, 1).unsqueeze(0)  # (1,C,gh,gw)
    x = F.interpolate(x, size=(out_h, out_w), mode=mode, align_corners=True)
    return x.squeeze(0).permute(1, 2, 0).contiguous()


def _copy_ctrl(dst_param, src_tensor):
    """Copy src_tensor into the parameter dst_param in place, outside autograd."""
    with torch.no_grad():
        dst_param.copy_(src_tensor)

