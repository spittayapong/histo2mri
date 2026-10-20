
import torch
import torch.nn as nn


from typing import Tuple

# Median Absolute Deviation (MAD) -> robust to outliers

# ---- find patch correlation ---------------------------------------------------------


# ---- kernel caches ----
_gauss_cache = {}
_sobel_cache = {}


# ---- initial in-plane alignment ---------------------------------------------------------

# ---- 0. Helpers ----

# ---- Init curved sheet on top of G ---------------------------------------------------------
# ============== utilities ==============


def _tps_kernel_r2logr(r2, eps=1e-9):
    """
    TPS kernel in terms of squared distance r^2:
        U(r) = r^2 * log(r), with U(0) = 0.
    We pass r^2 in for numerical stability.
    """
    r = torch.clamp(r2, min=eps).sqrt()
    U = r2 * torch.log(r)
    # ensure exact 0 at r=0
    U = torch.where(r2 > 0, U, torch.zeros_like(U))
    return U


@torch.no_grad()
def _tps_residuals(tps, uv_all, xyz_all):
    """Compute residual norms for all points."""
    pred = tps(uv_all)                     # (N,3)
    return (xyz_all - pred).norm(dim=1)    # (N,)


# ============== 2) ThinPlateSplineSurface3D (PyTorch) ==============

class ThinPlateSplineSurface3D(nn.Module):
    """
    Vector-valued TPS surface S(u,v) = [x(u,v), y(u,v), z(u,v)],
    fit from (u,v) parameter points to (x,y,z) targets.

    - fit_from_uv_xyz(ctrl_uv, targets_xyz, ...)  -> solves TPS system
    - forward(uv_query)                           -> (M,3) xyz
    - predict(U, V)                               -> (X,Y,Z) grids
    - normals(U, V)                               -> (3, *shape) unit normals

    All coordinates here are in *voxel* units (your pipeline is voxel-based).
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

        # PCA info placeholder (not used here, but kept for completeness)
        self.register_buffer("pca_origin", None, persistent=False)  # (3,)
        self.register_buffer("pca_axes", None, persistent=False)    # (3,3)

    # ---------- Fitting from (u,v) and (x,y,z) ----------

    def fit_from_uv_xyz(self,
                        ctrl_uv,
                        targets_xyz,
                        lam: float = 0.0,
                        normalize_uv: bool = False):
        """
        Fit TPS from explicit parameter points and 3D targets.
        Robust to duplicate uv / near-singular systems.
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

        # no PCA here; uv are already parameter coords
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

    # ---------- Internal helpers ----------

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

    # ---------- Forward & predict ----------

    def forward(self, uv_query: torch.Tensor) -> torch.Tensor:
        """
        uv_query : (M,2) tensor of (u,v) in same units as ctrl_uv.
        Returns  : (M,3) xyz in voxel coordinates.
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
        """
        Evaluate S(U,V) -> (X,Y,Z) on a grid or broadcasted arrays.
        """
        X = self._tps_eval_scalar(U, V, self.w_x, self.a_x)
        Y = self._tps_eval_scalar(U, V, self.w_y, self.a_y)
        Z = self._tps_eval_scalar(U, V, self.w_z, self.a_z)
        return X, Y, Z

    # ---------- Derivatives & normals ----------

    def partials(self, U, V):
        """
        First derivatives of S wrt (U,V): returns (S_u, S_v), each (3, *shape).
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
        """
        Surface unit normals at (U,V). Returns (3, *shape).
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

# ============== 1) Get (uv, xyz) from matches + G ==============


@torch.no_grad()
def _fit_tps_on_subset(uv, xyz, idx, device, dtype, lam, normalize_uv=False):
    """Fit TPS on subset indices idx and return fitted module."""
    tps = ThinPlateSplineSurface3D(device=device, dtype=dtype)
    tps.fit_from_uv_xyz(
        ctrl_uv=uv[idx],
        targets_xyz=xyz[idx],
        lam=lam,
        normalize_uv=normalize_uv,
    )
    return tps
