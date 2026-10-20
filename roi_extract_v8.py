import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
import torch
from tqdm import tqdm
import inspect
import math
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import os


# HOG + MIND late-fusion patch matcher (GPU)
def _to_numpy_image(x):
    if sitk is not None and isinstance(x, sitk.Image):
        arr = sitk.GetArrayFromImage(x)
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _prep_inputs(proc_Ih, histo_mask, slab_Im, slab_Mm):
    h = _to_numpy_image(proc_Ih)
    hm = _to_numpy_image(histo_mask)
    m = _to_numpy_image(slab_Im)
    mm = _to_numpy_image(slab_Mm)

    if h.ndim == 3: h = h[0]
    if hm.ndim == 3: hm = hm[0]
    if m.ndim == 2: m = m[None, ...]
    if mm.ndim == 2: mm = mm[None, ...]

    return h.astype(np.float32), (hm > 0), m.astype(np.float32), (mm > 0)


def _sobel_xy(img_hw, device):
    x = torch.as_tensor(img_hw, device=device, dtype=torch.float32)[None, None]
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=torch.float32).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=torch.float32).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)[0, 0]
    gy = F.conv2d(x, ky, padding=1)[0, 0]
    return gx, gy


def _unfold_hw(t_hw, patch, stride):
    return F.unfold(t_hw[None, None], kernel_size=(patch, patch), stride=(stride, stride)).squeeze(0).T


@torch.no_grad()
def _mind_ssc_2d(img, radius=2, dilation=1, eps=1e-8):
    # img: [B,1,H,W] -> [B,8,H,W]
    B, C, H, W = img.shape
    offsets = torch.tensor(
        [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]],
        device=img.device, dtype=torch.int64
    ) * int(dilation)

    pad = int(radius) + int(dilation)
    img_pad = F.pad(img, (pad, pad, pad, pad), mode="reflect")

    ssd_list = []
    for k in range(offsets.shape[0]):
        dy = int(offsets[k, 0].item())
        dx = int(offsets[k, 1].item())
        shifted = img_pad[:, :, pad+dy:pad+dy+H, pad+dx:pad+dx+W]
        diff2 = (img - shifted) ** 2
        ssd = F.avg_pool2d(diff2, kernel_size=2*radius+1, stride=1, padding=radius)
        ssd_list.append(ssd)

    ssd = torch.cat(ssd_list, dim=1)
    var = ssd.mean(dim=1, keepdim=True).clamp_min(eps)
    desc = torch.exp(-ssd / var)
    desc = F.normalize(desc, p=2, dim=1)
    return desc


# --- keep for matcher ---
def _norm01_torch(x, eps=1e-8):
    x = x.to(torch.float32)
    m = torch.isfinite(x)
    out = torch.zeros_like(x)
    if m.sum() <= 1:
        return out
    mn = x[m].min()
    mx = x[m].max()
    out[m] = (x[m] - mn) / (mx - mn + eps)
    return out


def _hog_from_unfolded(gx_u, gy_u, mk_u, patch=64, cell_size=8, bins=18, eps=1e-6, signed=True):
    assert patch % cell_size == 0
    N, P = gx_u.shape
    n_cells = patch // cell_size
    Fdim = n_cells * n_cells * bins
    device = gx_u.device
    dtype = gx_u.dtype

    mag = torch.sqrt(gx_u * gx_u + gy_u * gy_u + eps) * mk_u.to(dtype)

    ang = torch.atan2(gy_u, gx_u)  # [-pi, pi]
    if signed:
        ang = torch.remainder(ang + 2 * np.pi, 2 * np.pi)  # [0, 2pi)
        bf = ang * (bins / (2 * np.pi))
    else:
        ang = torch.remainder(ang, np.pi)  # [0, pi)
        bf = ang * (bins / np.pi)

    b0 = torch.floor(bf).long() % bins
    w1 = bf - torch.floor(bf)
    w0 = 1.0 - w1
    b1 = (b0 + 1) % bins

    yy, xx = torch.meshgrid(torch.arange(patch, device=device), torch.arange(patch, device=device), indexing="ij")
    cell_idx = ((yy // cell_size) * n_cells + (xx // cell_size)).reshape(-1).long()

    idx0 = cell_idx[None, :] * bins + b0
    idx1 = cell_idx[None, :] * bins + b1

    desc = torch.zeros((N, Fdim), device=device, dtype=dtype)
    desc.scatter_add_(1, idx0, mag * w0)
    desc.scatter_add_(1, idx1, mag * w1)
    return F.normalize(desc, p=2, dim=1, eps=eps)

@torch.no_grad()
def _mind_desc_from_patch_batch(patches_bpp, masks_bp, radius=2, dilation=1, eps=1e-8):
    # patches_bpp: [B,p,p], masks_bp: [B,P] bool
    B, p, _ = patches_bpp.shape
    x = patches_bpp[:, None, :, :]  # [B,1,p,p]
    d = _mind_ssc_2d(x, radius=radius, dilation=dilation, eps=eps)  # [B,D,p,p]
    D = d.shape[1]
    d = d.reshape(B, D, p * p)
    d = d * masks_bp[:, None, :].to(d.dtype)
    d = d.reshape(B, -1)
    d = F.normalize(d, p=2, dim=1, eps=eps)
    return d


# @torch.no_grad()
# def run_patch_matching_hog_mind_fusion_gpu(
#     proc_Ih,
#     histo_mask,
#     slab_Im,
#     slab_Mm,
#     hist_patch=64,
#     hist_stride=32,
#     mri_patch=64,
#     mri_stride=4,
#     cell_size=8,
#     bins=18,
#     min_histo_mask_frac=0.5,
#     min_mri_mask_frac=0.5,
#     min_overlap_frac=0.25,
#     topk_hog=64,
#     w_hog=0.6,
#     w_mind=0.4,
#     mind_radius=2,
#     mind_dilation=1,
#     hist_batch=64,
#     mri_chunk=4096,
#     z_min=None,
#     z_max=None,
#     device="cuda:0"
# ):
#     assert hist_patch == mri_patch, "This function assumes same patch size for hist and MRI."
#     patch = hist_patch
#     P = patch * patch
#     min_overlap = int(P * float(min_overlap_frac))

#     # dev = torch.device(device if ("cuda" in str(device) and torch.cuda.is_available()) else "cpu")
#     dev=device

#     hist_img, hist_msk, mri_img, mri_msk = _prep_inputs(proc_Ih, histo_mask, slab_Im, slab_Mm)

#     # -------- Hist bank --------
#     h_gx, h_gy = _sobel_xy(hist_img, dev)
#     h_im = torch.as_tensor(hist_img, device=dev, dtype=torch.float32)
#     h_mk = torch.as_tensor(hist_msk.astype(np.float32), device=dev)

#     h_gx_u = _unfold_hw(h_gx, patch, hist_stride)
#     h_gy_u = _unfold_hw(h_gy, patch, hist_stride)
#     h_im_u = _unfold_hw(h_im, patch, hist_stride)          # [Nh_all,P]
#     h_mk_u = _unfold_hw(h_mk, patch, hist_stride) > 0.5    # [Nh_all,P] bool

#     keep_h = (h_mk_u.float().mean(dim=1) >= float(min_histo_mask_frac))
#     if not torch.any(keep_h):
#         raise RuntimeError("No histology patches survived mask filtering.")

#     Hh, Wh = hist_img.shape
#     ny_h = (Hh - patch) // hist_stride + 1
#     nx_h = (Wh - patch) // hist_stride + 1
#     yy_h, xx_h = torch.meshgrid(torch.arange(ny_h, device=dev), torch.arange(nx_h, device=dev), indexing="ij")
#     h_y0 = (yy_h * hist_stride).reshape(-1)
#     h_x0 = (xx_h * hist_stride).reshape(-1)
#     h_yc = h_y0 + patch // 2
#     h_xc = h_x0 + patch // 2

#     h_gx_u = h_gx_u[keep_h]
#     h_gy_u = h_gy_u[keep_h]
#     h_im_u = h_im_u[keep_h]
#     h_mk_u = h_mk_u[keep_h]
#     hist_center = torch.stack([h_yc[keep_h], h_xc[keep_h]], dim=1).to(torch.int32)
#     hist_topleft = torch.stack([h_y0[keep_h], h_x0[keep_h]], dim=1).to(torch.int32)

#     h_hog = _hog_from_unfolded(h_gx_u, h_gy_u, h_mk_u, patch=patch, cell_size=cell_size, bins=bins)  # [Nh,Fh]
#     h_patches = h_im_u.reshape(-1, patch, patch)  # [Nh,p,p]
#     h_mind = _mind_desc_from_patch_batch(h_patches, h_mk_u, radius=mind_radius, dilation=mind_dilation)  # [Nh,Fm]
#     Nh = h_hog.shape[0]

#     # -------- MRI bank (HOG + coords + masks + raw patchs for MIND) --------
#     m_hog_list, m_mk_list, m_top_list, m_center_list, m_patch_list = [], [], [], [], []
#     Z = mri_img.shape[0]

#     for z in tqdm(range(Z), desc="Build MRI bank"):
#         if z_min is not None and z < z_min:
#             continue
#         if z_max is not None and z > z_max:
#             continue

#         m_gx, m_gy = _sobel_xy(mri_img[z], dev)
#         m_im = torch.as_tensor(mri_img[z], device=dev, dtype=torch.float32)
#         m_m = torch.as_tensor(mri_msk[z].astype(np.float32), device=dev)

#         m_gx_u = _unfold_hw(m_gx, patch, mri_stride)
#         m_gy_u = _unfold_hw(m_gy, patch, mri_stride)
#         m_im_u = _unfold_hw(m_im, patch, mri_stride)
#         m_mk_u = _unfold_hw(m_m, patch, mri_stride) > 0.5

#         Hm, Wm = mri_img[z].shape
#         ny_m = (Hm - patch) // mri_stride + 1
#         nx_m = (Wm - patch) // mri_stride + 1
#         yy_m, xx_m = torch.meshgrid(torch.arange(ny_m, device=dev), torch.arange(nx_m, device=dev), indexing="ij")
#         m_y0 = (yy_m * mri_stride).reshape(-1)
#         m_x0 = (xx_m * mri_stride).reshape(-1)
#         m_yc = m_y0 + patch // 2
#         m_xc = m_x0 + patch // 2

#         center_ok = (m_m[m_yc, m_xc] > 0.5)
#         frac_ok = (m_mk_u.float().mean(dim=1) >= float(min_mri_mask_frac))
#         keep_m = center_ok & frac_ok
#         if not torch.any(keep_m):
#             continue

#         gxk = m_gx_u[keep_m]
#         gyk = m_gy_u[keep_m]
#         mkk = m_mk_u[keep_m]
#         pim = m_im_u[keep_m].reshape(-1, patch, patch)

#         hogk = _hog_from_unfolded(gxk, gyk, mkk, patch=patch, cell_size=cell_size, bins=bins).half().cpu()
#         m_hog_list.append(hogk)
#         m_mk_list.append(mkk.cpu())
#         m_patch_list.append(pim.cpu())

#         n = int(keep_m.sum().item())
#         zcol = torch.full((n, 1), z, dtype=torch.int32, device=dev).cpu()
#         top = torch.cat([zcol, m_y0[keep_m].to(torch.int32).cpu().unsqueeze(1), m_x0[keep_m].to(torch.int32).cpu().unsqueeze(1)], dim=1)
#         cen = torch.cat([zcol, m_yc[keep_m].to(torch.int32).cpu().unsqueeze(1), m_xc[keep_m].to(torch.int32).cpu().unsqueeze(1)], dim=1)
#         m_top_list.append(top)
#         m_center_list.append(cen)

#     if len(m_hog_list) == 0:
#         raise RuntimeError("No MRI patches survived mask filtering.")

#     m_hog = torch.cat(m_hog_list, dim=0)             # CPU [Nm,Fh] float16
#     m_mk = torch.cat(m_mk_list, dim=0)               # CPU [Nm,P] bool
#     m_patches = torch.cat(m_patch_list, dim=0)       # CPU [Nm,p,p] float32
#     m_top = torch.cat(m_top_list, dim=0).cpu().numpy()     # [Nm,3]
#     m_center = torch.cat(m_center_list, dim=0).cpu().numpy()
#     Nm = m_hog.shape[0]

#     # MIND cache for MRI candidates (index -> desc cpu float16)
#     mind_cache = {}

#     def get_mri_mind_desc(idxs):
#         # idxs: list[int]
#         miss = [i for i in idxs if i not in mind_cache]
#         if len(miss) > 0:
#             batch = torch.stack([m_patches[i] for i in miss], dim=0).to(dev, dtype=torch.float32)  # [B,p,p]
#             masks = torch.stack([m_mk[i] for i in miss], dim=0).to(dev)                              # [B,P] bool
#             desc = _mind_desc_from_patch_batch(batch, masks, radius=mind_radius, dilation=mind_dilation)  # [B,F]
#             desc = desc.half().cpu()
#             for k, idx in enumerate(miss):
#                 mind_cache[int(idx)] = desc[k]
#         return torch.stack([mind_cache[int(i)] for i in idxs], dim=0)  # CPU [K,F] float16

#     # -------- Matching --------
#     best_score = torch.full((Nh,), -float("inf"), device=dev, dtype=torch.float32)
#     best_idx = torch.full((Nh,), -1, device=dev, dtype=torch.long)

#     for hs in tqdm(range(0, Nh, hist_batch), desc="Match HOG+MIND"):
#         he = min(hs + hist_batch, Nh)
#         hh = h_hog[hs:he].to(torch.float32)       # [Hb,Fh] GPU
#         hmasks = h_mk_u[hs:he]                    # [Hb,P] bool GPU

#         for r in range(he - hs):
#             i = hs + r
#             hdesc_hog = hh[r:r+1]                 # [1,Fh]
#             hmask = hmasks[r]                     # [P]
#             hdesc_mind = h_mind[i:i+1].to(torch.float32)

#             # 1) top-k by HOG
#             run_vals = torch.full((topk_hog,), -float("inf"), device=dev, dtype=torch.float32)
#             run_idx = torch.full((topk_hog,), -1, device=dev, dtype=torch.long)

#             for ms in range(0, Nm, mri_chunk):
#                 me = min(ms + mri_chunk, Nm)
#                 md = m_hog[ms:me].to(dev, dtype=torch.float32, non_blocking=True)   # [Mb,Fh]
#                 mm_chunk = m_mk[ms:me].to(dev, non_blocking=True)                    # [Mb,P] bool

#                 overlap = (mm_chunk & hmask.unsqueeze(0)).sum(dim=1)
#                 valid = overlap >= min_overlap

#                 sim = (hdesc_hog @ md.T).squeeze(0)                                  # [Mb]
#                 sim[~valid] = -float("inf")

#                 k = min(topk_hog, me - ms)
#                 cvals, cidx = torch.topk(sim, k=k)
#                 cidx = cidx + ms

#                 all_vals = torch.cat([run_vals, cvals], dim=0)
#                 all_idx = torch.cat([run_idx, cidx], dim=0)
#                 run_vals, pos = torch.topk(all_vals, k=topk_hog)
#                 run_idx = all_idx[pos]

#             sel = run_idx[run_idx >= 0].unique().detach().cpu().tolist()
#             if len(sel) == 0:
#                 continue

#             # 2) rerank selected with HOG + MIND
#             md_sel_hog = m_hog[sel].to(dev, dtype=torch.float32, non_blocking=True)           # [K,Fh]
#             hog_sim = (hdesc_hog @ md_sel_hog.T).squeeze(0)                                    # [K]

#             md_sel_mind = get_mri_mind_desc(sel).to(dev, dtype=torch.float32, non_blocking=True)  # [K,Fm]
#             mind_sim = (hdesc_mind @ md_sel_mind.T).squeeze(0)                                 # [K] cosine

#             hog_n = _norm01_torch(hog_sim)
#             mind_n = _norm01_torch(mind_sim)
#             combo = w_hog * hog_n + w_mind * mind_n

#             b = int(torch.argmax(combo).item())
#             j = int(sel[b])

#             if combo[b] > best_score[i]:
#                 best_score[i] = combo[b]
#                 best_idx[i] = j

#     # -------- Pack --------
#     hc = hist_center.detach().cpu().numpy()
#     ht = hist_topleft.detach().cpu().numpy()
#     bi = best_idx.detach().cpu().numpy()
#     bs = best_score.detach().cpu().numpy()

#     matches = []
#     for i in range(Nh):
#         j = int(bi[i])
#         matches.append({
#             "hist_index": int(i),
#             "hist_center_yx": (int(hc[i, 0]), int(hc[i, 1])),
#             "hist_topleft_yx": (int(ht[i, 0]), int(ht[i, 1])),
#             "mri_center_zyx": tuple(m_center[j].tolist()) if j >= 0 else (-1, -1, -1),
#             "mri_coords": tuple(m_center[j].tolist()) if j >= 0 else (-1, -1, -1),
#             "mri_topleft_zyx": tuple(m_top[j].tolist()) if j >= 0 else (-1, -1, -1),
#             "score": float(bs[i]),
#         })

#     return {
#         "matches": matches,
#         "hist_img": hist_img,
#         "hist_mask": hist_msk,
#         "mri_img": mri_img,
#         "mri_mask": mri_msk,
#         "device_used": str(dev),
#         "metric": "HOG+MIND_fusion",
#     }

# @torch.no_grad()
# def run_patch_matching_hog_mind_fusion_gpu(
#     proc_Ih,
#     histo_mask,
#     slab_Im,
#     slab_Mm,
#     hist_patch=64,
#     hist_stride=32,
#     mri_patch=64,
#     mri_stride=4,
#     cell_size=8,
#     bins=18,
#     min_histo_mask_frac=0.5,
#     min_mri_mask_frac=0.5,
#     min_overlap_frac=0.25,
#     topk_hog=64,
#     w_hog=0.6,
#     w_mind=0.4,
#     mind_radius=2,
#     mind_dilation=1,
#     hist_batch=64,
#     mri_chunk=4096,
#     z_min=None,
#     z_max=None,
#     device="cuda:0",
#     keep_mri_bank_on_gpu=True,
#     mind_compute_batch=512,
# ):
#     assert hist_patch == mri_patch, "This function assumes same patch size for hist and MRI."
#     patch = int(hist_patch)
#     P = patch * patch
#     min_overlap = int(P * float(min_overlap_frac))

#     if isinstance(device, torch.device):
#         dev = device
#     else:
#         dev = torch.device(device)
#     if dev.type == "cuda" and not torch.cuda.is_available():
#         dev = torch.device("cpu")

#     hist_img, hist_msk, mri_img, mri_msk = _prep_inputs(proc_Ih, histo_mask, slab_Im, slab_Mm)

#     # -------- Hist bank --------
#     h_gx, h_gy = _sobel_xy(hist_img, dev)
#     h_im = torch.as_tensor(hist_img, device=dev, dtype=torch.float32)
#     h_mk = torch.as_tensor(hist_msk.astype(np.float32), device=dev)

#     h_gx_u = _unfold_hw(h_gx, patch, hist_stride)
#     h_gy_u = _unfold_hw(h_gy, patch, hist_stride)
#     h_im_u = _unfold_hw(h_im, patch, hist_stride)          # [Nh_all,P]
#     h_mk_u = _unfold_hw(h_mk, patch, hist_stride) > 0.5    # [Nh_all,P] bool

#     keep_h = (h_mk_u.float().mean(dim=1) >= float(min_histo_mask_frac))
#     if not torch.any(keep_h):
#         raise RuntimeError("No histology patches survived mask filtering.")

#     Hh, Wh = hist_img.shape
#     ny_h = (Hh - patch) // hist_stride + 1
#     nx_h = (Wh - patch) // hist_stride + 1
#     yy_h, xx_h = torch.meshgrid(
#         torch.arange(ny_h, device=dev),
#         torch.arange(nx_h, device=dev),
#         indexing="ij"
#     )
#     h_y0 = (yy_h * hist_stride).reshape(-1)
#     h_x0 = (xx_h * hist_stride).reshape(-1)
#     h_yc = h_y0 + patch // 2
#     h_xc = h_x0 + patch // 2

#     h_gx_u = h_gx_u[keep_h]
#     h_gy_u = h_gy_u[keep_h]
#     h_im_u = h_im_u[keep_h]
#     h_mk_u = h_mk_u[keep_h]
#     hist_center = torch.stack([h_yc[keep_h], h_xc[keep_h]], dim=1).to(torch.int32)
#     hist_topleft = torch.stack([h_y0[keep_h], h_x0[keep_h]], dim=1).to(torch.int32)

#     h_hog = _hog_from_unfolded(
#         h_gx_u, h_gy_u, h_mk_u, patch=patch, cell_size=cell_size, bins=bins
#     )  # [Nh,Fh]
#     h_patches = h_im_u.reshape(-1, patch, patch)  # [Nh,p,p]
#     h_mind = _mind_desc_from_patch_batch(
#         h_patches, h_mk_u, radius=mind_radius, dilation=mind_dilation
#     )  # [Nh,Fm]
#     Nh = h_hog.shape[0]

#     # -------- MRI bank --------
#     m_hog_list, m_mk_list, m_top_list, m_center_list, m_patch_list = [], [], [], [], []
#     Z = mri_img.shape[0]

#     for z in tqdm(range(Z), desc="Build MRI bank"):
#         if z_min is not None and z < z_min:
#             continue
#         if z_max is not None and z > z_max:
#             continue

#         m_gx, m_gy = _sobel_xy(mri_img[z], dev)
#         m_im = torch.as_tensor(mri_img[z], device=dev, dtype=torch.float32)
#         m_m = torch.as_tensor(mri_msk[z].astype(np.float32), device=dev)

#         m_gx_u = _unfold_hw(m_gx, patch, mri_stride)
#         m_gy_u = _unfold_hw(m_gy, patch, mri_stride)
#         m_im_u = _unfold_hw(m_im, patch, mri_stride)
#         m_mk_u = _unfold_hw(m_m, patch, mri_stride) > 0.5

#         Hm, Wm = mri_img[z].shape
#         ny_m = (Hm - patch) // mri_stride + 1
#         nx_m = (Wm - patch) // mri_stride + 1
#         yy_m, xx_m = torch.meshgrid(
#             torch.arange(ny_m, device=dev),
#             torch.arange(nx_m, device=dev),
#             indexing="ij"
#         )
#         m_y0 = (yy_m * mri_stride).reshape(-1)
#         m_x0 = (xx_m * mri_stride).reshape(-1)
#         m_yc = m_y0 + patch // 2
#         m_xc = m_x0 + patch // 2

#         center_ok = (m_m[m_yc, m_xc] > 0.5)
#         frac_ok = (m_mk_u.float().mean(dim=1) >= float(min_mri_mask_frac))
#         keep_m = center_ok & frac_ok
#         if not torch.any(keep_m):
#             continue

#         gxk = m_gx_u[keep_m]
#         gyk = m_gy_u[keep_m]
#         mkk = m_mk_u[keep_m]
#         pim = m_im_u[keep_m].reshape(-1, patch, patch)

#         hogk = _hog_from_unfolded(
#             gxk, gyk, mkk, patch=patch, cell_size=cell_size, bins=bins
#         ).half().cpu()

#         m_hog_list.append(hogk)
#         m_mk_list.append(mkk.cpu())
#         m_patch_list.append(pim.cpu())

#         n = int(keep_m.sum().item())
#         zcol = torch.full((n, 1), z, dtype=torch.int32, device=dev).cpu()
#         top = torch.cat(
#             [
#                 zcol,
#                 m_y0[keep_m].to(torch.int32).cpu().unsqueeze(1),
#                 m_x0[keep_m].to(torch.int32).cpu().unsqueeze(1),
#             ],
#             dim=1,
#         )
#         cen = torch.cat(
#             [
#                 zcol,
#                 m_yc[keep_m].to(torch.int32).cpu().unsqueeze(1),
#                 m_xc[keep_m].to(torch.int32).cpu().unsqueeze(1),
#             ],
#             dim=1,
#         )
#         m_top_list.append(top)
#         m_center_list.append(cen)

#     if len(m_hog_list) == 0:
#         raise RuntimeError("No MRI patches survived mask filtering.")

#     m_hog_cpu = torch.cat(m_hog_list, dim=0)       # CPU [Nm,Fh] float16
#     m_mk_cpu = torch.cat(m_mk_list, dim=0)         # CPU [Nm,P] bool
#     m_patches = torch.cat(m_patch_list, dim=0)     # CPU [Nm,p,p] float32
#     m_top = torch.cat(m_top_list, dim=0).cpu().numpy()
#     m_center = torch.cat(m_center_list, dim=0).cpu().numpy()
#     Nm = m_hog_cpu.shape[0]

#     # Try to keep full MRI banks on GPU.
#     bank_on_gpu = False
#     m_hog_bank = m_hog_cpu
#     m_mk_bank = m_mk_cpu
#     if keep_mri_bank_on_gpu and dev.type == "cuda":
#         try:
#             m_hog_bank = m_hog_cpu.to(dev, non_blocking=True)
#             m_mk_bank = m_mk_cpu.to(dev, non_blocking=True)
#             bank_on_gpu = True
#         except RuntimeError:
#             m_hog_bank = m_hog_cpu
#             m_mk_bank = m_mk_cpu
#             torch.cuda.empty_cache()

#     # MIND cache: index -> desc (CPU float16)
#     mind_cache = {}

#     def get_mri_mind_desc(idxs, out_device=dev):
#         if isinstance(idxs, torch.Tensor):
#             idx_list = idxs.detach().cpu().tolist()
#         else:
#             idx_list = [int(i) for i in idxs]

#         miss = [i for i in idx_list if i not in mind_cache]
#         if len(miss) > 0:
#             for s in range(0, len(miss), int(mind_compute_batch)):
#                 miss_chunk = miss[s:s + int(mind_compute_batch)]
#                 idx_cpu = torch.tensor(miss_chunk, dtype=torch.long)

#                 batch = m_patches[idx_cpu].to(dev, dtype=torch.float32, non_blocking=True)
#                 if bank_on_gpu:
#                     idx_gpu = idx_cpu.to(dev, non_blocking=True)
#                     masks = m_mk_bank[idx_gpu]
#                 else:
#                     masks = m_mk_cpu[idx_cpu].to(dev, non_blocking=True)

#                 desc = _mind_desc_from_patch_batch(
#                     batch, masks, radius=mind_radius, dilation=mind_dilation
#                 ).half().cpu()

#                 for k, idx in enumerate(miss_chunk):
#                     mind_cache[int(idx)] = desc[k]

#         out = torch.stack([mind_cache[int(i)] for i in idx_list], dim=0)  # CPU float16
#         if out_device is None:
#             return out
#         return out.to(out_device, dtype=torch.float32, non_blocking=True)

#     # -------- Matching --------
#     best_score = torch.full((Nh,), -float("inf"), device=dev, dtype=torch.float32)
#     best_idx = torch.full((Nh,), -1, device=dev, dtype=torch.long)

#     for hs in tqdm(range(0, Nh, hist_batch), desc="Match HOG+MIND"):
#         he = min(hs + hist_batch, Nh)
#         Hb = he - hs

#         hh = h_hog[hs:he].to(torch.float32)        # [Hb,Fh] dev
#         hmasks = h_mk_u[hs:he]                     # [Hb,P] bool dev
#         hmasks_f = hmasks.to(torch.float32)        # [Hb,P] float dev
#         hmind = h_mind[hs:he].to(torch.float32)    # [Hb,Fm] dev

#         run_vals = torch.full((Hb, topk_hog), -float("inf"), device=dev, dtype=torch.float32)
#         run_idx = torch.full((Hb, topk_hog), -1, device=dev, dtype=torch.long)

#         # Batched HOG search over MRI chunks
#         for ms in range(0, Nm, mri_chunk):
#             me = min(ms + mri_chunk, Nm)
#             Mb = me - ms

#             if bank_on_gpu:
#                 md = m_hog_bank[ms:me].to(torch.float32)     # [Mb,Fh] dev
#                 mm_chunk = m_mk_bank[ms:me]                  # [Mb,P] bool dev
#             else:
#                 md = m_hog_cpu[ms:me].to(dev, dtype=torch.float32, non_blocking=True)
#                 mm_chunk = m_mk_cpu[ms:me].to(dev, non_blocking=True)

#             sim = hh @ md.T                                  # [Hb,Mb]
#             overlap = hmasks_f @ mm_chunk.to(torch.float32).T  # [Hb,Mb]
#             sim[overlap < min_overlap] = -float("inf")

#             k = min(topk_hog, Mb)
#             cvals, cidx = torch.topk(sim, k=k, dim=1)       # [Hb,k]
#             cidx = cidx + ms

#             all_vals = torch.cat([run_vals, cvals], dim=1)
#             all_idx = torch.cat([run_idx, cidx], dim=1)
#             run_vals, pos = torch.topk(all_vals, k=topk_hog, dim=1)
#             run_idx = torch.gather(all_idx, 1, pos)

#         # Rerank with HOG + MIND
#         for r in range(Hb):
#             i = hs + r
#             sel = run_idx[r]
#             sel = sel[sel >= 0]
#             if sel.numel() == 0:
#                 continue
#             sel = torch.unique(sel)

#             hdesc_hog = hh[r:r + 1]
#             hdesc_mind = hmind[r:r + 1]

#             if bank_on_gpu:
#                 md_sel_hog = m_hog_bank[sel].to(torch.float32)
#             else:
#                 sel_cpu = sel.detach().cpu()
#                 md_sel_hog = m_hog_cpu[sel_cpu].to(dev, dtype=torch.float32, non_blocking=True)

#             hog_sim = (hdesc_hog @ md_sel_hog.T).squeeze(0)          # [K]
#             md_sel_mind = get_mri_mind_desc(sel, out_device=dev)     # [K,Fm]
#             mind_sim = (hdesc_mind @ md_sel_mind.T).squeeze(0)       # [K]

#             hog_n = _norm01_torch(hog_sim)
#             mind_n = _norm01_torch(mind_sim)
#             combo = w_hog * hog_n + w_mind * mind_n

#             b = int(torch.argmax(combo).item())
#             j = int(sel[b].item())

#             if combo[b] > best_score[i]:
#                 best_score[i] = combo[b]
#                 best_idx[i] = j

#     # -------- Pack --------
#     hc = hist_center.detach().cpu().numpy()
#     ht = hist_topleft.detach().cpu().numpy()
#     bi = best_idx.detach().cpu().numpy()
#     bs = best_score.detach().cpu().numpy()

#     matches = []
#     for i in range(Nh):
#         j = int(bi[i])
#         matches.append(
#             {
#                 "hist_index": int(i),
#                 "hist_center_yx": (int(hc[i, 0]), int(hc[i, 1])),
#                 "hist_topleft_yx": (int(ht[i, 0]), int(ht[i, 1])),
#                 "mri_center_zyx": tuple(m_center[j].tolist()) if j >= 0 else (-1, -1, -1),
#                 "mri_coords": tuple(m_center[j].tolist()) if j >= 0 else (-1, -1, -1),
#                 "mri_topleft_zyx": tuple(m_top[j].tolist()) if j >= 0 else (-1, -1, -1),
#                 "score": float(bs[i]),
#             }
#         )

#     return {
#         "matches": matches,
#         "hist_img": hist_img,
#         "hist_mask": hist_msk,
#         "mri_img": mri_img,
#         "mri_mask": mri_msk,
#         "device_used": str(dev),
#         "metric": "HOG+MIND_fusion",
#     }

def _masked_minmax_normalize_patch_batch(patches_bpp, masks_bp=None, eps=1e-8):
    # patches_bpp: [B,p,p], masks_bp: [B,P] bool or None
    B, p, _ = patches_bpp.shape
    x = patches_bpp.to(torch.float32).reshape(B, -1)

    if masks_bp is None:
        mn = x.min(dim=1, keepdim=True).values
        mx = x.max(dim=1, keepdim=True).values
        y = (x - mn) / (mx - mn + eps)
        return y.view(B, p, p)

    m = masks_bp.reshape(B, -1).bool()
    finfo = torch.finfo(x.dtype)

    mn = torch.where(m, x, torch.full_like(x, finfo.max)).min(dim=1, keepdim=True).values
    mx = torch.where(m, x, torch.full_like(x, finfo.min)).max(dim=1, keepdim=True).values

    any_valid = m.any(dim=1, keepdim=True)
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    mn = torch.where(any_valid, mn, x_min)
    mx = torch.where(any_valid, mx, x_max)

    y = (x - mn) / (mx - mn + eps)
    y = y * m.to(y.dtype)
    return y.view(B, p, p)


@torch.no_grad()
def _ngf_similarity_from_patch_batch(hist_patch_pp, hist_mask_p, mri_patches_bpp, mri_masks_bp, eps=1e-8):
    # hist_patch_pp: [p,p]
    # hist_mask_p: [P] bool
    # mri_patches_bpp: [B,p,p]
    # mri_masks_bp: [B,P] bool
    B, p, _ = mri_patches_bpp.shape
    dev = mri_patches_bpp.device

    hpatch = hist_patch_pp.to(dev, dtype=torch.float32).reshape(1, p, p)
    hmask = hist_mask_p.reshape(1, -1).bool().to(dev)

    mpatch = mri_patches_bpp.to(dev, dtype=torch.float32)
    mmask = mri_masks_bp.reshape(B, -1).bool().to(dev)

    hnorm = _masked_minmax_normalize_patch_batch(hpatch, hmask, eps=eps)
    mnorm = _masked_minmax_normalize_patch_batch(mpatch, mmask, eps=eps)

    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        device=dev, dtype=torch.float32
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        device=dev, dtype=torch.float32
    ).view(1, 1, 3, 3)

    grad_Ix = F.conv2d(hnorm[:, None], kx, padding=1)   # [1,1,p,p]
    grad_Iy = F.conv2d(hnorm[:, None], ky, padding=1)
    grad_Jx = F.conv2d(mnorm[:, None], kx, padding=1)   # [B,1,p,p]
    grad_Jy = F.conv2d(mnorm[:, None], ky, padding=1)

    valid = (hmask.expand(B, -1) & mmask).view(B, 1, p, p).to(torch.float32)

    dot = grad_Ix.expand(B, -1, -1, -1) * grad_Jx + grad_Iy.expand(B, -1, -1, -1) * grad_Jy
    normI = grad_Ix.pow(2) + grad_Iy.pow(2)
    normJ = grad_Jx.pow(2) + grad_Jy.pow(2)

    ngf_map = (dot.pow(2)) / (normI.expand(B, -1, -1, -1) * normJ + eps)

    denom = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return (ngf_map * valid).sum(dim=(1, 2, 3)) / denom

@torch.no_grad()
def build_mri_bank_hog_mind_ngf(
    slab_Im,
    slab_Mm,
    mri_patch,
    mri_stride,
    cell_size,
    bins,
    min_mri_mask_frac,
    z_min,
    z_max,
    device,
    keep_mri_bank_on_gpu=True,
):
    patch = int(mri_patch)

    if isinstance(device, torch.device):
        dev = device
    else:
        dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    mri_img = _to_numpy_image(slab_Im).astype(np.float32)
    mri_msk = (_to_numpy_image(slab_Mm) > 0)

    if mri_img.ndim == 2:
        mri_img = mri_img[None, ...]
    if mri_msk.ndim == 2:
        mri_msk = mri_msk[None, ...]

    # -------- MRI bank --------
    m_hog_list, m_mk_list, m_top_list, m_center_list, m_patch_list = [], [], [], [], []
    Z = mri_img.shape[0]

    for z in tqdm(range(Z), desc="Build MRI bank"):
        if z_min is not None and z < z_min:
            continue
        if z_max is not None and z > z_max:
            continue

        m_gx, m_gy = _sobel_xy(mri_img[z], dev)
        m_im = torch.as_tensor(mri_img[z], device=dev, dtype=torch.float32)
        m_m = torch.as_tensor(mri_msk[z].astype(np.float32), device=dev)

        m_gx_u = _unfold_hw(m_gx, patch, mri_stride)
        m_gy_u = _unfold_hw(m_gy, patch, mri_stride)
        m_im_u = _unfold_hw(m_im, patch, mri_stride)
        m_mk_u = _unfold_hw(m_m, patch, mri_stride) > 0.5

        Hm, Wm = mri_img[z].shape
        ny_m = (Hm - patch) // mri_stride + 1
        nx_m = (Wm - patch) // mri_stride + 1
        yy_m, xx_m = torch.meshgrid(
            torch.arange(ny_m, device=dev),
            torch.arange(nx_m, device=dev),
            indexing="ij"
        )
        m_y0 = (yy_m * mri_stride).reshape(-1)
        m_x0 = (xx_m * mri_stride).reshape(-1)
        m_yc = m_y0 + patch // 2
        m_xc = m_x0 + patch // 2

        center_ok = (m_m[m_yc, m_xc] > 0.5)
        frac_ok = (m_mk_u.float().mean(dim=1) >= float(min_mri_mask_frac))
        keep_m = center_ok & frac_ok
        if not torch.any(keep_m):
            continue

        gxk = m_gx_u[keep_m]
        gyk = m_gy_u[keep_m]
        mkk = m_mk_u[keep_m]
        pim = m_im_u[keep_m].reshape(-1, patch, patch)

        hogk = _hog_from_unfolded(
            gxk, gyk, mkk, patch=patch, cell_size=cell_size, bins=bins
        ).half().cpu()

        m_hog_list.append(hogk)
        m_mk_list.append(mkk.cpu())
        m_patch_list.append(pim.half().cpu())

        n = int(keep_m.sum().item())
        zcol = torch.full((n, 1), z, dtype=torch.int32, device=dev).cpu()
        top = torch.cat(
            [
                zcol,
                m_y0[keep_m].to(torch.int32).cpu().unsqueeze(1),
                m_x0[keep_m].to(torch.int32).cpu().unsqueeze(1),
            ],
            dim=1,
        )
        cen = torch.cat(
            [
                zcol,
                m_yc[keep_m].to(torch.int32).cpu().unsqueeze(1),
                m_xc[keep_m].to(torch.int32).cpu().unsqueeze(1),
            ],
            dim=1,
        )
        m_top_list.append(top)
        m_center_list.append(cen)

    if len(m_hog_list) == 0:
        raise RuntimeError("No MRI patches survived mask filtering.")

    m_hog_cpu = torch.cat(m_hog_list, dim=0)       # CPU [Nm,Fh] float16
    m_mk_cpu = torch.cat(m_mk_list, dim=0)         # CPU [Nm,P] bool
    m_patches = torch.cat(m_patch_list, dim=0)     # CPU [Nm,p,p] float32
    m_top = torch.cat(m_top_list, dim=0).cpu().numpy()
    m_center = torch.cat(m_center_list, dim=0).cpu().numpy()
    Nm = m_hog_cpu.shape[0]

    # Try to keep full MRI banks on GPU.
    bank_on_gpu = False
    m_hog_bank = m_hog_cpu
    m_mk_bank = m_mk_cpu
    if keep_mri_bank_on_gpu and dev.type == "cuda":
        try:
            m_hog_bank = m_hog_cpu.to(dev, non_blocking=True)
            m_mk_bank = m_mk_cpu.to(dev, non_blocking=True)
            bank_on_gpu = True
        except RuntimeError:
            m_hog_bank = m_hog_cpu
            m_mk_bank = m_mk_cpu
            torch.cuda.empty_cache()

    return {
        "mri_img": mri_img,
        "mri_msk": mri_msk,
        "patch": patch,
        "mri_patch": patch,
        "mri_stride": int(mri_stride),
        "cell_size": int(cell_size),
        "bins": int(bins),
        "min_mri_mask_frac": float(min_mri_mask_frac),
        "z_min": z_min,
        "z_max": z_max,
        "device_used": str(dev),
        "m_hog_cpu": m_hog_cpu,
        "m_mk_cpu": m_mk_cpu,
        "m_patches": m_patches,
        "m_top": m_top,
        "m_center": m_center,
        "Nm": Nm,
        "bank_on_gpu": bank_on_gpu,
        "m_hog_bank": m_hog_bank,
        "m_mk_bank": m_mk_bank,
    }


@torch.no_grad()
def _sobel_xy_patch_batch(patches_bpp):
    # patches_bpp: [B,p,p]
    x = patches_bpp[:, None].to(torch.float32)  # [B,1,p,p]
    dev = x.device
    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        device=dev, dtype=torch.float32
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        device=dev, dtype=torch.float32
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)[:, 0]  # [B,p,p]
    gy = F.conv2d(x, ky, padding=1)[:, 0]  # [B,p,p]
    return gx, gy


@torch.no_grad()
def _pairwise_masked_hog_similarity(
    hist_patch_pp,
    mri_patches_bpp,
    pair_masks_bp,
    cell_size=8,
    bins=18,
):
    # hist_patch_pp: [p,p]
    # mri_patches_bpp: [B,p,p]
    # pair_masks_bp: [B,P] bool
    B, p, _ = mri_patches_bpp.shape
    dev = mri_patches_bpp.device

    hbatch = hist_patch_pp.to(dev, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

    h_gx, h_gy = _sobel_xy_patch_batch(hbatch)
    m_gx, m_gy = _sobel_xy_patch_batch(mri_patches_bpp)

    hdesc = _hog_from_unfolded(
        h_gx.reshape(B, -1),
        h_gy.reshape(B, -1),
        pair_masks_bp,
        patch=p,
        cell_size=cell_size,
        bins=bins,
    )
    mdesc = _hog_from_unfolded(
        m_gx.reshape(B, -1),
        m_gy.reshape(B, -1),
        pair_masks_bp,
        patch=p,
        cell_size=cell_size,
        bins=bins,
    )
    return (hdesc * mdesc).sum(dim=1)  # [B]


@torch.no_grad()
def _pairwise_masked_mind_similarity(
    hist_patch_pp,
    mri_patches_bpp,
    pair_masks_bp,
    radius=2,
    dilation=1,
):
    # hist_patch_pp: [p,p]
    # mri_patches_bpp: [B,p,p]
    # pair_masks_bp: [B,P] bool
    B, _, _ = mri_patches_bpp.shape
    dev = mri_patches_bpp.device

    hbatch = hist_patch_pp.to(dev, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

    hdesc = _mind_desc_from_patch_batch(
        hbatch,
        pair_masks_bp,
        radius=radius,
        dilation=dilation,
    )
    mdesc = _mind_desc_from_patch_batch(
        mri_patches_bpp.to(dev, dtype=torch.float32),
        pair_masks_bp,
        radius=radius,
        dilation=dilation,
    )
    return (hdesc * mdesc).sum(dim=1)  # [B]


@torch.no_grad()
def _pairwise_masked_ngf_similarity(
    hist_patch_pp,
    mri_patches_bpp,
    pair_masks_bp,
    eps=1e-8,
):
    # hist_patch_pp: [p,p]
    # mri_patches_bpp: [B,p,p]
    # pair_masks_bp: [B,P] bool
    B, p, _ = mri_patches_bpp.shape
    dev = mri_patches_bpp.device

    hbatch = hist_patch_pp.to(dev, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    masks = pair_masks_bp.to(dev).bool()

    hnorm = _masked_minmax_normalize_patch_batch(hbatch, masks, eps=eps)
    mnorm = _masked_minmax_normalize_patch_batch(mri_patches_bpp.to(dev, dtype=torch.float32), masks, eps=eps)

    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        device=dev, dtype=torch.float32
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        device=dev, dtype=torch.float32
    ).view(1, 1, 3, 3)

    grad_Ix = F.conv2d(hnorm[:, None], kx, padding=1)
    grad_Iy = F.conv2d(hnorm[:, None], ky, padding=1)
    grad_Jx = F.conv2d(mnorm[:, None], kx, padding=1)
    grad_Jy = F.conv2d(mnorm[:, None], ky, padding=1)

    valid = masks.view(B, 1, p, p).to(torch.float32)

    dot = grad_Ix * grad_Jx + grad_Iy * grad_Jy
    normI = grad_Ix.pow(2) + grad_Iy.pow(2)
    normJ = grad_Jx.pow(2) + grad_Jy.pow(2)

    ngf_map = (dot.pow(2)) / (normI * normJ + eps)
    denom = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return (ngf_map * valid).sum(dim=(1, 2, 3)) / denom


@torch.no_grad()
def run_patch_matching_hog_mind_fusion_gpu(
    proc_Ih,
    histo_mask,
    slab_Im,
    slab_Mm,
    hist_patch=64,
    hist_stride=32,
    mri_patch=64,
    mri_stride=4,
    cell_size=8,
    bins=18,
    min_histo_mask_frac=0.5,
    min_mri_mask_frac=0.5,
    min_overlap_frac=0.25,
    topk_hog=64,
    w_hog=0.6,
    w_mind=0.4,
    w_ngf=0.2,
    mind_radius=2,
    mind_dilation=1,
    hist_batch=64,
    mri_chunk=4096,
    z_min=None,
    z_max=None,
    device="cuda:0",
    keep_mri_bank_on_gpu=True,
    mri_bank=None,
):
    assert hist_patch == mri_patch, "This function assumes same patch size for hist and MRI."
    patch = int(hist_patch)
    P = patch * patch
    min_overlap = int(P * float(min_overlap_frac))

    total_w = float(w_hog) + float(w_mind) + float(w_ngf)
    if total_w <= 0:
        raise ValueError("At least one of w_hog, w_mind, w_ngf must be > 0.")

    if isinstance(device, torch.device):
        dev = device
    else:
        dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    hist_img, hist_msk, mri_img, mri_msk = _prep_inputs(proc_Ih, histo_mask, slab_Im, slab_Mm)

    # -------- Hist bank --------
    h_gx, h_gy = _sobel_xy(hist_img, dev)
    h_im = torch.as_tensor(hist_img, device=dev, dtype=torch.float32)
    h_mk = torch.as_tensor(hist_msk.astype(np.float32), device=dev)

    h_gx_u = _unfold_hw(h_gx, patch, hist_stride)
    h_gy_u = _unfold_hw(h_gy, patch, hist_stride)
    h_im_u = _unfold_hw(h_im, patch, hist_stride)          # [Nh_all,P]
    h_mk_u = _unfold_hw(h_mk, patch, hist_stride) > 0.5    # [Nh_all,P] bool

    keep_h = (h_mk_u.float().mean(dim=1) >= float(min_histo_mask_frac))
    if not torch.any(keep_h):
        raise RuntimeError("No histology patches survived mask filtering.")

    Hh, Wh = hist_img.shape
    ny_h = (Hh - patch) // hist_stride + 1
    nx_h = (Wh - patch) // hist_stride + 1
    yy_h, xx_h = torch.meshgrid(
        torch.arange(ny_h, device=dev),
        torch.arange(nx_h, device=dev),
        indexing="ij"
    )
    h_y0 = (yy_h * hist_stride).reshape(-1)
    h_x0 = (xx_h * hist_stride).reshape(-1)
    h_yc = h_y0 + patch // 2
    h_xc = h_x0 + patch // 2

    h_gx_u = h_gx_u[keep_h]
    h_gy_u = h_gy_u[keep_h]
    h_im_u = h_im_u[keep_h]
    h_mk_u = h_mk_u[keep_h]
    hist_center = torch.stack([h_yc[keep_h], h_xc[keep_h]], dim=1).to(torch.int32)
    hist_topleft = torch.stack([h_y0[keep_h], h_x0[keep_h]], dim=1).to(torch.int32)

    h_hog = _hog_from_unfolded(
        h_gx_u, h_gy_u, h_mk_u, patch=patch, cell_size=cell_size, bins=bins
    )  # [Nh,Fh]
    h_patches = h_im_u.reshape(-1, patch, patch)  # [Nh,p,p]
    h_mind = _mind_desc_from_patch_batch(
        h_patches, h_mk_u, radius=mind_radius, dilation=mind_dilation
    )  # [Nh,Fm]
    Nh = h_hog.shape[0]

    if mri_bank is None:
        mri_bank = build_mri_bank_hog_mind_ngf(
            slab_Im=slab_Im,
            slab_Mm=slab_Mm,
            mri_patch=mri_patch,
            mri_stride=mri_stride,
            cell_size=cell_size,
            bins=bins,
            min_mri_mask_frac=min_mri_mask_frac,
            z_min=z_min,
            z_max=z_max,
            device=dev,
            keep_mri_bank_on_gpu=keep_mri_bank_on_gpu,
        )

    mri_img = mri_bank["mri_img"]
    mri_msk = mri_bank["mri_msk"]
    m_hog_cpu = mri_bank["m_hog_cpu"]
    m_mk_cpu = mri_bank["m_mk_cpu"]
    m_patches = mri_bank["m_patches"]
    m_top = mri_bank["m_top"]
    m_center = mri_bank["m_center"]
    Nm = mri_bank["Nm"]
    bank_on_gpu = mri_bank["bank_on_gpu"]
    m_hog_bank = mri_bank["m_hog_bank"]
    m_mk_bank = mri_bank["m_mk_bank"]

    if int(mri_bank["patch"]) != patch:
        raise ValueError(
            f"mri_bank patch={mri_bank['patch']} does not match requested patch={patch}"
        )

    if "mri_stride" in mri_bank and int(mri_bank["mri_stride"]) != int(mri_stride):
        raise ValueError(
            f"mri_bank mri_stride={mri_bank['mri_stride']} does not match requested mri_stride={mri_stride}"
        )

    if "cell_size" in mri_bank and int(mri_bank["cell_size"]) != int(cell_size):
        raise ValueError(
            f"mri_bank cell_size={mri_bank['cell_size']} does not match requested cell_size={cell_size}"
        )

    if "bins" in mri_bank and int(mri_bank["bins"]) != int(bins):
        raise ValueError(
            f"mri_bank bins={mri_bank['bins']} does not match requested bins={bins}"
        )

    # -------- Matching --------
    best_score = torch.full((Nh,), -float("inf"), device=dev, dtype=torch.float32)
    best_idx = torch.full((Nh,), -1, device=dev, dtype=torch.long)

    for hs in tqdm(range(0, Nh, hist_batch), desc="Match HOG+MIND+NGF"):
        he = min(hs + hist_batch, Nh)
        Hb = he - hs

        hh = h_hog[hs:he].to(torch.float32)             # [Hb,Fh] dev
        hmasks = h_mk_u[hs:he]                          # [Hb,P] bool dev
        hmasks_f = hmasks.to(torch.float32)             # [Hb,P] float dev
        hmind = h_mind[hs:he].to(torch.float32)         # [Hb,Fm] dev
        hpatches = h_patches[hs:he].to(torch.float32)   # [Hb,p,p] dev

        run_vals = torch.full((Hb, topk_hog), -float("inf"), device=dev, dtype=torch.float32)
        run_idx = torch.full((Hb, topk_hog), -1, device=dev, dtype=torch.long)

        # Batched HOG search over MRI chunks
        for ms in range(0, Nm, mri_chunk):
            me = min(ms + mri_chunk, Nm)
            Mb = me - ms

            if bank_on_gpu:
                md = m_hog_bank[ms:me].to(torch.float32)     # [Mb,Fh] dev
                mm_chunk = m_mk_bank[ms:me]                  # [Mb,P] bool dev
            else:
                md = m_hog_cpu[ms:me].to(dev, dtype=torch.float32, non_blocking=True)
                mm_chunk = m_mk_cpu[ms:me].to(dev, non_blocking=True)

            sim = hh @ md.T
            overlap = hmasks_f @ mm_chunk.to(torch.float32).T
            sim[overlap < min_overlap] = -float("inf")

            k = min(topk_hog, Mb)
            cvals, cidx = torch.topk(sim, k=k, dim=1)
            cidx = cidx + ms

            all_vals = torch.cat([run_vals, cvals], dim=1)
            all_idx = torch.cat([run_idx, cidx], dim=1)
            run_vals, pos = torch.topk(all_vals, k=topk_hog, dim=1)
            run_idx = torch.gather(all_idx, 1, pos)

        # Rerank with pairwise-intersection-masked HOG + MIND + NGF
        for r in range(Hb):
            i = hs + r
            sel = run_idx[r]
            sel = sel[sel >= 0]
            if sel.numel() == 0:
                continue
            sel = torch.unique(sel)

            hpatch = hpatches[r]     # [p,p]
            hmask = hmasks[r]        # [P] bool
            sel_cpu = sel.detach().cpu()

            if bank_on_gpu:
                md_sel_mask = m_mk_bank[sel]
            else:
                md_sel_mask = m_mk_cpu[sel_cpu].to(dev, non_blocking=True)

            md_sel_patch = m_patches[sel_cpu].to(dev, dtype=torch.float32, non_blocking=True)

            # exact pairwise support for each candidate
            pair_mask = (hmask.unsqueeze(0) & md_sel_mask.bool())   # [K,P]

            # optional extra safety: skip candidates with too little pairwise support
            pair_ok = pair_mask.sum(dim=1) >= min_overlap
            if not torch.any(pair_ok):
                continue

            sel = sel[pair_ok]
            sel_cpu = sel_cpu[pair_ok.detach().cpu()]
            md_sel_patch = md_sel_patch[pair_ok]
            pair_mask = pair_mask[pair_ok]

            hog_sim = _pairwise_masked_hog_similarity(
                hpatch,
                md_sel_patch,
                pair_mask,
                cell_size=cell_size,
                bins=bins,
            )
            mind_sim = _pairwise_masked_mind_similarity(
                hpatch,
                md_sel_patch,
                pair_mask,
                radius=mind_radius,
                dilation=mind_dilation,
            )

            hog_n = _norm01_torch(hog_sim)
            mind_n = _norm01_torch(mind_sim)

            if float(w_ngf) > 0.0:
                ngf_sim = _pairwise_masked_ngf_similarity(
                    hpatch,
                    md_sel_patch,
                    pair_mask,
                )
                ngf_n = _norm01_torch(ngf_sim)
            else:
                ngf_n = torch.zeros_like(hog_n)

            combo = (
                float(w_hog) * hog_n +
                float(w_mind) * mind_n +
                float(w_ngf) * ngf_n
            ) / total_w

            b = int(torch.argmax(combo).item())
            j = int(sel[b].item())

            if combo[b] > best_score[i]:
                best_score[i] = combo[b]
                best_idx[i] = j


    # -------- Pack --------
    hc = hist_center.detach().cpu().numpy()
    ht = hist_topleft.detach().cpu().numpy()
    bi = best_idx.detach().cpu().numpy()
    bs = best_score.detach().cpu().numpy()

    matches = []
    for i in range(Nh):
        j = int(bi[i])
        matches.append(
            {
                "hist_index": int(i),
                "hist_center_yx": (int(hc[i, 0]), int(hc[i, 1])),
                "hist_topleft_yx": (int(ht[i, 0]), int(ht[i, 1])),
                "mri_center_zyx": tuple(m_center[j].tolist()) if j >= 0 else (-1, -1, -1),
                "mri_coords": tuple(m_center[j].tolist()) if j >= 0 else (-1, -1, -1),
                "mri_topleft_zyx": tuple(m_top[j].tolist()) if j >= 0 else (-1, -1, -1),
                "score": float(bs[i]),
            }
        )

    return {
        "matches": matches,
        "hist_img": hist_img,
        "hist_mask": hist_msk,
        "mri_img": mri_img,
        "mri_mask": mri_msk,
        "device_used": str(dev),
        "metric": "HOG+MIND+NGF_fusion",
    }





def _to_np2d(x):
    arr = _to_numpy_image(x)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float32)


def resize_hist_pair(hist_img_2d, hist_mask_2d, scale):
    H, W = hist_img_2d.shape
    new_h = max(32, int(round(H * scale)))
    new_w = max(32, int(round(W * scale)))

    x = torch.from_numpy((hist_img_2d).astype(np.float32))[None, None]
    m = torch.from_numpy((hist_mask_2d > 0).astype(np.float32))[None, None]

    xr = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)
    mr = F.interpolate(m, size=(new_h, new_w), mode="nearest")[0, 0].cpu().numpy() > 0.5
    return xr, mr


def robust_match_score(matches, top_frac=0.30, min_keep=10):
    if len(matches) == 0:
        return -np.inf
    s = np.array([m["score"] for m in matches], dtype=np.float32)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return -np.inf
    k = max(min_keep, int(np.ceil(top_frac * s.size)))
    top = np.sort(s)[-k:]
    return float(np.median(top))


def estimate_z_band(matches, score_quantile=0.7, k_mad=3.0, min_radius=8):
    z = np.array([m["mri_center_zyx"][0] for m in matches], dtype=np.int32)
    s = np.array([m["score"] for m in matches], dtype=np.float32)
    valid = np.isfinite(s) & (z >= 0)
    z, s = z[valid], s[valid]
    if len(z) == 0:
        return None, None, None

    thr = np.quantile(s, score_quantile)
    keep = s >= thr
    if keep.sum() < 10:
        idx = np.argsort(s)[-10:]
        z_top = z[idx]
    else:
        z_top = z[keep]

    zc = int(np.median(z_top))
    mad = float(np.median(np.abs(z_top - zc)))
    zr = max(min_radius, int(np.ceil(k_mad * mad)))
    return zc, max(0, zc - zr), zc + zr


# ---------- border padding helpers ----------
def _rb_pad_len(L, patch, stride):
    if L < patch:
        return patch - L
    rem = (L - patch) % stride
    return 0 if rem == 0 else (stride - rem)


def _pad2d_rb(arr2d, patch, stride):
    ph = _rb_pad_len(arr2d.shape[0], patch, stride)
    pw = _rb_pad_len(arr2d.shape[1], patch, stride)
    if ph == 0 and pw == 0:
        return arr2d, (0, 0)
    out = np.pad(arr2d, ((0, ph), (0, pw)), mode="constant", constant_values=0)
    return out, (ph, pw)


def _pad3d_zyx_rb(vol, patch, stride):
    ph = _rb_pad_len(vol.shape[1], patch, stride)  # Y
    pw = _rb_pad_len(vol.shape[2], patch, stride)  # X
    if ph == 0 and pw == 0:
        return vol, (0, 0)
    out = np.pad(vol, ((0, 0), (0, ph), (0, pw)), mode="constant", constant_values=0)
    return out, (ph, pw)


def run_patch_matching_hog_mind_fusion_gpu_with_border(
        proc_Ih, histo_mask,
        slab_Im, slab_Mm,
        hist_patch, hist_stride,
        mri_patch, mri_stride,
        z_min=None, z_max=None,
        mri_bank=None,
        **kwargs):
    """
    Wrapper: pad hist on right/bottom so border patches are included.
    If mri_bank is not provided, pad MRI too and let the matcher build its bank.
    If mri_bank is provided, reuse it directly.
    """
    Ih = _to_numpy_image(proc_Ih).astype(np.float32)
    Mh = (_to_numpy_image(histo_mask) > 0)

    if Ih.ndim == 3:
        Ih = Ih[0]
    if Mh.ndim == 3:
        Mh = Mh[0]

    hist_patch = int(hist_patch)
    hist_stride = int(hist_stride)
    mri_patch = int(mri_patch)
    mri_stride = int(mri_stride)

    Ih_p, hist_pad = _pad2d_rb(Ih, hist_patch, hist_stride)
    Mh_p, _ = _pad2d_rb(Mh, hist_patch, hist_stride)

    if mri_bank is None:
        Im = _to_numpy_image(slab_Im).astype(np.float32)
        Mm = (_to_numpy_image(slab_Mm) > 0)

        if Im.ndim == 2:
            Im = Im[None, ...]
        if Mm.ndim == 2:
            Mm = Mm[None, ...]

        Im_p, mri_pad = _pad3d_zyx_rb(Im, mri_patch, mri_stride)
        Mm_p, _ = _pad3d_zyx_rb(Mm, mri_patch, mri_stride)

        out = run_patch_matching_hog_mind_fusion_gpu(
            proc_Ih=Ih_p, histo_mask=Mh_p,
            slab_Im=Im_p, slab_Mm=Mm_p,
            hist_patch=hist_patch, hist_stride=hist_stride,
            mri_patch=mri_patch, mri_stride=mri_stride,
            z_min=z_min, z_max=z_max,
            mri_bank=None,
            **kwargs
        )
    else:
        out = run_patch_matching_hog_mind_fusion_gpu(
            proc_Ih=Ih_p, histo_mask=Mh_p,
            slab_Im=slab_Im, slab_Mm=slab_Mm,
            hist_patch=hist_patch, hist_stride=hist_stride,
            mri_patch=mri_patch, mri_stride=mri_stride,
            z_min=z_min, z_max=z_max,
            mri_bank=mri_bank,
            **kwargs
        )
        mri_pad = None

    out["padding_info"] = {
        "hist_pad_bottom_right": hist_pad,
        "mri_pad_bottom_right": mri_pad,
    }
    return out

# ---------- coarse scale search ----------
def _filter_cfg_for_matcher(cfg, matcher_fn):
    sig = inspect.signature(matcher_fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    out = dict(cfg)

    if "hist_kernel_batch" in params and "hist_batch" in out:
        out["hist_kernel_batch"] = out.pop("hist_batch")
    if "mri_chunk_size" in params and "mri_chunk" in out:
        out["mri_chunk_size"] = out.pop("mri_chunk")

    if accepts_var_kw:
        return out
    return {k: v for k, v in out.items() if k in params}

# def coarse_scale_search_hog_mind(
#     proc_Ih,
#     histo_mask,
#     slab_Im,
#     slab_Mm,
#     scales=None,                      # e.g. np.arange(1.00, 1.31, 0.05)
#     top_frac_score=0.30,
#     common_kwargs=None,               # matcher args
#     matcher_fn=run_patch_matching_hog_mind_fusion_gpu_with_border,
#     verbose=True,
# ):
#     if common_kwargs is None:
#         common_kwargs = {}

#     if scales is None:
#         scales = np.arange(1.00, 1.31, 0.05)
#     scales = np.array(sorted(set(float(s) for s in scales)), dtype=np.float32)

#     hist_img = _to_np2d(proc_Ih)
#     hist_msk = _to_np2d(histo_mask) > 0
#     Z = np.asarray(_to_numpy_image(slab_Im)).shape[0] if np.asarray(_to_numpy_image(slab_Im)).ndim == 3 else 1

#     records = []
#     best = dict(score=-np.inf, scale=None, result=None, z_center=None, z_min=None, z_max=None)

#     # Default matcher config (required by run_patch_matching_hog_mind_fusion_gpu_with_border)
#     run_cfg = dict(
#         hist_patch=96,
#         hist_stride=64,
#         mri_patch=96,
#         mri_stride=8,
#     )
#     # run_cfg = dict(
#     #     hist_patch=64,
#     #     hist_stride=32,
#     #     mri_patch=64,
#     #     mri_stride=8,
#     # )
#     run_cfg.update(common_kwargs)  # user overrides defaults if needed
#     run_cfg = _filter_cfg_for_matcher(run_cfg, matcher_fn)
    
#     Im = _to_numpy_image(slab_Im).astype(np.float32)
#     Mm = (_to_numpy_image(slab_Mm) > 0)
#     if Im.ndim == 2:
#         Im = Im[None, ...]
#     if Mm.ndim == 2:
#         Mm = Mm[None, ...]

#     Im_p, mri_pad = _pad3d_zyx_rb(Im, run_cfg["mri_patch"], run_cfg["mri_stride"])
#     Mm_p, _ = _pad3d_zyx_rb(Mm, run_cfg["mri_patch"], run_cfg["mri_stride"])

#     mri_bank = build_mri_bank_hog_mind_ngf(
#         slab_Im=Im_p,
#         slab_Mm=Mm_p,
#         mri_patch=run_cfg["mri_patch"],
#         mri_stride=run_cfg["mri_stride"],
#         cell_size=run_cfg.get("cell_size", 8),
#         bins=run_cfg.get("bins", 18),
#         min_mri_mask_frac=run_cfg.get("min_mri_mask_frac", 0.5),
#         z_min=None,
#         z_max=None,
#         device=run_cfg.get("device", "cuda:0"),
#         keep_mri_bank_on_gpu=run_cfg.get("keep_mri_bank_on_gpu", True),
#     )

#     for s in scales:
#         hs, ms = resize_hist_pair(hist_img, hist_msk, float(s))

        
#         if verbose:
#             print(f"[scale={s:.3f}] running...")

#         out = matcher_fn(
#             proc_Ih=hs,
#             histo_mask=ms,
#             slab_Im=slab_Im,
#             slab_Mm=slab_Mm,
#             mri_bank=mri_bank,
#             **run_cfg,
#         )

#         sc = robust_match_score(out["matches"], top_frac=top_frac_score)
#         zc, zmin, zmax = estimate_z_band(out["matches"])

#         if zmin is not None and zmax is not None:
#             zmin = int(max(0, zmin))
#             zmax = int(min(Z - 1, zmax))
#             if zmax < zmin:
#                 zmin, zmax = None, None

#         rec = dict(
#             scale=float(s),
#             score=float(sc),
#             n_matches=len(out["matches"]),
#             status="ok",
#             z_center=zc,
#             z_min=zmin,
#             z_max=zmax,
#         )
#         records.append(rec)
#         if verbose:
#             print(rec)

#         if sc > best["score"]:
#             best = dict(
#                 score=float(sc),
#                 scale=float(s),
#                 result=out,
#                 z_center=zc,
#                 z_min=zmin,
#                 z_max=zmax,
#             )

#         # except Exception as e:
#         #     rec = dict(
#         #         scale=float(s),
#         #         score=-np.inf,
#         #         n_matches=0,
#         #         status=f"error:{type(e).__name__}:{str(e)[:140]}",
#         #         z_center=None,
#         #         z_min=None,
#         #         z_max=None,
#         #     )
#         #     records.append(rec)
#         #     if verbose:
#         #         print(rec)

#     records = sorted(records, key=lambda r: (np.isfinite(r["score"]), r["score"]), reverse=True)
#     return best, records


# ---------- MRI bank builder (extracted from v6's inline loop, unchanged math) ----------
def build_mri_bank_hog_mind(
    slab_Im, slab_Mm,
    mri_patch, mri_stride,
    cell_size, bins,
    min_mri_mask_frac,
    z_min, z_max,
    device,
    keep_mri_bank_on_gpu=True,
):
    patch = int(mri_patch)

    if isinstance(device, torch.device):
        dev = device
    else:
        dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    mri_img = _to_numpy_image(slab_Im).astype(np.float32)
    mri_msk = (_to_numpy_image(slab_Mm) > 0)
    if mri_img.ndim == 2:
        mri_img = mri_img[None, ...]
    if mri_msk.ndim == 2:
        mri_msk = mri_msk[None, ...]

    m_hog_list, m_mk_list, m_top_list, m_center_list, m_patch_list = [], [], [], [], []
    Z = mri_img.shape[0]

    for z in tqdm(range(Z), desc="Build MRI bank"):
        if z_min is not None and z < z_min:
            continue
        if z_max is not None and z > z_max:
            continue

        m_gx, m_gy = _sobel_xy(mri_img[z], dev)
        m_im = torch.as_tensor(mri_img[z], device=dev, dtype=torch.float32)
        m_m = torch.as_tensor(mri_msk[z].astype(np.float32), device=dev)

        m_gx_u = _unfold_hw(m_gx, patch, mri_stride)
        m_gy_u = _unfold_hw(m_gy, patch, mri_stride)
        m_im_u = _unfold_hw(m_im, patch, mri_stride)
        m_mk_u = _unfold_hw(m_m, patch, mri_stride) > 0.5

        Hm, Wm = mri_img[z].shape
        ny_m = (Hm - patch) // mri_stride + 1
        nx_m = (Wm - patch) // mri_stride + 1
        yy_m, xx_m = torch.meshgrid(
            torch.arange(ny_m, device=dev),
            torch.arange(nx_m, device=dev),
            indexing="ij"
        )
        m_y0 = (yy_m * mri_stride).reshape(-1)
        m_x0 = (xx_m * mri_stride).reshape(-1)
        m_yc = m_y0 + patch // 2
        m_xc = m_x0 + patch // 2

        center_ok = (m_m[m_yc, m_xc] > 0.5)
        frac_ok = (m_mk_u.float().mean(dim=1) >= float(min_mri_mask_frac))
        keep_m = center_ok & frac_ok
        if not torch.any(keep_m):
            continue

        gxk = m_gx_u[keep_m]
        gyk = m_gy_u[keep_m]
        mkk = m_mk_u[keep_m]
        pim = m_im_u[keep_m].reshape(-1, patch, patch)

        hogk = _hog_from_unfolded(
            gxk, gyk, mkk, patch=patch, cell_size=cell_size, bins=bins
        ).half().cpu()

        m_hog_list.append(hogk)
        m_mk_list.append(mkk.cpu())
        m_patch_list.append(pim.cpu())

        n = int(keep_m.sum().item())
        zcol = torch.full((n, 1), z, dtype=torch.int32, device=dev).cpu()
        top = torch.cat([zcol, m_y0[keep_m].to(torch.int32).cpu().unsqueeze(1),
                          m_x0[keep_m].to(torch.int32).cpu().unsqueeze(1)], dim=1)
        cen = torch.cat([zcol, m_yc[keep_m].to(torch.int32).cpu().unsqueeze(1),
                          m_xc[keep_m].to(torch.int32).cpu().unsqueeze(1)], dim=1)
        m_top_list.append(top)
        m_center_list.append(cen)

    if len(m_hog_list) == 0:
        raise RuntimeError("No MRI patches survived mask filtering.")

    m_hog_cpu = torch.cat(m_hog_list, dim=0)
    m_mk_cpu = torch.cat(m_mk_list, dim=0)
    m_patches = torch.cat(m_patch_list, dim=0)
    m_top = torch.cat(m_top_list, dim=0).cpu().numpy()
    m_center = torch.cat(m_center_list, dim=0).cpu().numpy()
    Nm = m_hog_cpu.shape[0]

    bank_on_gpu = False
    m_hog_bank, m_mk_bank = m_hog_cpu, m_mk_cpu
    if keep_mri_bank_on_gpu and dev.type == "cuda":
        try:
            m_hog_bank = m_hog_cpu.to(dev, non_blocking=True)
            m_mk_bank = m_mk_cpu.to(dev, non_blocking=True)
            bank_on_gpu = True
        except RuntimeError:
            torch.cuda.empty_cache()

    return {
        "mri_img": mri_img, "mri_msk": mri_msk,
        "patch": patch, "mri_patch": patch, "mri_stride": int(mri_stride),
        "cell_size": int(cell_size), "bins": int(bins),
        "min_mri_mask_frac": float(min_mri_mask_frac),
        "z_min": z_min, "z_max": z_max, "device_used": str(dev),
        "m_hog_cpu": m_hog_cpu, "m_mk_cpu": m_mk_cpu, "m_patches": m_patches,
        "m_top": m_top, "m_center": m_center, "Nm": Nm,
        "bank_on_gpu": bank_on_gpu, "m_hog_bank": m_hog_bank, "m_mk_bank": m_mk_bank,
    }


# ---------- v6-style matcher, but accepts a precomputed mri_bank ----------
def run_patch_matching_hog_mind_fusion_gpu_cached(
    proc_Ih, histo_mask, slab_Im, slab_Mm,
    hist_patch=64, hist_stride=32, mri_patch=64, mri_stride=4,
    cell_size=8, bins=18,
    min_histo_mask_frac=0.5, min_mri_mask_frac=0.5, min_overlap_frac=0.25,
    topk_hog=64, w_hog=0.6, w_mind=0.4,
    mind_radius=2, mind_dilation=1,
    hist_batch=64, mri_chunk=4096,
    z_min=None, z_max=None,
    device="cuda:0", keep_mri_bank_on_gpu=True, mind_compute_batch=512,
    mri_bank=None,
):
    assert hist_patch == mri_patch, "This function assumes same patch size for hist and MRI."
    patch = int(hist_patch)
    P = patch * patch
    min_overlap = int(P * float(min_overlap_frac))

    if isinstance(device, torch.device):
        dev = device
    else:
        dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    hist_img, hist_msk, mri_img_, mri_msk_ = _prep_inputs(proc_Ih, histo_mask, slab_Im, slab_Mm)

    # -------- Hist bank (always built fresh: histology changes every call) --------
    h_gx, h_gy = _sobel_xy(hist_img, dev)
    h_im = torch.as_tensor(hist_img, device=dev, dtype=torch.float32)
    h_mk = torch.as_tensor(hist_msk.astype(np.float32), device=dev)

    h_gx_u = _unfold_hw(h_gx, patch, hist_stride)
    h_gy_u = _unfold_hw(h_gy, patch, hist_stride)
    h_im_u = _unfold_hw(h_im, patch, hist_stride)
    h_mk_u = _unfold_hw(h_mk, patch, hist_stride) > 0.5

    keep_h = (h_mk_u.float().mean(dim=1) >= float(min_histo_mask_frac))
    if not torch.any(keep_h):
        raise RuntimeError("No histology patches survived mask filtering.")

    Hh, Wh = hist_img.shape
    ny_h = (Hh - patch) // hist_stride + 1
    nx_h = (Wh - patch) // hist_stride + 1
    yy_h, xx_h = torch.meshgrid(torch.arange(ny_h, device=dev), torch.arange(nx_h, device=dev), indexing="ij")
    h_y0 = (yy_h * hist_stride).reshape(-1)
    h_x0 = (xx_h * hist_stride).reshape(-1)
    h_yc = h_y0 + patch // 2
    h_xc = h_x0 + patch // 2

    h_gx_u, h_gy_u, h_im_u, h_mk_u = h_gx_u[keep_h], h_gy_u[keep_h], h_im_u[keep_h], h_mk_u[keep_h]
    hist_center = torch.stack([h_yc[keep_h], h_xc[keep_h]], dim=1).to(torch.int32)
    hist_topleft = torch.stack([h_y0[keep_h], h_x0[keep_h]], dim=1).to(torch.int32)

    h_hog = _hog_from_unfolded(h_gx_u, h_gy_u, h_mk_u, patch=patch, cell_size=cell_size, bins=bins)
    h_patches = h_im_u.reshape(-1, patch, patch)
    h_mind = _mind_desc_from_patch_batch(h_patches, h_mk_u, radius=mind_radius, dilation=mind_dilation)
    Nh = h_hog.shape[0]

    # -------- MRI bank: reuse if given, else build once here --------
    if mri_bank is None:
        mri_bank = build_mri_bank_hog_mind(
            slab_Im=slab_Im, slab_Mm=slab_Mm,
            mri_patch=mri_patch, mri_stride=mri_stride,
            cell_size=cell_size, bins=bins, min_mri_mask_frac=min_mri_mask_frac,
            z_min=z_min, z_max=z_max, device=dev, keep_mri_bank_on_gpu=keep_mri_bank_on_gpu,
        )
    else:
        if int(mri_bank["patch"]) != patch:
            raise ValueError(f"mri_bank patch={mri_bank['patch']} != requested patch={patch}")
        if int(mri_bank["mri_stride"]) != int(mri_stride):
            raise ValueError(f"mri_bank mri_stride={mri_bank['mri_stride']} != requested mri_stride={mri_stride}")

    mri_img = mri_bank["mri_img"]
    mri_msk = mri_bank["mri_msk"]
    m_hog_cpu = mri_bank["m_hog_cpu"]
    m_mk_cpu = mri_bank["m_mk_cpu"]
    m_patches = mri_bank["m_patches"]
    m_top = mri_bank["m_top"]
    m_center = mri_bank["m_center"]
    Nm = mri_bank["Nm"]
    bank_on_gpu = mri_bank["bank_on_gpu"]
    m_hog_bank = mri_bank["m_hog_bank"]
    m_mk_bank = mri_bank["m_mk_bank"]

    mind_cache = {}

    def get_mri_mind_desc(idxs, out_device=dev):
        idx_list = idxs.detach().cpu().tolist() if isinstance(idxs, torch.Tensor) else [int(i) for i in idxs]
        miss = [i for i in idx_list if i not in mind_cache]
        if len(miss) > 0:
            for s in range(0, len(miss), int(mind_compute_batch)):
                miss_chunk = miss[s:s + int(mind_compute_batch)]
                idx_cpu = torch.tensor(miss_chunk, dtype=torch.long)
                batch = m_patches[idx_cpu].to(dev, dtype=torch.float32, non_blocking=True)
                if bank_on_gpu:
                    idx_gpu = idx_cpu.to(dev, non_blocking=True)
                    masks = m_mk_bank[idx_gpu]
                else:
                    masks = m_mk_cpu[idx_cpu].to(dev, non_blocking=True)
                desc = _mind_desc_from_patch_batch(batch, masks, radius=mind_radius, dilation=mind_dilation).half().cpu()
                for k, idx in enumerate(miss_chunk):
                    mind_cache[int(idx)] = desc[k]
        out = torch.stack([mind_cache[int(i)] for i in idx_list], dim=0)
        if out_device is None:
            return out
        return out.to(out_device, dtype=torch.float32, non_blocking=True)

    best_score = torch.full((Nh,), -float("inf"), device=dev, dtype=torch.float32)
    best_idx = torch.full((Nh,), -1, device=dev, dtype=torch.long)

    for hs in tqdm(range(0, Nh, hist_batch), desc="Match HOG+MIND"):
        he = min(hs + hist_batch, Nh)
        Hb = he - hs
        hh = h_hog[hs:he].to(torch.float32)
        hmasks_f = h_mk_u[hs:he].to(torch.float32)
        hmind = h_mind[hs:he].to(torch.float32)

        run_vals = torch.full((Hb, topk_hog), -float("inf"), device=dev, dtype=torch.float32)
        run_idx = torch.full((Hb, topk_hog), -1, device=dev, dtype=torch.long)

        for ms in range(0, Nm, mri_chunk):
            me = min(ms + mri_chunk, Nm)
            if bank_on_gpu:
                md = m_hog_bank[ms:me].to(torch.float32)
                mm_chunk = m_mk_bank[ms:me]
            else:
                md = m_hog_cpu[ms:me].to(dev, dtype=torch.float32, non_blocking=True)
                mm_chunk = m_mk_cpu[ms:me].to(dev, non_blocking=True)

            sim = hh @ md.T
            overlap = hmasks_f @ mm_chunk.to(torch.float32).T
            sim[overlap < min_overlap] = -float("inf")

            k = min(topk_hog, me - ms)
            cvals, cidx = torch.topk(sim, k=k, dim=1)
            cidx = cidx + ms
            all_vals = torch.cat([run_vals, cvals], dim=1)
            all_idx = torch.cat([run_idx, cidx], dim=1)
            run_vals, pos = torch.topk(all_vals, k=topk_hog, dim=1)
            run_idx = torch.gather(all_idx, 1, pos)

        for r in range(Hb):
            i = hs + r
            sel = run_idx[r]
            sel = sel[sel >= 0]
            if sel.numel() == 0:
                continue
            sel = torch.unique(sel)

            hdesc_hog = hh[r:r + 1]
            hdesc_mind = hmind[r:r + 1]
            if bank_on_gpu:
                md_sel_hog = m_hog_bank[sel].to(torch.float32)
            else:
                sel_cpu = sel.detach().cpu()
                md_sel_hog = m_hog_cpu[sel_cpu].to(dev, dtype=torch.float32, non_blocking=True)

            hog_sim = (hdesc_hog @ md_sel_hog.T).squeeze(0)
            md_sel_mind = get_mri_mind_desc(sel, out_device=dev)
            mind_sim = (hdesc_mind @ md_sel_mind.T).squeeze(0)

            hog_n = _norm01_torch(hog_sim)
            mind_n = _norm01_torch(mind_sim)
            combo = w_hog * hog_n + w_mind * mind_n

            b = int(torch.argmax(combo).item())
            j = int(sel[b].item())
            if combo[b] > best_score[i]:
                best_score[i] = combo[b]
                best_idx[i] = j

    hc = hist_center.detach().cpu().numpy()
    ht = hist_topleft.detach().cpu().numpy()
    bi = best_idx.detach().cpu().numpy()
    bs = best_score.detach().cpu().numpy()

    matches = []
    for i in range(Nh):
        j = int(bi[i])
        matches.append({
            "hist_index": int(i),
            "hist_center_yx": (int(hc[i, 0]), int(hc[i, 1])),
            "hist_topleft_yx": (int(ht[i, 0]), int(ht[i, 1])),
            "mri_center_zyx": tuple(m_center[j].tolist()) if j >= 0 else (-1, -1, -1),
            "mri_coords": tuple(m_center[j].tolist()) if j >= 0 else (-1, -1, -1),
            "mri_topleft_zyx": tuple(m_top[j].tolist()) if j >= 0 else (-1, -1, -1),
            "score": float(bs[i]),
        })

    return {
        "matches": matches, "hist_img": hist_img, "hist_mask": hist_msk,
        "mri_img": mri_img, "mri_mask": mri_msk, "device_used": str(dev),
        "metric": "HOG+MIND_fusion",
    }


# ---------- border-padded wrapper (mirrors your existing _with_border pattern) ----------
def run_patch_matching_hog_mind_fusion_gpu_with_border_cached(
        proc_Ih, histo_mask, slab_Im, slab_Mm,
        hist_patch, hist_stride, mri_patch, mri_stride,
        z_min=None, z_max=None, mri_bank=None, **kwargs):
    Ih = _to_numpy_image(proc_Ih).astype(np.float32)
    Mh = (_to_numpy_image(histo_mask) > 0)
    if Ih.ndim == 3: Ih = Ih[0]
    if Mh.ndim == 3: Mh = Mh[0]

    hist_patch, hist_stride = int(hist_patch), int(hist_stride)
    mri_patch, mri_stride = int(mri_patch), int(mri_stride)

    Ih_p, hist_pad = _pad2d_rb(Ih, hist_patch, hist_stride)
    Mh_p, _ = _pad2d_rb(Mh, hist_patch, hist_stride)

    if mri_bank is None:
        Im = _to_numpy_image(slab_Im).astype(np.float32)
        Mm = (_to_numpy_image(slab_Mm) > 0)
        if Im.ndim == 2: Im = Im[None, ...]
        if Mm.ndim == 2: Mm = Mm[None, ...]
        Im_p, mri_pad = _pad3d_zyx_rb(Im, mri_patch, mri_stride)
        Mm_p, _ = _pad3d_zyx_rb(Mm, mri_patch, mri_stride)
        out = run_patch_matching_hog_mind_fusion_gpu_cached(
            proc_Ih=Ih_p, histo_mask=Mh_p, slab_Im=Im_p, slab_Mm=Mm_p,
            hist_patch=hist_patch, hist_stride=hist_stride,
            mri_patch=mri_patch, mri_stride=mri_stride,
            z_min=z_min, z_max=z_max, mri_bank=None, **kwargs
        )
    else:
        out = run_patch_matching_hog_mind_fusion_gpu_cached(
            proc_Ih=Ih_p, histo_mask=Mh_p, slab_Im=slab_Im, slab_Mm=slab_Mm,
            hist_patch=hist_patch, hist_stride=hist_stride,
            mri_patch=mri_patch, mri_stride=mri_stride,
            z_min=z_min, z_max=z_max, mri_bank=mri_bank, **kwargs
        )
        mri_pad = None

    out["padding_info"] = {"hist_pad_bottom_right": hist_pad, "mri_pad_bottom_right": mri_pad}
    return out

def coarse_scale_search_hog_mind_cached(
    proc_Ih,
    histo_mask,
    slab_Im,
    slab_Mm,
    scales=None,                      # e.g. np.arange(1.00, 1.31, 0.05)
    top_frac_score=0.30,
    common_kwargs=None,               # matcher args
    matcher_fn=run_patch_matching_hog_mind_fusion_gpu_with_border_cached,
    verbose=True,
):
    if common_kwargs is None:
        common_kwargs = {}

    if scales is None:
        scales = np.arange(1.00, 1.31, 0.05)
    scales = np.array(sorted(set(float(s) for s in scales)), dtype=np.float32)

    hist_img = _to_np2d(proc_Ih)
    hist_msk = _to_np2d(histo_mask) > 0
    Z = np.asarray(_to_numpy_image(slab_Im)).shape[0] if np.asarray(_to_numpy_image(slab_Im)).ndim == 3 else 1

    records = []
    best = dict(score=-np.inf, scale=None, result=None, z_center=None, z_min=None, z_max=None)

    # Default matcher config (same defaults as your v6 coarse_scale_search_hog_mind)
    run_cfg = dict(
        hist_patch=96,
        hist_stride=64,
        mri_patch=96,
        mri_stride=8,
    )
    run_cfg.update(common_kwargs)  # user overrides defaults if needed
    run_cfg = _filter_cfg_for_matcher(run_cfg, matcher_fn)
    print(run_cfg)

    # -------- Build the MRI bank once: mri_patch/mri_stride never change across the scale loop --------
    Im = _to_numpy_image(slab_Im).astype(np.float32)
    Mm = (_to_numpy_image(slab_Mm) > 0)
    if Im.ndim == 2:
        Im = Im[None, ...]
    if Mm.ndim == 2:
        Mm = Mm[None, ...]

    Im_p, _ = _pad3d_zyx_rb(Im, run_cfg["mri_patch"], run_cfg["mri_stride"])
    Mm_p, _ = _pad3d_zyx_rb(Mm, run_cfg["mri_patch"], run_cfg["mri_stride"])

    mri_bank = build_mri_bank_hog_mind(
        slab_Im=Im_p,
        slab_Mm=Mm_p,
        mri_patch=run_cfg["mri_patch"],
        mri_stride=run_cfg["mri_stride"],
        cell_size=run_cfg.get("cell_size", 8),
        bins=run_cfg.get("bins", 18),
        min_mri_mask_frac=run_cfg.get("min_mri_mask_frac", 0.5),
        z_min=None,
        z_max=None,
        device=run_cfg.get("device", "cuda:0"),
        keep_mri_bank_on_gpu=run_cfg.get("keep_mri_bank_on_gpu", True),
    )

    for s in scales:
        hs, ms = resize_hist_pair(hist_img, hist_msk, float(s))

        if verbose:
            print(f"[scale={s:.3f}] running...")

        out = matcher_fn(
            proc_Ih=hs,
            histo_mask=ms,
            slab_Im=slab_Im,
            slab_Mm=slab_Mm,
            mri_bank=mri_bank,
            **run_cfg,
        )

        sc = robust_match_score(out["matches"], top_frac=top_frac_score)
        zc, zmin, zmax = estimate_z_band(out["matches"])

        if zmin is not None and zmax is not None:
            zmin = int(max(0, zmin))
            zmax = int(min(Z - 1, zmax))
            if zmax < zmin:
                zmin, zmax = None, None

        rec = dict(
            scale=float(s),
            score=float(sc),
            n_matches=len(out["matches"]),
            status="ok",
            z_center=zc,
            z_min=zmin,
            z_max=zmax,
        )
        records.append(rec)
        if verbose:
            print(rec)

        if sc > best["score"]:
            best = dict(
                score=float(sc),
                scale=float(s),
                result=out,
                z_center=zc,
                z_min=zmin,
                z_max=zmax,
            )

    records = sorted(records, key=lambda r: (np.isfinite(r["score"]), r["score"]), reverse=True)
    return best, records

def evaluate_fixed_scale_multi_patch(
    proc_Ih,
    histo_mask,
    slab_Im,
    slab_Mm,
    best_scale,
    patch_matrix,
    z_min=None,
    z_max=None,
    common_kwargs=None,
    matcher_fn=run_patch_matching_hog_mind_fusion_gpu_with_border,
):
    if common_kwargs is None:
        common_kwargs = {}

    hist_img, hist_msk, mri_img, mri_msk = _prep_inputs(
        proc_Ih, histo_mask, slab_Im, slab_Mm
    )

    hist_img = hist_img.astype(np.float32)
    hist_msk = hist_msk > 0
    mri_img = mri_img.astype(np.float32)
    mri_msk = mri_msk > 0

    hs, ms = resize_hist_pair(hist_img, hist_msk, float(best_scale))

    results = []

    import gc
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
    except Exception:
        libc = None

    for cfg in patch_matrix:
        p = int(cfg["patch"])
        hs_stride = int(cfg["hist_stride"])
        ms_stride = int(cfg["mri_stride"])

        cell_size = int(common_kwargs.get("cell_size", 8))
        bins = int(common_kwargs.get("bins", 18))
        min_mri_mask_frac = float(common_kwargs.get("min_mri_mask_frac", 0.5))
        device = common_kwargs.get("device", "cuda:0")

        try:
            Im_p, _ = _pad3d_zyx_rb(mri_img, p, ms_stride)
            Mm_p, _ = _pad3d_zyx_rb(mri_msk, p, ms_stride)

            mri_bank = build_mri_bank_hog_mind_ngf(
                slab_Im=Im_p,
                slab_Mm=Mm_p,
                mri_patch=p,
                mri_stride=ms_stride,
                cell_size=cell_size,
                bins=bins,
                min_mri_mask_frac=min_mri_mask_frac,
                z_min=z_min,
                z_max=z_max,
                device=device,
                keep_mri_bank_on_gpu=False,
            )

            run_kwargs = dict(common_kwargs)
            run_kwargs.update(dict(
                proc_Ih=hs,
                histo_mask=ms,
                slab_Im=mri_img,
                slab_Mm=mri_msk,
                hist_patch=p,
                hist_stride=hs_stride,
                mri_patch=p,
                mri_stride=ms_stride,
                z_min=z_min,
                z_max=z_max,
                mri_bank=mri_bank,
            ))

            out = matcher_fn(**run_kwargs)
            score = robust_match_score(out["matches"], top_frac=0.30)

            compact_out = {
                "matches": out["matches"],
                "padding_info": out.get("padding_info", None),
                "device_used": out.get("device_used", None),
                "metric": out.get("metric", None),
            }

            res = {
                "patch": p,
                "hist_stride": hs_stride,
                "mri_stride": ms_stride,
                "robust_score": score,
                "n_matches": len(out["matches"]),
                "_out": compact_out,
            }
            results.append(res)

            print(
                f"[patch={p}] robust_score={score:.4f}, "
                f"n_matches={len(out['matches'])}, "
                f"padding={compact_out['padding_info']}"
            )

            del compact_out
            del out
            del run_kwargs
            del mri_bank
            del Im_p
            del Mm_p
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if libc is not None:
                libc.malloc_trim(0)

        except Exception as e:
            print(f"[patch={p}] ERROR: {type(e).__name__}: {str(e)[:120]}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if libc is not None:
                libc.malloc_trim(0)

    if len(results) == 0:
        raise RuntimeError("All patch configs failed.")

    results = sorted(results, key=lambda r: r["robust_score"], reverse=True)

    print("\nBest config:", {
        k: results[0][k]
        for k in ["patch", "hist_stride", "mri_stride", "robust_score", "n_matches"]
    })
    return results


# ---------- Score stability ----------
def score_stability(matches, top_frac=0.30, min_keep=10):
    s = np.array([m["score"] for m in matches], dtype=np.float32)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return dict(n=0, median=np.nan, iqr=np.nan, p10=np.nan, p90=np.nan, median_top=np.nan, iqr_top=np.nan)

    k = max(min_keep, int(np.ceil(top_frac * s.size)))
    top = np.sort(s)[-k:]

    return dict(
        n=int(s.size),
        median=float(np.median(s)),
        iqr=float(np.percentile(s, 75) - np.percentile(s, 25)),
        p10=float(np.percentile(s, 10)),
        p90=float(np.percentile(s, 90)),
        median_top=float(np.median(top)),
        iqr_top=float(np.percentile(top, 75) - np.percentile(top, 25)),
    )


# ---------- Affine RANSAC (no external deps) ----------
def _fit_affine_xy(src_xy, dst_xy):
    # src_xy, dst_xy: [N,2], solve dst = A*[x,y,1]
    N = src_xy.shape[0]
    X = np.zeros((2 * N, 6), dtype=np.float64)
    y = np.zeros((2 * N,), dtype=np.float64)

    X[0::2, 0:2] = src_xy
    X[0::2, 2] = 1.0
    X[1::2, 3:5] = src_xy
    X[1::2, 5] = 1.0
    y[0::2] = dst_xy[:, 0]
    y[1::2] = dst_xy[:, 1]

    p, *_ = np.linalg.lstsq(X, y, rcond=None)
    M = np.array([[p[0], p[1], p[2]],
                  [p[3], p[4], p[5]]], dtype=np.float64)
    return M

def _apply_affine_xy(src_xy, M):
    return src_xy @ M[:, :2].T + M[:, 2][None, :]

# RANSAC finds a model with enough inliers (ok=True), not enough (ok=False)
def ransac_affine_from_matches(
    matches,
    reproj_thresh=12.0,
    n_iter=2000,
    min_samples=3,
    seed=0,
    z_mad_k=3.0,
    z_min_tol=2.0,
):
    if len(matches) < min_samples:
        return dict(ok=False, reason="too_few_matches")

    hist_xy, mri_xy, zvals, valid_idx = [], [], [], []
    for i, m in enumerate(matches):
        z, y, x = m["mri_center_zyx"]
        if z < 0:
            continue
        hy, hx = m["hist_center_yx"]
        hist_xy.append([hx, hy])
        mri_xy.append([x, y])
        zvals.append(z)
        valid_idx.append(i)

    if len(valid_idx) < min_samples:
        return dict(ok=False, reason="too_few_valid_matches")

    hist_xy = np.asarray(hist_xy, dtype=np.float64)
    mri_xy = np.asarray(mri_xy, dtype=np.float64)
    zvals = np.asarray(zvals, dtype=np.float64)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)

    zc = np.median(zvals)
    mad = np.median(np.abs(zvals - zc))
    z_tol = max(z_mad_k * mad, z_min_tol)
    z_keep = np.abs(zvals - zc) <= z_tol
    if z_keep.sum() >= min_samples:
        hist_xy, mri_xy, valid_idx = hist_xy[z_keep], mri_xy[z_keep], valid_idx[z_keep]

    N = hist_xy.shape[0]
    if N < min_samples:
        return dict(ok=False, reason="too_few_after_z_filter")

    rng = np.random.default_rng(seed)
    best_inliers, best_count, best_mederr, best_M = None, -1, np.inf, None

    for _ in range(n_iter):
        sample = rng.choice(N, size=min_samples, replace=False)
        M = _fit_affine_xy(hist_xy[sample], mri_xy[sample])
        pred = _apply_affine_xy(hist_xy, M)
        err = np.linalg.norm(pred - mri_xy, axis=1)
        inliers = err <= reproj_thresh
        cnt = int(inliers.sum())
        if cnt < min_samples:
            continue
        mederr = float(np.median(err[inliers]))
        if (cnt > best_count) or (cnt == best_count and mederr < best_mederr):
            best_count, best_mederr, best_inliers, best_M = cnt, mederr, inliers, M

    if best_inliers is None:
        return dict(ok=False, reason="no_model")

    M_refit = _fit_affine_xy(hist_xy[best_inliers], mri_xy[best_inliers])
    pred = _apply_affine_xy(hist_xy, M_refit)
    err = np.linalg.norm(pred - mri_xy, axis=1)
    inliers = err <= reproj_thresh
    inlier_global_idx = valid_idx[inliers]

    return dict(
        ok=True, M=M_refit, inlier_idx=inlier_global_idx,
        inlier_count=int(inliers.sum()), total_valid=int(N),
        inlier_ratio=float(inliers.mean()),
        median_reproj=float(np.median(err[inliers])) if inliers.any() else np.inf,
    )


# ---------- Compare all patch configs ----------
def compare_patch_results_ransac(
    res_list,
    reproj_thresh=12.0,
    n_iter=2000,
    top_frac=0.30,
    seed=0,
):
    rows = []
    for r in res_list:
        matches = r["_out"]["matches"]
        # Computes score distribution quality
        st = score_stability(matches, top_frac=top_frac)
        rr = ransac_affine_from_matches(
            matches,
            reproj_thresh=reproj_thresh,
            n_iter=n_iter,
            seed=seed,
        )

        row = dict(
            patch=r["patch"],
            hist_stride=r["hist_stride"],
            mri_stride=r["mri_stride"],
            robust_score=r["robust_score"],
            n_matches=r["n_matches"],
            score_median_top=st["median_top"],
            score_iqr_top=st["iqr_top"],
            ransac_ok=rr.get("ok", False),
            inlier_count=rr.get("inlier_count", 0),
            inlier_ratio=rr.get("inlier_ratio", 0.0),
            median_reproj=rr.get("median_reproj", np.inf),
            _ransac=rr,
            _matches=matches,
            _out=r["_out"],
        )
        rows.append(row)

    # rank: prioritize geometry first, then stable score
    rows_sorted = sorted(
        rows,
        key=lambda d: (
            d["ransac_ok"],
            d["inlier_count"],
            d["inlier_ratio"],
            -d["median_reproj"],
            d["score_median_top"],
            -d["score_iqr_top"],
        ),
        reverse=True,
    )

    print("=== Patch config comparison (best first) ===")
    for i, d in enumerate(rows_sorted):
        print(
            f"[{i}] patch={d['patch']:>3} hs={d['hist_stride']:>2} ms={d['mri_stride']:>2} | "
            f"inliers={d['inlier_count']:>4}/{d['n_matches']:<4} ({d['inlier_ratio']:.3f}) | "
            f"med_reproj={d['median_reproj']:.2f} | "
            f"score_top_med={d['score_median_top']:.4f} score_top_iqr={d['score_iqr_top']:.4f}"
        )
    return rows_sorted


# ---------- Combine inliers from top-K configs ----------
def combine_inliers_from_top(rows_sorted, top_k=2, reproj_thresh=12.0, n_iter=3000, seed=0, grid=4):
    pooled = []
    for d in rows_sorted[:top_k]:
        rr = d["_ransac"]
        if not rr.get("ok", False):
            continue
        idx = set(rr["inlier_idx"].tolist())
        for i, m in enumerate(d["_matches"]):
            if i in idx:
                pooled.append(m)

    if len(pooled) == 0:
        return dict(ok=False, reason="no_inliers_from_topk")

    # dedup by hist center location (quantized), not hist_index
    best_by_loc = {}
    for m in pooled:
        hy, hx = m["hist_center_yx"]
        key = (int(round(hy / grid)), int(round(hx / grid)))
        if (key not in best_by_loc) or (m["score"] > best_by_loc[key]["score"]):
            best_by_loc[key] = m
    merged = list(best_by_loc.values())

    rr = ransac_affine_from_matches(merged, reproj_thresh=reproj_thresh, n_iter=n_iter, seed=seed)
    return dict(ok=rr.get("ok", False), merged_matches=merged, ransac=rr)

def z_band_from_inliers(inlier_matches, k_mad=3.0, min_radius=8):
    z = np.array([m["mri_center_zyx"][0] for m in inlier_matches], dtype=np.int32)
    z = z[z >= 0]
    if len(z) == 0:
        raise RuntimeError("No valid z in inlier_matches.")

    z_center = int(np.median(z))
    mad = float(np.median(np.abs(z - z_center)))
    z_radius = max(min_radius, int(np.ceil(k_mad * mad)))

    z_min = max(0, z_center - z_radius)
    z_max = z_center + z_radius
    return z_center, z_min, z_max, z_radius

def scale_histo_with_best_scale(proc_Ih, proc_Mh, best_scale, preserve_physical_size=True):
    """
    Returns:
      h_scaled_sitk (sitk.Image), m_scaled_sitk (sitk.Image)

    Notes:
    - 2D input stays 2D.
    - 3D scalar input is collapsed to one 2D plane for scaling, then returned as [1,H,W].
    - This matches the rest of this file, which later uses arr[0] for histology.
    """
    if sitk is None:
        raise ImportError("SimpleITK is required for returning SITK images.")

    is_h_sitk = isinstance(proc_Ih, sitk.Image)
    is_m_sitk = isinstance(proc_Mh, sitk.Image)

    h_arr = sitk.GetArrayFromImage(proc_Ih) if is_h_sitk else np.asarray(proc_Ih)
    m_arr = sitk.GetArrayFromImage(proc_Mh) if is_m_sitk else np.asarray(proc_Mh)

    # Keep output dimensionality compatible with downstream code.
    h_return_3d = (h_arr.ndim == 3)
    m_return_3d = (m_arr.ndim == 3)

    # Collapse to 2D for interpolation.
    # If your stack has multiple real slices and you want a different one,
    # change [0] to the desired slice index.
    h2d = _to_np2d(proc_Ih)
    m2d = (_to_np2d(proc_Mh) > 0).astype(np.float32)

    h2d = h2d.astype(np.float32)

    H, W = h2d.shape
    new_h = max(32, int(round(H * float(best_scale))))
    new_w = max(32, int(round(W * float(best_scale))))

    ht = torch.from_numpy(h2d)[None, None]
    mt = torch.from_numpy(m2d)[None, None]

    h_scaled_2d = F.interpolate(
        ht, size=(new_h, new_w), mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy().astype(np.float32)

    m_scaled_2d = (
        F.interpolate(mt, size=(new_h, new_w), mode="nearest")[0, 0].cpu().numpy() > 0.5
    ).astype(np.uint8)

    if h_return_3d:
        h_out = sitk.GetImageFromArray(h_scaled_2d[None, ...])
    else:
        h_out = sitk.GetImageFromArray(h_scaled_2d)

    if m_return_3d:
        m_out = sitk.GetImageFromArray(m_scaled_2d[None, ...])
    else:
        m_out = sitk.GetImageFromArray(m_scaled_2d)

    if is_h_sitk:
        h_out.SetOrigin(proc_Ih.GetOrigin())
        h_out.SetDirection(proc_Ih.GetDirection())
        sp = list(proc_Ih.GetSpacing())
        if preserve_physical_size:
            sp[0] = sp[0] * (W / float(new_w))
            sp[1] = sp[1] * (H / float(new_h))
        h_out.SetSpacing(tuple(sp))

    if is_m_sitk:
        m_out.SetOrigin(proc_Mh.GetOrigin())
        m_out.SetDirection(proc_Mh.GetDirection())
        sp = list(proc_Mh.GetSpacing())
        if preserve_physical_size:
            sp[0] = sp[0] * (W / float(new_w))
            sp[1] = sp[1] * (H / float(new_h))
        m_out.SetSpacing(tuple(sp))

    return h_out, m_out


def ransac_plane_fit_z_from_matches_robust(
    matches, score_key="score", z_min=None, z_max=None,
    dist_thresh=8.0, n_iter=2000, seed=0,
    knn_k=3, knn_mad_k=3.5,    # spatial isolation filter
):
    pts_zyx = np.array([m["mri_center_zyx"] for m in matches], dtype=np.float64)
    sc = np.array([m.get(score_key, 0.0) for m in matches], dtype=np.float64)

    valid = np.isfinite(sc) & np.isfinite(pts_zyx).all(axis=1) & (pts_zyx[:, 0] >= 0)
    if z_min is not None: valid &= (pts_zyx[:, 0] >= z_min)
    if z_max is not None: valid &= (pts_zyx[:, 0] <= z_max)

    valid_idx = np.where(valid)[0]
    pts = pts_zyx[valid]
    if len(pts) < 3:
        return dict(ok=False, reason="too_few_points")

    # ---------- Stage 1: spatial isolation in XY ----------
    # reject points with bad (x,y) whose (x,y) are too isolated from neighbors based on kNN distance + MAD threshold
    xy = np.stack([pts[:, 2], pts[:, 1]], axis=1)  # [N,2]
    N = xy.shape[0]
    k = min(knn_k, max(1, N - 1))

    if N > 3 and k >= 1:
        D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)
        dk = np.partition(D, kth=k-1, axis=1)[:, k-1]  # k-th nearest distance

        med = np.median(dk)
        mad = np.median(np.abs(dk - med)) + 1e-8
        thr = med + knn_mad_k * 1.4826 * mad
        spatial_keep = dk <= thr
    else:
        spatial_keep = np.ones(N, dtype=bool)

    pts2 = pts[spatial_keep]
    idx2 = valid_idx[spatial_keep]
    if len(pts2) < 3:
        return dict(ok=False, reason="too_few_after_spatial_filter")

    xyz = np.stack([pts2[:, 2], pts2[:, 1], pts2[:, 0]], axis=1)  # (x,y,z)
    N2 = xyz.shape[0]
    rng = np.random.default_rng(seed)

    # ---------- Stage 2: RANSAC plane ----------
    # reject points with bad z relative to the fitted plane
    best_inliers = None
    best_cnt = -1

    for _ in range(n_iter):
        s = rng.choice(N2, size=3, replace=False)
        p1, p2, p3 = xyz[s]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n = n / nn
        d = -np.dot(n, p1)

        dist = np.abs(xyz @ n + d)
        inl = dist <= dist_thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inl

    if best_inliers is None or best_cnt < 3:
        return dict(ok=False, reason="no_model")

    in_xyz = xyz[best_inliers]
    A = np.stack([in_xyz[:, 0], in_xyz[:, 1], np.ones(len(in_xyz))], axis=1)
    z = in_xyz[:, 2]
    a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]

    # map inliers back to original matches
    inlier_match_idx = idx2[best_inliers]

    return dict(
        ok=True,
        a=float(a), b=float(b), c=float(c),
        inlier_idx=inlier_match_idx,                 # indices in original matches
        inlier_count=int(len(inlier_match_idx)),
        total_count=int(len(valid_idx)),
        inlier_ratio=float(len(inlier_match_idx) / max(1, len(valid_idx))),
    )

def _extract_match_points(
    matches,
    score_key="score",
    z_min=None,
    z_max=None,
    return_idx=False,
    dtype=np.float32,
):
    """
    Extract valid MRI match points and scores from match dicts.
    Returns:
      - return_idx=False: (pts, sc)
      - return_idx=True : (pts, sc, idx_valid)
    where:
      pts: [N,3] as (z,y,x)
      sc : [N]
      idx_valid: indices in original `matches`
    """
    if len(matches) == 0:
        pts = np.empty((0, 3), dtype=dtype)
        sc = np.empty((0,), dtype=dtype)
        if return_idx:
            return pts, sc, np.empty((0,), dtype=np.int64)
        return pts, sc

    pts = np.array([m["mri_center_zyx"] for m in matches], dtype=dtype)  # [N,3] (z,y,x)
    sc = np.array([m.get(score_key, 0.0) for m in matches], dtype=dtype)

    valid = np.isfinite(sc) & np.isfinite(pts).all(axis=1) & (pts[:, 0] >= 0)
    if z_min is not None:
        valid &= (pts[:, 0] >= z_min)
    if z_max is not None:
        valid &= (pts[:, 0] <= z_max)

    pts = pts[valid]
    sc = sc[valid]

    if return_idx:
        idx_valid = np.flatnonzero(valid).astype(np.int64)
        return pts, sc, idx_valid
    return pts, sc

def knn_outlier_mask_plane_xy_iqr(
    pts_zyx,
    k=8,
    plane_mad_k=2.5,      # stricter: 2.0
    tukey_k=1.5,          # stricter: 1.0
    cc_radius_scale=2.5,  # stricter: 2.0
    min_component_size=6,
):
    """
    Robust outlier mask for pts [z,y,x].
    Returns:
      keep2 : (N,) bool
      r     : (N,) local in-plane kNN residual (larger = more outlier)
      fence : (lo, hi) Tukey fence for r on plane-inliers

    Steps:
      1) remove points far from fitted 3D plane (MAD)
      2) in-plane local kNN-IQR outlier removal
      3) keep largest connected component on plane
    """
    pts = np.asarray(pts_zyx, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=np.float64), (0.0, 0.0)

    # zyx -> xyz
    X = pts[:, [2, 1, 0]]

    # ---- Step 1: plane MAD filter ----
    X0 = X.mean(axis=0)
    Xc = X - X0
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    e1, e2, nrm = Vt[0], Vt[1], Vt[2]
    nrm = nrm / (np.linalg.norm(nrm) + 1e-12)

    d_plane = np.abs((X - X0) @ nrm)
    med = np.median(d_plane)
    mad = np.median(np.abs(d_plane - med))
    sigma = 1.4826 * mad + 1e-8
    thr_plane = med + plane_mad_k * sigma
    keep_plane = d_plane <= thr_plane

    if keep_plane.sum() < 3:
        return keep_plane.copy(), np.full(n, np.inf), (0.0, 0.0)

    # ---- Project to plane uv ----
    P = np.column_stack([(X - X0) @ e1, (X - X0) @ e2])

    # ---- Step 2: local kNN-IQR in plane ----
    idx_plane = np.where(keep_plane)[0]
    Pp = P[idx_plane]

    tree = cKDTree(Pp)
    kk = min(k + 1, len(Pp))
    dists, _ = tree.query(Pp, k=kk)
    if kk <= 1:
        r_plane = np.zeros(len(Pp), dtype=np.float64)
    else:
        r_plane = np.median(dists[:, 1:], axis=1)

    q1, q3 = np.percentile(r_plane, [25, 75])
    iqr = max(q3 - q1, 1e-8)
    lo = max(0.0, q1 - tukey_k * iqr)
    hi = q3 + tukey_k * iqr

    keep_knn_plane = (r_plane >= lo) & (r_plane <= hi)

    # map r back to full array
    r = np.full(n, np.inf, dtype=np.float64)
    r[idx_plane] = r_plane

    keep2 = np.zeros(n, dtype=bool)
    keep2[idx_plane[keep_knn_plane]] = True

    # ---- Step 3: keep largest connected component on plane ----
    idx_keep = np.where(keep2)[0]
    if len(idx_keep) >= max(3, min_component_size):
        Pk = P[idx_keep]

        # auto radius from current local scale
        r_keep = r[idx_keep]
        med_r = np.median(r_keep[np.isfinite(r_keep)]) if np.any(np.isfinite(r_keep)) else 0.0
        radius = max(1e-8, cc_radius_scale * med_r)

        tree_k = cKDTree(Pk)
        nbrs = tree_k.query_ball_point(Pk, r=radius)

        visited = np.zeros(len(Pk), dtype=bool)
        comps = []
        for i in range(len(Pk)):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in nbrs[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(comp)

        largest = max(comps, key=len)
        if len(largest) >= min_component_size:
            keep_cc = np.zeros(len(Pk), dtype=bool)
            keep_cc[np.array(largest, dtype=int)] = True

            keep2_new = np.zeros(n, dtype=bool)
            keep2_new[idx_keep[keep_cc]] = True
            keep2 = keep2_new

    return keep2, r, (lo, hi)

def coarse_patch_search_hog_mind(
    proc_Ih, proc_Mh,
    slab_Im, slab_Mm,
    best_scale,
    patch_matrix,
    z_min=None, z_max=None,
    common_kwargs=None,
):
    res_list = evaluate_fixed_scale_multi_patch(
        proc_Ih=proc_Ih,
        histo_mask=proc_Mh,
        slab_Im=slab_Im,
        slab_Mm=slab_Mm,
        best_scale=best_scale,
        patch_matrix=patch_matrix,
        z_min=None, z_max=None,
        common_kwargs=common_kwargs,
    )

    rows_sorted = compare_patch_results_ransac(
        res_list,
        reproj_thresh=12.0,  # tune to your pixel scale
        n_iter=2000,
        top_frac=0.30,
        seed=42
    )

    # provisional best
    best_row = rows_sorted[0]
    best_out = best_row["_out"]
    best_inliers = best_row["_ransac"]["inlier_idx"] if best_row["_ransac"]["ok"] else []

    print("Chosen config:", best_row["patch"], best_row["hist_stride"], best_row["mri_stride"])
    print("Inliers:", len(best_inliers))

    inlier_matches = [best_out["matches"][i] for i in best_inliers]
    z_center, z_min, z_max, z_radius = z_band_from_inliers(inlier_matches)
    print("z_center:", z_center, "z_min:", z_min, "z_max:", z_max, "z_radius:", z_radius)

    return best_row["patch"], best_row["hist_stride"], best_row["mri_stride"], z_center, z_min, z_max

def filter_outliers(
    all_matches,
    z_min=None,
    z_max=None,
):
    # Coarse pre-filter outliers.
    coarse_idx = ransac_plane_fit_z_from_matches_robust(
        all_matches,
        z_min=z_min,
        z_max=z_max,
        # dist_thresh=5.0,      # For H&E NB_C-Block
        dist_thresh=20.0,       # For others
        knn_k=3,
        knn_mad_k=3.5,
    )

    # pts_all, _, idx_all = _extract_match_points(all_matches, z_min=z_min, z_max=z_max, return_idx=True)
    # xy = np.stack([pts_all[:,2], pts_all[:,1]], 1).astype(np.float64)
    # D = np.sqrt(((xy[:,None,:]-xy[None,:,:])**2).sum(2)); np.fill_diagonal(D, np.inf)
    # for kk in (3, 5, 8):
    #     dk = np.partition(D, kk-1, axis=1)[:, kk-1]
    #     med = np.median(dk); mad = np.median(np.abs(dk-med))+1e-8
    #     print(f"k={kk} thr(mad_k=2.5)={med+2.5*1.4826*mad:8.2f}  top5 dk={np.sort(dk)[-5:].round(1)}")

    # a,b,c = coarse_idx["a"], coarse_idx["b"], coarse_idx["c"]
    # dz = np.abs(a*pts_all[:,2] + b*pts_all[:,1] + c - pts_all[:,0])
    # print("worst |z residual| to RANSAC plane:", np.sort(dz)[-5:].round(2))


    coarse_matches = [all_matches[i] for i in coarse_idx["inlier_idx"]]

    pts, sc, idx_valid = _extract_match_points(
        all_matches, z_min=z_min, z_max=z_max, return_idx=True
    )



    inlier_set = set(coarse_idx["inlier_idx"])
    inlier_mask = np.array([i in inlier_set for i in idx_valid], dtype=bool)

    pts_in = pts[inlier_mask]
    sc_in  = sc[inlier_mask]
    idx_in = idx_valid[inlier_mask]

    # Stricter outlier cleanup
    clean_idx, _, _ = knn_outlier_mask_plane_xy_iqr(
        pts_in, k=8, plane_mad_k=3.0, tukey_k=3.0, cc_radius_scale=3.0        # For others
        # pts_in, k=3, plane_mad_k=2.0, tukey_k=1.5, cc_radius_scale=2.0          # For H&E NB_C-Block
    )
    pts_clean = pts_in[clean_idx]
    sc_clean  = sc_in[clean_idx]
    idx_clean = idx_in[clean_idx]
    print("removed:", (~clean_idx).sum(), "kept:", clean_idx.sum())

    clean_matches = [all_matches[i] for i in idx_clean]

    return clean_matches, coarse_matches, pts_clean, sc_clean

# Visualization Debugger
# def show_matched_pairs(matches, hist_img, mri_img, patch=64, ncols=3, max_pairs=None, sort_by_score=True, save_file=None):
    
#     vals = [m for m in matches if m["mri_center_zyx"][0] >= 0]
#     if len(vals) == 0:
#         print("No valid matches.")
#         return

#     if sort_by_score:
#         vals = sorted(vals, key=lambda d: d["score"], reverse=True)
#     if max_pairs is not None:
#         vals = vals[:max_pairs]

#     n = len(vals)
#     rows = math.ceil(n / ncols)
#     fig, axes = plt.subplots(rows, ncols * 2, figsize=(6 * ncols, 3 * rows))
#     axes = np.array(axes).reshape(rows, ncols * 2)

#     # hide all
#     for r in range(rows):
#         for c in range(ncols * 2):
#             axes[r, c].axis("off")

#     for k, m in enumerate(vals):
#         r = k // ncols
#         g = k % ncols
#         c0 = 2 * g
#         c1 = c0 + 1

#         hy0, hx0 = m["hist_topleft_yx"]
#         mz, my0, mx0 = m["mri_topleft_zyx"]

#         hp = hist_img[hy0:hy0 + patch, hx0:hx0 + patch]
#         mp = mri_img[mz, my0:my0 + patch, mx0:mx0 + patch]

#         axes[r, c0].imshow(np.flipud(hp), cmap="gray")
#         axes[r, c0].set_title(f"Hist idx={m['hist_index']} center={m['hist_center_yx']}", fontsize=9)
#         axes[r, c0].axis("off")

#         axes[r, c1].imshow(np.flipud(mp), cmap="gray")
#         axes[r, c1].set_title(f"MRI {m['mri_center_zyx']} score={m['score']:.3f}", fontsize=9)
#         axes[r, c1].axis("off")

#     plt.tight_layout()

#     if save_file is not None:
#         os.makedirs(os.path.dirname(save_file), exist_ok=True) if os.path.dirname(save_file) else None
#         fig.savefig(save_file, dpi=200, bbox_inches="tight")
#         print(f"Saved figure to: {save_file}")

#     plt.show()
def show_matched_pairs(
    matches,
    hist_img,
    mri_img,
    patch=64,
    ncols=3,
    max_pairs=None,
    sort_by_score=True,
    save_file=None,
    hist_mask=None,
    mri_mask=None,
):
    vals = [m for m in matches if m["mri_center_zyx"][0] >= 0]
    if len(vals) == 0:
        print("No valid matches.")
        return

    if sort_by_score:
        vals = sorted(vals, key=lambda d: d["score"], reverse=True)
    if max_pairs is not None:
        vals = vals[:max_pairs]

    show_overlap = (hist_mask is not None) and (mri_mask is not None)
    panels_per_pair = 3 if show_overlap else 2

    n = len(vals)
    rows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        rows,
        ncols * panels_per_pair,
        figsize=(3 * panels_per_pair * ncols, 3 * rows)
    )
    axes = np.array(axes).reshape(rows, ncols * panels_per_pair)

    for r in range(rows):
        for c in range(ncols * panels_per_pair):
            axes[r, c].axis("off")

    for k, m in enumerate(vals):
        r = k // ncols
        g = k % ncols

        c0 = panels_per_pair * g
        c1 = c0 + 1
        c2 = c0 + 2 if show_overlap else None

        hy0, hx0 = m["hist_topleft_yx"]
        mz, my0, mx0 = m["mri_topleft_zyx"]

        hp = hist_img[hy0:hy0 + patch, hx0:hx0 + patch]
        mp = mri_img[mz, my0:my0 + patch, mx0:mx0 + patch]

        axes[r, c0].imshow(np.flipud(hp), cmap="gray")
        axes[r, c0].set_title(
            f"Hist idx={m['hist_index']} center={m['hist_center_yx']}",
            fontsize=9
        )
        axes[r, c0].axis("off")

        axes[r, c1].imshow(np.flipud(mp), cmap="gray")
        axes[r, c1].set_title(
            f"MRI {m['mri_center_zyx']} score={m['score']:.3f}",
            fontsize=9
        )
        axes[r, c1].axis("off")

        if show_overlap:
            hm = hist_mask[hy0:hy0 + patch, hx0:hx0 + patch] > 0
            mm = mri_mask[mz, my0:my0 + patch, mx0:mx0 + patch] > 0

            overlap = hm & mm
            overlap_frac = overlap.mean()

            axes[r, c2].imshow(np.flipud(overlap.astype(np.uint8) * 255), cmap="gray", vmin=0, vmax=255)
            axes[r, c2].set_title(f"Overlap frac={overlap_frac:.3f}", fontsize=9)
            axes[r, c2].axis("off")

    plt.tight_layout()

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True) if os.path.dirname(save_file) else None
        fig.savefig(save_file, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_file}")

    plt.show()



def fit_plane_z_ax_by_c(pts_zyx, weights=None):
    z = pts_zyx[:, 0].astype(np.float64)
    y = pts_zyx[:, 1].astype(np.float64)
    x = pts_zyx[:, 2].astype(np.float64)
    A = np.stack([x, y, np.ones_like(x)], axis=1)

    if weights is not None:
        w = np.sqrt(np.clip(weights, 1e-8, None))
        Aw = A * w[:, None]
        zw = z * w
        a, b, c = np.linalg.lstsq(Aw, zw, rcond=None)[0]
    else:
        a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]

    return float(a), float(b), float(c)

def _to_zyx_volume(vol):
    if sitk is not None and isinstance(vol, sitk.Image):
        arr = sitk.GetArrayFromImage(vol)  # [Z,Y,X]
    else:
        arr = np.asarray(vol)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume [Z,Y,X], got shape={arr.shape}")
    return arr.astype(np.float32)

def _norm01_np(a, p1=1, p99=99):
    lo, hi = np.percentile(a, p1), np.percentile(a, p99)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)

def _make_xz_canvas(vol_zyx, z_min, z_max, mode="mip"):
    Z, Y, X = vol_zyx.shape
    z_min = int(max(0, z_min))
    z_max = int(min(Z - 1, z_max))
    band = vol_zyx[z_min:z_max + 1]  # [zb,Y,X]

    if mode == "mean":
        proj = band.mean(axis=1)  # [zb,X]
    else:
        proj = band.max(axis=1)   # [zb,X] mip over Y

    canvas = np.zeros((Z, X), dtype=np.float32)  # full Z range
    canvas[z_min:z_max + 1, :] = proj
    return canvas

def _make_yz_canvas(vol_zyx, z_min, z_max, mode="mip"):
    Z, Y, X = vol_zyx.shape
    z_min = int(max(0, z_min))
    z_max = int(min(Z - 1, z_max))
    band = vol_zyx[z_min:z_max + 1]  # [zb,Y,X]

    if mode == "mean":
        proj = band.mean(axis=2)  # [zb,Y]
    else:
        proj = band.max(axis=2)   # [zb,Y] mip over X

    canvas = np.zeros((Z, Y), dtype=np.float32)  # full Z range
    canvas[z_min:z_max + 1, :] = proj
    return canvas

@torch.no_grad()
def resample_mri_on_plane_gpu(
    mri_vol, a, b, c,               # plane: z = a*x + b*y + c
    x_range=None, y_range=None,     # (min,max), in voxel index
    step=1.0,
    device="cuda:0",
    padding_mode="zeros"
):
    vol = _to_zyx_volume(mri_vol)
    Z, Y, X = vol.shape

    if x_range is None: x_range = (0, X - 1)
    if y_range is None: y_range = (0, Y - 1)

    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])

    xs = np.arange(x0, x1 + 1e-6, step, dtype=np.float32)
    ys = np.arange(y0, y1 + 1e-6, step, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)  # [H,W]
    ZZ = a * XX + b * YY + c

    valid = (ZZ >= 0) & (ZZ <= (Z - 1))

    # normalize to [-1, 1] for grid_sample (align_corners=True)
    xn = 2.0 * (XX / max(X - 1, 1)) - 1.0
    yn = 2.0 * (YY / max(Y - 1, 1)) - 1.0
    zn = 2.0 * (ZZ / max(Z - 1, 1)) - 1.0

    grid = np.stack([xn, yn, zn], axis=-1)              # [H,W,3] (x,y,z)
    grid_t = torch.from_numpy(grid).to(device)[None, None]  # [1,1,H,W,3]

    vol_t = torch.from_numpy(vol).to(device)[None, None]    # [1,1,Z,Y,X]

    samp = F.grid_sample(
        vol_t, grid_t, mode="bilinear",
        padding_mode=padding_mode, align_corners=True
    )  # [1,1,1,H,W]

    plane_img = samp[0, 0, 0].detach().cpu().numpy()
    plane_img[~valid] = np.nan
    return plane_img, XX, YY, ZZ, valid

def visualize_roi_from_matches(
    matches,
    mri_vol,
    z_min=None,
    z_max=None,
    score_key="score",
    max_points=50000,
    plane_stride=16,
    mode="mip",           # "mip" or "mean" for XZ/YZ background
    elev=20,
    azim=35,
    use_score_weight=True,
    figsize=(18, 5),
    save_file=None,
):
    """
    One figure, 3 columns:
      [0] 3D points + fitted plane
      [1] XZ overlay
      [2] YZ overlay
    """
    # expects your existing helpers:
    # _to_zyx_volume, _extract_match_points, _fit_plane_z_eq_ax_by_c,
    # _make_xz_canvas, _make_yz_canvas
    vol = _to_zyx_volume(mri_vol)  # [Z,Y,X]
    Z, Y, X = vol.shape

    if z_min is None:
        z_min = 0
    if z_max is None:
        z_max = Z - 1
    z_min = int(max(0, z_min))
    z_max = int(min(Z - 1, z_max))

    pts, sc = _extract_match_points(matches, score_key=score_key, z_min=z_min, z_max=z_max)
    if len(pts) < 3:
        print("Not enough valid points for visualization.")
        return None

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, sc = pts[idx], sc[idx]

    # weighted plane fit
    w = None
    if use_score_weight:
        s = sc - np.min(sc)
        w = s / (np.max(s) + 1e-8)
    a, b, c = fit_plane_z_ax_by_c(pts, weights=w)

    # plane mesh for 3D panel
    xx = np.arange(0, X, plane_stride, dtype=np.float32)
    yy = np.arange(0, Y, plane_stride, dtype=np.float32)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.clip(a * XX + b * YY + c, 0, Z - 1)

    # XZ/YZ canvases
    def _norm01_img(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)

    canvas_xz = _norm01_img(_make_xz_canvas(vol, z_min, z_max, mode=mode))
    canvas_yz = _norm01_img(_make_yz_canvas(vol, z_min, z_max, mode=mode))

    z, y, x = pts[:, 0], pts[:, 1], pts[:, 2]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0])

    # (1) 3D
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    p0 = ax0.scatter(x, y, z, c=sc, s=8, cmap="turbo", alpha=0.9)
    ax0.plot_surface(XX, YY, ZZ, alpha=0.28, color="cyan", linewidth=0, antialiased=False)
    ax0.set_xlim(0, X - 1); ax0.set_ylim(0, Y - 1); ax0.set_zlim(0, Z - 1)
    ax0.set_xlabel("X"); ax0.set_ylabel("Y"); ax0.set_zlabel("Z")
    ax0.set_title("3D points + fitted plane")
    ax0.view_init(elev=elev, azim=azim)

    # (2) XZ
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(canvas_xz, cmap="gray", origin="lower", extent=[0, X - 1, 0, Z - 1], aspect="auto")
    p1 = ax1.scatter(x, z, c=sc, s=10, cmap="turbo", alpha=0.9)
    ax1.set_xlim(0, X - 1); ax1.set_ylim(0, Z - 1)
    ax1.set_xlabel("X"); ax1.set_ylabel("Z")
    ax1.set_title(f"XZ overlay ({mode})")

    # (3) YZ
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(canvas_yz, cmap="gray", origin="lower", extent=[0, Y - 1, 0, Z - 1], aspect="auto")
    p2 = ax2.scatter(y, z, c=sc, s=10, cmap="turbo", alpha=0.9)
    ax2.set_xlim(0, Y - 1); ax2.set_ylim(0, Z - 1)
    ax2.set_xlabel("Y"); ax2.set_ylabel("Z")
    ax2.set_title(f"YZ overlay ({mode})")

    # one shared colorbar
    fig.colorbar(p2, ax=[ax0, ax1, ax2], shrink=0.8, label=score_key)

    if save_file is not None:
        import os
        save_dir = os.path.dirname(save_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_file, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_file}")

    print(f"Fitted plane: z = {a:.5f}*x + {b:.5f}*y + {c:.5f}")
    plt.show()

    return {"a": float(a), "b": float(b), "c": float(c), "n_points": int(len(pts))}

def visualize_matches_on_image(
    clean_matches,
    scaled_proc_Ih,
    pts_clean,        # [N,3] as (z,y,x)
    slab_Im,
    device,
    margin_xy=12,
    pad_xy=32,
    step=1.0,
    save_file=None,
):
    # ---------- Left panel: histology support ----------
    h = sitk.GetArrayFromImage(scaled_proc_Ih) if hasattr(scaled_proc_Ih, "GetSize") else np.asarray(scaled_proc_Ih)
    if h.ndim == 3:
        h = h[0]

    pts_hist = np.array([[m["hist_center_yx"][1], m["hist_center_yx"][0]] for m in clean_matches], dtype=np.float32)  # (x,y)
    if len(pts_hist) < 3:
        raise ValueError("Need >=3 clean_matches for histology hull.")

    hull_h = ConvexHull(pts_hist)
    poly_h = pts_hist[hull_h.vertices]
    path_h = Path(poly_h)

    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    inside_h = path_h.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(H, W)
    hist_cov = float(inside_h.mean())

    # ---------- Build plane image ----------
    a, b, c = fit_plane_z_ax_by_c(pts_clean)

    x_min = max(0, int(np.floor(pts_clean[:, 2].min() - margin_xy)))
    x_max = int(np.ceil(pts_clean[:, 2].max() + margin_xy))
    y_min = max(0, int(np.floor(pts_clean[:, 1].min() - margin_xy)))
    y_max = int(np.ceil(pts_clean[:, 1].max() + margin_xy))

    x0, x1 = x_min - pad_xy, x_max + pad_xy
    y0, y1 = y_min - pad_xy, y_max + pad_xy

    plane_img, XX, YY, ZZ, valid = resample_mri_on_plane_gpu(
        mri_vol=slab_Im,
        a=a, b=b, c=c,
        x_range=(x0, x1),
        y_range=(y0, y1),
        step=step,
        device=device,
    )

    # ---------- Right panel: plane support ----------
    pts_xy = np.array(
        [[m["mri_center_zyx"][2], m["mri_center_zyx"][1]] for m in clean_matches if m["mri_center_zyx"][0] >= 0],
        dtype=np.float32
    )
    if len(pts_xy) < 3:
        raise ValueError("Need >=3 valid MRI points for plane hull.")

    Hp, Wp = plane_img.shape
    px0, py0 = float(XX[0, 0]), float(YY[0, 0])
    sx = float(XX[0, 1] - XX[0, 0]) if Wp > 1 else 1.0
    sy = float(YY[1, 0] - YY[0, 0]) if Hp > 1 else 1.0

    px = (pts_xy[:, 0] - px0) / sx
    py = (pts_xy[:, 1] - py0) / sy
    pts_px = np.stack([px, py], axis=1)

    in_img = (px >= 0) & (px < Wp) & (py >= 0) & (py < Hp)
    pts_px_in = pts_px[in_img]
    if len(pts_px_in) < 3:
        raise ValueError("Too few points inside plane_img.")

    hull_p = ConvexHull(pts_px_in)
    poly_p = pts_px_in[hull_p.vertices]
    path_p = Path(poly_p)

    yyg, xxg = np.mgrid[0:Hp, 0:Wp]
    inside_p = path_p.contains_points(np.c_[xxg.ravel(), yyg.ravel()]).reshape(Hp, Wp)

    plane_cov_all = float(inside_p.mean())
    plane_cov_valid = float((inside_p & np.isfinite(plane_img)).sum() / max(np.isfinite(plane_img).sum(), 1))

    # ---------- One figure, 2 columns ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Histology panel
    axes[0].imshow(h, cmap="gray", origin="lower")
    axes[0].scatter(pts_hist[:, 0], pts_hist[:, 1], s=8, c="lime")
    axes[0].plot(np.r_[poly_h[:, 0], poly_h[0, 0]], np.r_[poly_h[:, 1], poly_h[0, 1]], c="cyan", lw=1.2)
    axes[0].set_title(f"Histology support (hull cov={hist_cov:.3f})")
    axes[0].axis("off")

    # Plane panel
    axes[1].imshow(np.nan_to_num(plane_img, nan=0.0), cmap="gray", origin="lower")
    axes[1].scatter(pts_px_in[:, 0], pts_px_in[:, 1], s=8, c="lime")
    axes[1].plot(np.r_[poly_p[:, 0], poly_p[0, 0]], np.r_[poly_p[:, 1], poly_p[0, 1]], c="cyan", lw=1.2)
    axes[1].set_title(f"Plane support (all={plane_cov_all:.3f}, valid={plane_cov_valid:.3f})")
    axes[1].axis("off")

    plt.tight_layout()

    if save_file is not None:
        import os
        save_dir = os.path.dirname(save_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_file, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_file}")

    plt.show()

    print(f"hull coverage (hist): {hist_cov:.6f}")
    print(f"hull coverage (plane all pixels): {plane_cov_all:.6f}")
    print(f"hull coverage (plane valid pixels): {plane_cov_valid:.6f}")
    print(f"plane: z = {a:.6f}*x + {b:.6f}*y + {c:.6f}")

    return {
        "plane_img": plane_img,
        "x0": int(x0),
        "x1": int(x1),
        "y0": int(y0),
        "y1": int(y1)
    }


def check_roi(
    slab_Im,
    scaled_proc_Ih,
    pts_clean,
    sc_clean,
    all_matches,
    best_patch,
    coarse_matches,
    clean_matches,
    z_min,
    z_max,
    save_file=None,
    save_file2=None,
    save_file3=None,
    device="cuda:0",
):
    # show_matched_pairs(all_matches["matches"], all_matches["hist_img"], all_matches["mri_img"], patch=best_patch, ncols=3)
    # show_matched_pairs(coarse_matches, all_matches["hist_img"], all_matches["mri_img"], patch=best_patch, ncols=3)
    # show_matched_pairs(clean_matches, all_matches["hist_img"], all_matches["mri_img"], patch=best_patch, ncols=3, save_file=save_file)
    show_matched_pairs(
        clean_matches,
        all_matches["hist_img"],
        all_matches["mri_img"],
        patch=best_patch,
        ncols=2,
        save_file=save_file,
        hist_mask=all_matches["hist_mask"],
        mri_mask=all_matches["mri_mask"],
    )



    viz = visualize_roi_from_matches(
        matches=clean_matches,   # or inlier_matches
        mri_vol=slab_Im,
        z_min=z_min,
        z_max=z_max,
        mode="mip",
        save_file=save_file2,
    )

    out = visualize_matches_on_image(
        clean_matches=clean_matches,
        scaled_proc_Ih=scaled_proc_Ih,
        pts_clean=pts_clean,
        slab_Im=slab_Im,
        device=device,
        save_file=save_file3,
    )
    plane_img = out["plane_img"]
    x0, x1, y0, y1 = out["x0"], out["x1"], out["y0"], out["y1"]

    return plane_img, x0, x1, y0, y1

def pad2d_to_canvas(img2d, Hc, Wc, oy, ox, fill=0.0):
    out = np.full((Hc, Wc), fill, dtype=img2d.dtype)
    h, w = img2d.shape
    out[oy:oy+h, ox:ox+w] = img2d
    return out

def create_common_canvas(scaled_proc_Ih, scaled_proc_Mh, plane_img):
    arr_h = sitk.GetArrayFromImage(scaled_proc_Ih).astype(np.float32)
    hist2d = arr_h[0] if arr_h.ndim == 3 else arr_h

    arr_mh = sitk.GetArrayFromImage(scaled_proc_Mh).astype(np.float32)
    mask2d = arr_mh[0] if arr_mh.ndim == 3 else arr_mh

    mri2d = plane_img.astype(np.float32)
    Hh, Wh = hist2d.shape
    Hm, Wm = mri2d.shape
    Hc = max(Hh, Hm)
    Wc = max(Wh, Wm)
    # choose placement (centered here)
    oy_h = (Hc - Hh) // 2
    ox_h = (Wc - Wh) // 2
    oy_m = (Hc - Hm) // 2
    ox_m = (Wc - Wm) // 2
    hist_common = pad2d_to_canvas(hist2d, Hc, Wc, oy_h, ox_h, fill=0.0)
    mask_common = pad2d_to_canvas(mask2d, Hc, Wc, oy_h, ox_h, fill=0.0)
    mri_common  = pad2d_to_canvas(mri2d, Hc, Wc, oy_m, ox_m, fill=0.0)

    print("hist original size:", (Hh, Wh))
    print("mri original size:", (Hm, Wm))
    print("common canvas:", (Hc, Wc))
    print("hist offset (oy,ox):", (oy_h, ox_h))
    print("mri offset  (oy,ox):", (oy_m, ox_m))

    return Hc, Wc, oy_h, ox_h, oy_m, ox_m, hist_common, mri_common, mask_common

def check_common_canvas(
    hist_common,
    mri_common,
    save_file=None,
    title_left="Hist Common",
    title_right="MRI Common",
    figsize=(12, 6),
    cmap="gray",
):
    def _to_2d(x):
        try:
            import SimpleITK as sitk
            if isinstance(x, sitk.Image):
                a = sitk.GetArrayFromImage(x)
            else:
                a = np.asarray(x)
        except Exception:
            a = np.asarray(x)

        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2:
            raise ValueError(f"Expected 2D or [1,H,W], got shape={a.shape}")
        return a.astype(np.float32)

    def _norm01(a):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)

    h = _norm01(_to_2d(hist_common))
    m = _norm01(_to_2d(mri_common))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(h, cmap=cmap, origin="lower")
    axes[0].set_title(title_left)
    # axes[0].axis("off")

    axes[1].imshow(m, cmap=cmap, origin="lower")
    axes[1].set_title(title_right)
    # axes[1].axis("off")

    plt.tight_layout()

    if save_file is not None:
        save_dir = os.path.dirname(save_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_file, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_file}")

    plt.show()
