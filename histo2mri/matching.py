"""HOG + MIND late-fusion patch matching between histology and an MRI slab, with scale and patch-size search."""
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
import torch
from tqdm import tqdm
import inspect
from .features import _hog_from_unfolded, _mind_desc_from_patch_batch, _norm01_torch, _pairwise_masked_hog_similarity, _pairwise_masked_mind_similarity, _pairwise_masked_ngf_similarity, _prep_inputs, _sobel_xy, _to_np2d, _to_numpy_image, _unfold_hw


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


def match_patches(
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


def search_scale(
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
    matcher_fn=match_patches,
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


def rescale_histology(proc_Ih, proc_Mh, best_scale, preserve_physical_size=True):
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


def search_patch_config(
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

