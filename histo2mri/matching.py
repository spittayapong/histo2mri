"""Match histology patches to MRI-slab patches with HOG + MIND (+ NGF) late fusion.

Matching stage of the pipeline (after preprocessing, before outlier filtering): it searches the
histology scale (search_scale) and patch size/stride (search_patch_config), rescales the histology
(rescale_histology) and matches patches (match_patches). Histology positions are (y, x) pixel
indices of the matched 2D image; MRI positions are (z, y, x) voxel indices into the slab array.
"""
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
import torch
from tqdm import tqdm
import inspect
from .features import _hog_from_unfolded, _mind_desc_from_patch_batch, _norm01_torch, _pairwise_masked_hog_similarity, _pairwise_masked_mind_similarity, _pairwise_masked_ngf_similarity, _prep_inputs, _sobel_xy, _to_np2d, _to_numpy_image, _unfold_hw
import logging

logger = logging.getLogger(__name__)


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
    """Build the MRI patch bank used by the plain matcher (run_patch_matching_hog_mind_fusion_gpu).

    Every slice z of the slab in [z_min, z_max] is cut into mri_patch x mri_patch patches on a
    grid with step mri_stride. A patch is kept when its centre voxel lies inside the MRI mask and
    at least min_mri_mask_frac of its voxels are inside the mask. For each kept patch the bank
    stores a masked HOG descriptor (Sobel gradients, L2-normalised), the patch mask, the raw
    intensity patch (float16) and its top-left and centre positions. The plain matcher uses the
    raw patches to recompute HOG, MIND and NGF on pairwise intersection masks.
    It differs from build_mri_bank_hog_mind only in storing the raw patches as float16 (instead of
    float32) and in explicitly falling back to the CPU tensors when the GPU copy fails.

    Args:
        slab_Im: MRI slab image (SimpleITK image or array), 3D (z, y, x) or 2D (y, x).
        slab_Mm: MRI slab mask with the same shape as slab_Im; voxels > 0 are tissue.
        mri_patch: Patch side length in slab voxels.
        mri_stride: Grid step between patch top-left corners, in slab voxels.
        cell_size: HOG cell side length in voxels; must divide mri_patch.
        bins: Number of HOG orientation bins (signed, over 0 to 2*pi).
        min_mri_mask_frac: Minimum fraction of a patch's voxels that must be inside the mask.
        z_min: First slab slice to include (inclusive), or None for no lower limit.
        z_max: Last slab slice to include (inclusive), or None for no upper limit.
        device: torch device or device string; falls back to CPU if CUDA is not available.
        keep_mri_bank_on_gpu: If True and the device is CUDA, also copy the HOG descriptors and
            masks to the GPU (falls back to CPU on an out-of-memory RuntimeError).

    Returns:
        dict with
            mri_img, mri_msk: the slab as float32 / bool arrays of shape (Z, H, W);
            patch, mri_patch, mri_stride, cell_size, bins, min_mri_mask_frac, z_min, z_max:
                the build parameters;
            device_used: device string;
            m_hog_cpu: [Nm, Fh] float16 HOG descriptors on CPU;
            m_mk_cpu: [Nm, patch*patch] bool patch masks on CPU;
            m_patches: [Nm, patch, patch] float16 raw intensity patches on CPU;
            m_top, m_center: [Nm, 3] int32 arrays of (z, y, x) top-left and centre voxels;
            Nm: number of bank patches;
            bank_on_gpu: whether m_hog_bank / m_mk_bank are on the GPU;
            m_hog_bank, m_mk_bank: the descriptors and masks used for matching (GPU or CPU).

    Raises:
        RuntimeError: If no MRI patch passes the mask filter.
    """
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

    for z in tqdm(range(Z), desc="Build MRI bank", disable=not logger.isEnabledFor(logging.DEBUG)):
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
    m_patches = torch.cat(m_patch_list, dim=0)     # CPU [Nm,p,p] float16
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
    """Match each histology patch to its best MRI-slab patch (two-stage HOG + MIND + NGF fusion).

    Histology patches are taken on a grid with step hist_stride and kept when at least
    min_histo_mask_frac of their pixels are inside the histology mask. Stage 1 (shortlist): the
    dot product of masked HOG descriptors is computed against the whole MRI bank, in chunks of
    mri_chunk; pairs whose masks overlap in fewer than min_overlap_frac * patch**2 pixels are
    excluded, and the topk_hog best MRI patches are kept per histology patch. Stage 2 (rerank):
    for each shortlisted pair, HOG, MIND and NGF similarities are recomputed from the raw patches
    restricted to the intersection of the two masks; each similarity is min-max normalised over
    that histology patch's candidates, and the weighted mean
    (w_hog*HOG + w_mind*MIND + w_ngf*NGF) / (w_hog + w_mind + w_ngf) picks the best candidate.
    Because of the per-candidate-set normalisation, the score lies in [0, 1] and measures how
    strongly the descriptors agree on one candidate, not an absolute similarity.
    No padding is applied here; match_patches wraps this function and pads the borders.

    Args:
        proc_Ih: Prepared histology image (SimpleITK image or array), 2D (y, x) or (1, y, x).
        histo_mask: Histology mask with the same shape; pixels > 0 are tissue.
        slab_Im: MRI slab image, 3D (z, y, x) or 2D. Only used if mri_bank is None.
        slab_Mm: MRI slab mask with the same shape. Only used if mri_bank is None.
        hist_patch: Histology patch side length in pixels; must equal mri_patch.
        hist_stride: Grid step between histology patches, in pixels.
        mri_patch: MRI patch side length in voxels.
        mri_stride: Grid step between MRI patches, in voxels.
        cell_size: HOG cell side length; must divide the patch size.
        bins: Number of HOG orientation bins.
        min_histo_mask_frac: Minimum in-mask fraction for a histology patch to be matched.
        min_mri_mask_frac: Minimum in-mask fraction for an MRI patch to enter the bank.
        min_overlap_frac: Minimum mask overlap (fraction of patch**2 pixels) for a candidate pair.
        topk_hog: Number of HOG shortlist candidates per histology patch.
        w_hog: Fusion weight of the pairwise-masked HOG similarity.
        w_mind: Fusion weight of the pairwise-masked MIND-SSC similarity.
        w_ngf: Fusion weight of the pairwise-masked NGF similarity; 0 skips the NGF computation.
        mind_radius: MIND patch (box filter) radius.
        mind_dilation: MIND neighbour offset dilation.
        hist_batch: Number of histology patches processed per batch.
        mri_chunk: Number of MRI bank patches per chunk in the shortlist stage.
        z_min: First slab slice for the bank (inclusive); only used if mri_bank is None.
        z_max: Last slab slice for the bank (inclusive); only used if mri_bank is None.
        device: torch device or device string; falls back to CPU if CUDA is not available.
        keep_mri_bank_on_gpu: Passed to build_mri_bank_hog_mind_ngf when the bank is built here.
        mri_bank: Optional precomputed bank from build_mri_bank_hog_mind_ngf. Its patch,
            mri_stride, cell_size and bins must match the arguments.

    Returns:
        dict with
            matches: list with one match dict per kept histology patch, with keys hist_index
                (index among kept patches), hist_center_yx, hist_topleft_yx (histology pixels),
                mri_center_zyx, mri_topleft_zyx (slab voxels), mri_coords (same as
                mri_center_zyx) and score (fused score; -inf with MRI positions (-1, -1, -1) if no
                candidate survived);
            hist_img, hist_mask: the histology image (float32) and mask (bool) that were matched;
            mri_img, mri_mask: the slab arrays stored in the bank, (Z, H, W);
            device_used: device string;
            metric: "HOG+MIND+NGF_fusion".

    Raises:
        ValueError: If all fusion weights are zero or the bank parameters do not match.
        RuntimeError: If no histology patch passes the mask filter.
    """
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

    for hs in tqdm(range(0, Nh, hist_batch), desc="Match HOG+MIND+NGF", disable=not logger.isEnabledFor(logging.DEBUG)):
        he = min(hs + hist_batch, Nh)
        Hb = he - hs

        hh = h_hog[hs:he].to(torch.float32)             # [Hb,Fh] dev
        hmasks = h_mk_u[hs:he]                          # [Hb,P] bool dev
        hmasks_f = hmasks.to(torch.float32)             # [Hb,P] float dev
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

            # Pixels valid in both patches; all rerank similarities use only these.
            pair_mask = (hmask.unsqueeze(0) & md_sel_mask.bool())   # [K,P]

            # Drop candidates whose intersection is too small. This also removes shortlist
            # entries that stage 1 had set to -inf but that topk still returned.
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
    """Resize a 2D histology image and its mask by a uniform scale factor.

    The image is resized bilinearly and the mask with nearest-neighbour interpolation. Each output
    side is round(side * scale), but at least 32 pixels.

    Args:
        hist_img_2d: Histology image, numpy array (H, W).
        hist_mask_2d: Histology mask, numpy array (H, W); values > 0 are tissue.
        scale: Scale factor (> 1 enlarges).

    Returns:
        Tuple (image, mask): float32 array (H', W') and bool array (H', W').
    """
    H, W = hist_img_2d.shape
    new_h = max(32, int(round(H * scale)))
    new_w = max(32, int(round(W * scale)))

    x = torch.from_numpy((hist_img_2d).astype(np.float32))[None, None]
    m = torch.from_numpy((hist_mask_2d > 0).astype(np.float32))[None, None]

    xr = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)
    mr = F.interpolate(m, size=(new_h, new_w), mode="nearest")[0, 0].cpu().numpy() > 0.5
    return xr, mr


def robust_match_score(matches, top_frac=0.30, min_keep=10):
    """Summarise a match set by the median score of its best matches.

    Takes the finite match scores, keeps the k highest with k = max(min_keep,
    ceil(top_frac * n)), and returns their median. This is the score maximised by search_scale and
    used to rank configurations in evaluate_fixed_scale_multi_patch.

    Args:
        matches: List of match dicts with a "score" key.
        top_frac: Fraction of the finite scores to keep.
        min_keep: Minimum number of scores to keep.

    Returns:
        float median of the top scores, or -inf if there is no finite score.
    """
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
    """Estimate the band of slab slices in which the histology section most likely lies.

    Uses the matches with a valid MRI position and a finite score. The z centre is the median
    slice of the best-scoring matches (score >= the score_quantile quantile, or the 10 best if
    fewer than 10 pass); the half-width is max(min_radius, ceil(k_mad * MAD)) of those slices.

    Args:
        matches: List of match dicts with "mri_center_zyx" and "score".
        score_quantile: Score quantile above which matches are used.
        k_mad: Multiplier of the median absolute deviation of z.
        min_radius: Minimum half-width of the band, in slices.

    Returns:
        Tuple (z_center, z_min, z_max) in slab slice indices; z_min is clipped at 0, z_max is not
        clipped. (None, None, None) if there is no valid match.
    """
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
    """Return the end padding after which a patch grid with this stride covers all of length L."""
    if L < patch:
        return patch - L
    rem = (L - patch) % stride
    return 0 if rem == 0 else (stride - rem)


def _pad2d_rb(arr2d, patch, stride):
    """Zero-pad a 2D array at the bottom and right; return (padded, (pad_y, pad_x))."""
    ph = _rb_pad_len(arr2d.shape[0], patch, stride)
    pw = _rb_pad_len(arr2d.shape[1], patch, stride)
    if ph == 0 and pw == 0:
        return arr2d, (0, 0)
    out = np.pad(arr2d, ((0, ph), (0, pw)), mode="constant", constant_values=0)
    return out, (ph, pw)


def _pad3d_zyx_rb(vol, patch, stride):
    """Zero-pad a (z, y, x) volume at the bottom (y) and right (x); return (padded, (pad_y, pad_x))."""
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
    """Match histology patches to the MRI slab, including patches at the right and bottom borders.

    A patch grid only reaches the far border when (side - patch) is a multiple of the stride. The
    histology image and mask are therefore zero-padded at the bottom and right (and up to at least
    one patch) before calling run_patch_matching_hog_mind_fusion_gpu (HOG shortlist, then
    pairwise-masked HOG + MIND + NGF rerank). Padding only at the bottom and right keeps all
    pixel coordinates the same. If mri_bank is None the slab and its mask are padded in y and x
    the same way and the matcher builds the bank; otherwise the given bank is used as is.
    This is the matcher used for the final patch matching and by search_patch_config.

    Args:
        proc_Ih: Histology image (SimpleITK image or array), 2D (y, x) or (1, y, x).
        histo_mask: Histology mask with the same shape; pixels > 0 are tissue.
        slab_Im: MRI slab image, (z, y, x) or 2D.
        slab_Mm: MRI slab mask with the same shape.
        hist_patch: Histology patch side length in pixels (must equal mri_patch).
        hist_stride: Histology patch grid step in pixels.
        mri_patch: MRI patch side length in voxels.
        mri_stride: MRI patch grid step in voxels.
        z_min: First slab slice for the bank (inclusive), or None.
        z_max: Last slab slice for the bank (inclusive), or None.
        mri_bank: Optional precomputed bank from build_mri_bank_hog_mind_ngf.
        **kwargs: Further arguments for run_patch_matching_hog_mind_fusion_gpu (descriptor
            settings, fusion weights, batch sizes, device).

    Returns:
        The matcher's output dict (see run_patch_matching_hog_mind_fusion_gpu) plus
        padding_info = {"hist_pad_bottom_right": (pad_y, pad_x), "mri_pad_bottom_right":
        (pad_y, pad_x) or None when a bank was given}.
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
    """Drop config keys that matcher_fn does not accept (all are kept if it accepts **kwargs)."""
    params = inspect.signature(matcher_fn).parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    out = dict(cfg)
    if accepts_var_kw:
        return out
    return {k: v for k, v in out.items() if k in params}


# ---------- MRI bank builder for the cached matcher ----------
def build_mri_bank_hog_mind(
    slab_Im, slab_Mm,
    mri_patch, mri_stride,
    cell_size, bins,
    min_mri_mask_frac,
    z_min, z_max,
    device,
    keep_mri_bank_on_gpu=True,
):
    """Build the MRI patch bank used by the cached matcher (search_scale).

    Same patch extraction and filtering as build_mri_bank_hog_mind_ngf: patches on a grid with
    step mri_stride in every slice in [z_min, z_max], kept when the centre voxel is in the mask
    and at least min_mri_mask_frac of the patch is in the mask. The bank stores masked HOG
    descriptors (float16), patch masks and raw patches (float32), from which the cached matcher
    computes MRI MIND descriptors on demand. Build it once and pass it to every call of
    run_patch_matching_hog_mind_fusion_gpu_cached that uses the same patch size and stride.

    Args:
        slab_Im: MRI slab image (SimpleITK image or array), 3D (z, y, x) or 2D (y, x).
        slab_Mm: MRI slab mask with the same shape; voxels > 0 are tissue.
        mri_patch: Patch side length in slab voxels.
        mri_stride: Grid step between patch top-left corners, in voxels.
        cell_size: HOG cell side length in voxels; must divide mri_patch.
        bins: Number of HOG orientation bins.
        min_mri_mask_frac: Minimum fraction of a patch's voxels inside the mask.
        z_min: First slice to include (inclusive), or None.
        z_max: Last slice to include (inclusive), or None.
        device: torch device or device string; falls back to CPU if CUDA is not available.
        keep_mri_bank_on_gpu: If True and the device is CUDA, also copy HOG descriptors and masks
            to the GPU (stays on CPU if that raises a RuntimeError).

    Returns:
        dict with the same keys, and the same dtypes, as build_mri_bank_hog_mind_ngf
        [Nm, patch, patch].

    Raises:
        RuntimeError: If no MRI patch passes the mask filter.
    """
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

    for z in tqdm(range(Z), desc="Build MRI bank", disable=not logger.isEnabledFor(logging.DEBUG)):
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
        # float16, as build_mri_bank_hog_mind_ngf stores them: this is the largest part of
        # the bank, and both readers cast it to float32 when they take a batch out of it.
        m_patch_list.append(pim.half().cpu())

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


# ---------- Cached HOG + MIND matcher (reusable MRI bank, used by the scale search) ----------
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
    """Match each histology patch to its best MRI patch with a cheaper HOG + MIND fusion.

    Faster variant of run_patch_matching_hog_mind_fusion_gpu for repeated calls against the same
    MRI bank (search_scale calls it once per candidate scale). The HOG shortlist stage is the
    same. The rerank differs from the plain path:
      * HOG and MIND descriptors are computed once per patch with that patch's own mask, not
        recomputed on the pairwise intersection mask; the HOG rerank similarity is the same
        descriptor dot product used for the shortlist;
      * MRI MIND descriptors are computed only for shortlisted patches, in batches of
        mind_compute_batch, and cached (float16, CPU) for the rest of this call;
      * there is no NGF term, and the score is w_hog*HOG + w_mind*MIND of the per-candidate-set
        min-max normalised similarities, not divided by the weight sum (in [0, 1] when the
        weights sum to 1);
      * the mask-overlap check is applied only in the shortlist stage.
    No padding is applied here; see run_patch_matching_hog_mind_fusion_gpu_with_border_cached.

    Args:
        proc_Ih: Histology image (SimpleITK image or array), 2D (y, x) or (1, y, x).
        histo_mask: Histology mask with the same shape; pixels > 0 are tissue.
        slab_Im: MRI slab image (z, y, x). Only used if mri_bank is None.
        slab_Mm: MRI slab mask. Only used if mri_bank is None.
        hist_patch: Histology patch side length in pixels; must equal mri_patch.
        hist_stride: Histology patch grid step in pixels.
        mri_patch: MRI patch side length in voxels.
        mri_stride: MRI patch grid step in voxels.
        cell_size: HOG cell side length.
        bins: Number of HOG orientation bins.
        min_histo_mask_frac: Minimum in-mask fraction for a histology patch to be matched.
        min_mri_mask_frac: Minimum in-mask fraction for an MRI bank patch (bank built here only).
        min_overlap_frac: Minimum mask overlap (fraction of patch**2) in the shortlist stage.
        topk_hog: Number of HOG shortlist candidates per histology patch.
        w_hog: Weight of the HOG similarity.
        w_mind: Weight of the MIND-SSC similarity.
        mind_radius: MIND patch (box filter) radius.
        mind_dilation: MIND neighbour offset dilation.
        hist_batch: Number of histology patches per batch.
        mri_chunk: Number of MRI bank patches per chunk in the shortlist stage.
        z_min: First slab slice for the bank (inclusive); only used if mri_bank is None.
        z_max: Last slab slice for the bank (inclusive); only used if mri_bank is None.
        device: torch device or device string; falls back to CPU if CUDA is not available.
        keep_mri_bank_on_gpu: Passed to build_mri_bank_hog_mind when the bank is built here.
        mind_compute_batch: Number of MRI patches per batch when filling the MIND cache.
        mri_bank: Optional bank from build_mri_bank_hog_mind; its patch and mri_stride must match.

    Returns:
        dict with matches, hist_img, hist_mask, mri_img, mri_mask and device_used as in
        run_patch_matching_hog_mind_fusion_gpu, and metric "HOG+MIND_fusion".

    Raises:
        ValueError: If the bank's patch size or stride does not match.
        RuntimeError: If no histology patch passes the mask filter.
    """
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
        """Return MIND descriptors [len(idxs), Fm] of the given bank patches, computing missing ones."""
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

    for hs in tqdm(range(0, Nh, hist_batch), desc="Match HOG+MIND", disable=not logger.isEnabledFor(logging.DEBUG)):
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


# ---------- Border-padded wrapper for the cached matcher ----------
def run_patch_matching_hog_mind_fusion_gpu_with_border_cached(
        proc_Ih, histo_mask, slab_Im, slab_Mm,
        hist_patch, hist_stride, mri_patch, mri_stride,
        z_min=None, z_max=None, mri_bank=None, **kwargs):
    """Run the cached HOG + MIND matcher with bottom/right border padding.

    Same padding as match_patches, but calls run_patch_matching_hog_mind_fusion_gpu_cached. If
    mri_bank is given, the slab arguments are not used (the bank must have been built from a
    slab padded for the same mri_patch and mri_stride, as search_scale does). This is the default
    matcher of search_scale.

    Args:
        proc_Ih: Histology image, 2D (y, x) or (1, y, x).
        histo_mask: Histology mask with the same shape.
        slab_Im: MRI slab image (z, y, x).
        slab_Mm: MRI slab mask.
        hist_patch: Histology patch side length in pixels (must equal mri_patch).
        hist_stride: Histology patch grid step in pixels.
        mri_patch: MRI patch side length in voxels.
        mri_stride: MRI patch grid step in voxels.
        z_min: First slab slice for a bank built here (inclusive), or None.
        z_max: Last slab slice for a bank built here (inclusive), or None.
        mri_bank: Optional bank from build_mri_bank_hog_mind.
        **kwargs: Further arguments for run_patch_matching_hog_mind_fusion_gpu_cached.

    Returns:
        The cached matcher's output dict plus padding_info, as in match_patches.
    """
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
    """Find the histology scale factor that best matches the MRI slab (coarse scale search).

    For each candidate scale the histology and mask are resized (resize_hist_pair) and matched
    against the slab with matcher_fn (by default the cached, border-padded HOG + MIND matcher).
    The MRI bank (build_mri_bank_hog_mind, all slices, slab padded for the MRI patch and stride)
    is built only once, because the MRI patch size and stride do not change across scales.
    The scale with the highest robust_match_score (median of the top top_frac_score match
    scores) is selected. The matcher config defaults to hist_patch=mri_patch=96, hist_stride=64,
    mri_stride=8; entries of common_kwargs override it.

    Args:
        proc_Ih: Prepared histology image (SimpleITK image or array), 2D or (1, y, x).
        histo_mask: Prepared histology mask with the same shape.
        slab_Im: MRI slab image (z, y, x).
        slab_Mm: MRI slab mask.
        scales: Candidate scale factors (duplicates removed, sorted); default 1.00 to 1.30 in
            steps of 0.05.
        top_frac_score: Fraction of best matches used by robust_match_score.
        common_kwargs: Matcher arguments (descriptor settings, weights, device, ...). With the
            default matcher every key is forwarded, so each must be a parameter of
            run_patch_matching_hog_mind_fusion_gpu_cached (w_ngf, for example, is not).
        matcher_fn: Matcher with the signature of
            run_patch_matching_hog_mind_fusion_gpu_with_border_cached; it receives mri_bank.
        verbose: Print progress and the per-scale record.

    Returns:
        Tuple (best, records):
            best: dict with score, scale, result (matcher output at that scale), z_center,
                z_min, z_max (slice band from estimate_z_band, clipped to the slab; None if not
                available);
            records: one dict per scale (scale, score, n_matches, status, z_center, z_min,
                z_max), sorted by score, best first.
    """
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

    # Default coarse patch configuration
    run_cfg = dict(
        hist_patch=96,
        hist_stride=64,
        mri_patch=96,
        mri_stride=8,
    )
    run_cfg.update(common_kwargs)  # user overrides defaults if needed
    run_cfg = _filter_cfg_for_matcher(run_cfg, matcher_fn)
    logger.debug(run_cfg)

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
            logger.debug(f'[scale={s:.3f}] running...')

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
            logger.debug(rec)

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
    """Run patch matching at a fixed histology scale for several patch configurations.

    The histology is resized once by best_scale. For each configuration in patch_matrix, an MRI
    bank (build_mri_bank_hog_mind_ngf, slab padded for that patch size and stride, kept on CPU)
    is built and matcher_fn (by default the plain HOG + MIND + NGF matcher match_patches) is run.
    Each configuration is scored with robust_match_score (top 30%). A configuration that raises
    an exception is reported and skipped. GPU and host memory are released after each
    configuration because the banks are large.

    Args:
        proc_Ih: Prepared histology image (SimpleITK image or array), 2D or (1, y, x).
        histo_mask: Prepared histology mask with the same shape.
        slab_Im: MRI slab image (z, y, x).
        slab_Mm: MRI slab mask.
        best_scale: Histology scale factor (from search_scale).
        patch_matrix: List of dicts with keys patch, hist_stride, mri_stride.
        z_min: First slab slice for the bank (inclusive), or None.
        z_max: Last slab slice for the bank (inclusive), or None.
        common_kwargs: Extra matcher arguments; cell_size, bins, min_mri_mask_frac and device
            are also used to build the bank.
        matcher_fn: Matcher that accepts mri_bank (default match_patches).

    Returns:
        List of result dicts sorted by robust_score (best first), each with patch, hist_stride,
        mri_stride, robust_score, n_matches and _out (matches, padding_info, device_used,
        metric).

    Raises:
        RuntimeError: If every configuration failed.
    """
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

            logger.debug(f"[patch={p}] robust_score={score:.4f}, n_matches={len(out['matches'])}, padding={compact_out['padding_info']}")

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
            logger.warning(f'[patch={p}] ERROR: {type(e).__name__}: {str(e)[:120]}')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if libc is not None:
                libc.malloc_trim(0)

    if len(results) == 0:
        raise RuntimeError("All patch configs failed.")

    results = sorted(results, key=lambda r: r["robust_score"], reverse=True)

    # Highest robust_score only; search_patch_config chooses by RANSAC consistency instead
    # (logged as "Chosen patch config").
    logger.info('Highest robust score: %s', {k: results[0][k] for k in ['patch', 'hist_stride', 'mri_stride', 'robust_score', 'n_matches']})
    return results


# ---------- Score stability ----------
def score_stability(matches, top_frac=0.30, min_keep=10):
    """Summarise the distribution of match scores, overall and for the best matches.

    The top set is the k highest finite scores, with k = max(min_keep, ceil(top_frac * n)).

    Args:
        matches: List of match dicts with a "score" key.
        top_frac: Fraction of finite scores in the top set.
        min_keep: Minimum size of the top set.

    Returns:
        dict with n (number of finite scores), median, iqr, p10, p90 of all finite scores, and
        median_top, iqr_top of the top set. All statistics are NaN if there is no finite score.
    """
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
    """Least-squares 2D affine M (2x3) with dst = M @ [x, y, 1]; src_xy and dst_xy are [N, 2]."""
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
    """Apply a 2x3 affine M to points src_xy [N, 2]; return [N, 2]."""
    return src_xy @ M[:, :2].T + M[:, 2][None, :]


def ransac_affine_from_matches(
    matches,
    reproj_thresh=12.0,
    n_iter=2000,
    min_samples=3,
    seed=0,
    z_mad_k=3.0,
    z_min_tol=2.0,
):
    """Fit a 2D affine map from histology (x, y) to MRI in-plane (x, y) with RANSAC.

    Used to judge how geometrically consistent a match set is. Matches without an MRI position
    are skipped. Matches whose slab slice z is far from the median slice (more than
    max(z_mad_k * MAD, z_min_tol)) are removed, unless fewer than min_samples would remain.
    The slice index is otherwise ignored: the model maps hist_center_yx to the (y, x) part of
    mri_center_zyx. Each iteration fits an affine to min_samples random matches; the model with
    the most inliers (reprojection error <= reproj_thresh), ties broken by the lower median
    inlier error, is refitted on its inliers, and the inliers are recomputed.

    Args:
        matches: List of match dicts with hist_center_yx and mri_center_zyx.
        reproj_thresh: Inlier threshold on the MRI in-plane error, in slab voxels.
        n_iter: Number of RANSAC iterations.
        min_samples: Matches per hypothesis (3 for an affine) and minimum inlier count.
        seed: Random seed.
        z_mad_k: Multiplier of the z MAD for the slice pre-filter.
        z_min_tol: Minimum slice tolerance of the pre-filter.

    Returns:
        dict with ok. If ok is False, reason says why ("too_few_matches",
        "too_few_valid_matches", "too_few_after_z_filter" or "no_model"). If ok is True:
        M (2x3 affine, histology xy -> MRI xy), inlier_idx (indices into matches), inlier_count,
        total_valid (matches after the z pre-filter), inlier_ratio (inlier_count / total_valid)
        and median_reproj (median inlier error).
    """
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
    best_inliers, best_count, best_mederr = None, -1, np.inf

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
            best_count, best_mederr, best_inliers = cnt, mederr, inliers

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
    """Rank patch configurations by the geometric consistency of their matches.

    For each result of evaluate_fixed_scale_multi_patch, fits a RANSAC affine
    (ransac_affine_from_matches) and computes score statistics (score_stability). Configurations
    are sorted, best first, lexicographically by: RANSAC succeeded, inlier count, inlier ratio,
    lower median reprojection error, higher median top score, lower IQR of the top scores.
    The ranking table is printed.

    Args:
        res_list: Results from evaluate_fixed_scale_multi_patch.
        reproj_thresh: RANSAC inlier threshold, in slab voxels.
        n_iter: Number of RANSAC iterations.
        top_frac: Fraction of best scores used by score_stability.
        seed: RANSAC random seed.

    Returns:
        List of row dicts sorted best first, with patch, hist_stride, mri_stride, robust_score,
        n_matches, score_median_top, score_iqr_top, ransac_ok, inlier_count, inlier_ratio,
        median_reproj, and _ransac (full RANSAC result), _matches and _out.
    """
    rows = []
    for r in res_list:
        matches = r["_out"]["matches"]
        # Score distribution statistics (tie-breakers in the ranking below)
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

    logger.debug('=== Patch config comparison (best first) ===')
    for i, d in enumerate(rows_sorted):
        logger.debug(f"[{i}] patch={d['patch']:>3} hs={d['hist_stride']:>2} ms={d['mri_stride']:>2} | inliers={d['inlier_count']:>4}/{d['n_matches']:<4} ({d['inlier_ratio']:.3f}) | med_reproj={d['median_reproj']:.2f} | score_top_med={d['score_median_top']:.4f} score_top_iqr={d['score_iqr_top']:.4f}")
    return rows_sorted


# ---------- Slice band from RANSAC inliers ----------

def z_band_from_inliers(inlier_matches, k_mad=3.0, min_radius=8):
    """Estimate the slab slice band from a set of (inlier) matches.

    The centre is the median slice of the valid matches; the half-width is
    max(min_radius, ceil(k_mad * MAD)).

    Args:
        inlier_matches: List of match dicts with mri_center_zyx.
        k_mad: Multiplier of the median absolute deviation of z.
        min_radius: Minimum half-width, in slices.

    Returns:
        Tuple (z_center, z_min, z_max, z_radius) in slab slice indices; z_min is clipped at 0,
        z_max is not clipped.

    Raises:
        RuntimeError: If no match has a valid slice index.
    """
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
    """Resize the prepared histology and its mask by the best scale and return SimpleITK images.

    Uses the same resizing as resize_hist_pair (bilinear image, nearest-neighbour mask, each side
    round(side * best_scale) but at least 32 pixels). A 2D input gives a 2D output; a 3D input
    is resized from its first plane and returned with shape (1, H', W'). For SimpleITK inputs the
    origin and direction are copied; the spacing is copied too and, with
    preserve_physical_size, its x and y entries are multiplied by old/new size so that the
    physical extent of the image stays the same.

    Args:
        proc_Ih: Prepared histology image (SimpleITK image or array), 2D or (1, H, W).
        proc_Mh: Prepared histology mask (SimpleITK image or array); values > 0 are tissue.
        best_scale: Scale factor (from search_scale).
        preserve_physical_size: If True, adjust the spacing to keep the physical size.

    Returns:
        Tuple (h_scaled, m_scaled) of SimpleITK images: float32 image and uint8 {0, 1} mask.
    """
    is_h_sitk = isinstance(proc_Ih, sitk.Image)
    is_m_sitk = isinstance(proc_Mh, sitk.Image)

    h_arr = sitk.GetArrayFromImage(proc_Ih) if is_h_sitk else np.asarray(proc_Ih)
    m_arr = sitk.GetArrayFromImage(proc_Mh) if is_m_sitk else np.asarray(proc_Mh)

    # Return the same dimensionality as the input.
    h_return_3d = (h_arr.ndim == 3)
    m_return_3d = (m_arr.ndim == 3)

    # Collapse to 2D for interpolation (_to_np2d keeps only the first plane of a 3D input).
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
    """Choose the patch size and strides for fine matching, and the slab slice band.

    Runs the plain matcher at the fixed scale for every configuration in patch_matrix
    (evaluate_fixed_scale_multi_patch), then ranks the configurations by RANSAC affine
    consistency (compare_patch_results_ransac: inlier count first, then inlier ratio,
    reprojection error and score statistics; reprojection threshold 12 voxels, seed 42). The
    slice band is computed from the RANSAC inliers of the best configuration (z_band_from_inliers).

    Args:
        proc_Ih: Prepared histology image (not yet rescaled), 2D or (1, y, x).
        proc_Mh: Prepared histology mask.
        slab_Im: MRI slab image (z, y, x).
        slab_Mm: MRI slab mask.
        best_scale: Histology scale factor from search_scale.
        patch_matrix: List of dicts with keys patch, hist_stride, mri_stride.
        z_min: Not used; the search always runs over all slab slices.
        z_max: Not used; see z_min.
        common_kwargs: Matcher arguments passed to evaluate_fixed_scale_multi_patch.

    Returns:
        Tuple (patch, hist_stride, mri_stride, z_center, z_min, z_max): the chosen patch size
        and strides (pixels / voxels) and the slice band (slab slice indices; z_min clipped at
        0, z_max not clipped).

    Raises:
        RuntimeError: If every configuration failed, or if the best configuration has no
            RANSAC inliers (z_band_from_inliers gets an empty list).
    """
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
        reproj_thresh=12.0,  # slab voxels
        n_iter=2000,
        top_frac=0.30,
        seed=42
    )

    # Best-ranked configuration
    best_row = rows_sorted[0]
    best_out = best_row["_out"]
    if not best_row["_ransac"]["ok"]:
        raise ValueError("Patch-configuration search failed: no configuration gave a RANSAC "
                         f"model ({best_row['_ransac'].get('reason', 'unknown reason')})")
    best_inliers = best_row["_ransac"]["inlier_idx"]

    logger.info('Chosen patch config: patch=%s hist_stride=%s mri_stride=%s', best_row['patch'], best_row['hist_stride'], best_row['mri_stride'])
    logger.info('Inliers: %s', len(best_inliers))

    inlier_matches = [best_out["matches"][i] for i in best_inliers]
    z_center, z_min, z_max, z_radius = z_band_from_inliers(inlier_matches)
    logger.info('z_center: %s z_min: %s z_max: %s z_radius: %s', z_center, z_min, z_max, z_radius)

    return best_row["patch"], best_row["hist_stride"], best_row["mri_stride"], z_center, z_min, z_max

