import numpy as np
from skimage.util import view_as_windows
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from scipy.ndimage import map_coordinates
from utilities import sample_mri_cubic_3d
from typing import Tuple

# Median Absolute Deviation (MAD) -> robust to outliers
def mad_based_filter(values, thresh=3.5):
    median = np.median(values)
    diff = np.abs(values - median)
    med_abs_deviation = np.median(diff)
    # normal distribution = 0.6745 * std
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score < thresh

# ---- find patch correlation ---------------------------------------------------------

def prepare_mri_vectors(mri_volume, patch_size, z_window=1, stride=4, device='cuda'):
    """
    Prepare MRI vectors from 3D volume using unfolding.
    """
    mri_tensor = torch.tensor(mri_volume, dtype=torch.float32, device=device)  # (Z, H, W)
    Z, H, W = mri_tensor.shape
    mri_tensor = mri_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, H, W)

    # Normalize patch/stride to 2D tuples
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size
    if isinstance(stride, int):
        stride_y = stride_x = stride
    else:
        stride_y, stride_x = stride

    patches = []
    coords = []

    half_z = z_window // 2
    for z in range(half_z, Z - half_z):
        slice_stack = mri_tensor[:, :, z - half_z : z + half_z + 1]     # (1, 1, z_window, H, W)
        patch_mean = slice_stack.mean(dim=2)                            # (1, 1, H, W)

        # Extract 2D patches using unfold
        patch_tensor = F.unfold(patch_mean, kernel_size=(patch_h,patch_w), stride=(stride_y, stride_x))  # (1, patch_area, num_patches)
        patch_tensor = patch_tensor.squeeze(0).T  # (num_patches, patch_area)

        # Keep only non-flat patches
        stds = patch_tensor.std(dim=1)
        keep_mask = stds > 1e-6
        patch_tensor = patch_tensor[keep_mask]

        # Normalize each patch to [0, 1]
        mins = patch_tensor.min(dim=1, keepdim=True)[0]
        maxs = patch_tensor.max(dim=1, keepdim=True)[0]
        patch_tensor = (patch_tensor - mins) / (maxs - mins + 1e-8)

        patches.append(patch_tensor)

        # Compute (z, y_center, x_center) for each patch location
        # Build grid of top-left origins with same ordering as unfold (x varies fastest)
        y0 = torch.arange(0, H - patch_h + 1, stride_y, device=device, dtype=torch.float32)
        x0 = torch.arange(0, W - patch_w + 1, stride_x, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y0, x0, indexing='ij')

        y_center = y_grid + patch_h / 2.0
        x_center = x_grid + patch_w / 2.0

        yx = torch.stack([y_center.flatten(), x_center.flatten()], dim=1)  # (num_xy, 2)
        yx = yx[keep_mask]

        z_vec = torch.full((yx.shape[0],), float(z), device=device)
        coords.append(torch.stack([z_vec, yx[:, 0], yx[:, 1]], dim=1))  # (N, 3) as (z, y_center, x_center)

    # Concatenate across slices
    mri_vectors = torch.cat(patches, dim=0)  # (M, patch_area)
    coords_list = torch.cat(coords, dim=0).tolist()  # Python list of (z, y_center, x_center)

    return mri_vectors, coords_list

def prepare_mri_vectors_batched(
    mri_volume,
    patch_size,
    z_window=1,
    stride=4,
    device='cuda',
    z_batch=8,
    out_dtype=torch.float32,   # set torch.float16 to cut CPU/GPU memory
    mri_mask=None,             # optional: center-in-mask filtering here
):
    """
    Batched MRI patch extraction:
    - never keeps full (N, patch_area) tensor on GPU
    - stores patch vectors on CPU
    - optional center mask filtering during extraction
    """
    dev = torch.device(device)

    vol_cpu = torch.as_tensor(mri_volume, dtype=torch.float32, device='cpu')
    mask_cpu = None
    if mri_mask is not None:
        mask_cpu = torch.as_tensor(mri_mask, dtype=torch.bool, device='cpu')

    Z, H, W = vol_cpu.shape

    if isinstance(patch_size, int):
        ph = pw = patch_size
    else:
        ph, pw = patch_size

    if isinstance(stride, int):
        sy = sx = stride
    else:
        sy, sx = stride

    half_z = z_window // 2
    z_start = half_z
    z_end = Z - half_z

    # Precompute XY centers in unfold order (x fastest)
    y0 = torch.arange(0, H - ph + 1, sy, dtype=torch.float32, device='cpu')
    x0 = torch.arange(0, W - pw + 1, sx, dtype=torch.float32, device='cpu')
    y_grid, x_grid = torch.meshgrid(y0, x0, indexing='ij')
    base_yx = torch.stack([y_grid.flatten() + ph / 2.0, x_grid.flatten() + pw / 2.0], dim=1)  # (num_xy,2)

    patches_cpu = []
    coords_cpu = []

    for z0 in range(z_start, z_end, z_batch):
        z1 = min(z0 + z_batch, z_end)

        # bring only the needed slab for this z-chunk to GPU
        slab_cpu = vol_cpu[z0 - half_z : z1 + half_z]  # (z_batch + z_window - 1, H, W)
        slab_gpu = slab_cpu.to(dev, non_blocking=False)

        for z in range(z0, z1):
            local = z - z0
            stack = slab_gpu[local : local + z_window].unsqueeze(0).unsqueeze(0)  # (1,1,z_window,H,W)
            patch_mean = stack.mean(dim=2)  # (1,1,H,W)

            patch = F.unfold(patch_mean, kernel_size=(ph, pw), stride=(sy, sx)).squeeze(0).T  # (num_xy, ph*pw)

            keep = patch.std(dim=1) > 1e-6
            if not torch.any(keep):
                del stack, patch_mean, patch, keep
                continue

            patch = patch[keep]
            yx = base_yx[keep.cpu()]

            if mask_cpu is not None:
                yi = yx[:, 0].long()
                xi = yx[:, 1].long()
                center_keep = mask_cpu[z, yi, xi]
                if not torch.any(center_keep):
                    del stack, patch_mean, patch, keep, yx, yi, xi, center_keep
                    continue
                patch = patch[center_keep.to(dev)]
                yx = yx[center_keep]

            mins = patch.min(dim=1, keepdim=True).values
            maxs = patch.max(dim=1, keepdim=True).values
            patch = (patch - mins) / (maxs - mins + 1e-8)

            z_vec = torch.full((yx.shape[0],), float(z), dtype=torch.float32, device='cpu')
            coords = torch.stack([z_vec, yx[:, 0], yx[:, 1]], dim=1)

            patches_cpu.append(patch.to('cpu', dtype=out_dtype))
            coords_cpu.append(coords)

            del stack, patch_mean, patch, keep, yx, mins, maxs, z_vec, coords

        del slab_cpu, slab_gpu
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    if len(patches_cpu) == 0:
        raise ValueError("No MRI patches were extracted (check mask/stride/patch_size).")

    mri_vectors = torch.cat(patches_cpu, dim=0)      # CPU tensor
    coords_list = torch.cat(coords_cpu, dim=0).tolist()
    return mri_vectors, coords_list


def prepare_hist_vector(hist_patch,device='cuda'):
    return torch.tensor(hist_patch.flatten(), dtype=torch.float32, device=device)

# ---- kernel caches ----
_gauss_cache = {}
_sobel_cache = {}

def _key(device, dtype, *extra):
    return (str(device), str(dtype)) + tuple(extra)

def get_gaussian_window(window_size=11, sigma=1.5, device='cuda', dtype=torch.float32):
    k = _key(device, dtype, window_size, float(sigma))
    win = _gauss_cache.get(k)
    if win is None:
        coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        win = (g[:, None] @ g[None, :]).view(1, 1, window_size, window_size)
        _gauss_cache[k] = win
    return win

def get_sobel_kernels(device='cuda', dtype=torch.float32):
    k = _key(device, dtype)
    pair = _sobel_cache.get(k)
    if pair is None:
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=dtype, device=device).view(1,1,3,3)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=dtype, device=device).view(1,1,3,3)
        pair = (sx, sy)
        _sobel_cache[k] = pair
    return pair

def clear_kernel_caches():
    _gauss_cache.clear()
    _sobel_cache.clear()

def compute_ssim_similarity(hist_vector, mri_vectors, image_size=64,
                                   window_size=11, C1=0.01**2, C2=0.03**2, eps=1e-8):
    B, N = mri_vectors.shape
    H = W = image_size
    device = hist_vector.device
    dtype  = hist_vector.dtype

    hist_img = hist_vector.view(1, 1, H, W)
    mri_imgs = mri_vectors.view(B, 1, H, W)

    # Normalize to [0,1]
    hist_img = (hist_img - hist_img.min()) / (hist_img.max() - hist_img.min() + eps)
    mri_min = mri_imgs.amin(dim=(1,2,3), keepdim=True)
    mri_max = mri_imgs.amax(dim=(1,2,3), keepdim=True)
    mri_imgs = (mri_imgs - mri_min) / (mri_max - mri_min + eps)

    # ---- cached Gaussian window
    window = get_gaussian_window(window_size, sigma=1.5, device=device, dtype=dtype)

    mu1 = F.conv2d(hist_img.expand(B, -1, -1, -1), window, padding=window_size//2)
    mu2 = F.conv2d(mri_imgs, window, padding=window_size//2)

    mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(hist_img.pow(2).expand(B, -1, -1, -1), window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(mri_imgs.pow(2), window, padding=window_size//2) - mu2_sq
    sigma12  = F.conv2d(hist_img.expand(B, -1, -1, -1) * mri_imgs, window, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)
    return ssim_map.mean(dim=(1, 2, 3))  # (B,)

def compute_ngf_similarity(hist_vector, mri_vectors, image_size=64, eps=1e-8):
    B, N = mri_vectors.shape
    H = W = image_size
    device = hist_vector.device
    dtype  = hist_vector.dtype

    hist_img = hist_vector.view(1, 1, H, W)
    mri_imgs = mri_vectors.view(B, 1, H, W)

    # Normalize to [0,1]
    hist_img = (hist_img - hist_img.min()) / (hist_img.max() - hist_img.min() + eps)
    mri_min = mri_imgs.amin(dim=(1,2,3), keepdim=True)
    mri_max = mri_imgs.amax(dim=(1,2,3), keepdim=True)
    mri_imgs = (mri_imgs - mri_min) / (mri_max - mri_min + eps)

    # ---- cached Sobel kernels
    sobel_x, sobel_y = get_sobel_kernels(device=device, dtype=dtype)

    grad_Ix = F.conv2d(hist_img, sobel_x, padding=1)
    grad_Iy = F.conv2d(hist_img, sobel_y, padding=1)
    grad_Jx = F.conv2d(mri_imgs, sobel_x, padding=1)
    grad_Jy = F.conv2d(mri_imgs, sobel_y, padding=1)

    dot   = grad_Ix * grad_Jx + grad_Iy * grad_Jy
    normI = grad_Ix**2 + grad_Iy**2
    normJ = grad_Jx**2 + grad_Jy**2

    ngf_map = (dot**2) / (normI * normJ + eps)
    return ngf_map.view(B, -1).mean(dim=1)  # (B,)

def compute_ssim_similarity_masked(
    hist_vector, mri_vectors, hist_mask_vector, mri_mask_vectors,
    image_size=64, window_size=11, C1=0.01**2, C2=0.03**2, eps=1e-8
):
    B, N = mri_vectors.shape
    H = W = image_size
    device = hist_vector.device
    dtype  = hist_vector.dtype

    hist_img = hist_vector.view(1, 1, H, W)
    mri_imgs = mri_vectors.view(B, 1, H, W)

    # build intersection mask per slice
    hist_mask = hist_mask_vector.view(1, 1, H, W) > 0
    mri_masks = mri_mask_vectors.view(B, 1, H, W) > 0
    mask = (hist_mask & mri_masks).float()

    # intensity normalize inside each image
    hist_img = (hist_img - hist_img.min()) / (hist_img.max() - hist_img.min() + eps)
    mri_min = mri_imgs.amin(dim=(1,2,3), keepdim=True)
    mri_max = mri_imgs.amax(dim=(1,2,3), keepdim=True)
    mri_imgs = (mri_imgs - mri_min) / (mri_max - mri_min + eps)

    window = get_gaussian_window(window_size, sigma=1.5, device=device, dtype=dtype)

    mu1 = F.conv2d(hist_img.expand(B, -1, -1, -1), window, padding=window_size//2)
    mu2 = F.conv2d(mri_imgs, window, padding=window_size//2)

    mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(hist_img.pow(2).expand(B, -1, -1, -1), window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(mri_imgs.pow(2), window, padding=window_size//2) - mu2_sq
    sigma12  = F.conv2d(hist_img.expand(B, -1, -1, -1) * mri_imgs, window, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)

    masked_sum = (ssim_map * mask).sum(dim=(1,2,3))
    mask_norm  = mask.sum(dim=(1,2,3)).clamp(min=1.0)  # avoid div0
    return masked_sum / mask_norm  # (B,)

def compute_ngf_similarity_masked(
    hist_vector, mri_vectors, hist_mask_vector, mri_mask_vectors,
    image_size=64, eps=1e-8
):
    B, N = mri_vectors.shape
    H = W = image_size
    device = hist_vector.device
    dtype  = hist_vector.dtype

    hist_img = hist_vector.view(1, 1, H, W)
    mri_imgs = mri_vectors.view(B, 1, H, W)

    hist_mask = hist_mask_vector.view(1, 1, H, W) > 0
    mri_masks = mri_mask_vectors.view(B, 1, H, W) > 0
    mask = (hist_mask & mri_masks).float()

    # normalize inside whole image (masking handled in aggregation)
    hist_img = (hist_img - hist_img.min()) / (hist_img.max() - hist_img.min() + eps)
    mri_min = mri_imgs.amin(dim=(1,2,3), keepdim=True)
    mri_max = mri_imgs.amax(dim=(1,2,3), keepdim=True)
    mri_imgs = (mri_imgs - mri_min) / (mri_max - mri_min + eps)

    sobel_x, sobel_y = get_sobel_kernels(device=device, dtype=dtype)

    grad_Ix = F.conv2d(hist_img, sobel_x, padding=1)
    grad_Iy = F.conv2d(hist_img, sobel_y, padding=1)
    grad_Jx = F.conv2d(mri_imgs, sobel_x, padding=1)
    grad_Jy = F.conv2d(mri_imgs, sobel_y, padding=1)

    dot   = grad_Ix * grad_Jx + grad_Iy * grad_Jy
    normI = grad_Ix**2 + grad_Iy**2
    normJ = grad_Jx**2 + grad_Jy**2

    ngf_map = (dot**2) / (normI * normJ + eps)  # (B,1,H,W)

    masked_sum = (ngf_map * mask).sum(dim=(1,2,3))
    mask_norm  = mask.sum(dim=(1,2,3)).clamp(min=1.0)
    return masked_sum / mask_norm  # (B,)

def compute_nmi_similarity_masked(
    hist_vector,
    mri_vectors,
    hist_mask_vector,
    mri_mask_vectors,
    image_size=64,
    num_bins=64,
    eps=1e-8,
):
    """
    Masked normalized mutual information over intersection of histo/mri masks.
    Returns: (B,) NMI scores.
    """
    B, N = mri_vectors.shape
    H = W = image_size

    hist_img = hist_vector.view(1, 1, H, W)
    mri_imgs = mri_vectors.view(B, 1, H, W)

    hist_mask = hist_mask_vector.view(1, 1, H, W) > 0
    mri_masks = mri_mask_vectors.view(B, 1, H, W) > 0
    masks = hist_mask & mri_masks  # (B,1,H,W) via broadcasting

    # normalize to [0,1] (same as other metrics)
    hist_img = (hist_img - hist_img.min()) / (hist_img.max() - hist_img.min() + eps)
    mri_min = mri_imgs.amin(dim=(1,2,3), keepdim=True)
    mri_max = mri_imgs.amax(dim=(1,2,3), keepdim=True)
    mri_imgs = (mri_imgs - mri_min) / (mri_max - mri_min + eps)

    bin_ids = torch.linspace(0, 1, num_bins + 1, device=hist_img.device, dtype=hist_img.dtype)

    scores = []
    for b in range(B):
        m = masks[b, 0]
        if m.sum() == 0:
            scores.append(torch.tensor(0.0, device=hist_img.device))
            continue

        h_vals = hist_img[0, 0][m]
        m_vals = mri_imgs[b, 0][m]

        h_bin = torch.clamp((h_vals * (num_bins - 1)).long(), 0, num_bins - 1)
        m_bin = torch.clamp((m_vals * (num_bins - 1)).long(), 0, num_bins - 1)

        joint_idx = h_bin * num_bins + m_bin
        joint = torch.bincount(joint_idx, minlength=num_bins * num_bins).float().view(num_bins, num_bins)
        joint = joint / joint.sum().clamp(min=1.0)

        p_h = joint.sum(dim=1)
        p_m = joint.sum(dim=0)

        H_h = -(p_h * torch.log(p_h + eps)).sum()
        H_m = -(p_m * torch.log(p_m + eps)).sum()
        H_joint = -(joint * torch.log(joint + eps)).sum()

        nmi = (H_h + H_m) / (H_joint + eps)
        scores.append(nmi)

    return torch.stack(scores, dim=0)


@torch.no_grad()
def score_hist_patch_batched(
    hist_vector,                 # (4096,)
    mri_vectors_cpu,             # (N, 4096) on CPU
    batch_size=4096,
    device='cuda',
    image_size=64,
    alpha=0.5,
    use_fp16=True,
):
    is_cuda = (device.type == 'cuda')

    # Prefer FP16 only if we have cached kernels (which build weights in the right dtype).
    have_cached = ('compute_ssim_similarity_cached' in globals()
                   and 'compute_ngf_similarity_cached' in globals())
    ssim_fn = globals().get('compute_ssim_similarity_cached',
                            globals().get('compute_ssim_similarity'))
    ngf_fn  = globals().get('compute_ngf_similarity_cached',
                            globals().get('compute_ngf_similarity'))

    prefer_fp16 = bool(is_cuda and use_fp16 and have_cached)

    # Put hist_vector on device once with the chosen dtype
    tgt_dtype = torch.float16 if prefer_fp16 else torch.float32
    hv = hist_vector.to(device, dtype=tgt_dtype, non_blocking=True)

    N = mri_vectors_cpu.shape[0]

    # -------- Pass 1: global mins/maxes --------
    ngf_min = float('inf'); ngf_max = float('-inf')
    ssim_min = float('inf'); ssim_max = float('-inf')

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        mb = mri_vectors_cpu[start:end].to(device, dtype=tgt_dtype, non_blocking=True)

        ngf_scores  = ngf_fn(hv, mb, image_size=image_size)
        ssim_scores = ssim_fn(hv, mb, image_size=image_size)

        nmin = float(ngf_scores.min()); nmax = float(ngf_scores.max())
        smin = float(ssim_scores.min()); smax = float(ssim_scores.max())
        if nmin < ngf_min: ngf_min = nmin
        if nmax > ngf_max: ngf_max = nmax
        if smin < ssim_min: ssim_min = smin
        if smax > ssim_max: ssim_max = smax

        del mb, ngf_scores, ssim_scores

    ngf_rng  = max(ngf_max - ngf_min, 1e-8)
    ssim_rng = max(ssim_max - ssim_min, 1e-8)

    # -------- Pass 2: normalized scores + argmax --------
    best_idx = -1
    best_score = float('-inf')
    offset = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        mb = mri_vectors_cpu[start:end].to(device, dtype=tgt_dtype, non_blocking=True)

        ngf_scores  = ngf_fn(hv, mb, image_size=image_size)
        ssim_scores = ssim_fn(hv, mb, image_size=image_size)

        ngf_norm  = (ngf_scores  - ngf_min)  / ngf_rng
        ssim_norm = (ssim_scores - ssim_min) / ssim_rng
        scores = alpha * ngf_norm + (1.0 - alpha) * ssim_norm

        b_idx = int(torch.argmax(scores))
        b_val = float(scores[b_idx])
        if b_val > best_score:
            best_score = b_val
            best_idx = offset + b_idx

        offset += (end - start)
        del mb, ngf_scores, ssim_scores, ngf_norm, ssim_norm, scores

    return best_idx, best_score


def find_patch_corr(
    histo_img,
    mri_vol,
    patch_size,
    stride,
    device,
    mri_batch_size=4096,
    alpha=1.0,
    use_fp16=True,
    histo_mask=None,
    mri_mask=None,
    min_histo_mask_fraction=0.5,
):
    """
    Compute patch correlations between histology and MRI, restricted to tissue masks.

    New args:
        histo_mask: 2D mask (same HxW as histo_img), 1/True = tissue, 0/False = background.
        mri_mask:   3D mask (same ZxYxX as mri_vol), 1/True = tissue, 0/False = background.
        min_histo_mask_fraction: minimum fraction of tissue pixels in a histology patch
                                 for it to be used (0.0–1.0).
    """
    print("\n-------- Find Correspondence Points btw Histology and MRI --------")
    ORIGIN = (0, 0)

    def _to_hw_pair(val):
        return (val, val) if isinstance(val, int) else tuple(val[:2])

    patch_h, patch_w = _to_hw_pair(patch_size)
    stride_y, stride_x = _to_hw_pair(stride)
    origin_y, origin_x = _to_hw_pair(ORIGIN)

    arr_proc_Ih = sitk.GetArrayFromImage(histo_img)[0]
    arr_proc_Mh = sitk.GetArrayFromImage(histo_mask)[0]
    arr_mri_vol = sitk.GetArrayFromImage(mri_vol)
    arr_mri_mask = sitk.GetArrayFromImage(mri_mask)


    # -------------------------------
    # 1) Build MRI candidates
    # -------------------------------
    # mri_vectors_cpu, coords_list = prepare_mri_vectors(
    #     arr_mri_vol, patch_size=patch_size, device=device
    # )

    mri_vectors_cpu, coords_list = prepare_mri_vectors_batched(
        arr_mri_vol,
        patch_size=patch_size,
        stride=4,  # important: use your function arg, not default 4
        z_window=1,
        device=device,
        z_batch=64,  # try 4 if still tight
        out_dtype=torch.float16 if use_fp16 else torch.float32,
        mri_mask=arr_mri_mask,  # optional early filtering
    )

    # Optionally restrict MRI candidates to those whose CENTER is inside the MRI mask
    if arr_mri_mask is not None:
        # Assume coords_list is a list of (z, y, x) or similar
        filtered_vectors = []
        filtered_coords = []

        for idx, coord in enumerate(coords_list):
            # Be robust to tuple/list/np.array
            z, y, x = map(int, coord[:3])
            if (
                0 <= z < arr_mri_mask.shape[0]
                and 0 <= y < arr_mri_mask.shape[1]
                and 0 <= x < arr_mri_mask.shape[2]
                and arr_mri_mask[z, y, x]  # non-zero / True = keep
            ):
                filtered_vectors.append(mri_vectors_cpu[idx])
                filtered_coords.append(coord)

        if len(filtered_vectors) == 0:
            raise ValueError("No MRI candidates remain after applying arr_mri_mask.")

        # If mri_vectors_cpu is a tensor, this will still work (list -> stacked later).
        if isinstance(mri_vectors_cpu, torch.Tensor):
            mri_vectors_cpu = torch.stack(filtered_vectors, dim=0)
        else:
            # e.g. numpy array or list
            mri_vectors_cpu = np.stack(filtered_vectors, axis=0)

        coords_list = filtered_coords

    # -------------------------------
    # 2) Extract histology patches
    # -------------------------------
    patches = view_as_windows(arr_proc_Ih, (patch_size, patch_size), step=stride)
    n_y, n_x = patches.shape[:2]

    # If we have a histology mask, extract corresponding patch windows
    if arr_proc_Mh is not None:
        mask_patches = view_as_windows(arr_proc_Mh, (patch_size, patch_size), step=stride)
    else:
        mask_patches = None

    matches = []

    with torch.no_grad():
        for i in tqdm(range(n_y)):        # row (y)
            for j in range(n_x):          # col (x)

                # -------------------------------
                # 2a) Mask-based patch filtering
                # -------------------------------
                if mask_patches is not None:
                    patch_mask = mask_patches[i, j]

                    # Assume mask is 0/1 or False/True.
                    # Compute fraction of pixels inside tissue.
                    frac_tissue = float(patch_mask.mean())

                    # Skip patches that are mostly background
                    if frac_tissue < min_histo_mask_fraction:
                        continue

                # -------------------------------
                # 2b) Correlation for this patch
                # -------------------------------
                hist_patch = patches[i, j]
                hist_vector = prepare_hist_vector(hist_patch, device=device)  # (patch_h*patch_w,)

                best_idx, best_score = score_hist_patch_batched(
                    hist_vector=hist_vector,
                    mri_vectors_cpu=mri_vectors_cpu,
                    batch_size=mri_batch_size,
                    device=device,
                    image_size=patch_size,
                    alpha=alpha,
                    use_fp16=use_fp16
                )

                best_coords = coords_list[best_idx] if (best_idx >= 0 and best_score > 0) else (-1, -1, -1)

                center_x = origin_x + j * stride_x + patch_w / 2.0
                center_y = origin_y + i * stride_y + patch_h / 2.0

                matches.append({
                    'hist_patch_idx': (i, j),
                    'hist_center_yx': (float(center_y), float(center_x)),
                    'mri_coords': best_coords,
                    'correlation': float(best_score),
                })

    clear_kernel_caches()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return matches





@torch.no_grad()
def score_hist_patch_batched_masked(
    hist_vector,                 # (4096,)
    mri_vectors_cpu,             # (N, 4096) on CPU
    hist_mask_vector=None,       # (4096,) or None
    mri_mask_vectors_cpu=None,   # (N, 4096) on CPU or None
    batch_size=4096,
    device=torch.device('cuda'),
    image_size=64,
    alpha=0.5,                   # weight for NGF; (1-alpha) for NMI
    use_fp16=True,
):
    is_cuda = (device.type == 'cuda')
    use_masked = hist_mask_vector is not None and mri_mask_vectors_cpu is not None

    ngf_fn = compute_ngf_similarity_masked if use_masked else compute_ngf_similarity
    nmi_fn = compute_nmi_similarity_masked

    tgt_dtype = torch.float16 if (is_cuda and use_fp16) else torch.float32
    hv = hist_vector.to(device, dtype=tgt_dtype, non_blocking=True)
    hm = hist_mask_vector.to(device, dtype=tgt_dtype, non_blocking=True) if use_masked else None

    N = mri_vectors_cpu.shape[0]
    best_idx = -1
    best_score = float('-inf')
    offset = 0

    # Pass 1: mins/maxes
    ngf_min = float('inf'); ngf_max = float('-inf')
    nmi_min = float('inf'); nmi_max = float('-inf')
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        mb = mri_vectors_cpu[start:end].to(device, dtype=tgt_dtype, non_blocking=True)
        mm = mri_mask_vectors_cpu[start:end].to(device, dtype=tgt_dtype, non_blocking=True) if use_masked else None

        ngf_scores = ngf_fn(hv, mb, hm, mm, image_size=image_size) if use_masked \
                     else ngf_fn(hv, mb, image_size=image_size)
        nmi_scores = nmi_fn(hv, mb, hm if use_masked else torch.ones_like(hv),
                            mm if use_masked else torch.ones_like(mb),
                            image_size=image_size)

        ngf_min = min(ngf_min, float(ngf_scores.min())); ngf_max = max(ngf_max, float(ngf_scores.max()))
        nmi_min = min(nmi_min, float(nmi_scores.min())); nmi_max = max(nmi_max, float(nmi_scores.max()))
        del mb, mm, ngf_scores, nmi_scores

    ngf_rng = max(ngf_max - ngf_min, 1e-8)
    nmi_rng = max(nmi_max - nmi_min, 1e-8)

    # Pass 2: normalize + argmax
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        mb = mri_vectors_cpu[start:end].to(device, dtype=tgt_dtype, non_blocking=True)
        mm = mri_mask_vectors_cpu[start:end].to(device, dtype=tgt_dtype, non_blocking=True) if use_masked else None

        ngf_scores = ngf_fn(hv, mb, hm, mm, image_size=image_size) if use_masked \
                     else ngf_fn(hv, mb, image_size=image_size)
        nmi_scores = nmi_fn(hv, mb, hm if use_masked else torch.ones_like(hv),
                            mm if use_masked else torch.ones_like(mb),
                            image_size=image_size)

        ngf_norm = (ngf_scores - ngf_min) / ngf_rng
        nmi_norm = (nmi_scores - nmi_min) / nmi_rng
        scores = alpha * ngf_norm + (1.0 - alpha) * nmi_norm

        b_idx = int(torch.argmax(scores))
        b_val = float(scores[b_idx])
        if b_val > best_score:
            best_score = b_val
            best_idx = offset + b_idx

        offset += (end - start)
        del mb, mm, ngf_scores, nmi_scores, ngf_norm, nmi_norm, scores

    return best_idx, best_score

def find_patch_corr_masked(
    histo_img,
    mri_vol,
    patch_size,
    stride,
    device,
    mri_batch_size=8192,
    alpha=0.5,
    use_fp16=False,
    histo_mask=None,
    mri_mask=None,
    min_histo_mask_fraction=0.5,
):
    print("\n-------- Find Correspondence Points btw Histology and MRI --------")
    ORIGIN = (0, 0)

    def _to_hw_pair(val):
        return (val, val) if isinstance(val, int) else tuple(val[:2])

    patch_h, patch_w = _to_hw_pair(patch_size)
    stride_y, stride_x = _to_hw_pair(stride)
    origin_y, origin_x = _to_hw_pair(ORIGIN)

    arr_proc_Ih   = sitk.GetArrayFromImage(histo_img)[0]
    arr_proc_Mh   = sitk.GetArrayFromImage(histo_mask)[0] if histo_mask is not None else None
    arr_mri_vol   = sitk.GetArrayFromImage(mri_vol)
    arr_mri_mask  = sitk.GetArrayFromImage(mri_mask) if mri_mask is not None else None

    # --- helper to build MRI vectors + corresponding mask vectors
    def prepare_mri_vectors_with_masks(vol, vol_mask, patch_size, stride, device):
        mri_tensor  = torch.tensor(vol, dtype=torch.float32, device=device)    # (Z,H,W)
        mask_tensor = torch.tensor(vol_mask if vol_mask is not None else np.ones_like(vol),
                                   dtype=torch.float32, device=device)
        Z, H, W = mri_tensor.shape
        mri_tensor  = mri_tensor.unsqueeze(0).unsqueeze(0)   # (1,1,Z,H,W)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,Z,H,W)

        if isinstance(patch_size, int):
            ph = pw = patch_size
        else:
            ph, pw = patch_size
        if isinstance(stride, int):
            sy = sx = stride
        else:
            sy, sx = stride

        patches = []
        mask_patches = []
        coords = []

        for z in range(Z):
            slice_stack = mri_tensor[:, :, z:z+1]          # (1,1,1,H,W)
            mask_stack  = mask_tensor[:, :, z:z+1]

            patch_tensor = F.unfold(slice_stack.squeeze(2), kernel_size=(ph, pw), stride=(sy, sx)).squeeze(0).T
            mask_tensor2 = F.unfold(mask_stack.squeeze(2),  kernel_size=(ph, pw), stride=(sy, sx)).squeeze(0).T

            stds = patch_tensor.std(dim=1)
            keep_mask = stds > 1e-6

            patches.append(patch_tensor[keep_mask])
            mask_patches.append(mask_tensor2[keep_mask])

            y0 = torch.arange(0, H - ph + 1, sy, device=device, dtype=torch.float32)
            x0 = torch.arange(0, W - pw + 1, sx, device=device, dtype=torch.float32)
            y_grid, x_grid = torch.meshgrid(y0, x0, indexing='ij')
            yx = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)[keep_mask]
            z_vec = torch.full((yx.shape[0],), float(z), device=device)
            coords.append(torch.stack([z_vec, yx[:, 0] + ph/2, yx[:, 1] + pw/2], dim=1))

        mri_vectors       = torch.cat(patches, dim=0).cpu()
        mri_mask_vectors  = torch.cat(mask_patches, dim=0).cpu()
        coords_list       = torch.cat(coords, dim=0).tolist()
        return mri_vectors, mri_mask_vectors, coords_list

    # 1) Build MRI candidates + mask vectors
    mri_vectors_cpu, mri_mask_vectors_cpu, coords_list = prepare_mri_vectors_with_masks(
        arr_mri_vol, arr_mri_mask, patch_size=patch_size, stride=stride, device=device
    )

    # center-in-mask filtering
    if arr_mri_mask is not None:
        keep_idxs = []
        for idx, coord in enumerate(coords_list):
            z, y, x = map(int, coord[:3])
            if (0 <= z < arr_mri_mask.shape[0] and 0 <= y < arr_mri_mask.shape[1]
                and 0 <= x < arr_mri_mask.shape[2] and arr_mri_mask[z, y, x]):
                keep_idxs.append(idx)
        if not keep_idxs:
            raise ValueError("No MRI candidates remain after applying arr_mri_mask.")
        keep_idxs_t = torch.tensor(keep_idxs, dtype=torch.long)
        mri_vectors_cpu      = mri_vectors_cpu[keep_idxs_t]
        mri_mask_vectors_cpu = mri_mask_vectors_cpu[keep_idxs_t]
        coords_list          = [coords_list[i] for i in keep_idxs]

    # 2) Extract histology patches (+ masks)
    patches = view_as_windows(arr_proc_Ih, (patch_size, patch_size), step=stride)
    n_y, n_x = patches.shape[:2]
    mask_patches = view_as_windows(arr_proc_Mh, (patch_size, patch_size), step=stride) if arr_proc_Mh is not None else None

    matches = []
    with torch.no_grad():
        for i in tqdm(range(n_y)):        # row (y)
            for j in range(n_x):          # col (x)
                if mask_patches is not None:
                    patch_mask = mask_patches[i, j]
                    if float(patch_mask.mean()) < min_histo_mask_fraction:
                        continue
                    hist_mask_vector = torch.tensor(patch_mask.flatten(), dtype=torch.float32, device=device)
                else:
                    hist_mask_vector = None

                hist_patch = patches[i, j]
                hist_vector = prepare_hist_vector(hist_patch, device=device)

                best_idx, best_score = score_hist_patch_batched_masked(
                    hist_vector=hist_vector,
                    mri_vectors_cpu=mri_vectors_cpu,
                    hist_mask_vector=hist_mask_vector,
                    mri_mask_vectors_cpu=mri_mask_vectors_cpu,
                    batch_size=mri_batch_size,
                    device=device,
                    image_size=patch_size,
                    alpha=alpha,
                    use_fp16=use_fp16,
                )
                # print(f"idx: {best_idx} score: {best_score}")

                # best_coords = coords_list[best_idx] if (best_idx >= 0 and best_score > 0) else (-1, -1, -1)
                best_coords = coords_list[best_idx] if best_idx >= 0 else (-1, -1, -1)
                center_x = origin_x + j * stride_x + patch_w / 2.0
                center_y = origin_y + i * stride_y + patch_h / 2.0

                matches.append({
                    'hist_patch_idx': (i, j),
                    'hist_center_yx': (float(center_y), float(center_x)),
                    'mri_coords': best_coords,
                    'correlation': float(best_score),
                })

    clear_kernel_caches()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return matches






def filter_matches_by_z_mad(matches, thresh=2.0):
    """
    Filter matches based on z using a MAD-based outlier filter.
    Expects each match to have 'mri_coords' = (z, y, x).
    Returns a filtered list of matches.
    """
    Pzyx = np.array([m['mri_coords'] for m in matches])
    z = Pzyx[:, 0]
    mask = mad_based_filter(z, thresh=thresh)
    match_filtered = [m for m, keep in zip(matches, mask) if keep]
    return match_filtered


# ---- initial in-plane alignment ---------------------------------------------------------

# ---- 0. Helpers ----
def _fit_plane_tls_xyz(X_xyz):
    """
    Total least squares plane fit to 3D points X (N,3) in (x,y,z).
    Returns plane origin X0 (on plane), and orthonormal basis (e1, e2, n).
    """
    X = np.asarray(X_xyz, float)
    assert X.ndim == 2 and X.shape[1] == 3 and X.shape[0] >= 3
    X0 = X.mean(0)
    Xc = X - X0
    # SVD of centered points; right singular vectors are principal axes
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    e1, e2, n = Vt  # e1/e2 span plane, n is normal
    # Right-handed fix
    if np.linalg.det(np.vstack([e1, e2, n])) < 0:
        e2 = -e2
    # Normalize
    e1 /= np.linalg.norm(e1); e2 /= np.linalg.norm(e2); n /= np.linalg.norm(n)
    return X0, e1, e2, n

def _project_to_plane_xy(X_xyz, X0, e1, e2):
    """
    Project 3D points to 2D plane coordinates P = [(X-X0)·e1, (X-X0)·e2].
    """
    X = np.asarray(X_xyz, float)
    return np.column_stack([(X - X0) @ e1, (X - X0) @ e2])

def build_arrays_from_matches(matches_filtered):
    """
    Input: list of dicts with keys:
      - m['mri_coords']     = (z, y, x)  voxels
      - m['hist_center_yx'] = (y, x)     pixels
    Returns:
      X_zyx: (N,3), U_yx: (N,2)
    """
    X_zyx = []
    U_yx  = []
    for m in matches_filtered:
        z, y, x = m['mri_coords']
        cy, cx  = m['hist_center_yx']
        if (z, y, x) == (-1, -1, -1):
            continue
        X_zyx.append((z, y, x))
        U_yx.append((cy, cx))
    return np.asarray(X_zyx, float), np.asarray(U_yx, float)

def decompose_affine_to_rotation_and_scale(A2):
    """
    Polar decomposition: A2 = R @ S
    R = pure rotation (2x2, det≈+1)
    S = symmetric (scale + shear)
    """
    U, svals, Vt = np.linalg.svd(A2)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
        svals[-1] *= -1

    S = Vt.T @ np.diag(svals) @ Vt
    return R, S

def decompose_affine_to_uniform_scale_and_rotation(A2):
    """
    Decompose A2 into:
        A2 = s * R @ A_residual
    where:
        s           = uniform scale (scalar)
        R           = pure rotation (2x2, det≈+1)
        A_residual  = remaining anisotropy (2x2)
    """
    R, S = decompose_affine_to_rotation_and_scale(A2)

    # eigenvalues of S give anisotropic scales
    eigvals, _ = np.linalg.eig(S)
    # uniform scale = geometric mean of eigenvalues
    s = float(np.sqrt(abs(eigvals[0] * eigvals[1])))

    # residual anisotropy after removing uniform scale
    A_residual = S / s

    return s, R, A_residual

def _affine_2d(src, dst):
    """
    Full 2D affine fit: dst ≈ src @ A^T + t.
    src, dst: (N,2)
    Returns:
      A: (2,2)
      t: (2,)
    """
    X = np.asarray(src, float)
    Y = np.asarray(dst, float)
    assert X.shape == Y.shape and X.shape[1] == 2 and X.shape[0] >= 3

    N = X.shape[0]
    X_aug = np.hstack([X, np.ones((N, 1))])  # (N,3)

    # Y ≈ X_aug @ B  where B is (3,2) = [[A^T],[t]]
    B, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
    A_T = B[:2, :]   # (2,2)
    t   = B[2, :]    # (2,)
    A   = A_T.T
    return A, t


def fit_global_from_matches_affine(matches_filtered, 
                                   hist_shape_yx, 
                                   hist_mask_yx=None,            
                                   center_to_mask=True,
                                   match_scale_to_mask=True,
                                   q=95):
    """
    Fit:
      - best plane in MRI
      - full 2D affine A2,t_xy from plane coords -> hist coords

    Also computes:
      - s (uniform scale)
      - R2 (pure rotation)

    Returns G_affine with keys:
      "X0","e1","e2","n",
      "A2","t_xy",
      "s","R2","A_residual",
      "reprojection"
    """
    print("\n-------- Estimate Initial Global Alignment --------")
    H, W = hist_shape_yx

    # (a) arrays
    X_zyx, U_yx = build_arrays_from_matches(matches_filtered)
    X_xyz = X_zyx[:, [2, 1, 0]]  # (x,y,z)

    # (b) plane
    X0, e1, e2, n = _fit_plane_tls_xyz(X_xyz)

    # (c) project to plane
    P = _project_to_plane_xy(X_xyz, X0, e1, e2)  # (N,2)

    # (d) hist coords (y,x) -> (x, y), NO flip
    U_math = np.column_stack([U_yx[:, 1], U_yx[:, 0]])  # (N,2)

    # (e) affine
    A2, t_xy = _affine_2d(P, U_math)
    if center_to_mask or match_scale_to_mask:
        if hist_mask_yx is not None:
            # yy, xx = np.nonzero(hist_mask_yx > 0)
            m = np.asarray(hist_mask_yx)
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            m = (m > 0.5)  # avoid >0 halo
            yy, xx = np.nonzero(m)
            if len(xx) > 0:
                Uc = np.array([xx.mean(), yy.mean()], dtype=float)  # target center (x,y)
            else:
                Uc = np.array([(W - 1)/2.0, (H - 1)/2.0], dtype=float)
        else:
            Uc = np.array([(W - 1)/2.0, (H - 1)/2.0], dtype=float)

        Pc = P.mean(axis=0)

        U_hat = P @ A2.T + t_xy
        Uh_c = U_hat.mean(axis=0)

        gamma = 1.0
        if match_scale_to_mask and hist_mask_yx is not None:
            r_hat = np.sqrt(((U_hat - Uh_c) ** 2).sum(axis=1))
            r_hat = np.percentile(r_hat, q)

            Um = np.stack([xx, yy], axis=1).astype(float)
            r_m = np.sqrt(((Um - Uc) ** 2).sum(axis=1))
            r_m = np.percentile(r_m, q)

            if r_hat > 1e-6:
                gamma = float(r_m / r_hat)

        gamma = float(np.clip(gamma, 0.85, 1.15))  # prevent overscale jump

        A2 = gamma * A2
        if center_to_mask:
            t_xy = Uc - Pc @ A2.T

    # (f) decompose A2 into s, R2, A_residual
    s_uni, R2, A_residual = decompose_affine_to_uniform_scale_and_rotation(A2)

    # (g) reprojection
    U_hat = P @ A2.T + t_xy
    err   = np.linalg.norm(U_hat - U_math, axis=1)

    A2_invT = np.linalg.inv(A2).T
    residual_U = (U_math - U_hat)

    repro = {
        "mean_px":   float(err.mean()),
        "median_px": float(np.median(err)),
        "max_px":    float(err.max()),
        "num":       int(P.shape[0]),
    }

    G_affine = {
        "X0": X0,
        "e1": e1,
        "e2": e2,
        "n":  n,
        "A2": A2,
        "t_xy": t_xy,
        # similarity-like pieces for reuse in your TPS code:
        "s":  s_uni,
        "R2": R2,
        "A_residual": A_residual,
        "reprojection": repro,
        "A2_invT": A2_invT,
        "P_lm": P,
        "U_lm": U_math,
        "U_hat": U_hat,
        "residual_U": residual_U,
    }
    return G_affine

def warp_mri_to_hist_canvas_affine(mri_zyx, G, hist_shape_yx,
                                    kernel: str = "catmullrom",
                                    padding_mode: str = "zeros",
                                    device: str | torch.device = "cuda",
                                    dtype: torch.dtype = torch.float32,
                                    order=1, border_value=np.nan):
    """
    Warps MRI (Z,Y,X) onto the histology canvas (H,W) using a 2D affine model.

    Uses the inverse of: U = P @ A2^T + t, with U = (x, y) in image coords.
    """
    H, W = hist_shape_yx

    # Hist grid in image coords
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    Yg, Xg = np.meshgrid(ys, xs, indexing="ij")  # (H,W)

    # Convert to math coords: U = (x, y)
    U_math = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)  # (HW,2)

    # Inverse affine: P = (U - t) @ (A2^{-1})^T
    A2, t = G["A2"], G["t_xy"]
    # A2_inv = np.linalg.inv(A2)
    # P = (U_math - t) @ A2_inv.T  # (HW,2)
    if "A2_invT" in G:
        P = (U_math - t) @ G["A2_invT"]
    else:
        P = (U_math - t) @ np.linalg.inv(A2).T

    # Lift to 3D in (x,y,z): X = X0 + p_x*e1 + p_y*e2
    X0, e1, e2 = G["X0"], G["e1"], G["e2"]
    X_xyz = X0 + P[:, 0:1] * e1 + P[:, 1:2] * e2  # (HW,3)

    # reshape to (H,W,3) for your sampler
    P_vox_xyz = X_xyz.reshape(H, W, 3)

    # --- Torch: vol to (1,1,Z,Y,X)
    vol_t = torch.as_tensor(mri_zyx, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    # --- Torch: points (H,W,3)
    P_t = torch.as_tensor(P_vox_xyz, dtype=dtype, device=device)

    # --- Sample with your custom function (expects x,y,z order)
    out_1_1_hw = sample_mri_cubic_3d(
        vol_1_1_zyx=vol_t,
        P_vox_xyz=P_t,
        kernel=kernel,
        padding_mode=padding_mode,
    )  # (1,1,H,W)

    mri_on_hist_new = out_1_1_hw[0, 0].detach().cpu().numpy()

    return mri_on_hist_new


# ---- Init curved sheet on top of G ---------------------------------------------------------
# ============== utilities ==============

def _to_device(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)

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

def _build_hist_grid(H, W, device):
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    return y, x

@torch.no_grad()
def _tps_residuals(tps, uv_all, xyz_all):
    """Compute residual norms for all points."""
    pred = tps(uv_all)                     # (N,3)
    return (xyz_all - pred).norm(dim=1)    # (N,)

# def make_G_tensors(G, device="cuda"):
#     return {
#         "s":  _to_device(G["s"], device),
#         "R2": _to_device(G["R2"], device),
#         "t":  _to_device(G["t_xy"], device),
#         "X0": _to_device(G["X0"], device),
#         "e1": _to_device(G["e1"], device),
#         "e2": _to_device(G["e2"], device),
#     }

def make_G_tensors(G, device="cuda"):
    out = {
        "s":  _to_device(G["s"], device),
        "R2": _to_device(G["R2"], device),
        "t":  _to_device(G["t_xy"], device),
        "X0": _to_device(G["X0"], device),
        "e1": _to_device(G["e1"], device),
        "e2": _to_device(G["e2"], device),
    }

    # full affine inverse if available
    if "A2_invT" in G:
        out["A2_invT"] = _to_device(G["A2_invT"], device)
    elif "A2" in G:
        A2 = _to_device(G["A2"], device)
        out["A2_invT"] = torch.linalg.inv(A2).T

    # affine residual (fixed in Stage2 unless you later learn it)
    if "A_residual" in G:
        out["A_residual"] = _to_device(G["A_residual"], device)
    else:
        out["A_residual"] = torch.eye(2, device=out["R2"].device, dtype=out["R2"].dtype)

    return out

# def uv_from_hist_grid_inverse_G(H, W, Gt, device="cuda"):
#     """
#     Build hist pixel grid and map (x, y) through inverse similarity to (u,v).
#     Returns: (HW,2) parameter coords.
#     """
#     y, x = _build_hist_grid(H, W, device)
#     U_math = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)  # (HW,2)
#     # Inverse: P = ((U - t) @ R2) / s
#     P = ((U_math - Gt["t"]) @ Gt["R2"]) / Gt["s"]
#     return P  # (HW,2)

def uv_from_hist_grid_inverse_G(H, W, Gt, device="cuda"):
    y, x = _build_hist_grid(H, W, device)
    U_math = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)  # (HW,2)

    if "A2_invT" in Gt:
        # full affine inverse
        P = (U_math - Gt["t"]) @ Gt["A2_invT"]
    else:
        # fallback similarity inverse
        P = ((U_math - Gt["t"]) @ Gt["R2"]) / Gt["s"]
    return P

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

def uv_from_matches_and_G(matches_filtered, G, device="cuda"):
    """
    Convert match list + similarity transform G into:
      - uv : (N,2) parameter coords (after inverse G)
      - X_xyz : (N,3) MRI voxel coords (x,y,z)

    matches_filtered: list of dicts with keys:
        'mri_coords'      -> (z,y,x) voxel indices
        'hist_center_yx'  -> (y,x)   histology pixel coords
    G: dict with
        "s", "R2", "t_xy"  (similarity transform pieces)
    """
    pts_hist = []   # histology (x,y)
    pts_xyz  = []   # MRI (x,y,z) in voxel indices

    for m in matches_filtered:
        z, y, x   = m['mri_coords']      # (z,y,x) in voxels
        cy, cx    = m['hist_center_yx']  # (y,x) in hist pixels
        if (z, y, x) == (-1, -1, -1):
            continue
        pts_hist.append((cx, cy))        # hist (x,y)
        pts_xyz.append((x, y, z))        # MRI  (x,y,z) in voxels

    U = torch.as_tensor(np.array(pts_hist, float),
                        device=device, dtype=torch.float32)  # (N,2)
    X_xyz = torch.as_tensor(np.array(pts_xyz, float),
                            device=device, dtype=torch.float32)  # (N,3)

    # # similarity params from G
    # s  = torch.as_tensor(G["s"],     device=device, dtype=torch.float32)
    # R2 = torch.as_tensor(G["R2"],    device=device, dtype=torch.float32)
    # t  = torch.as_tensor(G["t_xy"],  device=device, dtype=torch.float32)

    # # Inverse similarity: U ≈ s*(P @ R^T) + t  ⇒ P ≈ ((U - t) @ R) / s
    # uv = ((U - t) @ R2) / s  # (N,2) parameter coords
    t = torch.as_tensor(G["t_xy"], device=device, dtype=torch.float32)

    if "A2_invT" in G:
        A2_invT = torch.as_tensor(G["A2_invT"], device=device, dtype=torch.float32)
        uv = (U - t) @ A2_invT
    else:
        s  = torch.as_tensor(G["s"],  device=device, dtype=torch.float32)
        R2 = torch.as_tensor(G["R2"], device=device, dtype=torch.float32)
        uv = ((U - t) @ R2) / s

    return uv, X_xyz

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

def robust_fit_tps_ransac(
    matches_filtered,
    G,
    device="cuda",
    # RANSAC controls:
    n_iters: int = 200,
    sample_size: int = 30,        # TPS needs more than a minimal sample; 20–60 typical
    inlier_thresh: float = 5.0,   # voxels
    min_inliers: int = 20,
    # TPS controls:
    lam_hyp: float = 1e-3,        # λ used inside hypotheses (can be a bit > 0)
    lam_final: float = 1e-3,      # λ for final refit on consensus set
    normalize_uv: bool = False,
    # scoring:
    score_type: str = "inliers",  # "inliers" or "truncated"
    truncated_cap: float = 10.0,  # used if score_type="truncated"
    seed: int = 0
):
    """
    RANSAC-like TPS:
      - sample subset -> fit TPS -> score inliers by residual < inlier_thresh
      - keep best hypothesis
      - refit TPS on best inliers (final)

    Returns
    -------
    tps_final, uv_in, xyz_in, inlier_mask, residuals, best_score
    """
    print("\n-------- Estimate Initial Curved Sheet --------")

    device = torch.device(device)

    # Step 1: matches -> controls
    uv_ctrl, xyz_ctrl = uv_from_matches_and_G(matches_filtered, G, device=device)
    N = uv_ctrl.shape[0]
    if N < max(3, sample_size):
        raise ValueError(f"Not enough points for RANSAC TPS: N={N}, sample_size={sample_size}")

    dtype = uv_ctrl.dtype
    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(seed)

    best_inlier_mask = None
    best_score = None
    best_residuals = None

    # Precompute for speed
    all_idx = torch.arange(N, device=device)

    for it in range(n_iters):
        # Random subset
        perm = torch.randperm(N, generator=g_cpu, device="cpu")[:sample_size]
        idx = all_idx[perm.to(device)]

        # Fit hypothesis TPS
        try:
            tps_h = _fit_tps_on_subset(
                uv_ctrl, xyz_ctrl, idx,
                device=device, dtype=dtype,
                lam=lam_hyp, normalize_uv=normalize_uv
            )
        except RuntimeError:
            # singular system or solve issues -> skip
            continue

        # Score hypothesis
        residuals = _tps_residuals(tps_h, uv_ctrl, xyz_ctrl)
        inlier_mask = residuals <= inlier_thresh
        num_inliers = int(inlier_mask.sum())

        if num_inliers < min_inliers:
            continue

        if score_type == "inliers":
            score = num_inliers  # maximize
            better = (best_score is None) or (score > best_score)
        elif score_type == "truncated":
            # minimize robust cost: sum(min(r, cap))
            cost = torch.clamp(residuals, max=truncated_cap).sum().item()
            score = -cost  # maximize negative cost
            better = (best_score is None) or (score > best_score)
        else:
            raise ValueError("score_type must be 'inliers' or 'truncated'")

        if better:
            best_score = score
            best_inlier_mask = inlier_mask.clone()
            best_residuals = residuals.clone()

            # Optional early stop if almost all points are inliers
            if num_inliers > 0.9 * N:
                break

    if best_inlier_mask is None:
        # Fallback: no good hypothesis found -> fit on all points
        print("RANSAC TPS: no valid hypothesis found; falling back to all points.")
        best_inlier_mask = torch.ones(N, device=device, dtype=torch.bool)
        best_residuals = torch.full((N,), float("nan"), device=device, dtype=dtype)

    uv_in  = uv_ctrl[best_inlier_mask]
    xyz_in = xyz_ctrl[best_inlier_mask]

    # Final refit on inliers
    tps_final = ThinPlateSplineSurface3D(device=device, dtype=torch.float32)
    tps_final.fit_from_uv_xyz(
        ctrl_uv=uv_in,
        targets_xyz=xyz_in,
        lam=lam_final,
        normalize_uv=normalize_uv
    ).to(device)

    # Recompute residuals with final model (so residuals correspond to returned tps_final)
    residuals_final = _tps_residuals(tps_final, uv_ctrl, xyz_ctrl)

    return tps_final, uv_in, xyz_in, best_inlier_mask, residuals_final, best_score


@torch.no_grad()
def robust_fit_tps_ransac_mad(
    matches_filtered,
    G,
    device="cuda",
    # Robustness selection:
    method: str = "ransac",      # "ransac" (legacy), "mad" (MAD trim), "all" (no robust step)
    # RANSAC controls (used only when method == "ransac"):
    n_iters: int = 200,
    sample_size: int = 30,
    inlier_thresh: float = 5.0,
    min_inliers: int = 20,
    score_type: str = "inliers",   # or "truncated"
    truncated_cap: float = 10.0,
    seed: int = 0,
    # MAD controls (used only when method == "mad"):
    mad_thresh: float = 3.5,       # threshold in MAD units
    mad_iterations: int = 1,
    # TPS controls:
    lam_hyp: float = 1e-3,
    lam_final: float = 1e-3,
    normalize_uv: bool = False,
):
    """
    Flexible TPS fitting:
      - method="ransac": original RANSAC loop.
      - method="mad":    iterative MAD trimming on residuals (no random subsets).
      - method="all":    fit once on all points.
    Returns: tps_final, uv_in, xyz_in, inlier_mask, residuals_final, best_score
    """
    print(f"\n-------- Estimate Initial Curved Sheet (method={method}) --------")
    device = torch.device(device)
    uv_ctrl, xyz_ctrl = uv_from_matches_and_G(matches_filtered, G, device=device)
    N = uv_ctrl.shape[0]
    if N < 3:
        raise ValueError(f"Not enough points to fit TPS: N={N}")

    def _final_refit(inlier_mask, lam):
        uv_in = uv_ctrl[inlier_mask]
        xyz_in = xyz_ctrl[inlier_mask]
        tps = ThinPlateSplineSurface3D(device=device, dtype=torch.float32)
        tps.fit_from_uv_xyz(
            ctrl_uv=uv_in, targets_xyz=xyz_in, lam=lam, normalize_uv=normalize_uv
        ).to(device)
        residuals_final = _tps_residuals(tps, uv_ctrl, xyz_ctrl)
        return tps, uv_in, xyz_in, residuals_final

    # --- No robustness ---
    if method == "all":
        inlier_mask = torch.ones(N, device=device, dtype=torch.bool)
        tps_final, uv_in, xyz_in, residuals_final = _final_refit(inlier_mask, lam_final)
        best_score = int(inlier_mask.sum())
        return tps_final, uv_in, xyz_in, inlier_mask, residuals_final, best_score

    # --- MAD trimming ---
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

    # --- RANSAC (legacy) ---
    if method != "ransac":
        raise ValueError("method must be one of {'ransac', 'mad', 'all'}")
    if N < max(3, sample_size):
        raise ValueError(f"Not enough points for RANSAC TPS: N={N}, sample_size={sample_size}")

    dtype = uv_ctrl.dtype
    g_cpu = torch.Generator(device="cpu"); g_cpu.manual_seed(seed)
    best_inlier_mask = None; best_score = None
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
        print("RANSAC TPS: no valid hypothesis found; falling back to all points.")
        best_inlier_mask = torch.ones(N, device=device, dtype=torch.bool)

    tps_final, uv_in, xyz_in, residuals_final = _final_refit(best_inlier_mask, lam_final)
    return tps_final, uv_in, xyz_in, best_inlier_mask, residuals_final, best_score




def resample_mri_from_G_TPS_torch(
    mri_vol,
    G,
    tps3d_module,
    hist_shape_yx,
    kernel="catmull-rom",
    padding_mode="zeros",
    mri_mask_vol=None,
    mask_threshold=0.5,
    mask_fill_value=0.0,
):
    """
    mri_vol : (1,1,Z,Y,X) torch tensor in voxel space
    mri_mask_vol : optional (1,1,Z,Y,X) torch tensor (binary/prob mask) in voxel space
    G       : dict with "s","R2","t_xy","X0","e1","e2"
    tps3d_module : ThinPlateSplineSurface3D (fitted)
    hist_shape_yx : (H,W)

    Returns:
      warped_mri       : (1,1,H,W)
      xyz_norm         : (H,W,3) in [0,1] for viz
      warped_mask      : (1,1,H,W) or None
      warped_mri_masked: (1,1,H,W) or None
    """
    device = mri_vol.device
    H, W = hist_shape_yx
    Zdim, Ydim, Xdim = mri_vol.shape[-3:]

    # --- G tensors
    Gt = make_G_tensors(G, device=device)

    # --- (u,v) for each hist pixel
    uv = uv_from_hist_grid_inverse_G(H, W, Gt, device=device)  # (HW,2)

    # --- TPS sheet points in voxel coords (x,y,z)
    Xyz = tps3d_module(uv)                 # (HW,3)
    P_vox_xyz = Xyz.reshape(H, W, 3)       # (H,W,3) (safer than view)

    # --- Warp MRI with your cubic sampler
    warped_mri = sample_mri_cubic_3d(
        mri_vol,
        P_vox_xyz,
        kernel=kernel,
        padding_mode=padding_mode,
    )  # expected (1,1,H,W)

    # --- xyz_norm in [0,1] for visualization
    xn01 = P_vox_xyz[..., 0] / (Xdim - 1.0)
    yn01 = P_vox_xyz[..., 1] / (Ydim - 1.0)
    zn01 = P_vox_xyz[..., 2] / (Zdim - 1.0)
    xyz_norm = torch.stack([xn01, yn01, zn01], dim=-1)  # (H,W,3)

    warped_mask = None
    warped_mri_masked = None

    # --- If mask is provided: warp it (nearest) and apply it
    if mri_mask_vol is not None:
        # Build grid for grid_sample: (1, 1, H, W, 3) in [-1,1]
        # Use Xyz flat then reshape to force correct rank.
        xg = 2.0 * (Xyz[:, 0] / (Xdim - 1.0)) - 1.0
        yg = 2.0 * (Xyz[:, 1] / (Ydim - 1.0)) - 1.0
        zg = 2.0 * (Xyz[:, 2] / (Zdim - 1.0)) - 1.0
        grid = torch.stack([xg, yg, zg], dim=-1).reshape(1, 1, H, W, 3)

        # Warp mask with nearest neighbor. Output: (1,1,1,H,W) -> squeeze D_out
        warped_mask = F.grid_sample(
            mri_mask_vol.float(),
            grid,
            mode="nearest",
            padding_mode="zeros",   # masks should usually go to 0 outside
            align_corners=True,
        ).squeeze(2)  # (1,1,H,W)

        # Binarize (optional but recommended for masks)
        warped_mask_bin = (warped_mask >= mask_threshold).to(warped_mri.dtype)

        # Apply mask to warped MRI
        warped_mri_masked = warped_mri * warped_mask_bin

        # Optionally fill outside with a constant instead of 0:
        if mask_fill_value != 0.0:
            warped_mri_masked = torch.where(
                warped_mask_bin > 0,
                warped_mri_masked,
                torch.as_tensor(mask_fill_value, device=device, dtype=warped_mri.dtype),
            )

    return warped_mri, xyz_norm, warped_mask, warped_mri_masked

