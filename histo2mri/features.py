"""Patch descriptors and masked patch similarities for histology-MRI matching (torch, GPU).

Used by matching to score histology patches against in-plane MRI slab
patches. Provides HOG (orientation histograms of Sobel gradients), a MIND-style
self-similarity descriptor over an 8-neighbourhood, and normalised gradient
fields (NGF). The pairwise functions compare one histology patch with a batch
of MRI patches, each pair restricted to its own mask. Images are 2D (y, x) and
patches flattened row-major; P = patch * patch.
"""
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
import torch


def _to_numpy_image(x):
    """Return a sitk image or array as numpy, dropping a leading singleton axis."""
    if isinstance(x, sitk.Image):
        arr = sitk.GetArrayFromImage(x)
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _prep_inputs(proc_Ih, histo_mask, slab_Im, slab_Mm):
    """Return (histology [H, W] float32, its bool mask, MRI [Z, Y, X] float32, its bool mask)."""
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
    """Return 3x3 Sobel gradients (gx, gy) of a 2D image as [H, W] tensors (zero padding)."""
    x = torch.as_tensor(img_hw, device=device, dtype=torch.float32)[None, None]
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=torch.float32).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=torch.float32).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)[0, 0]
    gy = F.conv2d(x, ky, padding=1)[0, 0]
    return gx, gy


def _unfold_hw(t_hw, patch, stride):
    """Extract patch x patch windows at the given stride from [H, W]; returns [N, P] (row-major)."""
    return F.unfold(t_hw[None, None], kernel_size=(patch, patch), stride=(stride, stride)).squeeze(0).T


@torch.no_grad()
def _mind_2d(img, radius=2, dilation=1, eps=1e-8):
    """Compute a MIND-style self-similarity descriptor with 8 channels per pixel.

    For each of the 8 neighbour offsets (scaled by dilation), the squared
    difference to the shifted image is box-averaged over (2*radius+1)^2; the
    descriptor is exp(-ssd / mean_ssd) with the mean over channels, L2
    normalised across channels. Reflect padding at the image border.
    """
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


def _norm01_torch(x, eps=1e-8):
    """Min-max scale the finite entries of a tensor to [0, 1]; non-finite entries become 0."""
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
    """Compute HOG descriptors for a batch of flattened gradient patches.

    Gradient magnitude, zeroed outside mk_u, is voted into `bins` orientation
    bins (signed: [0, 2*pi), unsigned: [0, pi)) with linear interpolation
    between the two nearest bins, per cell_size x cell_size cell. The
    concatenated cell histograms are L2 normalised as a whole (no block
    normalisation).

    Args:
        gx_u: [N, P] x gradients of N patches.
        gy_u: [N, P] y gradients.
        mk_u: [N, P] bool mask.
        patch: patch edge length (must be a multiple of cell_size).
        cell_size: cell edge length in pixels.
        bins: number of orientation bins.
        eps: stabiliser for the magnitude and the normalisation.
        signed: use signed (2*pi) or unsigned (pi) orientations.

    Returns:
        [N, (patch // cell_size)^2 * bins] L2-normalised descriptors.
    """
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
    """Compute masked MIND descriptors for a batch of patches; returns [B, 8 * P], L2 normalised."""
    # patches_bpp: [B,p,p], masks_bp: [B,P] bool
    B, p, _ = patches_bpp.shape
    x = patches_bpp[:, None, :, :]  # [B,1,p,p]
    d = _mind_2d(x, radius=radius, dilation=dilation, eps=eps)  # [B,D,p,p]
    D = d.shape[1]
    d = d.reshape(B, D, p * p)
    d = d * masks_bp[:, None, :].to(d.dtype)
    d = d.reshape(B, -1)
    d = F.normalize(d, p=2, dim=1, eps=eps)
    return d


def _masked_minmax_normalize_patch_batch(patches_bpp, masks_bp=None, eps=1e-8):
    """Min-max scale each patch over its masked pixels and zero the rest; returns [B, p, p].

    A patch with an empty mask falls back to its whole-patch min/max (and is then all zero).
    """
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
def _sobel_xy_patch_batch(patches_bpp):
    """Return 3x3 Sobel gradients (gx, gy), each [B, p, p], of a patch batch (zero padding)."""
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
    """Return the HOG cosine similarity [B] of one histology patch with B MRI patches.

    Gradients are taken on the full patches; each pair's HOG uses only the
    pixels in its pair mask.
    """
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
    """Return the MIND cosine similarity [B] of one histology patch with B MRI patches.

    Descriptors are computed on the full patches and restricted to each pair's
    mask before normalising.
    """
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
    """Return the masked NGF similarity [B] of one histology patch with B MRI patches.

    Both patches are min-max normalised inside the pair mask (zero outside);
    the score is the mean over masked pixels of
    (grad I . grad J)^2 / (|grad I|^2 |grad J|^2 + eps), in [0, 1].
    """
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


def _to_np2d(x):
    """Return a sitk image or array as a 2D float32 array (first slice if 3D)."""
    arr = _to_numpy_image(x)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float32)

