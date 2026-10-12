"""Test-time augmentation over the 8 exact symmetries of a square image (the dihedral group D4).

Seeds are photographed from above, so a seed rotated by 90 degrees or mirrored is still a
valid seed image. Rotations by multiples of 90 degrees on a square tensor are lossless.
"""
import numpy as np
import torch


def d4_views(x):
    """Identity, 3 rotations, and the same 4 after a horizontal flip."""
    views = [torch.rot90(x, k, dims=(-2, -1)) for k in range(4)]
    flipped = torch.flip(x, dims=(-1,))
    return views + [torch.rot90(flipped, k, dims=(-2, -1)) for k in range(4)]


@torch.no_grad()
def predict_probs(model, images, normalize, use_tta, batch_size=32):
    """Haploid probability per image, averaged over the 8 views when use_tta is set."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        xb = images[i:i + batch_size]
        views = d4_views(xb) if use_tta else [xb]
        probs = [torch.softmax(model(normalize(v)), 1)[:, 1] for v in views]
        out.append(torch.stack(probs).mean(0))
    return torch.cat(out).numpy()


def choose_threshold(y_val, p_val, lo=0.2, hi=0.8):
    """Threshold that maximises validation accuracy (ties: closest to 0.5), limited to [lo, hi]."""
    cands = np.unique(np.concatenate([p_val, [0.5]]))
    mids = np.concatenate([[0.5], (cands[:-1] + cands[1:]) / 2])
    mids = mids[(mids >= lo) & (mids <= hi)]
    if len(mids) == 0:
        return 0.5
    accs = np.array([((p_val >= t).astype(int) == y_val).mean() for t in mids])
    best = np.flatnonzero(accs == accs.max())
    return float(mids[best[np.argmin(np.abs(mids[best] - 0.5))]])
