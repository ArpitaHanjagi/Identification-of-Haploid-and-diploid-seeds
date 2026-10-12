"""Extract frozen ResNet50 (ImageNet) features once, so heads train quickly on CPU.

Seeds have varied aspect ratios on a white background, so each image is padded
to a square with white before resizing; a center crop could cut off the embryo,
which is where the R1-nj marker is. Training images also get augmented views
(rotation, flips, mild brightness/contrast). Hue is left alone because colour
is the signal.
"""
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

from common import FEATURES_NPZ, IMG_SIZE, N_AUGMENTED_VIEWS, SPLIT_SEED, SPLITS_CSV, keep_awake

WHITE = (255, 255, 255)
PARTS_DIR = FEATURES_NPZ.parent / "feature_parts"


def pad_to_square(img: Image.Image) -> Image.Image:
    side = max(img.size)
    return ImageOps.pad(img, (side, side), color=WHITE)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
clean_tf = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])
augment_tf = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.RandomRotation(180, fill=WHITE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    normalize,
])


@torch.no_grad()
def extract(backbone, paths, tf, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
        feats.append(backbone(batch).numpy())
    return np.concatenate(feats).astype(np.float32)


def cached(name, compute):
    """Save each stage to its own file so an interruption only loses that stage."""
    path = PARTS_DIR / f"{name}.npy"
    if path.exists():
        print(f"{name}: already done, skipping", flush=True)
        return np.load(path)
    t0 = time.time()
    arr = compute()
    np.save(path, arr)
    print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    return arr


def main():
    keep_awake()
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SPLITS_CSV)
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = torch.nn.Identity()
    backbone.eval()

    out = {}
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        paths = part["path"].tolist()
        out[f"{split}_x"] = cached(f"{split}_x", lambda: extract(backbone, paths, clean_tf))
        out[f"{split}_y"] = part["label"].to_numpy()
        if split == "train":
            for v in range(N_AUGMENTED_VIEWS):
                # Seed each view separately so a rerun reproduces the same augmentations.
                def compute(v=v):
                    torch.manual_seed(SPLIT_SEED + v)
                    return extract(backbone, paths, augment_tf)
                out[f"train_aug{v}_x"] = cached(f"train_aug{v}_x", compute)

    np.savez_compressed(FEATURES_NPZ, **out)
    print(f"saved {FEATURES_NPZ}", flush=True)


if __name__ == "__main__":
    main()
