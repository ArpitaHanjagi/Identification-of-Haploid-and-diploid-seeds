"""Classify seed photos with a trained model, for demos and quick checks.

Examples:
    python predict.py photo1.jpg photo2.jpg     # one or more photos
    python predict.py path\\to\\folder            # every photo in a folder
    python predict.py --from-test 10            # 10 random test-set photos, shown with their true labels

By default it uses each head's final configuration from results/final_decisions.json (chosen on
validation), with the same preprocessing and test-time augmentation as the reported results.
"""
import argparse
import importlib
import json
import warnings
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from common import DATA_DIR, RESULTS_DIR, SPLITS_CSV
from tta import predict_probs

warnings.filterwarnings("ignore", category=UserWarning)
ft = importlib.import_module("04_finetune")
PHOTO_TYPES = {".jpg", ".jpeg", ".png"}


def load_photo(path):
    """Same preprocessing as training: pad to a white square, resize to 224 x 224 (uint8 tensor)."""
    im = Image.open(path).convert("RGB")
    side = max(im.size)
    im = ImageOps.pad(im, (side, side), color=(ft.WHITE,) * 3).resize((ft.IMG_SIZE, ft.IMG_SIZE), Image.BICUBIC)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1)


def expand(paths):
    out = []
    for p in map(Path, paths):
        out += sorted(f for f in p.iterdir() if f.suffix.lower() in PHOTO_TYPES) if p.is_dir() else [p]
    return out


def locate(path, cls):
    """splits.csv holds absolute paths from the PC that made the split. On another PC, look for
    the same file under DATA_DIR/<class>/ instead (set HAPLOID_DATA_DIR to the dataset folder)."""
    if Path(path).exists():
        return Path(path)
    moved = DATA_DIR / cls / PureWindowsPath(path).name
    if not moved.exists():
        raise SystemExit(f"cannot find {PureWindowsPath(path).name}: set HAPLOID_DATA_DIR to the folder "
                         f"that holds the haploid/ and diploid/ subfolders (now: {DATA_DIR})")
    return moved


def final_choice(head):
    """Seeds, TTA and threshold that 06_report.py chose for this head on validation."""
    d = json.loads((RESULTS_DIR / "final_decisions.json").read_text())[head]
    return (d["seeds"] if d["use_ensemble"] else [d["seed"]]), d["tta"], d["threshold"]


def main():
    ap = argparse.ArgumentParser(description="Classify maize seed photos as haploid or diploid.")
    ap.add_argument("photos", nargs="*", help="photo files or folders")
    ap.add_argument("--head", choices=["qcnn", "classical"], default="qcnn")
    ap.add_argument("--seed", type=int, help="use this one training run instead of the final choice")
    ap.add_argument("--no-tta", action="store_true", help="judge each photo once instead of 8 ways")
    ap.add_argument("--from-test", type=int, metavar="N", help="pick N random photos from the test set")
    ap.add_argument("--sample-seed", type=int, default=0, help="which random test photos --from-test picks")
    ap.add_argument("--device", default="auto", help="auto (default), cuda, mps or cpu")
    args = ap.parse_args()

    seeds, tta, threshold = final_choice(args.head)
    if args.seed is not None:
        seeds = [args.seed]
    tta = tta and not args.no_tta
    if args.from_test:
        test = pd.read_csv(SPLITS_CSV).query("split == 'test'")
        test = test.sample(min(args.from_test, len(test)), random_state=args.sample_seed)
        paths, truth = [locate(p, c) for p, c in zip(test["path"], test["class"])], test["class"].tolist()
    else:
        paths, truth = expand(args.photos), None
    if not paths:
        ap.error("give photo files, a folder, or --from-test N")

    device = ft.pick_device(args.device)
    images = torch.stack([load_photo(p) for p in paths])
    probs = []
    for s in seeds:
        model = ft.SeedClassifier(args.head)
        weights = RESULTS_DIR / "finetune" / f"{args.head}_seed{s}" / "best_model.pt"
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        probs.append(predict_probs(model.place(device), images, ft.normalize, use_tta=tta))
    p_haploid = np.mean(probs, axis=0)

    runs = f"seed {seeds[0]}" if len(seeds) == 1 else f"average of seeds {', '.join(map(str, seeds))}"
    print(f"\nModel: ResNet18 + {'QCNN' if args.head == 'qcnn' else 'classical'} head ({runs}), "
          f"{'with' if tta else 'without'} TTA, threshold {threshold:g}, device {device}\n")
    name_w = max(len(p.name) for p in paths)
    print(f"{'photo':{name_w}}  P(haploid)  prediction" + ("  true label  result" if truth else ""))
    for i, path in enumerate(paths):
        pred = "haploid" if p_haploid[i] >= threshold else "diploid"
        row = f"{path.name:{name_w}}  {p_haploid[i]:10.3f}  {pred:10}"
        if truth:
            row += f"  {truth[i]:10}  {'correct' if pred == truth[i] else 'WRONG'}"
        print(row)
    if truth:
        right = sum(("haploid" if q >= threshold else "diploid") == t for q, t in zip(p_haploid, truth))
        print(f"\n{right} of {len(paths)} correct ({100 * right / len(paths):.1f}%)")


if __name__ == "__main__":
    main()
