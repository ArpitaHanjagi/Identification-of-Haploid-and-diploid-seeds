"""Compute plain and D4-TTA probabilities on validation and test for every finished run.

Only validation accuracy is printed; test predictions are saved for 06_report.py,
which applies decisions that were made on validation alone.

    python 05_evaluate.py                          # ResNet18 at 224 px (the reported results)
    python 05_evaluate.py --variant resnet50_384   # runs made with --backbone resnet50 --img-size 384
"""
import argparse
import importlib

import numpy as np
import torch

from common import DEFAULT_VARIANT, keep_awake, runs_root
from tta import predict_probs

ft = importlib.import_module("04_finetune")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=DEFAULT_VARIANT, help="<backbone>_<img size>, e.g. resnet50_384")
    ap.add_argument("--force", action="store_true", help="recompute runs that already have probs.npz")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    keep_awake()
    device = ft.pick_device(args.device)
    amp = ft.pick_amp("auto", device)
    print(f"device: {device}  amp: {amp}", flush=True)
    root = runs_root("finetune", args.variant)
    data = {}
    for run in sorted(root.glob("*_seed*")):
        out = run / "probs.npz"
        if not (run / "test_result.json").exists():
            print(f"skip {run.name}: training not finished", flush=True)
            continue
        if out.exists() and not args.force:
            print(f"skip {run.name}: probs.npz exists", flush=True)
            continue
        model, cfg = ft.load_trained(run, run.name.split("_seed")[0])
        model.place(device, amp)
        if cfg["img_size"] not in data:
            d = ft.load_images(cfg["img_size"])
            split = np.array(d["split"])
            data[cfg["img_size"]] = {s: (d["images"][torch.from_numpy(split == s)],
                                         d["labels"][torch.from_numpy(split == s)].numpy()) for s in ["val", "test"]}
        xy = data[cfg["img_size"]]
        res = {f"y_{s}": y for s, (_, y) in xy.items()}
        for s, (x, _) in xy.items():
            res[f"{s}_plain"] = predict_probs(model, x, ft.normalize, use_tta=False)
            res[f"{s}_tta"] = predict_probs(model, x, ft.normalize, use_tta=True)
        np.savez(out, **res)
        acc = lambda p: ((p >= 0.5) == res["y_val"]).mean()  # noqa: E731
        print(f"{run.name}: val plain {acc(res['val_plain']):.4f}  val TTA {acc(res['val_tta']):.4f}", flush=True)


if __name__ == "__main__":
    main()
