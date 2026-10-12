"""Compute plain and D4-TTA probabilities on validation and test for every finished run.

Only validation accuracy is printed; test predictions are saved for 06_report.py,
which applies decisions that were made on validation alone.
"""
import importlib
import sys

import numpy as np
import torch

from common import RESULTS_DIR, keep_awake
from tta import predict_probs

ft = importlib.import_module("04_finetune")


def main():
    keep_awake()
    data = ft.load_images()
    split = np.array(data["split"])
    sel = {s: torch.from_numpy(split == s) for s in ["val", "test"]}
    x = {s: data["images"][m] for s, m in sel.items()}
    y = {s: data["labels"][m].numpy() for s, m in sel.items()}

    for run in sorted((RESULTS_DIR / "finetune").glob("*_seed*")):
        best, out = run / "best_model.pt", run / "probs.npz"
        if not (run / "test_result.json").exists():
            print(f"skip {run.name}: training not finished", flush=True)
            continue
        if out.exists() and "--force" not in sys.argv:
            print(f"skip {run.name}: probs.npz exists", flush=True)
            continue
        model = ft.SeedClassifier(run.name.split("_seed")[0])
        model.load_state_dict(torch.load(best))
        res = {f"y_{s}": y[s] for s in x}
        for s in x:
            res[f"{s}_plain"] = predict_probs(model, x[s], ft.normalize, use_tta=False)
            res[f"{s}_tta"] = predict_probs(model, x[s], ft.normalize, use_tta=True)
        np.savez(out, **res)
        acc = lambda p: ((p >= 0.5) == y["val"]).mean()  # noqa: E731
        print(f"{run.name}: val plain {acc(res['val_plain']):.4f}  val TTA {acc(res['val_tta']):.4f}", flush=True)


if __name__ == "__main__":
    main()
