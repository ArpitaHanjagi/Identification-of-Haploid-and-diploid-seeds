"""Check that this PC is ready to train: GPU, PyTorch build, dataset, split, folds, weights.

    python check_setup.py               # report problems; changes nothing
    python check_setup.py --download    # also download the ImageNet weights of every backbone

Run it on a new PC (for example the workstation) after installing the packages and copying
the dataset, and before the first long run. Every line starts with OK, NOTE or PROBLEM.
"""
import argparse
import importlib
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from common import DATA_DIR, FOLDS_CSV, SPLITS_CSV, check_folds, cv_folds

ft = importlib.import_module("04_finetune")
problems = []


def ok(msg):
    print(f"OK       {msg}")


def note(msg):
    print(f"NOTE     {msg}")


def problem(msg):
    problems.append(msg)
    print(f"PROBLEM  {msg}")


def check_gpu():
    import pennylane
    import torchvision
    print(f"Python {platform.python_version()}, PyTorch {torch.__version__} (CUDA build {torch.version.cuda}), "
          f"torchvision {torchvision.__version__}, PennyLane {pennylane.__version__}, {platform.system()}")
    threads = ft.auto_threads()
    print(f"CPU: {os.cpu_count()} logical processors -> auto threads {threads}")
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            ok("Apple GPU (MPS) available; training uses it automatically (no bf16, no workers needed)")
        elif torch.version.cuda is None and sys.platform != "darwin":
            problem("PyTorch is the CPU-only build. For an NVIDIA GPU install requirements-cu130.txt "
                    "(Blackwell, RTX 50, most GPUs) or requirements.txt (older GPUs and drivers)")
        else:
            problem("no GPU found by PyTorch. Check the NVIDIA driver with nvidia-smi")
        return
    d = torch.device("cuda")
    name, cap = torch.cuda.get_device_name(d), torch.cuda.get_device_capability(d)
    mem = torch.cuda.get_device_properties(d).total_memory / 2 ** 30
    print(f"GPU: {name}, compute capability {cap[0]}.{cap[1]}, {mem:.0f} GB")
    archs = torch.cuda.get_arch_list()
    if f"sm_{cap[0]}{cap[1]}" not in archs:
        problem(f"this PyTorch build has no code for compute capability {cap[0]}.{cap[1]} (it has {', '.join(archs)}). "
                f"For Blackwell (12.0) install requirements-cu130.txt")
        return
    try:
        a = torch.randn(1024, 1024, device=d)
        (a @ a).sum().item()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (a @ a).float().sum().item()
    except Exception as e:  # noqa: BLE001
        problem(f"the GPU failed a small test calculation: {e}")
        return
    ok(f"GPU works with this PyTorch build; --amp auto will use {ft.pick_amp('auto', d)}, "
       f"--workers auto {min(8, threads)}")
    if mem < 8:
        note(f"{mem:.0f} GB of GPU memory: bigger backbones or sizes above 224 px may need --batch-size 16")


def check_data():
    counts = {c: len(list((DATA_DIR / c).glob("*.jpg"))) for c in ["haploid", "diploid"]}
    if counts != {"haploid": 1230, "diploid": 1770}:
        problem(f"dataset at {DATA_DIR} has {counts}; expected 1230 haploid and 1770 diploid. "
                f"Set HAPLOID_DATA_DIR to the folder with the haploid/ and diploid/ subfolders")
        return
    ok(f"dataset: {DATA_DIR} (1230 haploid, 1770 diploid)")
    if not SPLITS_CSV.exists():
        problem("results/splits.csv is missing: run 01_make_splits.py")
        return
    df = pd.read_csv(SPLITS_CSV)
    test = df.query("split == 'test'")["class"].value_counts().to_dict()
    missing = sum(not Path(p).exists() for p in df["path"])
    if test != {"diploid": 266, "haploid": 184} or missing:
        problem(f"results/splits.csv does not fit this PC (test set {test}, {missing} paths not found). "
                f"Run 01_make_splits.py again; the test set must be 184 haploid + 266 diploid")
        return
    ok("split: results/splits.csv matches this dataset (test set 184 haploid + 266 diploid)")
    if FOLDS_CSV.exists():
        try:
            check_folds(df, cv_folds(df["label"].values, df["split"].values), save_if_missing=False)
            ok("cross-validation folds are identical to results/cv/folds.csv")
        except SystemExit as e:
            problem(str(e))
    else:
        note("results/cv/folds.csv not found; the first cross-validation run will create it")


def check_weights(download):
    from torchvision import models
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    absent = []
    for name, (arch, weights, _, display) in ft.BACKBONES.items():
        w = models.get_model_weights(arch)[weights]
        if (hub / Path(w.url).name).exists():
            continue
        if download:
            print(f"downloading {display} ImageNet weights ...", flush=True)
            w.get_state_dict(progress=False)
        else:
            absent.append(name)
    if absent:
        note(f"ImageNet weights not downloaded yet for {', '.join(absent)}. They download on first use "
             f"(needs internet), or run check_setup.py --download. Offline PC: copy {hub} from another PC")
    else:
        ok(f"ImageNet weights of all backbones are in {hub}")


def check_qcnn():
    model = ft.SeedClassifier("qcnn", pretrained=False)
    t = time.time()
    z = torch.rand(32, 8, requires_grad=True)
    model.head(z).sum().backward()
    ok(f"QCNN simulator works on the CPU ({time.time() - t:.2f} s for one batch, forward and backward)")


def main():
    ap = argparse.ArgumentParser(description="Check that this PC is ready to train.")
    ap.add_argument("--download", action="store_true", help="download the ImageNet weights of every backbone")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    check_gpu()
    check_data()
    check_weights(args.download)
    check_qcnn()
    print("\nReady to train." if not problems else f"\n{len(problems)} problem(s) to fix before training.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
