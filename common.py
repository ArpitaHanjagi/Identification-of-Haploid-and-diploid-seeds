"""Shared paths, settings and model definitions for the maize ploidy experiments."""
import ctypes
import os
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pennylane as qml
import torch
from torch import nn

# Folder with haploid/ and diploid/ subfolders. Set HAPLOID_DATA_DIR to use another
# location (the Kaggle notebook does this).
DATA_DIR = Path(os.environ.get(
    "HAPLOID_DATA_DIR",
    r"S:\Codes and Programs\Programs\Dev\Haploid_Diploid_Classification\haploid_and_diploid_maize_seeds_dataset"))
PROJECT_DIR = Path(__file__).resolve().parent
SPLITS_CSV = PROJECT_DIR / "results" / "splits.csv"
FEATURES_NPZ = PROJECT_DIR / "results" / "resnet50_features.npz"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"

# Haploid is the positive class (index 1), following Sabadin et al. (2021).
CLASSES = ["diploid", "haploid"]
SPLIT_SEED = 42
CV_FOLDS = 5  # cross-validation folds over the 2,550 non-test images
IMG_SIZE = 224
N_AUGMENTED_VIEWS = 4  # extra augmented copies of each training image

N_QUBITS = 4
N_QLAYERS = 2
FEATURE_DIM = 2048  # ResNet50 global-average-pooled output


def cv_folds(labels, split, n_folds: int = CV_FOLDS) -> np.ndarray:
    """Cross-validation fold (0 .. n_folds-1) of every image; -1 for the test images.

    Only the train + validation images (2,550) are folded, stratified by class and
    shuffled with SPLIT_SEED, so the 450 test images never enter cross-validation.
    """
    from sklearn.model_selection import StratifiedKFold
    labels, split = np.asarray(labels), np.asarray(split)
    fold = np.full(len(split), -1)
    pool = np.flatnonzero(split != "test")
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=SPLIT_SEED)
    for k, (_, held_out) in enumerate(skf.split(pool, labels[pool])):
        fold[pool[held_out]] = k
    return fold


FOLDS_CSV = RESULTS_DIR / "cv" / "folds.csv"
DEFAULT_VARIANT = "resnet18_224"  # backbone and image size of the reported results


def variant_name(backbone: str, img_size: int, tag: str = "") -> str:
    """e.g. resnet50_384, or resnet18_224_ws for runs made with --tag ws."""
    return f"{backbone}_{img_size}" + (f"_{tag}" if tag else "")


def parse_variant(variant: str):
    """(backbone, img_size, tag) from a variant name such as efficientnet_v2_s_384_ws."""
    import re
    m = re.fullmatch(r"(?P<backbone>.+?)_(?P<size>\d+)(?:_(?P<tag>[A-Za-z0-9-]+))?", variant)
    if not m:
        raise ValueError(f"not a variant name: {variant} (expected <backbone>_<img size>[_<tag>])")
    return m["backbone"], int(m["size"]), m["tag"] or ""


def runs_root(kind: str, variant: str = DEFAULT_VARIANT, smoke: bool = False) -> Path:
    """Folder that holds the runs of one backbone and image size.

    kind is "finetune" (fixed split) or "cv". The original ResNet18 at 224 px keeps its
    original place (results/finetune, results/cv); every other variant gets a subfolder,
    for example results/cv/resnet50_384.
    """
    root = RESULTS_DIR / (f"{kind}_smoke" if smoke else kind)
    return root if variant == DEFAULT_VARIANT else root / variant


def report_dir(variant: str = DEFAULT_VARIANT) -> Path:
    """Where 06_report.py writes a variant's final table, decisions and statistics: results/ for
    ResNet18 at 224 px (the reported results), results/finetune/<variant>/ otherwise."""
    return RESULTS_DIR if variant == DEFAULT_VARIANT else runs_root("finetune", variant)


def file_key(path, cls) -> str:
    """class/file name of an image, the same on every PC (splits.csv holds absolute paths)."""
    return f"{cls}/{PureWindowsPath(path).name}"  # PureWindowsPath splits on both / and \


def check_folds(splits, fold, save_if_missing: bool = True):
    """Make sure the cross-validation folds are the ones in results/cv/folds.csv.

    Comparisons between cross-validation runs, including runs made on different PCs, are
    only valid on identical folds. If folds.csv does not exist yet, it is written.
    """
    import pandas as pd
    keys = [file_key(p, c) for p, c in zip(splits["path"], splits["class"])]
    mine = {k: int(f) for k, f in zip(keys, fold) if f >= 0}
    if not FOLDS_CSV.exists():
        if save_if_missing:
            FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"index": np.flatnonzero(np.asarray(fold) >= 0), "file": [k.split("/")[1] for k in mine],
                          "class": [k.split("/")[0] for k in mine], "fold": list(mine.values())}
                         ).to_csv(FOLDS_CSV, index=False)
        return
    saved = pd.read_csv(FOLDS_CSV)
    theirs = {f"{c}/{f}": int(k) for f, c, k in zip(saved["file"], saved["class"], saved["fold"])}
    if mine != theirs:
        differ = sum(mine.get(k) != v for k, v in theirs.items()) + len(mine.keys() - theirs.keys())
        raise SystemExit(f"the cross-validation folds differ from {FOLDS_CSV} for {differ} images. Re-make the "
                         f"split with 01_make_splits.py on the full dataset (test set: 184 haploid + 266 diploid) "
                         f"and check HAPLOID_DATA_DIR.")


def keep_awake():
    """Ask Windows not to sleep while this process runs (released on exit)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def quantum_layer() -> nn.Module:
    """4-qubit variational circuit: angle embedding + strongly entangling layers."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return qml.qnn.TorchLayer(circuit, {"weights": (N_QLAYERS, N_QUBITS, 3)})


QCNN_QUBITS = 8


def _qcnn_conv(w, a, b):
    """Two-qubit convolution block (6 parameters), shared across a layer."""
    qml.RY(w[0], wires=a)
    qml.RY(w[1], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(w[2], wires=a)
    qml.RY(w[3], wires=b)
    qml.CNOT(wires=[b, a])
    qml.RZ(w[4], wires=a)
    qml.RZ(w[5], wires=b)


def _qcnn_pool(w, src, sink):
    """Pooling: controlled rotations move information from src into sink; src is dropped."""
    qml.CRZ(w[0], wires=[src, sink])
    qml.PauliX(wires=src)
    qml.CRX(w[1], wires=[src, sink])
    qml.PauliX(wires=src)


def qcnn_layer() -> nn.Module:
    """8-qubit quantum convolutional neural network (Cong et al. 2019; Hur et al. 2022).

    Three convolution + pooling stages reduce 8 qubits to 4, 2, then 1. Weights are
    shared within each stage, like a CNN kernel, giving 3 x (6 + 2) = 24 parameters.
    Output: <Z> of the final qubit, in [-1, 1].
    """
    dev = qml.device("default.qubit", wires=QCNN_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, conv_w, pool_w):
        qml.AngleEmbedding(inputs, wires=range(QCNN_QUBITS), rotation="Y")
        active = list(range(QCNN_QUBITS))
        for stage in range(3):
            n = len(active)
            pairs = [(active[i], active[i + 1]) for i in range(0, n, 2)]
            if n > 2:  # second, offset row of convolutions (with wrap-around)
                pairs += [(active[i], active[(i + 1) % n]) for i in range(1, n, 2)]
            for a, b in pairs:
                _qcnn_conv(conv_w[stage], a, b)
            for i in range(0, n, 2):
                _qcnn_pool(pool_w[stage], active[i], active[i + 1])
            active = active[1::2]
        return qml.expval(qml.PauliZ(active[0]))

    return qml.qnn.TorchLayer(circuit, {"conv_w": (3, 6), "pool_w": (3, 2)})


class ToAngle01(nn.Module):
    """Map any real value to a rotation angle in (0, pi).

    RY(+t) and RY(-t) give almost the same measurement, so angles in (-pi, pi)
    throw away sign information; (0, pi) keeps the encoding one-to-one.
    """

    def forward(self, x):
        return torch.sigmoid(x) * torch.pi


class ExpvalToLogits(nn.Module):
    """Turn one expectation value z in [-1, 1] into two class logits (+-(k*z + b)/2)."""

    def __init__(self, k: float = 3.0):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(k))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        s = self.k * z + self.b
        return torch.stack([-s, s], dim=1) / 2


class ScaleToAngle(nn.Module):
    """Map tanh outputs in (-1, 1) to rotation angles in (-pi, pi)."""

    def forward(self, x):
        return x * torch.pi


def build_head(kind: str, dropout: float = 0.3) -> nn.Module:
    """Classifier heads trained on frozen ResNet50 features.

    linear:    2048 -> 2 (standard linear probe)
    classical: 2048 -> 64 -> 4 (tanh) -> 2
    hybrid:    2048 -> 64 -> 4 (tanh * pi) -> 4-qubit circuit -> 2
    classical and hybrid share the same bottleneck, so the only difference
    between them is the quantum circuit.
    """
    if kind == "linear":
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(FEATURE_DIM, 2))
    trunk = [
        nn.Dropout(dropout),
        nn.Linear(FEATURE_DIM, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, N_QUBITS),
        nn.Tanh(),
    ]
    if kind == "classical":
        return nn.Sequential(*trunk, nn.Linear(N_QUBITS, 2))
    if kind == "hybrid":
        return nn.Sequential(*trunk, ScaleToAngle(), quantum_layer(), nn.Linear(N_QUBITS, 2))
    raise ValueError(f"unknown head: {kind}")
