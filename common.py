"""Shared paths, settings and model definitions for the maize ploidy experiments."""
import ctypes
import sys
from pathlib import Path

import pennylane as qml
import torch
from torch import nn

DATA_DIR = Path(r"C:\Users\arpit\Downloads\haploid_and_diploid_maize_seeds_dataset")
PROJECT_DIR = Path(__file__).resolve().parent
SPLITS_CSV = PROJECT_DIR / "results" / "splits.csv"
FEATURES_NPZ = PROJECT_DIR / "results" / "resnet50_features.npz"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"

# Haploid is the positive class (index 1), following Sabadin et al. (2021).
CLASSES = ["diploid", "haploid"]
SPLIT_SEED = 42
IMG_SIZE = 224
N_AUGMENTED_VIEWS = 4  # extra augmented copies of each training image

N_QUBITS = 4
N_QLAYERS = 2
FEATURE_DIM = 2048  # ResNet50 global-average-pooled output


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
