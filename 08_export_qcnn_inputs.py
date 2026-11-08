"""Export what the quantum-hardware test needs from a trained QCNN model.

For every test image, the classical part of the model (backbone -> 512 -> 8 -> sigmoid * pi)
produces the 8 rotation angles that enter the circuit. This script saves those angles,
the circuit's trained weights, the readout parameters and the exact (noise-free) circuit
output from PennyLane, so that 09_quantum_hardware.py can run the same circuit on a
simulator or on IBM quantum hardware without PyTorch.

    python 08_export_qcnn_inputs.py                         # final QCNN of the reported results
    python 08_export_qcnn_inputs.py --variant resnet50_384  # final QCNN of another variant

Test-time augmentation is not used: it would need 8 circuits per image. The exported
"exact" accuracy is therefore that of the model without TTA.
"""
import argparse
import importlib
import json

import numpy as np
import torch

from common import DEFAULT_VARIANT, RESULTS_DIR, report_dir, runs_root

ft = importlib.import_module("04_finetune")


def out_dir(variant=DEFAULT_VARIANT):
    return RESULTS_DIR / "quantum" if variant == DEFAULT_VARIANT else RESULTS_DIR / "quantum" / variant


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--seed", type=int, help="QCNN seed to use (default: the one chosen in final_decisions.json)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    seed = args.seed
    if seed is None:
        seed = json.loads((report_dir(args.variant) / "final_decisions.json").read_text())["qcnn"]["seed"]
    run = runs_root("finetune", args.variant) / f"qcnn_seed{seed}"
    model, cfg = ft.load_trained(run, "qcnn")
    device = ft.pick_device(args.device)
    model.place(device)  # full precision, so the angles match the reported model exactly
    model.eval()

    data = ft.load_images(cfg["img_size"])
    test = np.flatnonzero(np.array(data["split"]) == "test")
    x, y = data["images"][test], data["labels"][test].numpy()

    angles, z_exact = [], []
    to_angle, circuit, readout = model.head
    for i in range(0, len(x), 64):
        xb = ft.normalize(x[i:i + 64].to(device))
        a = to_angle(model.reduce(model.backbone(xb)).float().cpu())
        angles.append(a.double().numpy())
        z_exact.append(circuit(a).double().numpy())
    angles, z_exact = np.concatenate(angles), np.concatenate(z_exact)
    k, b = float(readout.k), float(readout.b)
    p_exact = 1 / (1 + np.exp(-(k * z_exact + b)))  # softmax of the two logits +-(k z + b) / 2

    # The exported model must reproduce the reported probabilities (seed run, no TTA).
    probs = run / "probs.npz"
    if probs.exists():
        diff = np.abs(np.load(probs)["test_plain"] - p_exact).max()
        print(f"largest difference from the reported test probabilities (no TTA): {diff:.1e}")
        if diff > 1e-4:
            raise SystemExit("the exported model does not reproduce the reported probabilities")

    out = out_dir(args.variant)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "qcnn_inputs.npz", angles=angles, y=y, index=test, z_exact=z_exact, p_exact=p_exact,
             conv_w=circuit.conv_w.detach().double().numpy(), pool_w=circuit.pool_w.detach().double().numpy(),
             k=k, b=b)
    acc = ((p_exact >= 0.5).astype(int) == y).mean()
    meta = {"variant": args.variant, "seed": seed, "backbone": cfg["backbone"], "img_size": cfg["img_size"],
            "n_test": int(len(y)), "exact_accuracy_no_tta": float(acc)}
    (out / "qcnn_inputs.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out / 'qcnn_inputs.npz'}: {len(y)} test images, exact accuracy without TTA {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
