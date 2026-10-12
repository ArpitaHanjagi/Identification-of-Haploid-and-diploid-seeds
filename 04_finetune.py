"""Fine-tune a ResNet18 end-to-end together with a QCNN (or classical) head.

Usage:
    python 04_finetune.py --head qcnn --seed 0
    python 04_finetune.py --head classical --seed 0

Heads (both take the 512-d ResNet18 feature through the same 512 -> 8 linear layer):
    qcnn:      sigmoid * pi angles -> 8-qubit QCNN (24 params) -> <Z> -> 2 logits
    classical: sigmoid values      -> linear layer (18 params)  -> 2 logits

The model is selected on validation accuracy; the test set is scored once, at the
end. A checkpoint is written every epoch, so an interrupted run resumes.

The ResNet18 runs on the GPU when there is one (CUDA, or Apple MPS); the head always
runs on the CPU. It is tiny, and the quantum simulator works in float64/complex128,
which consumer GPUs run slowly and MPS does not support.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models
from torchvision.transforms import v2

from common import (QCNN_QUBITS, RESULTS_DIR, SPLITS_CSV, ExpvalToLogits,
                    ToAngle01, keep_awake, qcnn_layer)

IMG_SIZE = 224
CACHE = RESULTS_DIR / "image_cache_224.pt"
WHITE = 255
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_images():
    """Pad each image to a white square, resize to 224 and keep all of them in memory (uint8)."""
    if CACHE.exists():
        return torch.load(CACHE)
    df = pd.read_csv(SPLITS_CSV)
    imgs = torch.empty((len(df), 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    for i, p in enumerate(df["path"]):
        im = Image.open(p).convert("RGB")
        side = max(im.size)
        im = ImageOps.pad(im, (side, side), color=(WHITE,) * 3).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        imgs[i] = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)
    data = {"images": imgs, "labels": torch.tensor(df["label"].values), "split": df["split"].tolist()}
    torch.save(data, CACHE)
    return data


augment = v2.Compose([
    v2.RandomRotation(180, fill=WHITE),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
    v2.ColorJitter(brightness=0.15, contrast=0.15),
])
normalize = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SeedClassifier(nn.Module):
    def __init__(self, head: str):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.reduce = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, QCNN_QUBITS))
        if head == "qcnn":
            self.head = nn.Sequential(ToAngle01(), qcnn_layer(), ExpvalToLogits())
        elif head == "classical":
            self.head = nn.Sequential(nn.Sigmoid(), nn.Linear(QCNN_QUBITS, 2))
        else:
            raise ValueError(head)
        self.device = torch.device("cpu")

    def place(self, device):
        """Move the CNN to `device`; the head stays on the CPU. Use this instead of .to()."""
        self.device = torch.device(device)
        self.backbone.to(self.device)
        self.reduce.to(self.device)
        return self

    def forward(self, x):
        """Takes a CPU batch and returns CPU logits, whatever device the CNN is on."""
        return self.head(self.reduce(self.backbone(x.to(self.device))).cpu())


def batches(images, labels, batch_size, train, gen=None):
    order = torch.randperm(len(images), generator=gen) if train else torch.arange(len(images))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        x = images[idx]
        if train:
            x = torch.stack([augment(img) for img in x])
        yield normalize(x), labels[idx]


@torch.no_grad()
def evaluate(model, images, labels, loss_fn):
    model.eval()
    logits = torch.cat([model(x) for x, _ in batches(images, labels, 64, train=False)])
    prob = torch.softmax(logits, 1)[:, 1]
    return loss_fn(logits, labels).item(), ((prob >= 0.5).long() == labels).float().mean().item(), prob.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["qcnn", "classical"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, to test the script end to end")
    ap.add_argument("--device", default="auto", help="auto (default), cuda, mps or cpu")
    args = ap.parse_args()

    keep_awake()
    torch.set_num_threads(4)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
          flush=True)
    run_dir = RESULTS_DIR / ("finetune_smoke" if args.smoke else "finetune") / f"{args.head}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_images()
    split = np.array(data["split"])
    x = {s: data["images"][torch.from_numpy(split == s)] for s in ["train", "val", "test"]}
    y = {s: data["labels"][torch.from_numpy(split == s)] for s in ["train", "val", "test"]}
    if args.smoke:  # random 64 per split, so both classes are present
        pick = {s: torch.randperm(len(v), generator=torch.Generator().manual_seed(1))[:64] for s, v in y.items()}
        x = {s: v[pick[s]] for s, v in x.items()}
        y = {s: v[pick[s]] for s, v in y.items()}

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    model = SeedClassifier(args.head).place(device)
    # The circuit and its readout need a much larger learning rate than the CNN
    # (0.05 vs 3e-4); with 0.01 the QCNN failed to learn in tests.
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-4},
        {"params": model.reduce.parameters(), "lr": 1e-3},
        {"params": model.head.parameters(), "lr": 5e-2 if args.head == "qcnn" else 1e-3},
    ], weight_decay=1e-4)
    steps_per_epoch = -(-len(x["train"]) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    counts = torch.bincount(y["train"], minlength=2).float()
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts), label_smoothing=0.05)

    ckpt_path = run_dir / "checkpoint.pt"
    start, history, best = 1, [], {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0}
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("epochs", args.epochs) != args.epochs:
            raise SystemExit(f"checkpoint was made with --epochs {ck['epochs']}; rerun with the same value "
                             f"or delete {run_dir}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])  # moves the optimizer state to each parameter's device
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["rng"])
        if ck.get("cuda_rng") is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start, history, best = ck["epoch"] + 1, ck["history"], ck["best"]
        print(f"resuming from epoch {ck['epoch']}", flush=True)

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        for xb, yb in batches(x["train"], y["train"], args.batch_size, train=True, gen=gen):
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            seen += len(yb)
        val_loss, val_acc, _ = evaluate(model, x["val"], y["val"], loss_fn)
        history.append({"epoch": epoch, "train_loss": run_loss / seen, "train_acc": correct / seen,
                        "val_loss": val_loss, "val_acc": val_acc, "seconds": round(time.time() - t0)})
        improved = val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"])
        if improved:
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "epoch": epoch, "epochs": args.epochs, "history": history, "best": best}, ckpt_path)
        print(f"{args.head} seed {args.seed} epoch {epoch}/{args.epochs}: train acc {correct / seen:.4f}  "
              f"val acc {val_acc:.4f}  val loss {val_loss:.4f}{'  *best*' if improved else ''}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Final: best-by-validation model, scored once on the untouched test set.
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    _, test_acc, prob = evaluate(model, x["test"], y["test"], loss_fn)
    from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
    yt = y["test"].numpy()
    tn, fp, fn, tp = confusion_matrix(yt, (prob >= 0.5).astype(int), labels=[0, 1]).ravel()
    result = {
        "head": args.head, "seed": args.seed, "best_epoch": best["epoch"],
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "val_acc": best["val_acc"],
        "test_accuracy": test_acc, "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp), "f1": 2 * tp / (2 * tp + fp + fn), "fnr": fn / (tp + fn),
        "auc": roc_auc_score(yt, prob), "mcc": matthews_corrcoef(yt, (prob >= 0.5).astype(int)),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }
    (run_dir / "test_result.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame({"y_true": yt, "prob_haploid": prob}).to_csv(run_dir / "test_probs.csv", index=False)
    print("TEST RESULT " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}),
          flush=True)


if __name__ == "__main__":
    main()
