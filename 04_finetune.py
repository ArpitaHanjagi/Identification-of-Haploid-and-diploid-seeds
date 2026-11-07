"""Fine-tune a pretrained CNN end-to-end together with a QCNN (or classical) head.

Usage:
    python 04_finetune.py --head qcnn --seed 0
    python 04_finetune.py --head classical --seed 0
    python 04_finetune.py --head qcnn --fold 2 --epochs 12              # cross-validation, see below
    python 04_finetune.py --head qcnn --fold 2 --backbone resnet50 --img-size 384

Model: an ImageNet-pretrained backbone (ResNet18 at 224 x 224 by default; see BACKBONES)
-> dropout -> linear layer to 8 values -> one of two heads:
    qcnn:      sigmoid * pi angles -> 8-qubit QCNN (24 params) -> <Z> -> 2 logits
    classical: sigmoid values      -> linear layer (18 params)  -> 2 logits

The model is selected on validation accuracy; the test set is scored once, at the
end. A checkpoint is written every epoch, so an interrupted run resumes.

With --fold k, the run is one fold of 5-fold cross-validation over the 2,550
train + validation images (common.cv_folds): fold k (510 images) is held out and the
other 2,040 are used for training. The test set is not loaded or scored. Plain and
TTA probabilities on the held-out fold are saved for the final-epoch model and for the
best-by-validation model (oof.npz); 07_cv_report.py summarises them.

Results go to results/finetune/<head>_seed<n>/ or results/cv/<head>_e<epochs>_fold<k>_seed<n>/
for ResNet18 at 224 px, and to a subfolder named <backbone>_<img size> for anything else,
for example results/cv/resnet50_384/. --tag adds a label to that name (resnet18_224_ws), to
keep runs from another PC apart from the ones already there. Each run folder has a
run_config.json.

Hardware. The backbone runs on the GPU if there is one (NVIDIA CUDA, or Apple MPS), else
on the CPU. The head always runs on the CPU: it is tiny, and PennyLane's simulator works
in float64/complex128 and mixes up devices when given GPU tensors. Three options only
change the speed, not the experiment, and default to "auto", which suits the PC;
run_config.json records what was used:
    --amp      bf16 mixed precision for the backbone. auto: on for NVIDIA GPUs from the
               RTX 30 series / Ampere on (including Blackwell), off for older ones and for
               Apple and CPU. (fp16 is not offered: on the GTX 1650 it gave NaN features at
               batch size 64 and ran 4x slower.)
    --workers  processes that prepare the augmented training images in parallel.
               auto: as many as --threads on a GPU PC, none on a CPU-only PC.
    --threads  CPU threads of the main process, which also runs the QCNN simulator.
               auto: half the logical CPUs, at most 8.
"""
import argparse
import contextlib
import json
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models
from torchvision.transforms import v2

from common import (CV_FOLDS, QCNN_QUBITS, RESULTS_DIR, SPLITS_CSV, ExpvalToLogits, ToAngle01, check_folds,
                    cv_folds, file_key, keep_awake, qcnn_layer, runs_root, variant_name)
from tta import predict_probs

IMG_SIZE = 224  # default image size; --img-size changes it
WHITE = 255
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# name: (torchvision model, ImageNet weights, classifier layer to remove, display name)
BACKBONES = {
    "resnet18": ("resnet18", "IMAGENET1K_V1", "fc", "ResNet18"),
    "resnet50": ("resnet50", "IMAGENET1K_V2", "fc", "ResNet50"),
    "convnext_tiny": ("convnext_tiny", "IMAGENET1K_V1", "classifier.2", "ConvNeXt-Tiny"),
    "efficientnet_b0": ("efficientnet_b0", "IMAGENET1K_V1", "classifier", "EfficientNet-B0"),
    "efficientnet_v2_s": ("efficientnet_v2_s", "IMAGENET1K_V1", "classifier", "EfficientNetV2-S"),
}
# Settings that define the experiment. A run only resumes from a checkpoint made with the same ones.
EXPERIMENT_KEYS = ["backbone", "img_size", "head", "seed", "epochs", "fold", "folds", "batch_size", "lr_backbone",
                   "smoke_images"]


def cache_path(img_size):
    return RESULTS_DIR / f"image_cache_{img_size}.pt"


def load_images(img_size=IMG_SIZE):
    """Pad each image to a white square, resize it and keep all of them in memory (uint8).

    The cache is rebuilt if it does not match results/splits.csv (e.g. after a new split).
    """
    df = pd.read_csv(SPLITS_CSV)
    keys = [file_key(p, c) for p, c in zip(df["path"], df["class"])]
    cache = cache_path(img_size)
    if cache.exists():
        data = torch.load(cache)
        if (data["split"] == df["split"].tolist() and data["labels"].tolist() == df["label"].tolist()
                and data.get("keys", keys) == keys):
            return data
        print(f"{cache.name} does not match {SPLITS_CSV.name}; rebuilding it", flush=True)
    imgs = torch.empty((len(df), 3, img_size, img_size), dtype=torch.uint8)
    for i, p in enumerate(df["path"]):
        im = Image.open(p).convert("RGB")
        side = max(im.size)
        im = ImageOps.pad(im, (side, side), color=(WHITE,) * 3).resize((img_size, img_size), Image.BICUBIC)
        imgs[i] = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)
    data = {"images": imgs, "labels": torch.tensor(df["label"].values), "split": df["split"].tolist(), "keys": keys}
    torch.save(data, cache)
    return data


def make_augment(img_size=IMG_SIZE):
    return v2.Compose([
        v2.RandomRotation(180, fill=WHITE),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
        v2.ColorJitter(brightness=0.15, contrast=0.15),
    ])


augment = make_augment()
normalize = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_amp(name, device):
    """Mixed precision for the backbone: "bf16" or "off"."""
    if name != "auto":
        if name == "bf16" and (device.type != "cuda" or torch.cuda.get_device_capability(device) < (8, 0)):
            raise SystemExit("--amp bf16 needs an NVIDIA RTX 30 series / Ampere or newer GPU (on the GTX 1650, "
                             "cuDNN found no bf16 convolution for ConvNeXt); use --amp off")
        return name
    return "bf16" if device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0) else "off"


def auto_threads():
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def build_backbone(name, pretrained=True):
    """Pretrained torchvision model with its ImageNet classifier removed, and its feature size."""
    arch, weights, classifier, _ = BACKBONES[name]
    net = models.get_model(arch, weights=weights if pretrained else None)
    removed = net.get_submodule(classifier)
    dim = [m for m in removed.modules() if isinstance(m, nn.Linear)][-1].in_features
    parent, _, leaf = classifier.rpartition(".")
    setattr(net.get_submodule(parent) if parent else net, leaf, nn.Identity())
    return net, dim


class SeedClassifier(nn.Module):
    def __init__(self, head: str, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone, dim = build_backbone(backbone, pretrained)
        self.reduce = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, QCNN_QUBITS))
        if head == "qcnn":
            self.head = nn.Sequential(ToAngle01(), qcnn_layer(), ExpvalToLogits())
        elif head == "classical":
            self.head = nn.Sequential(nn.Sigmoid(), nn.Linear(QCNN_QUBITS, 2))
        else:
            raise ValueError(head)
        self.device = torch.device("cpu")
        self.amp_dtype, self.channels_last = None, False

    def place(self, device, amp="off"):
        """Put the backbone on `device`; the head stays on the CPU. amp: "off" or "bf16".
        Use this instead of .to()."""
        self.device = torch.device(device)
        self.amp_dtype = {"off": None, "bf16": torch.bfloat16}[amp]
        self.backbone.to(self.device)
        self.reduce.to(self.device)
        # Channels-last layout speeds up bf16 convolutions on tensor cores (RTX 30 series and newer).
        self.channels_last = amp == "bf16"
        if self.channels_last:
            self.backbone.to(memory_format=torch.channels_last)
        return self

    def forward(self, x):
        """Takes a CPU batch and returns CPU logits, whatever device the backbone is on.

        x is either uint8 images (normalised here, on the device) or already normalised floats.
        """
        x = x.to(self.device, non_blocking=True)
        if x.dtype == torch.uint8:
            x = normalize(x)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        mixed = (torch.autocast(self.device.type, dtype=self.amp_dtype) if self.amp_dtype is not None
                 else contextlib.nullcontext())
        with mixed:
            z = self.reduce(self.backbone(x))
        return self.head(z.float().cpu())


def run_config(run_dir):
    """Settings a run was trained with. Runs made before run_config.json existed are ResNet18 at 224 px."""
    f = run_dir / "run_config.json"
    return json.loads(f.read_text()) if f.exists() else {"backbone": "resnet18", "img_size": IMG_SIZE}


def load_trained(run_dir, head, weights="best_model.pt"):
    """Rebuild a trained model from its run folder (no ImageNet download needed)."""
    cfg = run_config(run_dir)
    model = SeedClassifier(head, cfg["backbone"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / weights, map_location="cpu"))
    return model, cfg


def batches(images, labels, batch_size, train, gen=None, augment=augment):
    """Batches of uint8 images; training batches are shuffled with `gen` and augmented."""
    order = torch.randperm(len(images), generator=gen) if train else torch.arange(len(images))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        x = images[idx]
        if train:
            x = torch.stack([augment(img) for img in x])
        yield x, labels[idx]


class AugmentedImages(Dataset):
    """Training images, augmented one by one inside the DataLoader's worker processes."""

    def __init__(self, images, labels, augment):
        self.images, self.labels, self.augment = images, labels, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.augment(self.images[i]), self.labels[i]


class EpochShuffle(Sampler):
    """The same shuffled order as batches(): one permutation from `gen` per epoch."""

    def __init__(self, n, gen):
        self.n, self.gen = n, gen

    def __len__(self):
        return self.n

    def __iter__(self):
        yield from torch.randperm(self.n, generator=self.gen).tolist()


@torch.no_grad()
def evaluate(model, images, labels, loss_fn):
    model.eval()
    logits = torch.cat([model(x) for x, _ in batches(images, labels, 64, train=False)])
    prob = torch.softmax(logits, 1)[:, 1]
    return loss_fn(logits, labels).item(), ((prob >= 0.5).long() == labels).float().mean().item(), prob.numpy()


def device_name(device):
    return torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)


def finish_cv_run(args, model, device, run_dir, n_train, val_idx, x_val, y_val, history, best):
    """Save held-out-fold probabilities for the final-epoch and the best-by-validation model.

    The final-epoch model is the unbiased estimate: its epoch was fixed in advance. The
    best-epoch model was chosen on this same fold, so its accuracy is optimistic.
    """
    torch.save(model.state_dict(), run_dir / "last_model.pt")
    res = {"index": val_idx.numpy(), "y": y_val.numpy()}
    for which in ["last", "best"]:
        if which == "best":
            model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
        for key, use_tta in [("plain", False), ("tta", True)]:
            res[f"{which}_{key}"] = predict_probs(model, x_val, normalize, use_tta)
    np.savez(run_dir / "oof.npz", **res)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    acc = {f"{k}_accuracy": float(((res[k] >= 0.5).astype(int) == res["y"]).mean())
           for k in ["last_plain", "last_tta", "best_plain", "best_tta"]}
    result = {"head": args.head, "backbone": args.backbone, "img_size": args.img_size, "epochs": args.epochs,
              "fold": args.fold, "folds": args.folds, "seed": args.seed, "n_train": n_train,
              "n_val": len(y_val), "best_epoch": best["epoch"], "final_train_acc": history[-1]["train_acc"],
              "device": device_name(device), **acc}
    (run_dir / "cv_result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "checkpoint.pt").unlink(missing_ok=True)  # large; cv_result.json marks the run as done
    print("CV RESULT " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}),
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--head", choices=["qcnn", "classical"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone", choices=list(BACKBONES), default="resnet18")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE,
                    help="input size in pixels (default 224). The photos' longest side is 300-718 px, median "
                         "534, so sizes above about 448 mostly upscale.")
    ap.add_argument("--lr-backbone", type=float, default=3e-4, help="peak learning rate of the backbone")
    ap.add_argument("--tag", default="", help="label added to the results folder name, e.g. the PC's name "
                    "(letters, digits and -), to keep these runs apart from runs made elsewhere")
    ap.add_argument("--fold", type=int, default=None,
                    help="run cross-validation fold k (0-based) instead of the fixed split; never uses the test set")
    ap.add_argument("--folds", type=int, default=CV_FOLDS, help="number of cross-validation folds")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, to test the script end to end")
    ap.add_argument("--smoke-images", type=int, default=64,
                    help="images per split in a smoke run (e.g. 1024 to measure the speed of an epoch)")
    ap.add_argument("--device", default="auto", help="auto (default), cuda, mps or cpu")
    ap.add_argument("--amp", choices=["auto", "bf16", "off"], default="auto")
    ap.add_argument("--workers", default="auto", help="augmentation worker processes: auto or a number (0 = none)")
    ap.add_argument("--threads", default="auto", help="CPU threads of the main process: auto or a number")
    args = ap.parse_args()
    cv = args.fold is not None
    if cv and not 0 <= args.fold < args.folds:
        raise SystemExit(f"--fold must be between 0 and {args.folds - 1}")

    if args.tag and not re.fullmatch(r"[A-Za-z0-9-]+", args.tag):
        raise SystemExit("--tag may only contain letters, digits and -")
    variant = variant_name(args.backbone, args.img_size, args.tag)
    if cv:
        run_dir = runs_root("cv", variant, args.smoke) / f"{args.head}_e{args.epochs}_fold{args.fold}_seed{args.seed}"
        tag = f"{variant} {args.head} e{args.epochs} fold {args.fold} seed {args.seed}"
    else:
        run_dir = runs_root("finetune", variant, args.smoke) / f"{args.head}_seed{args.seed}"
        tag = f"{variant} {args.head} seed {args.seed}"
    if args.smoke:
        shutil.rmtree(run_dir, ignore_errors=True)  # smoke runs always start fresh
    elif cv and (run_dir / "cv_result.json").exists():
        print(f"skip {tag}: already finished ({run_dir / 'cv_result.json'})", flush=True)
        return

    keep_awake()
    device = pick_device(args.device)
    amp = pick_amp(args.amp, device)
    threads = auto_threads() if args.threads == "auto" else int(args.threads)
    workers = (min(8, threads) if device.type != "cpu" else 0) if args.workers == "auto" else int(args.workers)
    torch.set_num_threads(threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    print(f"device: {device}" + (f" ({device_name(device)})" if device.type == "cuda" else "")
          + f"  backbone: {args.backbone} at {args.img_size} px  amp: {amp}  workers: {workers}  "
          f"threads: {threads}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_images(args.img_size)
    split = np.array(data["split"])
    if cv:
        fold = cv_folds(data["labels"].numpy(), split, args.folds)
        if args.folds == CV_FOLDS:
            check_folds(pd.read_csv(SPLITS_CSV), fold, save_if_missing=not args.smoke)
        masks = {"train": (fold >= 0) & (fold != args.fold), "val": fold == args.fold}
    else:
        masks = {s: split == s for s in ["train", "val", "test"]}
    idx = {s: torch.from_numpy(np.flatnonzero(m)) for s, m in masks.items()}  # rows of splits.csv
    if args.smoke:  # a random subset of each split, so both classes are present
        idx = {s: v[torch.randperm(len(v), generator=torch.Generator().manual_seed(1))[:args.smoke_images]]
               for s, v in idx.items()}
    x = {s: data["images"][v] for s, v in idx.items()}
    y = {s: data["labels"][v] for s, v in idx.items()}

    config = {"backbone": args.backbone, "img_size": args.img_size, "head": args.head, "seed": args.seed,
              "epochs": args.epochs, "fold": args.fold, "folds": args.folds if cv else None,
              "batch_size": args.batch_size, "lr_backbone": args.lr_backbone,
              "smoke_images": args.smoke_images if args.smoke else None, "tag": args.tag,
              "amp": amp, "workers": workers, "threads": threads,
              "device": device_name(device), "torch": torch.__version__}

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    model = SeedClassifier(args.head, args.backbone).place(device, amp)
    # The circuit and its readout need a much larger learning rate than the CNN
    # (0.05 vs 3e-4); with 0.01 the QCNN failed to learn in tests.
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
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
        saved = ck.get("config", {"epochs": ck.get("epochs", args.epochs)})
        changed = [k for k in EXPERIMENT_KEYS if k in saved and saved[k] != config[k]]
        if changed:
            raise SystemExit(f"the checkpoint in {run_dir} was made with different settings "
                             f"({', '.join(f'{k}={saved[k]}' for k in changed)}); rerun with those or delete the folder")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])  # moves the optimizer state to each parameter's device
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["rng"])
        if ck.get("cuda_rng") is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start, history, best = ck["epoch"] + 1, ck["history"], ck["best"]
        print(f"resuming from epoch {ck['epoch']}", flush=True)
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    if workers > 0:  # augmentation in parallel processes; the shuffling stays in this process
        loader = DataLoader(AugmentedImages(x["train"], y["train"], make_augment(args.img_size)),
                            batch_size=args.batch_size, sampler=EpochShuffle(len(y["train"]), gen),
                            num_workers=workers, persistent_workers=True, pin_memory=device.type == "cuda")
        train_batches = lambda: loader  # noqa: E731
    else:
        aug = make_augment(args.img_size)
        train_batches = lambda: batches(x["train"], y["train"], args.batch_size, True, gen, aug)  # noqa: E731

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        for xb, yb in train_batches():
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
        mem = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else None
        history.append({"epoch": epoch, "train_loss": run_loss / seen, "train_acc": correct / seen,
                        "val_loss": val_loss, "val_acc": val_acc, "seconds": round(time.time() - t0),
                        "gpu_mem_gb": round(mem, 2) if mem is not None else None})
        improved = val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"])
        if improved:
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "epoch": epoch, "epochs": args.epochs, "config": config, "history": history, "best": best},
                   ckpt_path)
        print(f"{tag} epoch {epoch}/{args.epochs}: train acc {correct / seen:.4f}  "
              f"val acc {val_acc:.4f}  val loss {val_loss:.4f}{'  *best*' if improved else ''}  "
              f"({time.time() - t0:.0f}s" + (f", GPU memory {mem:.1f} GB)" if mem is not None else ")"), flush=True)

    if cv:
        finish_cv_run(args, model, device, run_dir, len(y["train"]), idx["val"], x["val"], y["val"], history, best)
        return

    # Final: best-by-validation model, scored once on the untouched test set.
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    _, test_acc, prob = evaluate(model, x["test"], y["test"], loss_fn)
    from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
    yt = y["test"].numpy()
    tn, fp, fn, tp = confusion_matrix(yt, (prob >= 0.5).astype(int), labels=[0, 1]).ravel()
    result = {
        "head": args.head, "backbone": args.backbone, "img_size": args.img_size, "seed": args.seed,
        "best_epoch": best["epoch"], "device": device_name(device),
        "val_acc": best["val_acc"],
        "test_accuracy": test_acc, "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp) if tp + fp else 0.0, "f1": 2 * tp / (2 * tp + fp + fn), "fnr": fn / (tp + fn),
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
