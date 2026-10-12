"""Train linear, classical and hybrid quantum heads on frozen ResNet50 features.

Each head is trained with 5 random seeds under identical settings, early-stopped
on validation loss, and evaluated once on the untouched test set. Haploid is the
positive class.
"""
import json
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, matthews_corrcoef, roc_auc_score,
                             roc_curve)
from torch import nn

from common import (CLASSES, FEATURES_NPZ, FIGURES_DIR, N_AUGMENTED_VIEWS,
                    RESULTS_DIR, build_head, keep_awake)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HEADS = ["linear", "classical", "hybrid"]
SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABELS = {"linear": "ResNet50 + linear", "classical": "ResNet50 + classical (4-unit)",
          "hybrid": "ResNet50 + quantum (4-qubit)"}


def load_data():
    d = np.load(FEATURES_NPZ)
    train_x = np.concatenate([d["train_x"]] + [d[f"train_aug{v}_x"] for v in range(N_AUGMENTED_VIEWS)])
    train_y = np.concatenate([d["train_y"]] * (N_AUGMENTED_VIEWS + 1))
    # Standardise with statistics from the clean training images only.
    mean, std = d["train_x"].mean(0), d["train_x"].std(0) + 1e-6
    t = lambda a: torch.tensor((a - mean) / std, dtype=torch.float32)  # noqa: E731
    return {
        "train": (t(train_x), torch.tensor(train_y)),
        "val": (t(d["val_x"]), torch.tensor(d["val_y"])),
        "test": (t(d["test_x"]), torch.tensor(d["test_y"])),
    }


def metrics(y_true, prob_haploid):
    y_pred = (prob_haploid >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "sensitivity": sens,
        "specificity": tn / (tn + fp),
        "precision": prec,
        "f1": 2 * prec * sens / (prec + sens) if prec + sens else 0.0,
        "fnr": fn / (tp + fn),
        "fdr": fp / (tp + fp) if tp + fp else 0.0,
        "auc": roc_auc_score(y_true, prob_haploid),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.softmax(model(x), dim=1)[:, 1].numpy()


def train_one(kind, seed, data):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_head(kind)
    x_tr, y_tr = data["train"]
    x_va, y_va = data["val"]
    counts = torch.bincount(y_tr, minlength=2).float()
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history, best, best_state, wait = [], float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(x_tr))
        run_loss, correct = 0.0, 0
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            out = model(x_tr[idx])
            loss = loss_fn(out, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == y_tr[idx]).sum().item()

        model.eval()
        with torch.no_grad():
            val_out = model(x_va)
            val_loss = loss_fn(val_out, y_va).item()
            val_acc = (val_out.argmax(1) == y_va).float().mean().item()
        history.append({"epoch": epoch, "train_loss": run_loss / len(perm), "train_acc": correct / len(perm),
                        "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best - 1e-4:
            best, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Haploid", "Diploid"])
    ax.set_yticks([0, 1], ["Haploid", "Diploid"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    keep_awake()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "models").mkdir(exist_ok=True)
    data = load_data()
    y_test = data["test"][1].numpy()

    if "--report-only" not in sys.argv:
        train_all(data, y_test)
    report(y_test)


def train_all(data, y_test):
    rows, histories, probs = [], [], {}
    for kind in HEADS:
        for seed in SEEDS:
            t0 = time.time()
            model, hist = train_one(kind, seed, data)
            prob = predict(model, data["test"][0])
            m = metrics(y_test, prob)
            m.update(head=kind, seed=seed, epochs=len(hist), seconds=round(time.time() - t0, 1),
                     params=sum(p.numel() for p in model.parameters()))
            rows.append(m)
            histories.append(hist.assign(head=kind, seed=seed))
            probs[f"{kind}_seed{seed}"] = prob
            torch.save(model.state_dict(), RESULTS_DIR / "models" / f"{kind}_seed{seed}.pt")
            print(f"{kind:9s} seed {seed}: acc {m['accuracy']:.4f}  sens {m['sensitivity']:.4f}  "
                  f"spec {m['specificity']:.4f}  auc {m['auc']:.4f}  epochs {len(hist)}  {m['seconds']}s",
                  flush=True)

    # Save everything the report needs, so reporting never requires retraining.
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "metrics_per_seed.csv", index=False)
    pd.concat(histories).to_csv(RESULTS_DIR / "histories.csv", index=False)
    pd.DataFrame(probs).assign(y_true=y_test).to_csv(RESULTS_DIR / "test_probs.csv", index=False)


def report(y_test):
    per_seed = pd.read_csv(RESULTS_DIR / "metrics_per_seed.csv")
    histories = pd.read_csv(RESULTS_DIR / "histories.csv")
    probs = pd.read_csv(RESULTS_DIR / "test_probs.csv")

    cols = ["accuracy", "sensitivity", "specificity", "precision", "f1", "fnr", "fdr", "auc", "mcc"]
    summary = per_seed.groupby("head", sort=False)[cols].agg(["mean", "std"])
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv")
    pretty = pd.DataFrame({c: [f"{100 * summary.loc[h, (c, 'mean')]:.2f} ± {100 * summary.loc[h, (c, 'std')]:.2f}"
                               if c != "mcc" else
                               f"{summary.loc[h, (c, 'mean')]:.3f} ± {summary.loc[h, (c, 'std')]:.3f}"
                               for h in HEADS] for c in cols}, index=[LABELS[h] for h in HEADS])
    pretty.insert(0, "trainable params", [int(per_seed.loc[per_seed["head"] == h, "params"].iloc[0])
                                          for h in HEADS])
    pretty.to_csv(RESULTS_DIR / "table_results.csv")
    print("\nTest set, mean ± std over 5 seeds (%):")
    print(pretty.to_string())

    # Confusion matrices summed over the 5 seeds, in the Sabadin layout (haploid first).
    for kind in HEADS:
        sub = per_seed[per_seed["head"] == kind]
        cm = np.array([[sub.tp.sum(), sub.fp.sum()], [sub.fn.sum(), sub.tn.sum()]]).T
        plot_confusion(cm, f"{LABELS[kind]}\n(test, summed over 5 seeds)", FIGURES_DIR / f"confusion_{kind}.png")

    # ROC curves for seed 0.
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for kind in HEADS:
        p = probs[f"{kind}_seed0"]
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{LABELS[kind]} (AUC {roc_auc_score(y_test, p):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (haploid)")
    ax.set_title("ROC curves, test set (seed 0)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_test.png", dpi=200)
    plt.close(fig)

    # Training curves for seed 0.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for kind in HEADS:
        h = histories[(histories["head"] == kind) & (histories["seed"] == 0)]
        axes[0].plot(h.epoch, h.train_acc, label=f"{kind} train")
        axes[0].plot(h.epoch, h.val_acc, "--", label=f"{kind} val")
        axes[1].plot(h.epoch, h.train_loss, label=f"{kind} train")
        axes[1].plot(h.epoch, h.val_loss, "--", label=f"{kind} val")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy (seed 0)")
    axes[1].set(xlabel="Epoch", ylabel="Weighted cross-entropy", title="Loss (seed 0)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=200)
    plt.close(fig)

    with open(RESULTS_DIR / "run_config.json", "w") as f:
        json.dump({"heads": HEADS, "seeds": SEEDS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                   "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
                   "augmented_views": N_AUGMENTED_VIEWS, "torch": torch.__version__}, f, indent=2)


if __name__ == "__main__":
    main()
