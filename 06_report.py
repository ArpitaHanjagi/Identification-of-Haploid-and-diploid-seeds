"""Final report for the fine-tuned models: validation-gated TTA, ensemble and threshold.

For each head (qcnn, classical) every choice is made by `decide`, which only ever
sees validation labels and probabilities. The chosen setting is then applied to the
test set exactly once.
"""
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve

from common import FIGURES_DIR, RESULTS_DIR
from tta import choose_threshold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HEADS = {"qcnn": "ResNet18 + QCNN (8 qubits)", "classical": "ResNet18 + classical head"}
ALTUNTAS = {"model": "Altuntaş et al. 2019 (VGG-19, reported)", "accuracy": 0.9422, "sensitivity": 0.9458,
            "specificity": 0.9397, "f1": 0.9307}


def ensemble(members):
    """Mean probability of several runs of the same head."""
    heads = {h for h, _ in members}
    if len(heads) != 1:
        raise ValueError(f"cannot mix heads in one ensemble: {sorted(heads)}")
    return np.mean([p for _, p in members], axis=0)


def _acc(y, p, t=0.5):
    return float(((p >= t).astype(int) == y).mean())


def decide(y_val, runs_val):
    """Choose TTA, single-vs-ensemble and threshold from validation data only.

    runs_val: {seed: {"plain": probs, "tta": probs}} for one head.
    """
    seeds = sorted(runs_val)
    tta = np.mean([_acc(y_val, runs_val[s]["tta"]) for s in seeds]) >= \
        np.mean([_acc(y_val, runs_val[s]["plain"]) for s in seeds])
    key = "tta" if tta else "plain"
    single = {s: _acc(y_val, runs_val[s][key]) for s in seeds}
    best_seed = max(seeds, key=lambda s: (single[s], -s))
    ens = np.mean([runs_val[s][key] for s in seeds], axis=0)
    use_ensemble = len(seeds) > 1 and _acc(y_val, ens) >= single[best_seed]
    chosen = ens if use_ensemble else runs_val[best_seed][key]
    return {"tta": bool(tta), "use_ensemble": bool(use_ensemble), "seed": int(best_seed),
            "seeds": seeds, "threshold": choose_threshold(y_val, chosen),
            "val_accuracy": _acc(y_val, chosen)}


def metrics(y, p, t=0.5):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": (tp + tn) / len(y), "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "precision": tp / (tp + fp) if tp + fp else 0.0, "f1": 2 * tp / (2 * tp + fp + fn),
            "fnr": fn / (tp + fn), "auc": roc_auc_score(y, p), "mcc": matthews_corrcoef(y, pred),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def load_runs(head):
    runs = {}
    for f in sorted((RESULTS_DIR / "finetune").glob(f"{head}_seed*/probs.npz")):
        seed = int(f.parent.name.split("_seed")[1])
        d = np.load(f)
        runs[seed] = {k: d[k] for k in d.files}
    return runs


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows, decisions, finals = [], {}, {}
    for head, label in HEADS.items():
        runs = load_runs(head)
        if not runs:
            print(f"no finished runs for {head}")
            continue
        y_val, y_test = next(iter(runs.values()))["y_val"], next(iter(runs.values()))["y_test"]
        for s, r in runs.items():
            for key in ["plain", "tta"]:
                rows.append({"model": f"{label}, seed {s}, {'TTA' if key == 'tta' else 'no TTA'}",
                             "val_accuracy": _acc(y_val, r[f"val_{key}"]), **metrics(y_test, r[f"test_{key}"])})
        d = decide(y_val, {s: {"plain": r["val_plain"], "tta": r["val_tta"]} for s, r in runs.items()})
        decisions[head] = d
        key = "tta" if d["tta"] else "plain"
        p_test = (ensemble([(head, runs[s][f"test_{key}"]) for s in d["seeds"]]) if d["use_ensemble"]
                  else runs[d["seed"]][f"test_{key}"])
        finals[head] = p_test
        what = f"{'ensemble of ' + str(len(d['seeds'])) + ' seeds' if d['use_ensemble'] else 'seed ' + str(d['seed'])}" \
               f"{', TTA' if d['tta'] else ''}"
        rows.append({"model": f"{label} FINAL ({what}, threshold 0.5)", "val_accuracy": d["val_accuracy"],
                     **metrics(y_test, p_test, 0.5)})
        rows.append({"model": f"{label} FINAL ({what}, val-chosen threshold {d['threshold']:.3f})",
                     "val_accuracy": d["val_accuracy"], **metrics(y_test, p_test, d["threshold"])})
    rows.append(ALTUNTAS)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "final_table.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "final_decisions.json").write_text(json.dumps(decisions, indent=2))
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(table[["model", "val_accuracy", "accuracy", "sensitivity", "specificity", "f1", "auc", "mcc"]]
              .round(4).to_string(index=False))
    print(json.dumps(decisions, indent=2))

    if "qcnn" in finals:
        y_test = load_runs("qcnn")[decisions["qcnn"]["seeds"][0]]["y_test"]
        t = 0.5
        cm = confusion_matrix(y_test, (finals["qcnn"] >= t).astype(int), labels=[1, 0])
        fig, ax = plt.subplots(figsize=(3.8, 3.4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=13, color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Haploid", "Diploid"])
        ax.set_yticks([0, 1], ["Haploid", "Diploid"])
        ax.set(xlabel="Predicted", ylabel="Actual", title="ResNet18 + QCNN, test set")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "final_confusion_qcnn.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        for head, p in finals.items():
            fpr, tpr, _ = roc_curve(y_test, p)
            ax.plot(fpr, tpr, label=f"{HEADS[head]} (AUC {roc_auc_score(y_test, p):.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="False positive rate", ylabel="True positive rate (haploid)", title="ROC, test set")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "final_roc.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for head in HEADS:
        f = RESULTS_DIR / "finetune" / f"{head}_seed0" / "history.csv"
        if f.exists():
            h = pd.read_csv(f)
            axes[0].plot(h.epoch, h.train_acc, label=f"{head} train")
            axes[0].plot(h.epoch, h.val_acc, "--", label=f"{head} val")
            axes[1].plot(h.epoch, h.train_loss, label=f"{head} train")
            axes[1].plot(h.epoch, h.val_loss, "--", label=f"{head} val")
    axes[0].axhline(0.9422, color="grey", lw=0.8, ls=":", label="Altuntaş 94.22%")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Fine-tuning accuracy (seed 0)")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Fine-tuning loss (seed 0)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_training_curves.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
