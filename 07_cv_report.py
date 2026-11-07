"""Summarise the 5-fold cross-validation runs (04_finetune.py --fold k).

Cross-validation uses only the 2,550 train + validation images; the test set is not
touched. Each image is held out exactly once, so the held-out ("out-of-fold")
predictions of the 5 folds cover all 2,550 images and can be pooled.

Primary numbers use the final-epoch model, whose epoch was fixed before training.
The best-epoch model was picked on the held-out fold itself, so its accuracy is
optimistic; it is reported only to show how much epoch selection adds.

Runs are grouped by variant (backbone and image size, e.g. resnet50_384), head, number
of epochs and seed. Paired comparisons: QCNN vs classical, longer vs shorter training,
and one variant vs another for the same head. A comparison between runs made on
different GPUs is marked, because hardware alone moves results by up to about a point.

Outputs: results/cv/cv_summary.csv, results/cv/cv_stats.json and
results/figures/cv_curves.png. results/cv/folds.csv is checked, never rewritten.
"""
import importlib
import json
import sys
import math
import re

import matplotlib
import numpy as np
import pandas as pd

from common import DEFAULT_VARIANT, FIGURES_DIR, RESULTS_DIR, SPLITS_CSV, check_folds, cv_folds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rep = importlib.import_module("06_report")  # metrics, bootstrap_ci, mcnemar

CV_DIR = RESULTS_DIR / "cv"
RUN_RE = re.compile(r"(?P<head>qcnn|classical)_e(?P<epochs>\d+)_fold(?P<fold>\d+)_seed(?P<seed>\d+)$")
VARIANTS = ["last_plain", "last_tta", "best_plain", "best_tta"]
LABEL = {"last_plain": "final epoch", "last_tta": "final epoch + TTA",
         "best_plain": "best epoch (optimistic)", "best_tta": "best epoch + TTA (optimistic)"}


def load_runs():
    """{(variant, head, epochs, seed): {fold: {"oof": {...}, "result": {...}, "history": DataFrame}}}

    ResNet18 at 224 px lives directly in results/cv; other variants in results/cv/<variant>/.
    """
    runs = {}
    folders = [(DEFAULT_VARIANT, d) for d in CV_DIR.iterdir() if d.is_dir() and RUN_RE.match(d.name)]
    folders += [(v.name, d) for v in CV_DIR.iterdir() if v.is_dir() and not RUN_RE.match(v.name)
                for d in v.iterdir() if d.is_dir() and RUN_RE.match(d.name)]
    for variant, d in sorted(folders, key=lambda t: (t[0], t[1].name)):
        if not (d / "cv_result.json").exists():
            continue
        m = RUN_RE.match(d.name)
        key = (variant, m["head"], int(m["epochs"]), int(m["seed"]))
        oof = np.load(d / "oof.npz")
        runs.setdefault(key, {})[int(m["fold"])] = {
            "oof": {k: oof[k] for k in oof.files},
            "result": json.loads((d / "cv_result.json").read_text()),
            "history": pd.read_csv(d / "history.csv")}
    return runs


def pooled(folds, variant):
    """Out-of-fold labels and probabilities of all folds, sorted by image index."""
    index = np.concatenate([f["oof"]["index"] for f in folds.values()])
    y = np.concatenate([f["oof"]["y"] for f in folds.values()])
    p = np.concatenate([f["oof"][variant] for f in folds.values()])
    order = np.argsort(index)
    return index[order], y[order], p[order]


def fold_acc(f, variant):
    return float(((f["oof"][variant] >= 0.5).astype(int) == f["oof"]["y"]).mean())


def name(key):
    variant, head, epochs, seed = key
    return f"{variant}, {head}, {epochs} epochs" + (f", seed {seed}" if seed else "")


def devices(folds):
    return sorted({f["result"].get("device", "?") for f in folds.values()})


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # names such as Altuntaş, also when output is redirected
    runs = load_runs()
    if not runs:
        raise SystemExit(f"no finished cross-validation runs in {CV_DIR}")
    n_folds = {r["result"]["folds"] for folds in runs.values() for r in folds.values()}.pop()

    df = pd.read_csv(SPLITS_CSV)
    if n_folds == 5:
        check_folds(df, cv_folds(df["label"].values, df["split"].values, n_folds))

    rows, stats = [], {"n_folds": n_folds, "threshold": 0.5, "configs": {}, "mcnemar": [], "fold_paired": []}
    for key, folds in sorted(runs.items()):
        complete = len(folds) == n_folds
        cfg = {"folds_done": sorted(folds), "complete": complete, "devices": devices(folds),
               "best_epoch": [folds[k]["result"]["best_epoch"] for k in sorted(folds)],
               "final_train_acc": [folds[k]["result"]["final_train_acc"] for k in sorted(folds)]}
        for v in VARIANTS:
            accs = [fold_acc(folds[k], v) for k in sorted(folds)]
            _, y, p = pooled(folds, v)
            m = rep.metrics(y, p)
            cfg[v] = {"fold_mean": float(np.mean(accs)),
                      "fold_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                      "per_fold": dict(zip(map(str, sorted(folds)), accs)),
                      "pooled": {k: float(val) for k, val in m.items()},
                      "pooled_ci95": rep.bootstrap_ci(y, p)}
            rows.append({"config": name(key), "variant": key[0], "head": key[1], "epochs": key[2], "seed": key[3],
                         "model": LABEL[v], "folds": len(folds), "n_images": len(y),
                         "fold_mean_acc": cfg[v]["fold_mean"], "fold_std_acc": cfg[v]["fold_std"],
                         **{k: m[k] for k in ["accuracy", "sensitivity", "specificity", "f1", "auc", "mcc"]},
                         "ci95_low": cfg[v]["pooled_ci95"][0], "ci95_high": cfg[v]["pooled_ci95"][1],
                         "devices": "; ".join(cfg["devices"])})
        stats["configs"][name(key)] = cfg

    # Paired comparisons on the same images, only between complete configurations.
    done = sorted(k for k, f in runs.items() if len(f) == n_folds)
    pairs = [(a, b) for a in done for b in done if a != b and (
        (a[0], a[2], a[3]) == (b[0], b[2], b[3]) and (a[1], b[1]) == ("qcnn", "classical")  # QCNN vs classical
        or (a[0], a[1], a[3]) == (b[0], b[1], b[3]) and a[2] > b[2]                         # longer vs shorter
        or (a[1], a[2], a[3]) == (b[1], b[2], b[3]) and a[0] < b[0])]                       # variant vs variant
    for a, b in pairs:
        same_gpu = devices(runs[a]) == devices(runs[b]) and len(devices(runs[a])) == 1
        for v in ["last_plain", "last_tta"]:
            ia, y, pa = pooled(runs[a], v)
            ib, _, pb = pooled(runs[b], v)
            assert (ia == ib).all()
            stats["mcnemar"].append({"a": name(a), "b": name(b), "variant": LABEL[v], "same_gpu": same_gpu,
                                     **rep.mcnemar(y, pa, pb)})
            diffs = [fold_acc(runs[a][k], v) - fold_acc(runs[b][k], v) for k in range(n_folds)]
            stats["fold_paired"].append({"a": name(a), "b": name(b), "variant": LABEL[v],
                                         "mean_diff": float(np.mean(diffs)), "std_diff": float(np.std(diffs, ddof=1)),
                                         "per_fold": diffs})

    table = pd.DataFrame(rows)
    table.to_csv(CV_DIR / "cv_summary.csv", index=False)
    (CV_DIR / "cv_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"{n_folds}-fold cross-validation on the train + validation images (test set not used)\n")
    for cname, cfg in stats["configs"].items():
        flag = "" if cfg["complete"] else f"  [incomplete: folds {cfg['folds_done']}]"
        print(f"{cname}{flag}   ({', '.join(cfg['devices'])})")
        print(f"  best epoch per fold: {cfg['best_epoch']}   final train acc: "
              f"{', '.join(f'{a:.3f}' for a in cfg['final_train_acc'])}")
        for v in VARIANTS:
            c = cfg[v]
            std = f" +- {100 * c['fold_std']:.2f}" if c["fold_std"] is not None else ""
            lo, hi = c["pooled_ci95"]
            print(f"  {LABEL[v]:30s} fold mean {100 * c['fold_mean']:.2f}{std}%   pooled "
                  f"{100 * c['pooled']['accuracy']:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]  "
                  f"AUC {c['pooled']['auc']:.4f}")
        print()
    if stats["mcnemar"]:
        print("Paired comparisons on the same out-of-fold images (A vs B):")
        for m, d in zip(stats["mcnemar"], stats["fold_paired"]):
            print(f"  {m['a']} vs {m['b']} ({m['variant']}): only A right {m['only_a_right']}, "
                  f"only B right {m['only_b_right']}, McNemar p = {m['p_value']:.3f};  "
                  f"fold-wise A - B = {100 * d['mean_diff']:+.2f} +- {100 * d['std_diff']:.2f} points"
                  + ("" if m["same_gpu"] else "  [different GPUs]"))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panels = sorted({(k[0], k[2]) for k in runs}, key=lambda t: (t[0] != DEFAULT_VARIANT, t[0], -t[1]))
    ncols = min(2, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), sharey=True, squeeze=False)
    for ax, (variant, epochs) in zip(axes.flat, panels):
        for key, folds in sorted(runs.items()):
            if (key[0], key[2]) != (variant, epochs):
                continue
            h = pd.concat([f["history"] for f in folds.values()]).groupby("epoch")
            line, = ax.plot(h.val_acc.mean().index, h.val_acc.mean(), label=f"{key[1]} held-out fold")
            ax.fill_between(h.val_acc.min().index, h.val_acc.min(), h.val_acc.max(), color=line.get_color(),
                            alpha=0.15)
            ax.plot(h.train_acc.mean().index, h.train_acc.mean(), "--", color=line.get_color(),
                    label=f"{key[1]} train")
        ax.set(xlabel="Epoch", title=f"{variant}, {epochs} epochs, mean of {n_folds} folds (band: min-max)")
        ax.title.set_fontsize(9)
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0.8, 1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cv_curves.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
