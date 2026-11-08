"""Run the whole workstation plan (FUTURE_SCOPE.md) with one command.

    python run_workstation_plan.py --speed-test     # 1. check the setup, time one epoch per variant
    python run_workstation_plan.py                  # 2. all cross-validation runs, then the report
    python run_workstation_plan.py --dry-run        # print the commands without running them

Order: setup check; the ResNet18 baseline at 224 px with --tag (default "ws"), so it does not
collide with the laptop's runs; then ResNet50 and ConvNeXt-Tiny at 224 and 384 px. Every
variant gets 5 folds x 2 heads x 12 epochs and is finished before the next one starts;
07_cv_report.py runs after each variant. Finished runs are skipped, so after an interruption
the same command continues where it stopped. It stops at the first failure.

Everything printed is also written to results/cv/workstation_log.txt (UTF-8).
Other settings stay at the plan's values (batch size 32, backbone learning rate 3e-4, seed 0);
--extra passes further options to every training run, e.g. --extra "--batch-size 16" if memory
runs out (then for every variant, so the comparison stays fair).

Choosing the final variant and training it on the fixed split stay manual steps (plan steps
5 and 6), because they score the test set.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
LOG = PROJECT / "results" / "cv" / "workstation_log.txt"
FOLDS, HEADS, EPOCHS = range(5), ["qcnn", "classical"], 12


def variants(args):
    v = [("resnet18", 224, args.tag), ("resnet50", 224, ""), ("resnet50", 384, ""),
         ("convnext_tiny", 224, ""), ("convnext_tiny", 384, "")]
    if args.efficientnet:
        v.append(("efficientnet_v2_s", 384, ""))
    return v


def run(cmd, args, log):
    line = " ".join(shlex.quote(c) if " " in c else c for c in cmd)
    log.write(f"\n$ {line}\n")
    print(f"\n$ {line}", flush=True)
    if args.dry_run:
        return ""
    out = []
    with subprocess.Popen(cmd, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONWARNINGS": "ignore", "PYTHONIOENCODING": "utf-8"}) as p:
        for text in p.stdout:
            print(text, end="", flush=True)
            log.write(text)
            log.flush()
            out.append(text)
    if p.returncode:
        log.write(f"FAILED (exit code {p.returncode})\n")
        raise SystemExit(f"\nstopped: the command above failed (exit code {p.returncode}). Fix the problem and "
                         f"run the same command again; finished runs are skipped.")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--speed-test", action="store_true", help="only check the setup and time each variant")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--smoke", action="store_true", help="tiny runs, to test this script (writes to results/cv_smoke)")
    ap.add_argument("--tag", default="ws", help="tag of the ResNet18 baseline run on this PC (default ws)")
    ap.add_argument("--efficientnet", action="store_true", help="also run EfficientNetV2-S at 384 px")
    ap.add_argument("--extra", default="", help="extra options for every training run, in quotes")
    args = ap.parse_args()
    py = sys.executable
    extra = shlex.split(args.extra)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} run_workstation_plan.py {' '.join(sys.argv[1:])}\n")
        if not args.smoke:
            run([py, "check_setup.py", "--download"], args, log)

        if args.speed_test:
            print("\nSpeed test: 2 epochs on 1,024 images per split, QCNN head (the slower one).")
            times = {}
            for backbone, size, _ in variants(args):
                out = run([py, "-u", "04_finetune.py", "--head", "qcnn", "--fold", "0", "--epochs", "2", "--smoke",
                           "--smoke-images", "1024", "--backbone", backbone, "--img-size", str(size), *extra], args, log)
                m = re.findall(r"epoch 2/2: .*?val acc ([\d.]+).*?\((\d+)s(?:, GPU memory ([\d.]+) GB)?\)", out)
                if m:
                    times[(backbone, size)] = (int(m[-1][1]), m[-1][2], float(m[-1][0]))
            if times:
                print("\nEstimated cross-validation time (12 epochs x 5 folds x 2 heads; the classical head is "
                      "faster, so this is an upper bound):")
                total = 0
                not_learning = []
                for (backbone, size), (sec, mem, val_acc) in times.items():
                    est = sec * 2040 / 1024 * EPOCHS * len(FOLDS) * len(HEADS) / 3600  # full folds have 2,040 images
                    total += est
                    print(f"  {backbone:15s} {size} px: {sec:3d} s per 1,024-image epoch, GPU memory {mem or '?'} GB"
                          f" -> about {est:.1f} h" + ("" if val_acc >= 0.6 else f"   NOT LEARNING (val acc {val_acc:.2f})"))
                    if val_acc < 0.6:
                        not_learning.append(f"{backbone} {size} px")
                print(f"  total: about {total:.1f} h")
                if not_learning:
                    print(f"\nWARNING: {', '.join(not_learning)} stayed at chance level (about 0.41 = always diploid) "
                          f"after 2 epochs. As fixed in the plan (FUTURE_SCOPE.md), lower the backbone learning rate "
                          f"for ALL variants and both heads, e.g. --extra \"--lr-backbone 1e-4\", and run the speed "
                          f"test again.")
            return

        for backbone, size, tag in variants(args):
            for fold in FOLDS:
                for head in HEADS:
                    cmd = [py, "-u", "04_finetune.py", "--head", head, "--fold", str(fold), "--epochs",
                           "1" if args.smoke else str(EPOCHS), "--backbone", backbone, "--img-size", str(size)]
                    if tag:
                        cmd += ["--tag", tag]
                    if args.smoke:
                        cmd += ["--smoke"]
                    run(cmd + extra, args, log)
            if not args.smoke:
                run([py, "07_cv_report.py"], args, log)
        print(f"\nAll cross-validation runs are done. The report is in results/cv/ (cv_summary.csv, cv_stats.json).\n"
              f"Next (manual, plan steps 5-6): choose the variant from the report, then train it on the fixed split.\n"
              f"Log: {LOG}")


if __name__ == "__main__":
    main()
