# Future scope

Experiments that could improve or extend the current results. None of these are part of the
reported results (see [README.md](README.md)). The time estimates assume the same GTX 1650, which
measured 13 s per epoch for the classical head and 22 s for the QCNN at 2,100 training images.

## Starting point

- **Final models:** the QCNN reaches 93.33% test accuracy (93.63 ± 0.34% over 3 seeds without TTA),
  and the classical head 94.22% (3-seed ensemble with TTA). Altuntaş et al. (2019) report 94.22% on a
  different split.
- **QCNN vs classical:** no significant difference (exact McNemar p = 0.22 for the final models).
- **No parameter advantage:** the QCNN head has 26 parameters and the classical head 18.

## Done: 30 epochs and 5-fold cross-validation (November 2026)

Both heads were trained at 30 and at 12 epochs on 5 folds of the 2,550 non-test images (the test set
was not used). Details and the full table are in the README, section "Cross-validation".

- **Training longer did not help significantly.** Held-out accuracy levels off at about 95.5% by
  epoch 10; only training accuracy keeps rising. 30 vs 12 epochs: classical +0.04 points (p = 1.0),
  QCNN +0.47 points (p = 0.16). The original "under-trained" reading was therefore wrong: held-out
  accuracy is flat from about epoch 10, so a late best epoch did not mean it was still rising.
- **QCNN vs classical: no significant difference.** Equal at 12 epochs; at 30 epochs the QCNN is
  0.47 points ahead (p = 0.09, one seed per fold, one of 8 comparisons). Re-check with 3 seeds per
  fold before reading anything into it.
- **TTA** adds about 0.4–0.6 points everywhere.

## Ground rules for any new experiment

1. **Treat both heads the same.** Any change (epochs, resolution, backbone, tuning budget) is applied
   to the QCNN and the classical head alike. Tuning one head harder than the other makes the
   comparison unfair.
2. **Protect the test set.** It has already been used twice (an earlier run on another PC, and the
   reported run). Make every choice on validation data or with cross-validation, and score the
   test set once at the end. Do not repeat runs until a test number looks better.
3. **Report averages and intervals.** Give the mean ± std over seeds or folds and a confidence
   interval, not only the best run. On 450 test images the 95% interval is about ±2 percentage
   points, so a difference under about 2 points is not evidence that one model is better.

## A. Improvements that should help both heads

| Idea | Why | Extra GPU time (est.) |
|---|---|---|
| ~~Train longer (25–30 epochs instead of 12)~~ | **Done; no significant gain** (see above). | — |
| ~~5-fold cross-validation~~ | **Done** (see above). The folds are in `results/cv/folds.csv` and can be reused. | — |
| ~~3 seeds per fold at 30 epochs~~ | Would check whether the QCNN's 0.47-point lead at 30 epochs is real or seed noise. | **Not planned** (decided November 2026). |
| **Higher input resolution (320–448)** | The R1-nj marker is a small coloured area on the embryo, and detail may be lost at 224. The photos' longest side has a median of 534 px (99.4% are at least 384), so up to about 448 adds real detail. | **Code ready** (`--img-size`); planned on the workstation, see below. |
| **Stronger backbone (ResNet50, ConvNeXt-Tiny, EfficientNet)** | The backbone does almost all of the work, so it is the most likely way to raise accuracy. | **Code ready** (`--backbone`); planned on the workstation, see below. Too big for the laptop's 4 GB GPU at 384 px. |
| ~~Equal-budget hyperparameter search~~ | Search learning rates and weight decay on validation with the same number of trials for each head. | **Not planned** (decided November 2026). |

Even with these, beating Altuntaş et al. could only be claimed as "higher accuracy", not
"significantly better", unless the gain is well above the roughly ±2-point interval. Their split is
also different.

### Workstation plan: stronger backbones and higher resolution

For an HP workstation with an NVIDIA RTX Blackwell GPU. Setup and commands are in the README,
section "Training on another PC or a workstation". These choices are fixed now, before any
training, so that the final model is not tuned to the results:

1. **Setup:** install `requirements-cu130.txt`; `check_setup.py --download` must end with
   `Ready to train.`; measure seconds per epoch for each variant with a smoke run.
2. **Baseline on the workstation:** ResNet18 at 224 px, 5 folds, 12 epochs, both heads, with
   `--tag ws`. It shows how far the new hardware alone moves the results compared with the laptop.
3. **Variants:** ResNet50 and ConvNeXt-Tiny, each at 224 and 384 px (4 variants), 5 folds, 12 epochs,
   both heads. Optionally EfficientNetV2-S at 384 px.
4. **Same settings for everything:** 12 epochs, batch size 32, backbone learning rate 3e-4, seed 0,
   identical for both heads. Any change is applied to all variants and both heads.
5. **Choosing the variant (cross-validation only):** the highest pooled final-epoch accuracy without
   TTA, averaged over the two heads. If a cheaper variant is within 0.5 points, take the cheaper one.
   One variant is chosen for both heads, so the QCNN vs classical comparison stays fair.
6. **Final model:** train the chosen variant on the fixed split (seeds 0, 1, 2 per head), then
   `05_evaluate.py --variant ...` and `06_report.py --variant ...`. TTA, seeds and threshold are chosen
   on validation, as before. This is the test set's third use, so it happens once, at the very end.
7. **Report all variants'** cross-validation results, not only the chosen one.

Time: the GTX 1650 needs about 35 min for ResNet18 at 224 px (5 folds, both heads, 12 epochs);
the workstation's times come from step 1.

## B. Experiments where the quantum head could show something new

| Idea | What it tests | Extra GPU time (est.) |
|---|---|---|
| ~~Low-data regime:~~ train on 10%, 25% and 50% of the training images, with 5 repeats each, for both heads | Whether the QCNN degrades more gracefully than the classical head with few labels. | **Not planned** (decided November 2026). |
| **Noise robustness:** evaluate the trained QCNN under simulated hardware noise (depolarising and readout errors) | Whether the accuracy holds on realistic near-term quantum hardware. | **Planned**, as the check before the real-hardware run. Under 1 h; inference only. |
| **Real quantum hardware:** run test-set inference on an 8-qubit device (e.g. the IBM Quantum free tier) | Whether the model works outside simulation. Would be a novel result for this application. | **Planned, the last step.** Limited by free QPU minutes; check the quota first. |
| ~~Larger or deeper circuits:~~ more qubits (via 16 angles) or repeated convolution layers | Whether circuit capacity limits accuracy. | **Not planned** (decided November 2026). |

A careful negative result ("no advantage under a fair comparison") is also publishable, particularly
at quantum machine learning venues.

## C. Literature and writing (planned)

- **Check later papers on the same dataset.** Several papers after Altuntaş et al. (2019) used these
  3,000 images, for example work based on pre-trained CNNs, intuitionistic fuzzy sets and GLCM
  features. Some may report more than 94.22%, so compare against the best published result, not only
  Altuntaş et al.
- **State the comparison caveats in the paper.** The test splits differ (450 images here, 900
  implied by their reported numbers), and the labels are putative ploidy from the R1-nj marker.

## Plan (decided November 2026)

Three steps remain: the literature and writing tasks (section C), the workstation training and the
real-hardware test. Everything else above is not planned, including any further work on accuracy
beyond the workstation plan.

1. ~~A: 30 epochs plus 5-fold CV for both heads~~ (done, November 2026; about 2 h of training on the
   GTX 1650 for both 30 and 12 epochs).
2. **A: the workstation plan above**, when the workstation is available. The code is ready and was
   tested on the laptop (all backbones, parallel augmentation, the CUDA 13.0 build). bf16 mixed
   precision could not be tested on real hardware (the GTX 1650 has no bf16 support); the first smoke
   run on the workstation confirms it.
3. **C: literature and writing**, now; it needs no training. Find the later papers on this dataset
   and compare with the best published result, and state the comparison caveats in the paper.
4. **B: real quantum hardware, the last step.** Inference only, with the final QCNN from step 2 (or
   the reported one): first under simulated noise of the chosen IBM device, then on the device itself
   if the simulated accuracy holds up. Compare with the noiseless simulator on the same test images.
