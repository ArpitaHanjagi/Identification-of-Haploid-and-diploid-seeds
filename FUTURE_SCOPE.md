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
   identical for both heads. Any change is applied to all variants and both heads. **Fixed in advance:**
   if any variant stays at chance level (about 41%, always "diploid") in the speed test, the backbone
   learning rate becomes 1e-4 for every variant and both heads (`--extra "--lr-backbone 1e-4"`). On the
   laptop at batch size 8, ConvNeXt-Tiny at 384 px collapsed at 3e-4 but reached 94.3% after two
   epochs at 1e-4; ResNet18, ResNet50 and ConvNeXt-Tiny at 224 px learned normally at 3e-4.
   `run_workstation_plan.py --speed-test` flags such a variant.
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
| ~~Noise robustness~~ under simulated hardware noise | Whether the accuracy holds on realistic near-term quantum hardware. | **Done** (November 2026) with the noise of IBM Fez, Torino and Brisbane: accuracy 93.56%, 93.56% and 93.11% vs 93.56% exact (at most 2 of 450 predictions change). See the README, section "Quantum hardware test". |
| **Real quantum hardware:** run test-set inference on an 8-qubit device (e.g. the IBM Quantum free tier) | Whether the model works outside simulation. Would be a novel result for this application. | **Planned. Code ready and tested up to submission** (`09_quantum_hardware.py ibm`); needs an IBM Quantum account. Start with a 10-image trial to see the quota use. |
| ~~Larger or deeper circuits:~~ more qubits (via 16 angles) or repeated convolution layers | Whether circuit capacity limits accuracy. | **Not planned** (decided November 2026). |

A careful negative result ("no advantage under a fair comparison") is also publishable, particularly
at quantum machine learning venues.

## C. Literature and writing (done, November 2026)

Later papers on the same 3,000 images (Altuntaş et al., 2019) report the following. Accuracy as reported
by the authors; the evaluation protocols differ, so the figures are not directly comparable.

| Study | Method | Evaluation | Accuracy (%) |
|---|---|---|---|
| Altuntaş, Cömert & Kocamaz (2019), *Comput. Electron. Agric.* 163 | VGG-19, transfer learning | single split (about 900 test images) | 94.22 |
| Dönmez (2020), *Celal Bayar Univ. J. Sci.* 16(3) | ResNet50 features + SVM | 10-fold CV | 91.4 |
| Dönmez (2022), *Data & Knowledge Eng.* 141 | features of six CNNs, selection and fusion, SVM | not checked | 96.74 † |
| Setiawan & Pramudita (2023), ICONNSMAL 2022 | ResNet50, background removed | 80:20 train/test, no validation set | 98.16 |
| Dönmez et al. (2023), *Multimedia Systems* 29(6) | hybrid transformer | not checked | 96.33 † |
| Ayaz, Kutlu & Cömert (2024), *Agronomy Journal* 116(3) | DeepMaizeNet (CNN with attention) | 5-fold CV | 94.13 |
| Dönmez, Diker, Elen & Ulu (2024), *Scientia Horticulturae* 337 | majority vote of five CNNs | not checked | 90.96 † |
| Cömert et al. (2025), *Electronics* 14(13) | Slot-Maize (CNN, slot attention, GRU, LSTM) | 80:20 train/test † | 96.97 |
| This project, test set | ResNet18 + QCNN / classical head | 70/15/15 split, 450 test images | 93.33 / 94.22 |
| This project, cross-validation | ResNet18 + QCNN or classical head | 5-fold CV, 2,550 non-test images | 95.3 |

† Not checked against the full text (publisher pages were not accessible): taken from an abstract
excerpt or as cited by Cömert et al. (2025). Check these before submitting the paper (deferred to the end
of the plan: it needs college library access).

- **What it means:** Altuntaş et al. is not the best published result. Four later studies report more
  than both of our test results (96–98%), but under different protocols; the highest (98.16%) has no
  separate validation set, so its test images probably also guided model selection. The QCNN vs
  classical conclusion is unaffected.
- **Done in the research summary** (`Research_Summary_Haploid_Diploid_QCNN.docx` and `.pdf`, outside the
  repository; the previous version is kept as `..._2026-10-19`): the later studies are cited in the
  introduction and discussion with a new Table 7 and 7 references; the comparison is worded as
  "comparable to the original VGG-19 baseline, below the 96–98% of several later studies"; the
  limitations and conclusion include the cross-validation results; and the comparison caveats
  (different splits, putative R1-nj labels) are stated.

## Plan (decided November 2026)

Remaining: the workstation training, the real-hardware run and, last, checking three accuracy figures
from the literature. Everything else above is not planned, including any further work on accuracy
beyond the workstation plan.

1. ~~A: 30 epochs plus 5-fold CV for both heads~~ (done, November 2026; about 2 h of training on the
   GTX 1650 for both 30 and 12 epochs).
2. **A: the workstation plan above**, when the workstation is available. The code is ready and was
   tested on the laptop (all backbones, parallel augmentation, the CUDA 13.0 build). bf16 mixed
   precision could not be tested on real hardware (the GTX 1650 has no bf16 support); the first smoke
   run on the workstation confirms it.
3. ~~C: literature and writing~~ (done, November 2026; see section C). Three accuracy figures still
   need checking against the full papers before submission.
4. **B: real quantum hardware.** Inference only, with the final QCNN from step 2 (or the reported
   one). The simulated-noise part is done for the reported QCNN (accuracy holds); the code for the
   real device is ready. Needs an IBM Quantum account: a 10-image trial first, then all 450 images.
   If step 2 produces a new final model, export it with `08_export_qcnn_inputs.py --variant ...` and
   repeat the noisy simulation before the real run.
5. **C: check the three unverified accuracy figures** in Table 7 of the research summary
   (Dönmez 2022, 2023, 2024) against the full papers, when college library access is available.
   Deferred to the end.
