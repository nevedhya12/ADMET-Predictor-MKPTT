# ADMET Property Prediction from SMILES

Predicts ADMET properties (Absorption, Distribution, Metabolism, Excretion,
Toxicity) from SMILES strings using Morgan fingerprints + RDKit physicochemical
descriptors, trained with XGBoost under scaffold-split cross-validation.

---

## 1. Overview

This project aims to solve the problem of high drug attrition rates and expensive,
time-consuming wet-lab ADMET experiments by screening candidate compounds
virtually before they are synthesized.

---

## 2. Final Results

Held-out set: 5-fold **scaffold split** cross-validation (Murcko scaffolds),
so every molecule is evaluated exactly once, always on a fold containing
structurally distinct scaffolds from its training set. Metrics match the
brief exactly: ROC-AUC/AUPRC for classification, R2/RMSE for regression.

|    Endpoin   |    Task Type   |      Metric 1       |        Metric 2    | Held-out Set Size |
|--------------|----------------|---------------------|--------------------|-------------------|
| Absorption   | Regression     | R2 = **0.632**      | RMSE = **0.460**   |               910 |  
| Distribution | Classification | ROC-AUC = **0.905** | AUPRC = **0.963**  |             2,030 | 
| Metabolism   | Classification | ROC-AUC = **0.876** | AUPRC = **0.773**  |            12,092 |
| Excretion    | Regression     | R2 = **0.027**      | RMSE = **49.03**   |             1,213 |
| Toxicity     | Classification | ROC-AUC = **0.841** | AUPRC = **0.905**  |               655 |
### Per-fold breakdown

<details>
<summary>Absorption (R2 / RMSE)</summary>

| Fold | R2 | RMSE | Size |
|------|----|------|------|
| 0 | 0.726 | 0.419 | 182 |
| 1 | 0.516 | 0.471 | 182 |
| 2 | 0.666 | 0.442 | 182 |
| 3 | 0.702 | 0.475 | 182 |
| 4 | 0.549 | 0.494 | 182 |

</details>

<details>
<summary>Distribution (ROC-AUC / AUPRC)</summary>

| Fold | ROC-AUC | AUPRC | Size |
|---|---|---|---|
| 0 | 0.880 | 0.940 | 406 |
| 1 | 0.871 | 0.956 | 406 |
| 2 | 0.923 | 0.966 | 406 |
| 3 | 0.918 | 0.971 | 406 |
| 4 | 0.933 | 0.983 | 406 |

</details>

<details>
<summary>Metabolism (ROC-AUC / AUPRC)</summary>

| Fold | ROC-AUC | AUPRC | Size |
|---|---|---|---|
| 0 | 0.875 | 0.768 | 2,419 |
| 1 | 0.880 | 0.747 | 2,419 |
| 2 | 0.872 | 0.780 | 2,418 |
| 3 | 0.890 | 0.785 | 2,418 |
| 4 | 0.861 | 0.783 | 2,418 |

</details>

<details>
<summary>Excretion (R2 / RMSE)</summary>

| Fold | R2 | RMSE | Size |
|---|---|---|---|
| 0 | 0.017 | 47.17 | 243 |
| 1 | -0.022 | 49.38 | 243 |
| 2 | -0.029 | 51.55 | 243 |
| 3 | 0.097 | 49.21 | 242 |
| 4 | 0.073 | 47.85 | 242 |

</details>

<details>
<summary>Toxicity (ROC-AUC / AUPRC)</summary>

| Fold | ROC-AUC | AUPRC | Size |
|---|---|---|---|
| 0 | 0.886 | 0.940 | 131 |
| 1 | 0.901 | 0.926 | 131 |
| 2 | 0.830 | 0.934 | 131 |
| 3 | 0.755 | 0.823 | 131 |
| 4 | 0.833 | 0.903 | 131 |

</details>

---

## 3. Data & Splitting Strategy

### 3.1 Datasets used

All five are ADME-Tendpoints pulled from the Therapeutics Data Commons
(TDC) `single_pred` module.

### 3.2 Why scaffold split (not random or cluster-based)

Random splitting lets near, identical and structurally trivial variants of the
same molecule land in both train and test, silently inflating scores by
letting the model "recognize" close relatives rather than generalize.
Scaffold splitting (via Murcko scaffolds) groups molecules by their core ring
system first, then distributes whole scaffold groups across different folds. So a test
fold can contain molecules whose *scaffold* was never seen in training. This
makes the model learn the underlying chemistry instead of "cheating by leaking", which matches 
TDC's recommended benchmark.

**This was not my first approach** — see the Iteration Log (Section 6) for
why we initially used UMAP-compressed-fingerprint-K-means-clustering instead,
and why we ultimately abandoned it for Murcko scaffold split.

### 3.3 Known / addressed leakage risks

- **UMAP and StandardScaler are currently on the full dataset
  (train + test combined) rather than within each fold.** This is known and
  unresolved leakage risk: a test molecule's UMAP embedding position and
  descriptor scaling are technically informed by having seen the full
  dataset, including itself and other test molecules, before any fold split
  happens. In practice this is unlikely to hand the model the literal
  answer (unlike target leakage would), but it likely makes reported scores
  mildly optimistic versus true held-out generalization. **Not fixed by
  submission time** due to time constraints — see Future Work. The correct
  fix is fitting UMAP/StandardScaler on `X_train` only per fold and calling
  `.transform()` (not `.fit_transform()`) on `X_test`.

---

## 4. Feature Engineering

### 4.1 Molecular fingerprints

Morgan/ECFP fingerprint, radius = 3, 2048 bits, same across all five endpoints.
And why radius = 3? Before concluding radius we looped over {1, 2, 3, 4, 5} radius per endpoint with scaffold folds and descriptors held identical across radii. The resulting score spread across radii was smaller than the fold-to-fold variance already present. Therefore, indicating radius is not a meaningful bottleneck for our model. and thus from the above set of 5 radius I kept radius=3 as a single shared value rather than different radii per-endpoint, since per-endpoint "optimal" radii from the looping are not clearly distinguishable from per-fold noise.

### 4.2 Physicochemical descriptors

Initially I only used 15 descriptors based on domain expertise but after performing math it came to 
conclusion that from over 200+ available descriptors, per-endpoint had like ~100 descriptors sharing a fair portion of importance.

### 4.3 Dimensionality reduction

Fingerprint (2048 bits) → UMAP (`metric='jaccard'`) → **10 dimensions**.

This value was chosen via running the model over different `n_components: {3, 5, 10, 20, 30,
50, 75, 100}`, measuring both **trustworthiness** (This metric is calculated by checking 
how much of the initial neighbours remained neighours after compression) and downstream weighted CV score. Trustworthiness flattened almost immediately across this range of (0.85–0.96 regardless of
dimensionality for every endpoint), indicating that the fingerprint's real
structural information lives in low dimension and thus wheather compressing to 3 or 100 dimensions made
negligible difference to information. So I decided to go with 10.

### 4.4 Feature pruning / noise reduction

After an initial full-descriptor run, per-endpoint XGBoost feature importance
was audited and descriptors contributing negligible (~0%) information gain
for a given endpoint were dropped **for that endpoint only**

---

## 5. Modeling Approach

- **Model:** XGBoost
  1. `XGBClassifier` for classification endpoints,
  2. `XGBRegressor` for regression endpoints.
- **Key hyperparameters:** 
  1. `n_estimators=100` 
  2. `learning_rate=0.1`
  3. `max_depth=6` 
  4. `random_state=42`
- **Class imbalance handling:** not yet applied — Metabolism's AUPRC/ROC-AUC
  gap suggests this may be worth adding (`scale_pos_weight`); see Future
  Work.
- **Target transforms:** regression targets are automatically log1p-transformed
  if their skew exceeds 1.0 (data driven decision, not hardcoded per
  endpoint Absorption's skew of -0.67 did not qualify, Excretion's skew of
  1.24 did). Predictions are inverse-transformed (`expm1`) back to original
  units before scoring, so all reported R2/RMSE are in real units.

---

## 6. Iteration Log
## **RUN-1 To RUN-4 Metrics are R2 & Accuracy, only from RUN-5 onwards metrics given in the brief used**

### Pre-RUN-1 — Initial design decisions
**Initialization:** Chose the TDC ADME-T dataset track over two alternative datasets provided in the biref. Initial splitting strategy: compress the 2048-bit fingerprint via UMAP, K-means cluster into 5 "chemical families" per endpoint, use clusters as CV folds.

### RUN-1 — Baseline: K-means clustering on raw 2048→3 UMAP fingerprint
**Pipeline:** First working pipeline, fingerprint only, no descriptors, UMAP
compressed to 3D, K-means (K=5) clustering used as CV folds.
**Result:** 
  => For Absorption it was 	-00.23
  => For Distribution it was 	 69.20%
  => For Metabolism it was	 66.69%
  => For Excretion it was 	-00.07
  => For Toxicity it was 	 67.67%
**Takeaway:** Regressors were highly sensitive to small clusters. created by clustering
**Post-Run Changes:** (1) more informative features beyond binary fingerprint bits, (2) finding
optimal choice of UMAP compression dimensionality instead of an assumed 2048→3.

### RUN-2 — Added 15 RDKit descriptors
**PipeLine:** Added 15 RDkit descriptors (these were chosen based on domain expertize), standard scaled these and concatenated with a optimal UMAP finding i.e. (3 → 10 dimensions barely affects information retained).
**Result:**
  => For Absorption it was 	-00.23
  => For Distribution it was 	 76.90%
  => For Metabolism it was	 75.79%
  => For Excretion it was 	-00.13
  => For Toxicity it was 	 76.64%
**Takeaway:** Classifiers improved significantly from descriptors, but regressors were still 
struggling, here I thought maybe clustering size that I chose earlier (K=5) is not optimal and as 
it was globally applied rather each database having their own cluster size.
**Post-Run Changes:** (1) Made a script to run the model and assess metrics based on different Ks
with multiseed averaging to find the best cluster size for each dataset.

### RUN-3 — Cluster count (K) tuning, validated across multiple random seeds
**PipeLine:** Looped over K: {3,...,10} per endpoint instead of assuming K=5.
Re-Assessed the "optimal K" findings across 5 different KMeans random
seeds after suspecting that may be the first result was a fluke of
`random_state=42`: 
only Absorption showed a stable optimal K with std = 0 across seeds, 
most other endpoints showed high variance, so only Absorption/Distribution/Metabolism 
K values were revised.
	- Absorption   = 5 -> 4
	- Distribution = 5 -> 5
	- Metabolism   = 5 -> 4
	- Excretion    = 5 -> 5
	- Toxicity     = 5 -> 5
**Result:**
  => For Absorption it was 	-00.13
  => For Distribution it was 	 76.90%
  => For Metabolism it was	 75.84%
  => For Excretion it was 	-00.13
  => For Toxicity it was 	 76.64%
**Takeaway:** Got a optimal K value for endpoints improving the results slightly.
**Post-Run Changes:** Trimmed off descriptors with less than 0.1% importance and also found the optimal 
fingerprint radius by ablation.

### RUN-4 — Per-endpoint descriptor pruning + fingerprint radius tuning
**PiepLine:** Audited per endpoint XGBoost feature importance and dropped
less than 0.1% importance giving descriptors per endpoint.
Tuned Morgan fingerprint radius (2 → 3) by ablation (found out that this was also not affecting the results too much so I just changed it from 2 to 3 for very slight improvement).
**Result:** 
  => For Absorption it was 	-00.06
  => For Distribution it was 	 76.85%
  => For Metabolism it was	 75.84%
  => For Excretion it was 	-00.10
  => For Toxicity it was 	 76.64%
**Takeaway:** Descriptor pruning gave a small Absorption improvement;
radius change had negligible effect. and because regressors scrores were so low, I though checking out
other models and found out that scaffold split is actually the most widely used splitting and thought
of giving it a chance
**Post-Run Changes:** Implemented scaffold splitting and removed the whole clustering folds.

### RUN-5 — Pivoted to scaffold split (Murcko scaffolds), full metric rework
**PipeLine:** Replaced K-means cluster-based CV entirely with scaffold split
(the standard approach). Also switched reported metrics to match the
brief exactly (ROC-AUC/AUPRC for classification, R2/RMSE for regression,
replacing the earlier accuracy/R2 metrics), added a single-class-fold
guard for undefined ROC-AUC/AUPRC cases, added log1p transform for skewed datasets (only needed for excreation)
**Result:** 
  => For Absorption it was 	 00.63
  => For Distribution it was 	 88.12%
  => For Metabolism it was	 81.30%
  => For Excretion it was 	 00.10
  => For Toxicity it was 	 81.83%
**Takeaway:** Scaffold split was the single largest driver of improvement
across every endpoint. The earlier hypothesis that unstable,
small-sized K-means clusters were the primary problem through RUN-1–4.
Investigated Excretion's remaining low R2 and 
found ~11.3% of its target values are censored at an assay detection
ceiling (150), making it to be a hard data-quality ceiling rather than a
fixable modeling issue (see Limitations).
**Post-Run Changes:** Rather using 15 RDkit descriptors I will once loop through all the 200 RDkit descriptors and check importance of all of them, incase I might ignore or leave a important descriptor for a specific endpoint.

### RUN-6 — Looped over all the available descriptors of RDKit to find the useful descriptor per endpoint
**PipeLine:** Before this run I was using only 15 descriptors based on domain expertise but math said otherwise and it came out that over 100 descriptors out of 200+ were playing a important role per endpoint and results were astonishing
**Result:** 
  => For Absorption it was 	 00.65
  => For Distribution it was 	 91.46%
  => For Metabolism it was	 89.01%
  => For Excretion it was 	 00.09
  => For Toxicity it was 	 85.91%
**Takeaway:** Using all the descriptors and trimming less important once from the full set was the right choice, and helped in capturing and generalizing the underlying chemistry, overfitting here is not a problem at all because of how we are splitting.
---

## 7. Limitations & Honest BackLogs

- **Global (not per-fold) UMAP/StandardScaler fitting.** Both are currently
  fit on the entire dataset before the CV loop, rather than fit on
  `X_train` and applied via `.transform()` to `X_test` per fold. This is a
  known, unresolved minor leakage risk, likely makes reported scores
  slightly optimistic versus true held-out generalization, though it does
  not hand the model direct answer information the way target leakage
  would. Not fixed by submission time.
- **Excretion's target is right-censored.** ~11.3% of Excretion (clearance) 
  values sit at exactly the dataset's observed maximum (150),
  consistent with an assay detection limit rather than a genuine measured
  value. This caps achievable R2/RMSE regardless of model quality, a
  perfect model can only output "150" for these molecules even if the true
  clearance was much higher, and will still be scored as wrong against the
  recorded (capped) ground truth. Log1p transform and prediction-clipping
  at the ceiling were applied to reduce per fold loop instability from this
  issue, but do not and cannot recover information that was never
  measured.
- **Small dataset sizes produce real fold-to-fold variance under scaffold
  split**, most visibly in Toxicity (655 molecules; ROC-AUC ranges from
  0.755 to 0.933 across its 5 folds) and Excretion (1,213 molecules). This
  is an honest property of scaffold-based evaluation on smaller datasets,
  not a bug, some scaffolds are inherently harder to generalize to than
  others, and small datasets have fewer scaffolds to average that
  difficulty over.
- **Metabolism's AUPRC (0.773) trails its ROC-AUC (0.876) by a notable
  margin**, a common signal of class imbalance (AUPRC's random baseline
  equals positive-class prevalence, while ROC-AUC's does not). Class
  balance has not yet been investigated or addressed for this endpoint.

---

## 8. Future Work

- **Fit UMAP and StandardScaler within each CV fold** rather than globally,
  which may cause slight leakage risk noted in Section 7. Deprioritized due to
  time, required restructuring the pipeline to fit fold-by-fold and
  increases UMAP fit count 5x (25 fits instead of 5 across all endpoints).
- **HDBSCAN as an alternative/complementary clustering approach** for
  identifying structurally atypical compounds (peptides, macrocycles) that
  don't fit any dense chemical cluster, rather than forcing them into the
  nearest one. Considered as a replacement to the K-means outlier-cluster
  problem seen in RUN-1–3, deprioritized after implementing scaffold split (RUN-5) solved
  the same underlying instability more directly and with less implementation
  risk under time pressure.
- **Count-based fingerprints** (`GetCountFingerprint` instead of binary
  bits) to capture repeated substructures (e.g. peptide backbone repeats)
  that binary fingerprints collapse to a single "present" flag. Requires
  switching UMAP's distance metric from `jaccard` (binary-only) to
  something magnitude-aware like `manhattan`, and re-validating the
  dimensionality ablation under the new representation, deprioritized for time.
- **Pretrained molecular embeddings** (e.g. ChemBERTa, MolCLR) as an
  additional or alternative feature block, pretrained on millions of
  diverse molecules, likely to generalize better to structurally unusual
  compounds than hand-built descriptors/fingerprints alone. but I thought that would be
  out of scope and also didn't had lot of time.

---

## 9. Repository Structure & Reproduction

<!-- TODO: confirm this matches your final repo layout exactly before submitting -->

```
.
├── data/
├── processed_data/              # legacy K-means-based pipeline outputs (RUN-1–4)
├── scaffold_dataset/            # final scaffold-split pipeline outputs (RUN-5)
├── featureConversionMain.py     # SMILES -> fingerprint extraction (legacy)
├── compress_and_cluster.py      # UMAP + K-means clustering (legacy, superseded)
├── add_descriptors.py           # RDKit descriptor generation + combination (legacy)
├── optimalCompressionDimension.py   # UMAP n_components ablation
├── OptimalClusterSize.py        # K-means cluster count ablation (legacy)
├── train_xgboost.py             # legacy training script (K-means CV)
├── master_scaffold_pipeline.py  # FINAL pipeline: scaffold split + descriptors + XGBoost
└── README.md
```

**To reproduce final results:**
```bash
python master_scaffold_pipeline.py
```

<!-- TODO: add any environment setup / requirements.txt instructions -->

---

## 10. Team / Acknowledgments

<!-- TODO: add team member names -->

Built with: [Therapeutics Data Commons (TDC)](https://tdcommons.ai/), RDKit,
XGBoost, UMAP, scikit-learn.