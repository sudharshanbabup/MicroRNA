# Negative-Control Invariant Concept Erasure (NICE) for Shortcut-Robust Serum-miRNA Detection

This repository contains the official code for Negative-Control Invariant Concept Erasure (**NICE**), a shortcut-robust detection framework designed for serum-miRNA detection of pancreatic cancer.

When cases are collected exclusively via a hospital biobank ($h=1$) and controls through volunteer cohorts ($h=0$), standard classifiers learn clinical collection artifacts rather than true biological signals. NICE leverages hospital sera from patients with non-target conditions (negative controls) to:
1. Identify the clinical collection pathway direction exclusively on the disease-free stratum (conditional concept erasure).
2. Fit a pathway- and class-balanced detector on the erased feature representation.
3. Establish worst-group conformal thresholds to guarantee false-positive rate control across distinct clinical pathways.

---

## Repository Structure

All necessary scripts are provided in the root directory:

* `prepare_data.py`: Downloads public NCBI GEO series and compiles the unified dataset (`data/mirna_dataset.npz`).
* `nice.py`: Implements NICE (conditional concept eraser, balanced ridge-logistic detector, worst-group conformal thresholding, shortcut audit).
* `baselines.py`: Implementation of comparison baselines (ERM-LR, XGBoost, MLP, JTT, OvR, Matched, RWG, Group DRO, IRM, V-REx, DANN, Deep CORAL, INLP).
* `run_experiments.py`: Real-data validation protocols (`cv`, `lonto`, `cross`, `abl`, `mneg`).
* `semi_synthetic.py`: Semi-synthetic stress test with controlled shortcut injection in real hospital sera.
* `theory_sim.py`: Monte-Carlo simulations verifying closed-form analytical results.
* `aggregate.py`: Computes summary statistics and tables from experiment result logs (`results/summary.json`).

---

## Dataset

The analysis utilizes four public NCBI GEO series based on the 3D-Gene human miRNA microarray platform:
* **Series A**: GSE59856 (Kojima et al., *PLoS ONE* 2015)
* **Series B**: GSE106817 (Yokoi et al., *Nat. Commun.* 2018)
* **Series C**: GSE113486 (Usuba et al., *Cancer Sci.* 2019)
* **Series D**: GSE124158 (Asano et al., *Nat. Commun.* 2019)

`prepare_data.py` downloads the series matrix files, extracts the 2,540 common miRNA probes, labels each serum (PC: Pancreatic Cancer, NC: Non-Cancer control, OC: Other Cancer hospital control, BEN: Benign disease control), and excludes duplicated patient samples.

---

## Requirements & Setup

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
* Python $\ge$ 3.8
* `numpy`
* `scipy`
* `pandas`
* `torch`
* `xgboost`

---

## Reproducing Experiments

1. **Download and prepare the dataset**:
   ```bash
   python prepare_data.py
   ```

2. **Execute experimental protocols**:
   ```bash
   # Stratified 5-fold cross-validation (20 repeats)
   python run_experiments.py cv 20

   # Leave-one-negative-type-out (5 repeats)
   python run_experiments.py lonto 5

   # Cross-study evaluation (20 repeats)
   python run_experiments.py cross 20

   # Ablation study (20 repeats)
   python run_experiments.py abl 20

   # Sample efficiency of negative controls (10 repeats)
   python run_experiments.py mneg 10

   # Semi-synthetic stress test on study B (20 repeats)
   python semi_synthetic.py 20 B

   # Theoretical Monte-Carlo simulations (2000 draws)
   python theory_sim.py 2000
   ```

3. **Aggregate results**:
   ```bash
   python aggregate.py
   ```
