# ADMET Prediction Pipeline Setup & Execution Guide

This repository contains a fully automated, data-driven pipeline for predicting molecular ADMET properties using an audited suite of RDKit 2D descriptors, structural Morgan Fingerprints (contracted via UMAP), and optimized XGBoost models. 

Follow these steps to reproduce the exact training environment, run validation audits, and launch the interactive prediction interface.

---

## 🛠️ 1. Prerequisites & Environment Setup

Ensure you have **Python 3.10 or 3.11** installed on your system. 

1. **Clone or Open the Repository Directory:**
   ```bash
   cd KBG-MootKaPaaniTeamToofani/
   ```

2. **Create a Clean Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   * **Windows (Command Prompt / PowerShell):**
     ```bash
     venv\Scripts\activate
     ```
   * **macOS / Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install Required Packages:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🔄 2. Verification & Execution Sequence

Because the final, fully-trained production models are already pre-packaged inside the `models/` directory, **you can skip directly to Section 3 to launch the app.** 

However, if you wish to audit, verify, or entirely retrain the pipeline from scratch, execute the python scripts in the following chronological order:

### [OPTIONAL] Step A: Run the Feature Audit
This script downloads datasets dynamically from the Therapeutic Data Commons (TDC), calculates the entire pool of ~200 RDKit 2D properties, runs a cross-validated Information Gain audit using XGBoost, and filters out noise by keeping descriptors clearing a strict 0.1% importance threshold.
```bash
python full_descriptor_audit.py
```
* **Expected Output:** Generates a `scaffold_dataset/` directory containing `descriptor_keep_lists.json` and granular `*_descriptor_importance.csv` metric rankings for every chemical endpoint category.

### [OPTIONAL] Step B: Validate the Robustness Framework
This pipeline parses the mathematical keep-lists generated in Step A, sets up a robust 5-Fold Murcko Scaffold Split (ensuring no structural leakage between training and validation sets), applies target transformations (such as log1p handling and assay ceiling clipping), and computes cross-validation performance.
```bash
python master_scaffold_pipeline.py
```
* **Expected Output:** Terminal output logging fold-by-fold metrics (R² / RMSE for regression, ROC-AUC / AUPRC for classification). It outputs and saves the finalized trimmed feature matrices (`*_X_trimmed.npy` and `*_y_trimmed.npy`) to disk.


## 🖥️ 3. Launching the Prediction Interface

Once your production models are successfully serialized, you can use the graphical interface to check properties on completely new, raw inputs.

```bash
python app.py
```
* **How to evaluate:** The command will output a local network address (typically `http://127.0.0.1:7860`). Open this link in any browser, input a validated **SMILES string** (e.g., `CC(=O)NC1=CC=C(O)C=C1` for Acetaminophen), and hit submit to view the predicted physical and toxicological parameters computed live.